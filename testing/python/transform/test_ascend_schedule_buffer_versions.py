# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Materialize certified iteration clocks into actual allocated TIR storage."""

import pytest
import tilelang  # noqa: F401
from tvm import tir, arith, ir
from tilelang.transform.ascend_schedule.ownership import EpochPlan, OwnerEpochPlan
from tilelang.transform.ascend_schedule.buffer_versions import materialize_iteration_versions


def fixture():
    buffer = tir.decl_buffer((16, 16), "float16", name="scratch", scope="wmma.matrix_a")
    i = tir.Var("i", "int32")
    pointer = buffer.access_ptr("w", offset=8)
    body = tir.Evaluate(tir.call_extern("handle", "probe", pointer))
    loop = tir.For(i, 3, 5, tir.ForKind.SERIAL, body)
    function = tir.PrimFunc([],
                            tir.Allocate(buffer.data, buffer.dtype, buffer.shape,
                                         tir.const(True, "bool"), loop))
    plan = EpochPlan(buffer.data, True, 2, (OwnerEpochPlan(loop, "iteration", object()),))
    return buffer, loop, function, plan


def test_expands_storage_and_offsets_every_access_by_owner_relative_epoch():
    buffer, loop, function, plan = fixture()
    result = materialize_iteration_versions(function, buffer, plan)
    assert [int(x) for x in result.body.extents] == [32, 16]
    ptr = result.body.body.body.value.args[1]
    assert ptr.args[1].same_as(buffer.data)
    for i in range(3, 8):
        offset = tir.stmt_functor.substitute(ptr.args[2], {loop.loop_var: tir.IntImm("int32", i)})
        assert int(arith.Analyzer().simplify(offset)) == 8 + ((i - 3) % 2) * 256
    assert int(ptr.args[3]) == 248  # Per-access extent is not allocation capacity.


def test_pre_flatten_root_block_allocation_is_versioned_and_certified():
    buffer, loop, function, plan = fixture()
    sibling = tir.decl_buffer((8,), "float32", name="sibling", scope="shared.ub")
    root = tir.Block([], [], [], "root", loop, alloc_buffers=[buffer, sibling])
    function = function.with_body(tir.BlockRealize([], True, root))
    result = materialize_iteration_versions(function, buffer, plan)
    enlarged, preserved = result.body.block.alloc_buffers
    assert [int(x) for x in enlarged.shape] == [32, 16]
    assert enlarged.data.same_as(buffer.data)
    assert preserved.same_as(sibling)
    certificate = result.attrs["tl.ascend.iteration_versions"][buffer.data]
    assert [int(certificate[0]), int(certificate[1])] == [2, 256]
    assert certificate[2].same_as(loop.loop_var)


def flattened_l1_fixture(tail=64, unsupported=False):
    buffer = tir.decl_buffer((8192,), "float16", name="kv", scope="shared.l1")
    gm = tir.decl_buffer((128, 128), "float16", name="gm")
    l0 = tir.decl_buffer((8192,), "float16", name="l0", scope="wmma.matrix_b")
    epoch = tir.Var("epoch", "int32")
    write = tir.Evaluate(
        tir.call_extern("handle", "tl::ascend::copy_gm_to_l1<half,64,128>", gm.access_ptr("r"),
                        buffer.access_ptr("w"), 128, tail, 128, 64, 128))
    read = tir.Evaluate(
        tir.call_extern("handle", "probe" if unsupported else "copy_l1_to_l0b<half,64,128,true>",
                        buffer.access_ptr("r"), l0.access_ptr("w"), 128, 64))
    owner = tir.For(epoch, 0, 3, tir.ForKind.SERIAL, tir.SeqStmt([write, read]))
    function = tir.PrimFunc([],
                            tir.BlockRealize([], True,
                                             tir.Block([], [], [],
                                                       "root",
                                                       owner,
                                                       alloc_buffers=[buffer])))
    plan = EpochPlan(buffer.data, True, 2, (OwnerEpochPlan(owner, "iteration", object()),))
    return buffer, owner, function, plan


def test_flattened_l1_ring_uses_native_geometry_and_rewrites_both_endpoints():
    buffer, owner, function, plan = flattened_l1_fixture()
    result = materialize_iteration_versions(function, buffer, plan)
    root = result.body.block
    assert tuple(map(int, root.alloc_buffers[0].shape)) == (16384,)
    write, read = root.body.body.seq
    pointers = (write.value.args[2], read.value.args[1])
    for epoch in range(3):
        offsets = [
            int(arith.Analyzer().simplify(
                tir.stmt_functor.substitute(pointer.args[2],
                                            {owner.loop_var: tir.IntImm("int32", epoch)})))
            for pointer in pointers
        ]
        assert offsets == [epoch % 2 * 8192] * 2
    assert all(int(pointer.args[3]) == 8192 for pointer in pointers)


@pytest.mark.parametrize("case", ["tail", "unknown_reader"])
def test_flattened_l1_geometry_is_revalidated_not_trusted_from_epoch_plan(case):
    buffer, _, function, plan = flattened_l1_fixture(
        tail=32 if case == "tail" else 64, unsupported=case == "unknown_reader")
    with pytest.raises(ValueError, match="L1 epoch"):
        materialize_iteration_versions(function, buffer, plan)


@pytest.mark.parametrize("case", ["malformed", "extra_pointer"])
def test_flattened_l1_cannot_skip_or_miscertify_pointer_occurrences(case):
    buffer, owner, function, _ = flattened_l1_fixture()
    write, read = owner.body.seq
    if case == "malformed":
        pointer = buffer.access_ptr("w")
        pointer = tir.Call(pointer.dtype, pointer.op, list(pointer.args[:-1]))
        write = tir.Evaluate(tir.call_extern("handle", "probe", pointer))
    else:
        args = list(write.value.args)
        args[6] = buffer.access_ptr("w")
        write = tir.Evaluate(tir.Call(write.value.dtype, write.value.op, args))
    owner = tir.For(owner.loop_var, owner.min, owner.extent, owner.kind, tir.SeqStmt([write, read]))
    function = function.with_body(
        tir.BlockRealize([], True, tir.Block([], [], [], "root", owner, alloc_buffers=[buffer])))
    plan = EpochPlan(buffer.data, True, 2, (OwnerEpochPlan(owner, "iteration", object()),))
    with pytest.raises(ValueError, match="malformed|DMA endpoint"):
        materialize_iteration_versions(function, buffer, plan)


def test_unrelated_root_regions_are_preserved_but_nested_target_views_are_rejected():
    buffer, owner, function, plan = flattened_l1_fixture()
    unrelated = tir.decl_buffer((8,), "float32", name="input")
    region = tir.BufferRegion(unrelated, [ir.Range(0, 8)])
    root = tir.Block([], [region], [], "root", owner, alloc_buffers=[buffer])
    function = function.with_body(tir.BlockRealize([], True, root))
    result = materialize_iteration_versions(function, buffer, plan)
    assert result.body.block.reads[0].same_as(region)
    target = tir.BufferRegion(buffer, [ir.Range(0, 8192)])
    nested = tir.BlockRealize([], True, tir.Block([], [target], [], "view", owner.body))
    owner = tir.For(owner.loop_var, owner.min, owner.extent, owner.kind, nested)
    function = function.with_body(
        tir.BlockRealize([], True, tir.Block([], [], [], "root", owner, alloc_buffers=[buffer])))
    plan = EpochPlan(buffer.data, True, 2, (OwnerEpochPlan(owner, "iteration", object()),))
    with pytest.raises(ValueError, match="block views"):
        materialize_iteration_versions(function, buffer, plan)


@pytest.mark.parametrize("case", ["inside_owner", "conditional", "logical_view", "sibling"])
def test_allocation_without_dominating_physical_lifetime_is_rejected(case):
    buffer, loop, function, plan = fixture()
    if case == "inside_owner":
        loop = tir.For(
            loop.loop_var, loop.min, loop.extent, loop.kind,
            tir.Allocate(buffer.data, buffer.dtype, buffer.shape, tir.const(True, "bool"),
                         loop.body))
        function = function.with_body(loop)
        plan = EpochPlan(buffer.data, True, 2, (OwnerEpochPlan(loop, "iteration", object()),))
    elif case == "conditional":
        enabled = tir.Var("enabled", "bool")
        function = function.with_body(
            tir.Allocate(buffer.data, buffer.dtype, buffer.shape, enabled, loop))
    elif case == "sibling":
        sibling = tir.BlockRealize([], True,
                                   tir.Block([], [], [],
                                             "sibling",
                                             tir.Evaluate(0),
                                             alloc_buffers=[buffer]))
        function = function.with_body(tir.SeqStmt([sibling, loop]))
    else:
        view = tir.BufferRegion(buffer, [ir.Range(0, 16), ir.Range(0, 16)])
        function = function.with_body(
            tir.BlockRealize([], True,
                             tir.Block([], [view], [], "root", loop, alloc_buffers=[buffer])))
    with pytest.raises(ValueError):
        materialize_iteration_versions(function, buffer, plan)


def test_narrow_offset_cannot_wrap_ring_slots_into_the_same_address():
    buffer = tir.decl_buffer((128,), "float16", name="scratch", scope="shared.ub")
    pointer = buffer.access_ptr("w")
    pointer = tir.Call(
        pointer.dtype, pointer.op,
        [*pointer.args[:2], tir.IntImm("uint8", 0), *pointer.args[3:]])
    loop = tir.For(
        tir.Var("epoch", "int32"), 0, 3, tir.ForKind.SERIAL,
        tir.Evaluate(tir.call_extern("handle", "probe", pointer)))
    function = tir.PrimFunc([],
                            tir.Allocate(buffer.data, buffer.dtype, buffer.shape,
                                         tir.const(True, "bool"), loop))
    plan = EpochPlan(buffer.data, True, 3, (OwnerEpochPlan(loop, "iteration", object()),))
    with pytest.raises(ValueError, match="offset arithmetic may overflow"):
        materialize_iteration_versions(function, buffer, plan)


@pytest.mark.parametrize("case", ["ineligible", "counter", "owner", "outside", "unaligned"])
def test_unproven_or_unsupported_versioning_fails_closed(case):
    buffer, loop, function, plan = fixture()
    if case == "ineligible":
        plan = EpochPlan(buffer.data, False, 1, plan.owners)
    elif case == "counter":
        plan = EpochPlan(buffer.data, True, 2, (OwnerEpochPlan(loop, "counter", object()),))
    elif case == "owner":
        plan = EpochPlan(buffer.data, True, 2, ())
    elif case == "outside":
        body = tir.SeqStmt([loop, loop.body])
        function = function.with_body(
            tir.Allocate(buffer.data, buffer.dtype, buffer.shape, tir.const(True, "bool"), body))
    elif case == "unaligned":
        buffer = tir.decl_buffer((17, 16), "float16", data=buffer.data, scope="wmma.matrix_a")
    with pytest.raises(ValueError):
        materialize_iteration_versions(function, buffer, plan)


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_native_planner_and_source_use_full_capacity_and_dynamic_slot(platform):
    import tvm

    buffer, loop, function, plan = fixture()
    source = tir.decl_buffer((16, 16), "float16", name="source", scope="shared.l1")
    copy = tir.Evaluate(
        tir.call_extern("handle", "tl::ascend::copy_l1_to_l0a<half,16,16,false>",
                        source.access_ptr("r"), buffer.access_ptr("w"), 16, 16))
    loop = tir.For(loop.loop_var, loop.min, loop.extent, loop.kind, copy)
    body = tir.Allocate(buffer.data, buffer.dtype, buffer.shape, tir.const(True, "bool"), loop)
    function = function.with_body(body)
    plan = EpochPlan(buffer.data, True, 2, (OwnerEpochPlan(loop, "iteration", object()),))
    function = materialize_iteration_versions(function, buffer, plan)
    body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, function.body)
    root = tir.Block([], [], [], "root", body, alloc_buffers=[source])
    function = function.with_body(tir.BlockRealize([], True, root)).with_attr(
        "global_symbol", "probe").with_attr("npu_platform", platform)
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    with tvm.transform.PassContext(config={"tl.ascend_memory_planning": True}):
        mod = tir.transform.BindTarget(target)(tvm.IRModule({"probe": function}))
        mod = tilelang.transform.AscendLowerOpaqueBlock()(mod)
        mod = tilelang.transform.Flatten2DBuffer()(mod)
        mod = tilelang.transform.FlattenBuffer()(mod)
        mod = tilelang.transform.AscendMemoryPlanning()(mod)
        generated = tvm.get_global_func("target.build.tilelang_ascend")(mod, target,
                                                                        platform).get_source()
    assert "scratch = ascend_l0a.GetWithOffset<half>(512," in generated
    assert "copy_l1_to_l0a<half,16,16,false>" in generated
    assert "256" in generated and "scratch[" in generated
    # Source-generation only: no unsynchronized kernel launch.


def test_raw_handle_escape_is_not_left_unversioned():
    buffer, loop, function, plan = fixture()
    body = tir.Evaluate(tir.call_extern("handle", "opaque_pointer_consumer", buffer.data))
    loop = tir.For(loop.loop_var, loop.min, loop.extent, loop.kind, body)
    function = function.with_body(
        tir.Allocate(buffer.data, buffer.dtype, buffer.shape, tir.const(True, "bool"), loop))
    plan = EpochPlan(buffer.data, True, 2, (OwnerEpochPlan(loop, "iteration", object()),))
    with pytest.raises(ValueError, match="raw buffer handles"):
        materialize_iteration_versions(function, buffer, plan)


def test_generated_gm_workspace_versions_keep_parameter_index_and_identity():
    workspace = tir.decl_buffer((16,), "float32", name="workspace", scope="global")
    handle = tir.Var("workspace_handle", "handle")
    i = tir.Var("epoch", "int32")
    loop = tir.For(i, 3, 5, tir.ForKind.SERIAL,
                   tir.Evaluate(tir.call_extern("handle", "probe", workspace.access_ptr("w"))))
    function = tir.PrimFunc([handle], loop, buffer_map={
        handle: workspace
    }).with_attr("auto_gm_indices", [0])
    plan = EpochPlan(workspace.data, True, 2, (OwnerEpochPlan(loop, "iteration", object()),))
    result = materialize_iteration_versions(function, workspace, plan)
    assert result.params[0].same_as(handle)
    assert result.buffer_map[handle].data.same_as(workspace.data)
    assert [int(x) for x in result.buffer_map[handle].shape] == [32]
    assert int(result.attrs["auto_gm_indices"][0]) == 0
    offset = result.body.body.value.args[1].args[2]
    for epoch in range(3, 8):
        bound = tir.stmt_functor.substitute(offset, {i: tir.IntImm("int32", epoch)})
        assert int(arith.Analyzer().simplify(bound)) == ((epoch - 3) % 2) * 16


def test_plain_gm_inputs_are_not_implicitly_versioned():
    workspace = tir.decl_buffer((16,), "float32", name="input", scope="global")
    handle = tir.Var("input_handle", "handle")
    loop = tir.For(
        tir.Var("epoch", "int32"), 0, 2, tir.ForKind.SERIAL,
        tir.Evaluate(tir.call_extern("handle", "probe", workspace.access_ptr("r"))))
    function = tir.PrimFunc([handle], loop, buffer_map={handle: workspace})
    plan = EpochPlan(workspace.data, True, 2, (OwnerEpochPlan(loop, "iteration", object()),))
    with pytest.raises(ValueError, match="generated scratch GM"):
        materialize_iteration_versions(function, workspace, plan)
