# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Materialize certified iteration clocks into actual allocated TIR storage."""

import pytest
import tilelang  # noqa: F401
from tvm import tir, arith
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
