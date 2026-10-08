# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Collective write coverage is group-relative, never whole-world initialization."""

import pytest
from tvm import tir

from test_ascend_schedule_cv_stage_waves import fixture, owner_of


def prepared(groups):
    # Reuse the real workspace lowering fixture, changing its grid BEFORE
    # lowering so its generated allocation and offsets reflect that grid.
    return fixture("A2", 3, groups=groups)[0]


def writer(function, core):
    from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks
    extraction = extract_tasks(owner_of(function).body)
    task = next(t for t in extraction.tasks
                if t.core == core and t.pipe == ("PIPE_FIX" if core == "C" else "PIPE_MTE3"))
    call = extraction.task_nodes[task.id].value
    buffer = next(
        view for view in function.buffer_map.values() if view.data.same_as(call.args[2].args[1]))
    return buffer, call


@pytest.mark.parametrize("groups", [1, 3])
@pytest.mark.parametrize("core", ["C", "V"])
def test_real_workspace_writer_certifies_partitions_not_full_world(groups, core):
    from tilelang.transform.ascend_schedule.workspace_epoch import certify_workspace_epoch_write
    function = prepared(groups)
    buffer, call = writer(function, core)
    proof = certify_workspace_epoch_write(function, buffer, call, core)
    assert proof.group_count == groups
    assert proof.group_pitch_elements == 256
    assert proof.version_pitch_elements == groups * 256
    assert proof.group_byte_extent == 512
    assert proof.participant_intervals == (((0, 512),) if core == "C" else ((0, 256), (256, 512)))
    assert proof.call.same_as(call) and proof.pointer.same_as(call.args[2])
    # The typed write certificate does not set Access.full_overwrite or
    # authorize arbitrary readers from another cid partition.
    assert not hasattr(proof, "full_overwrite")


@pytest.mark.parametrize(
    "bad",
    ["offset", "stride", "source_mask", "guard", "vid_offset", "participants", "plain_input"])
def test_partial_or_unowned_collective_write_is_rejected(bad):
    from tilelang.transform.ascend_schedule.workspace_epoch import certify_workspace_epoch_write
    function = prepared(3)
    core = "V" if bad in ("vid_offset", "participants") else "C"
    buffer, call = writer(function, core)
    if bad == "plain_input":
        function = function.with_attr("auto_gm_indices", [])
    elif bad == "guard":
        function = function.with_body(
            tir.stmt_functor.ir_transform(
                function.body, None, lambda n: tir.IfThenElse(tir.Var("enabled", "bool"), n, None)
                if isinstance(n, tir.Evaluate) and n.value.same_as(call) else None))
    elif bad == "participants":
        function = function.with_body(
            tir.stmt_functor.ir_transform(
                function.body, None, lambda n: tir.AttrStmt(n.node, n.attr_key, 1, n.body)
                if isinstance(n, tir.AttrStmt) and n.attr_key == "thread_extent" and n.node.
                thread_tag == "threadIdx.x" else None))
    else:
        args = list(call.args)
        if bad == "stride":
            args[3] = tir.IntImm("int32", 17)
        elif bad == "source_mask":
            pointer = args[1]
            args[1] = tir.Call(pointer.dtype, pointer.op,
                               [*pointer.args[:4], tir.IntImm("int32", 2)])
        else:
            pointer = args[2]
            offset = pointer.args[2] + 1
            if bad == "vid_offset":
                axes = []
                tir.stmt_functor.post_order_visit(
                    function.body, lambda n: axes.append(n.node.var)
                    if isinstance(n, tir.AttrStmt) and n.attr_key == "thread_extent" and n.node.
                    thread_tag == "threadIdx.x" else None)
                offset = pointer.args[2] - 128 * axes[0]
            args[2] = tir.Call(pointer.dtype, pointer.op,
                               [*pointer.args[:2], offset, *pointer.args[3:]])
        changed = tir.Call(call.dtype, call.op, args)
        function = function.with_body(
            tir.stmt_functor.ir_transform(function.body, None, lambda n: changed
                                          if n.same_as(call) else None))
        call = changed
    with pytest.raises(ValueError, match="workspace"):
        certify_workspace_epoch_write(function, buffer, call, core)


@pytest.mark.parametrize("bad", [
    "alias_view", "sibling_source", "dynamic_scope", "extra_axis", "malformed_axis",
    "source_offset", "source_elem_offset", "source_shape", "invalid_relu"
])
def test_uncertified_physical_geometry_and_ancestors_are_rejected(bad):
    from tilelang.transform.ascend_schedule.workspace_epoch import certify_workspace_epoch_write
    function = prepared(3)
    buffer, call = writer(function, "C")
    if bad == "alias_view":
        alias = tir.decl_buffer(buffer.shape, buffer.dtype, data=buffer.data, name="alias")
        handle = tir.Var("alias_handle", "handle")
        function = tir.PrimFunc([*function.params, handle],
                                function.body,
                                buffer_map={
                                    **dict(function.buffer_map), handle: alias
                                },
                                attrs=function.attrs)
    elif bad in ("dynamic_scope", "extra_axis", "malformed_axis"):
        if bad == "dynamic_scope":
            wrap = lambda n: tir.AttrStmt(
                tir.IntImm("int32", 0), "resource_scope", tir.Var("dynamic_core", "int32"), n)
        elif bad == "malformed_axis":
            wrap = lambda n: tir.AttrStmt(tir.Var("bad_axis", "int32"), "thread_extent", 2, n)
        else:
            axis = tir.IterVar(None, tir.Var("unknown_lane", "int32"), tir.IterVar.ThreadIndex,
                               "threadIdx.y")
            wrap = lambda n: tir.AttrStmt(axis, "thread_extent", 2, n)
        function = function.with_body(
            tir.stmt_functor.ir_transform(
                function.body, None, lambda n: wrap(n)
                if isinstance(n, tir.Evaluate) and n.value.same_as(call) else None))
    elif bad in ("source_offset", "invalid_relu"):
        args = list(call.args)
        if bad == "source_offset":
            pointer = args[1]
            args[1] = tir.Call(
                pointer.dtype, pointer.op,
                [*pointer.args[:2], tir.IntImm("int32", 1), *pointer.args[3:]])
        else:
            args[0] = tir.StringImm(args[0].value.replace("false>", "unknown>"))
        changed = tir.Call(call.dtype, call.op, args)
        function = function.with_body(
            tir.stmt_functor.ir_transform(function.body, None, lambda n: changed
                                          if n.same_as(call) else None))
        call = changed
    else:

        def change_allocation(node):
            if not isinstance(node, tir.Block):
                return None
            source = next(
                (view for view in node.alloc_buffers if view.data.same_as(call.args[1].args[1])),
                None)
            if source is None:
                return None
            allocations = [view for view in node.alloc_buffers if not view.same_as(source)]
            body = node.body
            if bad == "sibling_source":
                sibling = tir.BlockRealize([], True,
                                           tir.Block([], [], [],
                                                     "sibling",
                                                     tir.Evaluate(0),
                                                     alloc_buffers=[source]))
                body = tir.SeqStmt([sibling, body])
            else:
                source = tir.decl_buffer(
                    (32, 8) if bad == "source_shape" else source.shape,
                    source.dtype,
                    name=source.name,
                    data=source.data,
                    scope=source.scope(),
                    elem_offset=1 if bad == "source_elem_offset" else 0)
                allocations.append(source)
            return tir.Block(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                body,
                init=node.init,
                alloc_buffers=allocations,
                match_buffers=node.match_buffers,
                annotations=node.annotations)

        function = function.with_body(
            tir.stmt_functor.ir_transform(function.body, None, change_allocation))
    with pytest.raises(ValueError, match="workspace"):
        certify_workspace_epoch_write(function, buffer, call, "C")


@pytest.mark.parametrize("value", ["true", "false", "0", "1"])
def test_relu_does_not_change_full_write_coverage(value):
    from tilelang.transform.ascend_schedule.workspace_epoch import certify_workspace_epoch_write
    function = prepared(3)
    buffer, call = writer(function, "C")
    args = list(call.args)
    args[0] = tir.StringImm(args[0].value.replace("false>", value + ">"))
    changed = tir.Call(call.dtype, call.op, args)
    function = function.with_body(
        tir.stmt_functor.ir_transform(function.body, None, lambda n: changed
                                      if n.same_as(call) else None))
    assert certify_workspace_epoch_write(function, buffer, changed, "C").group_pitch_elements == 256


@pytest.mark.parametrize("bad", ["unaligned_ub", "aliased_axes", "conditional_source"])
def test_source_alignment_and_participant_identity_are_required(bad):
    from tilelang.transform.ascend_schedule.workspace_epoch import certify_workspace_epoch_write
    function = prepared(3)
    core = "V" if bad == "unaligned_ub" else "C"
    buffer, call = writer(function, core)
    if bad == "unaligned_ub":
        # Supply enough capacity so this fails on alignment, not bounds.
        def enlarge(node):
            if not isinstance(node, tir.Block):
                return None
            allocations = []
            for view in node.alloc_buffers:
                if view.data.same_as(call.args[1].args[1]):
                    view = tir.decl_buffer((256,),
                                           view.dtype,
                                           data=view.data,
                                           name=view.name,
                                           scope=view.scope())
                allocations.append(view)
            return tir.Block(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                node.body,
                init=node.init,
                alloc_buffers=allocations,
                match_buffers=node.match_buffers,
                annotations=node.annotations)

        function = function.with_body(tir.stmt_functor.ir_transform(function.body, None, enlarge))
        args = list(call.args)
        source = args[1]
        args[1] = tir.Call(
            source.dtype, source.op,
            [*source.args[:2], tir.IntImm("int32", 1), *source.args[3:]])
        changed = tir.Call(call.dtype, call.op, args)
        function = function.with_body(
            tir.stmt_functor.ir_transform(function.body, None, lambda n: changed
                                          if n.same_as(call) else None))
        call = changed
    elif bad == "conditional_source":
        function = function.with_body(
            tir.stmt_functor.ir_transform(
                function.body, None,
                lambda n: tir.BlockRealize(n.iter_values, tir.Var("active", "bool"), n.block)
                if isinstance(n, tir.BlockRealize) else None))
    else:
        axes = []
        tir.stmt_functor.post_order_visit(
            function.body, lambda n: axes.append(n.node.var) if isinstance(n, tir.AttrStmt) and n.
            attr_key == "thread_extent" and n.node.thread_tag == "blockIdx.x" else None)

        def alias_axis(node):
            if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent" and \
                    node.node.thread_tag == "threadIdx.x":
                axis = tir.IterVar(None, axes[0], tir.IterVar.ThreadIndex, "threadIdx.x")
                return tir.AttrStmt(axis, node.attr_key, node.value, node.body)
            return None

        function = function.with_body(
            tir.stmt_functor.ir_transform(function.body, None, alias_axis))
    with pytest.raises(ValueError, match="workspace"):
        certify_workspace_epoch_write(function, buffer, call, core)


@pytest.mark.parametrize("cols,groups", [(17, 3), (16, 1 << 24)])
def test_dma_row_pitch_and_group_address_overflow_fail_closed(cols, groups):
    from tilelang.transform.ascend_schedule.workspace_epoch import certify_workspace_epoch_write
    cid, vid = tir.Var("cid", "int32"), tir.Var("vid", "int32")
    pitch = 16 * cols
    gm = tir.decl_buffer((tir.IntImm("int64", groups * pitch),), "float16", name="scratch")
    ub = tir.decl_buffer((8, cols), "float16", name="source", scope="shared.ub")
    call = tir.call_extern("handle", f"tl::ascend::copy_ub_to_gm<half,{cols},8>",
                           ub.access_ptr("r"),
                           gm.access_ptr("w",
                                         offset=cid * pitch + vid * (pitch // 2)), cols, 8, cols)
    body = tir.BlockRealize([], True,
                            tir.Block([], [], [],
                                      "root",
                                      tir.AttrStmt(
                                          tir.IntImm("int32", 1), "resource_scope", 1,
                                          tir.Evaluate(call)),
                                      alloc_buffers=[ub]))
    for var, tag, count in ((vid, "threadIdx.x", 2), (cid, "blockIdx.x", groups)):
        axis = tir.IterVar(None, var, tir.IterVar.ThreadIndex, tag)
        body = tir.AttrStmt(axis, "thread_extent", count, body)
    handle = tir.Var("scratch_handle", "handle")
    function = tir.PrimFunc([handle], body, buffer_map={
        handle: gm
    }).with_attr("auto_gm_indices", [0])
    with pytest.raises(ValueError, match="workspace"):
        certify_workspace_epoch_write(function, gm, call, "V")


def test_narrowing_source_address_cast_checks_preconversion_bounds():
    from tilelang.transform.ascend_schedule.workspace_epoch import certify_workspace_epoch_write
    function = prepared(256)
    buffer, call = writer(function, "V")
    axes = []
    tir.stmt_functor.post_order_visit(
        function.body, lambda n: axes.append(n.node.var) if isinstance(n, tir.AttrStmt) and n.
        attr_key == "thread_extent" and n.node.thread_tag == "blockIdx.x" else None)

    def enlarge(node):
        if not isinstance(node, tir.Block):
            return None
        allocations = [
            tir.decl_buffer(
                (65536,), view.dtype, data=view.data, name=view.name, scope=view.scope())
            if view.data.same_as(call.args[1].args[1]) else view for view in node.alloc_buffers
        ]
        return tir.Block(
            node.iter_vars,
            node.reads,
            node.writes,
            node.name_hint,
            node.body,
            init=node.init,
            alloc_buffers=allocations,
            match_buffers=node.match_buffers,
            annotations=node.annotations)

    function = function.with_body(tir.stmt_functor.ir_transform(function.body, None, enlarge))
    args = list(call.args)
    source = args[1]
    args[1] = tir.Call(
        source.dtype, source.op,
        [*source.args[:2], tir.Cast("int16", axes[0] * 256), *source.args[3:]])
    changed = tir.Call(call.dtype, call.op, args)
    function = function.with_body(
        tir.stmt_functor.ir_transform(function.body, None, lambda n: changed
                                      if n.same_as(call) else None))
    with pytest.raises(ValueError, match="overflow"):
        certify_workspace_epoch_write(function, buffer, changed, "V")


def reader_pair(function, core):
    from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks
    extraction = extract_tasks(owner_of(function).body)
    task = next(t for t in extraction.tasks if t.core == core and t.pipe == "PIPE_MTE2")
    read = extraction.task_nodes[task.id].value
    buffer = next(
        view for view in function.buffer_map.values() if view.data.same_as(read.args[1].args[1]))
    writes = [
        extraction.task_nodes[t.id].value
        for t in extraction.tasks
        if t.pipe in ("PIPE_FIX", "PIPE_MTE3")
    ]
    write = next(call for call in writes if call.args[2].args[1].same_as(buffer.data))
    return buffer, write, read


@pytest.mark.parametrize("core", ["C", "V"])
@pytest.mark.parametrize("groups", [1, 3])
def test_real_workspace_readers_share_writer_group_and_owner(core, groups):
    from tilelang.transform.ascend_schedule.workspace_epoch import (certify_workspace_epoch_read,
                                                                    certify_workspace_epoch_write)
    function = prepared(groups)
    buffer, write, read = reader_pair(function, core)
    proof = certify_workspace_epoch_write(function, buffer, write, "V" if core == "C" else "C")
    result = certify_workspace_epoch_read(function, proof, read, core)
    assert result.owner.same_as(owner_of(function))
    assert result.writer.call.same_as(write)
    assert result.pointer.same_as(read.args[1])
    assert result.participant_intervals == (((0, 512),) if core == "C" else ((0, 256), (256, 512)))


@pytest.mark.parametrize("bad", [
    "other_group", "other_owner", "guard", "partial", "stride", "left_pad", "local_offset",
    "local_role"
])
def test_workspace_reader_must_match_partition_and_full_native_geometry(bad):
    from tilelang.transform.ascend_schedule.workspace_epoch import (certify_workspace_epoch_read,
                                                                    certify_workspace_epoch_write)
    function = prepared(3)
    buffer, write, read = reader_pair(function, "V")
    proof = certify_workspace_epoch_write(function, buffer, write, "C")
    if bad in ("other_owner", "guard"):

        def wrap(node):
            if isinstance(node, tir.Evaluate) and node.value.same_as(read):
                if bad == "guard":
                    return tir.IfThenElse(tir.Var("active", "bool"), node, None)
                return tir.For(tir.Var("other_epoch", "int32"), 0, 3, tir.ForKind.SERIAL, node)
            return None

        function = function.with_body(tir.stmt_functor.ir_transform(function.body, None, wrap))
    else:
        args = list(read.args)
        if bad == "other_group":
            pointer = args[1]
            args[1] = tir.Call(pointer.dtype, pointer.op,
                               [*pointer.args[:2], pointer.args[2] + 256, *pointer.args[3:]])
        elif bad in ("local_offset", "local_role"):
            pointer = args[2]
            fields = list(pointer.args)
            fields[2 if bad == "local_offset" else 4] = tir.IntImm("int32", 1)
            args[2] = tir.Call(pointer.dtype, pointer.op, fields)
        elif bad == "partial":
            args[4] = tir.IntImm("int32", 7)
        elif bad == "stride":
            args[3] = tir.IntImm("int32", 17)
        else:
            assert len(args) == 7  # padValue is present; leftPad is not.
            args.append(tir.IntImm("int32", 1))
        changed = tir.Call(read.dtype, read.op, args)
        function = function.with_body(
            tir.stmt_functor.ir_transform(function.body, None, lambda n: changed
                                          if n.same_as(read) else None))
        read = changed
    with pytest.raises(ValueError, match="workspace"):
        certify_workspace_epoch_read(function, proof, read, "V")


@pytest.mark.parametrize("core", ["C", "V"])
def test_storage_certificate_covers_every_pointer_and_orders_first_write(core):
    from tilelang.transform.ascend_schedule.workspace_epoch import certify_workspace_epoch
    function = prepared(3)
    buffer, write, read = reader_pair(function, core)
    certificate = certify_workspace_epoch(function, buffer)
    assert certificate.writer.call.same_as(write)
    assert len(certificate.readers) == 1
    assert certificate.readers[0].call.same_as(read)
    assert certificate.owner.same_as(owner_of(function))


@pytest.mark.parametrize(
    "bad", ["unknown_reader", "second_writer", "direct_load", "read_first", "handle_escape"])
def test_storage_certificate_rejects_unaudited_access_and_previous_epoch_reads(bad):
    from tilelang.transform.ascend_schedule.workspace_epoch import certify_workspace_epoch
    function = prepared(3)
    buffer, write, read = reader_pair(function, "V")
    owner = owner_of(function)
    statements = list(owner.body.seq)
    if bad == "unknown_reader":
        unknown = tir.Evaluate(
            tir.call_extern("handle", "unknown_workspace_consumer", buffer.access_ptr("r")))
        statements.append(tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, unknown))
    elif bad == "second_writer":
        statements.append(statements[0])
    elif bad == "direct_load":
        statements.append(
            tir.Evaluate(tir.BufferLoad(buffer, [tir.IntImm("int32", 0) for _ in buffer.shape])))
    elif bad == "handle_escape":
        handle = next(handle for handle, view in function.buffer_map.items()
                      if view.data.same_as(buffer.data))
        assert not handle.same_as(buffer.data)
        statements.append(
            tir.Evaluate(tir.call_extern("handle", "unknown_workspace_escape", handle)))
    else:
        statements[0], statements[1] = statements[1], statements[0]
    changed = tir.For(
        owner.loop_var,
        owner.min,
        owner.extent,
        owner.kind,
        tir.SeqStmt(statements),
        thread_binding=owner.thread_binding,
        annotations=owner.annotations)
    function = function.with_body(
        tir.stmt_functor.ir_transform(function.body, None, lambda n: changed
                                      if n.same_as(owner) else None))
    with pytest.raises(ValueError, match="workspace"):
        certify_workspace_epoch(function, buffer)


@pytest.mark.parametrize("endpoint", ["write", "read"])
def test_expression_conditional_endpoint_does_not_prove_unconditional_coverage(endpoint):
    from tilelang.transform.ascend_schedule.workspace_epoch import certify_workspace_epoch
    function = prepared(3)
    buffer, write, read = reader_pair(function, "V")
    selected = write if endpoint == "write" else read
    conditional = tir.if_then_else(
        tir.Var("enabled", "bool"), selected, tir.call_extern("handle", "noop"))
    function = function.with_body(
        tir.stmt_functor.ir_transform(function.body, None, lambda n: conditional
                                      if n.same_as(selected) else None))
    with pytest.raises(ValueError, match="workspace"):
        certify_workspace_epoch(function, buffer)


def test_reduction_initialization_is_not_an_every_epoch_writer():
    from tvm import ir
    from tilelang.transform.ascend_schedule.workspace_epoch import certify_workspace_epoch
    function = prepared(3)
    buffer, write, _ = reader_pair(function, "V")
    owner = owner_of(function)
    axis = tir.IterVar(ir.Range(0, 3), tir.Var("reduce_epoch", "int32"), tir.IterVar.CommReduce)

    def wrap(node):
        if isinstance(node, tir.Evaluate) and node.value.same_as(write):
            block = tir.Block([axis], [], [], "reduction", tir.Evaluate(0), init=node)
            return tir.BlockRealize([owner.loop_var], True, block)
        return None

    function = function.with_body(tir.stmt_functor.ir_transform(function.body, None, wrap))
    with pytest.raises(ValueError, match="workspace"):
        certify_workspace_epoch(function, buffer)


@pytest.mark.parametrize("kind", ["unbound_int64", "uint64_multiply"])
def test_unknown_integer_bounds_are_not_finite_address_proofs(kind):
    from tvm import arith, ir
    from tilelang.transform.ascend_schedule.workspace_epoch import _checked_address_arithmetic
    analyzer = arith.Analyzer()
    if kind == "unbound_int64":
        expression = tir.Var("unknown", "int64")
    else:
        cid = tir.Var("large_cid", "uint64")
        analyzer.bind(cid, ir.Range(0, 5))
        scale = tir.IntImm("uint64", 1 << 62)
        expression = ((cid * scale) // scale) * tir.IntImm("uint64", 256)
    with pytest.raises(ValueError, match="overflow|unbounded"):
        _checked_address_arithmetic(expression, analyzer)


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("sq", [128, 130])
@pytest.mark.parametrize("kv", [192, 130])
@pytest.mark.parametrize("causal", [False, True])
def test_native_fa_scheduler_input_has_certified_group_workspaces(platform, sq, kv, causal):
    """Actual native input geometry, not a full scheduler/hardware acceptance test."""
    from unittest.mock import patch
    import tvm
    from examples.flash_attention_auto128.fa_auto128 import (FlashAttentionConfig,
                                                             flash_attention_auto128_fwd,
                                                             pass_configs)
    from tilelang.engine.phase import LowerAndLegalize
    from tilelang.transform.ascend_schedule.conditions import normalize_conditions
    from tilelang.transform.ascend_schedule.core_scopes import normalize_core_scopes
    from tilelang.transform.ascend_schedule.gemm_decomposition import decompose_function_gemms
    from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks
    from tilelang.transform.ascend_schedule.uniform_clock import certify_uniform_clock
    from tilelang.transform.ascend_schedule.workspace_epoch import certify_workspace_epoch
    from tilelang.transform.ascend_schedule.workspace_tasks import prepare_cross_core_copies

    function = flash_attention_auto128_fwd.__wrapped__(
        FlashAttentionConfig(1, 1, sq, kv, 128, causal=causal)).with_attr("npu_platform", platform)
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    # Observe the real scheduler input before it creates slots, flags and waves.
    with patch("tilelang.transform.ascend_schedule.entry.schedule_module", lambda mod, *_: mod), \
            tvm.transform.PassContext(config=pass_configs):
        mod = prepare_cross_core_copies(LowerAndLegalize(tvm.IRModule({"main": function}), target))
    function = normalize_conditions(decompose_function_gemms(mod["main"], root_scratch=True))
    function = normalize_core_scopes(function, extract_tasks(function))
    extraction = extract_tasks(function)
    assert not extraction.scalar_bindings and not any(task.guard for task in extraction.tasks)
    generated = {int(index) for index in function.attrs["auto_gm_indices"]}
    proofs = [
        certify_workspace_epoch(function, function.buffer_map[param])
        for index, param in enumerate(function.params)
        if index in generated
    ]
    assert len(proofs) == 3
    assert {proof.writer.group_byte_extent for proof in proofs} == {8192, 4096, 16384}
    for proof in proofs:
        assert proof.writer.group_count == (sq + 31) // 32
        assert proof.writer.version_pitch_elements == proof.writer.group_pitch_elements * proof.writer.group_count
        assert len(proof.readers) == 1
        assert proof.owner.same_as(proofs[0].owner)
        certify_uniform_clock(function, proof.owner)
    from tilelang.transform.ascend_schedule.workspace_planning import (discover_workspace_epochs,
                                                                       plan_workspace_epochs)
    discovery = discover_workspace_epochs(function, extraction)
    assert not discovery.diagnostics and len(discovery.candidates) == 3
    assert all(
        candidate.clock is discovery.candidates[0].clock for candidate in discovery.candidates)
    for candidate in discovery.candidates:
        stages = {task.id: 0 for task in extraction.tasks}
        for task_id in candidate.reader_ids:
            stages[task_id] = 1
        # Per-workspace lifecycle/preflight only, not a globally executable
        # FA assignment or a claim that its other dependencies allow this map.
        plan = plan_workspace_epochs(candidate, stages)
        assert plan.eligible and plan.count == 2
        assert candidate.size_bytes == candidate.group_bytes * candidate.certificate.writer.group_count
        assert not any(access.full_overwrite
                       for task in discovery.tasks
                       for access in task.accesses
                       if access.storage is candidate.storage)
