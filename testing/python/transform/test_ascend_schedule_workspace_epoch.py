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
    "alias_view", "sibling_source", "dynamic_scope", "extra_axis", "source_offset",
    "source_elem_offset", "source_shape", "invalid_relu"
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
    elif bad in ("dynamic_scope", "extra_axis"):
        if bad == "dynamic_scope":
            wrap = lambda n: tir.AttrStmt(
                tir.IntImm("int32", 0), "resource_scope", tir.Var("dynamic_core", "int32"), n)
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


def test_relu_does_not_change_full_write_coverage():
    from tilelang.transform.ascend_schedule.workspace_epoch import certify_workspace_epoch_write
    function = prepared(3)
    buffer, call = writer(function, "C")
    args = list(call.args)
    args[0] = tir.StringImm(args[0].value.replace("false>", "true>"))
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
