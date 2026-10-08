# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Core placement follows lowered physical tasks, not legacy CombineCV."""

import pytest
import tilelang  # noqa: F401
from tilelang import language as T
from tvm import tir


def fixture():
    gm = tir.decl_buffer((256,), "float16", name="gm", scope="global")
    l1 = tir.decl_buffer((16, 16), "float16", name="l1", scope="shared.l1")
    ub = tir.decl_buffer((256,), "float16", name="ub", scope="shared.ub")
    cube = tir.Evaluate(
        tir.call_extern("handle", "copy_gm_to_l1<half,16,16>", gm.access_ptr("r"),
                        l1.access_ptr("w"), 16, 0, 0))
    vector = tir.Evaluate(T.tile.fill(ub, 1))
    owner = tir.For(
        tir.Var("epoch", "int32"), 0, 2, tir.ForKind.SERIAL, tir.SeqStmt([cube, vector]))
    root = tir.Block([], [], [], "root", owner, alloc_buffers=[l1, ub])
    return tir.PrimFunc([], tir.BlockRealize([], True, root)), cube, vector


def test_mixed_owner_keeps_neutral_cut_and_wraps_actual_operations():
    from tilelang.transform.ascend_schedule.core_scopes import normalize_core_scopes
    from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks

    function, cube, vector = fixture()
    result = normalize_core_scopes(function, extract_tasks(function))
    owner = result.body.block.body
    assert isinstance(owner, tir.For)
    for node, code, original in zip(owner.body.seq, (0, 1), (cube, vector)):
        assert node.attr_key == "resource_scope" and int(node.value) == code
        assert node.body.same_as(original)
    extraction = extract_tasks(result)
    assert [task.core for task in extraction.tasks] == ["C", "V"]
    assert all(context[0].same_as(owner) for context in extraction.loops.values())
    assert normalize_core_scopes(result, extraction).same_as(result)


def test_single_core_wrap_is_outside_owner_to_preserve_flat_loop_contract():
    from tilelang.transform.ascend_schedule.core_scopes import normalize_core_scopes
    from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks

    _, _, vector = fixture()
    owner = tir.For(tir.Var("epoch", "int32"), 0, 2, tir.ForKind.SERIAL, vector)
    function = tir.PrimFunc([], owner)
    result = normalize_core_scopes(function, extract_tasks(function))
    assert result.body.attr_key == "resource_scope" and int(result.body.value) == 1
    assert result.body.body.same_as(owner)


def test_single_core_scope_is_inside_participant_binding_and_executes_both_lanes():
    import numpy as np
    import tvm
    from tilelang.transform.ascend_schedule.entry import schedule_module
    from tilelang.simulator import (build_kernel_program, FunctionalSimulator, SimulatorConfig,
                                    BufferRegion, MemoryScope)

    ub = tir.decl_buffer((8,), "float32", name="lane_value", scope="shared.ub")
    axis = tir.IterVar(None, tir.Var("threadIdx_x", "int32"), tir.IterVar.ThreadIndex,
                       "threadIdx.x")
    root = tir.Block([], [], [],
                     "root",
                     tir.Evaluate(T.tile.fill(ub, axis.var + 1)),
                     alloc_buffers=[ub])
    body = tir.AttrStmt(axis, "thread_extent", 2, tir.BlockRealize([], True, root))
    function = tir.PrimFunc([], body).with_attr("npu_platform", "A2")
    scheduled = schedule_module(
        tvm.IRModule({"main": function}), "ascendc", {"tl.ascend_auto_schedule": True})["main"]
    assert scheduled.body.attr_key == "thread_extent"
    sim = FunctionalSimulator(
        build_kernel_program(scheduled, platform="A2"), SimulatorConfig(platform="A2"))
    sim.run()
    for index, owner in enumerate((0, sim.memory.vector1_owner(0))):
        np.testing.assert_array_equal(
            sim.read(BufferRegion("lane_value", MemoryScope.UB, (8,), "float32", core_id=owner)),
            np.full(8, index + 1))


def test_single_core_cannot_scope_across_separate_participant_groups():
    from tilelang.transform.ascend_schedule.core_scopes import normalize_core_scopes
    from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks

    _, _, vector = fixture()
    groups = []
    for index in (0, 1):
        axis = tir.IterVar(None, tir.Var(f"vid_{index}", "int32"), tir.IterVar.ThreadIndex,
                           "threadIdx.x")
        # Distinct Evaluate identities even when they call the same operation.
        groups.append(tir.AttrStmt(axis, "thread_extent", 2, tir.Evaluate(vector.value)))
    function = tir.PrimFunc([], tir.SeqStmt(groups))
    with pytest.raises(ValueError, match="common participant binding"):
        normalize_core_scopes(function, extract_tasks(function))


def test_opposing_nested_scope_and_duplicate_tasks_are_not_normalized_away():
    from tilelang.transform.ascend_schedule.core_scopes import normalize_core_scopes
    from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks

    function, _, _ = fixture()
    extraction = extract_tasks(function)
    contradictory = function.with_body(
        tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, function.body))
    with pytest.raises(ValueError, match="contradict"):
        normalize_core_scopes(contradictory, extraction)
    _, cube, _ = fixture()
    duplicate = tir.PrimFunc([], tir.SeqStmt([cube, cube]))
    with pytest.raises(ValueError, match="unique"):
        normalize_core_scopes(duplicate, extract_tasks(duplicate))


@pytest.mark.parametrize("pointer_scalar", [False, True])
def test_immediate_tensor_scalar_reads_fail_closed(pointer_scalar):
    from tilelang.transform.ascend_schedule.core_scopes import normalize_core_scopes
    from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks

    ub = tir.decl_buffer((8,), "float32", name="ub", scope="shared.ub")
    scalar = ub.access_ptr("r") if pointer_scalar else tir.BufferLoad(ub, [0])
    args = [ub.access_ptr("w"), ub.access_ptr("r"), scalar]
    if pointer_scalar:
        args.append(0)
    args.append(8)
    call = tir.call_intrin("handle", tir.op.Op.get("tl.ascend_muls"), *args)
    function = tir.PrimFunc([], tir.Evaluate(call))
    with pytest.raises(ValueError, match="scalar tensor read"):
        normalize_core_scopes(function, extract_tasks(function))
