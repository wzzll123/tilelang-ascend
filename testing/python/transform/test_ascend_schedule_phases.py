# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Prefix initialization and suffix output must not become per-iteration flags."""

import pytest
import numpy as np
import tilelang  # noqa: F401
import tilelang.language as T
import tvm
from tvm import tir


@pytest.mark.parametrize("iterations", [0, 1, 3])
@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_persistent_accumulator_survives_all_three_phases(iterations, platform):
    from tilelang.transform.ascend_schedule.phases import emit_three_phase_sync
    from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks
    from tilelang.simulator import (build_kernel_program, FunctionalSimulator, SimulatorConfig,
                                    BufferRegion, MemoryScope, MemoryHazardError,
                                    SimulationDeadlockError)

    acc = tir.decl_buffer((8,), "float32", name="acc", scope="shared.ub")
    ones = tir.decl_buffer((8,), "float32", name="ones", scope="shared.ub")
    output = tir.decl_buffer((8,), "float32", name="output", scope="global")
    handle = tir.Var("output_handle", "handle")
    add = tir.Evaluate(
        tir.call_intrin("handle", tir.op.Op.get("tl.ascend_add"), acc.access_ptr("w"),
                        acc.access_ptr("r"), ones.access_ptr("r"), 8))
    loop = tir.For(tir.Var("epoch", "int32"), 3, iterations, tir.ForKind.SERIAL, add)
    body = tir.SeqStmt([
        tir.Evaluate(T.tile.fill(acc, 0)),
        tir.Evaluate(T.tile.fill(ones, 1)), loop,
        tir.Evaluate(
            tir.call_extern("handle", "copy_ub_to_gm<float,8>", acc.access_ptr("r"),
                            output.access_ptr("w"), 8))
    ])
    root = tir.Block([], [], [],
                     "root",
                     tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, body),
                     alloc_buffers=[acc, ones])
    function = tir.PrimFunc([handle], tir.BlockRealize([], True, root), buffer_map={
        handle: output
    }).with_attr("npu_platform", platform)
    component = emit_three_phase_sync(function, extract_tasks(function))
    from tilelang.transform.ascend_schedule.entry import schedule_module
    result = schedule_module(
        tvm.IRModule({"main": function}), "ascendc", {"tl.ascend_auto_schedule": True})["main"]
    assert int(result.attrs["tl.ascend.schedule_materialized"]) == 1
    assert tvm.ir.structural_equal(component.body, result.body)
    if iterations == 1:
        from tilelang.transform.ascend_schedule.entry import AutoScheduleError
        first = body.seq[0]

        def duplicate_statement(node):
            if isinstance(node, tir.Evaluate) and isinstance(
                    node.value, tir.Call) and node.value.op.name == "tir.call_extern":
                return first
            return None

        duplicate = function.with_body(
            tir.stmt_functor.ir_transform(function.body, None, duplicate_statement))
        with pytest.raises(AutoScheduleError, match="unique physical node identities"):
            schedule_module(
                tvm.IRModule({"main": duplicate}), "ascendc", {"tl.ascend_auto_schedule": True})

        def misplaced_allocation(node):
            if isinstance(node, tir.Block) and node.name_hint == "root":
                sibling = tir.BlockRealize([], True,
                                           tir.Block([], [], [],
                                                     "sibling",
                                                     tir.Evaluate(0),
                                                     alloc_buffers=[acc]))
                return tir.Block([], [], [],
                                 "root",
                                 tir.SeqStmt([sibling, node.body]),
                                 alloc_buffers=[ones])
            return None

        wrong_lifetime = function.with_body(
            tir.stmt_functor.ir_transform(function.body, None, misplaced_allocation))
        with pytest.raises(AutoScheduleError, match="allocation must dominate"):
            schedule_module(
                tvm.IRModule({"main": wrong_lifetime}), "ascendc",
                {"tl.ascend_auto_schedule": True})
    for sync_only in (False, True):
        sim = FunctionalSimulator(
            build_kernel_program(result, platform=platform),
            SimulatorConfig(platform=platform, sync_only=sync_only, flag_balance_check="error"))
        sim.run()
        if not sync_only:
            np.testing.assert_array_equal(
                sim.read(BufferRegion("output", MemoryScope.GM, (8,), "float32")),
                np.full(8, iterations))
    assert [int(dim) for dim in result.body.block.alloc_buffers[0].shape] == [8]
    from tilelang.engine.phase import OptimizeForTarget
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    mod = tir.transform.BindTarget(target)(
        tvm.IRModule({"probe": result.with_attr("global_symbol", "probe")}))
    with tvm.transform.PassContext(config={"tl.ascend_auto_schedule": True}):
        mod = OptimizeForTarget(mod, target, platform)
    source = tvm.get_global_func("target.build.tilelang_ascend")(mod, target, platform).get_source()
    assert "SetFlag<" in source and "WaitFlag<" in source

    def remove_wait(node):
        if isinstance(node, tir.Evaluate) and isinstance(
                node.value, tir.Call) and node.value.op.name == "tl.ascend_wait_flag":
            return tir.Evaluate(0)
        return None

    mutated = result.with_body(tir.stmt_functor.ir_transform(result.body, None, remove_wait))
    from dataclasses import replace
    program = build_kernel_program(mutated, platform=platform)
    # An adversarial timing probe, not a claim about calibrated hardware cost.
    program = replace(
        program,
        cores=tuple(
            replace(
                core,
                tasks=tuple(
                    replace(task, duration_cycles=1000) if task.metadata.get("dst") is not None and
                    task.metadata["dst"].buffer in ("acc", "ones") else task
                    for task in core.tasks))
            for core in program.cores))
    with pytest.raises((MemoryHazardError, SimulationDeadlockError)):
        FunctionalSimulator(program, SimulatorConfig(platform=platform, sync_only=True)).run()
