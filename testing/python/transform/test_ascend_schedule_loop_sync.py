# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Short-loop synchronization must initialize, recycle and drain credits."""

import types
import pytest
import tilelang  # noqa: F401
from tvm import tir
from tilelang.transform.ascend_schedule.model import Task, Dependency
from tilelang.transform.ascend_schedule.local_sync import emit_loop_sync, LocalSyncError


def fixture(extent):
    ub = tir.decl_buffer((8,), "float32", name="ub", scope="shared.ub")
    output = tir.decl_buffer((8,), "float32", name="output", scope="global")
    statements = [
        tir.Evaluate(tir.call_extern("handle", "tl.ascend_fill", ub.access_ptr("w"), 1.0, 8)),
        tir.Evaluate(
            tir.call_extern("handle", "copy_ub_to_gm", ub.access_ptr("r"), output.access_ptr("w"),
                            8)),
    ]
    extraction = types.SimpleNamespace(
        tasks=(Task(0, "PIPE_V", "V"), Task(1, "PIPE_MTE3", "V")),
        task_nodes=dict(enumerate(statements)),
        loops={
            0: (),
            1: ()
        },
        scalar_bindings=())
    loop = tir.For(tir.Var("i", "int32"), 0, extent, tir.ForKind.SERIAL, tir.SeqStmt(statements))
    return extraction, loop, ub, output


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("extent", [0, 1, 2, 5])
def test_short_loop_true_execution_and_final_balance(platform, extent):
    from tilelang.simulator import build_kernel_program, FunctionalSimulator, SimulatorConfig

    extraction, loop, ub, output = fixture(extent)
    result = emit_loop_sync(
        extraction, loop,
        [Dependency(0, 1, "RAW"), Dependency(1, 0, "WAR", 1)], event_ids=(0, 1))
    root = tir.Block([], [], [],
                     "root",
                     tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, result),
                     alloc_buffers=[ub])
    function = tir.PrimFunc([output.data],
                            tir.BlockRealize([], True, root),
                            buffer_map={output.data: output})
    program = build_kernel_program(function, platform=platform)
    FunctionalSimulator(
        program, SimulatorConfig(platform=platform, sync_only=True,
                                 flag_balance_check="error")).run()


def test_unsupported_distance_is_not_silently_discarded():
    extraction, loop, _, _ = fixture(2)
    with pytest.raises(LocalSyncError, match="distance"):
        emit_loop_sync(extraction, loop, [Dependency(1, 0, "WAR", 2)], event_ids=(0, 1))


def test_deleted_ready_wait_exposes_memory_hazard():
    from tilelang.simulator import build_kernel_program, MemoryHazardError
    from tilelang.simulator.sync import validate_memory_synchronization

    extraction, loop, ub, output = fixture(2)
    result = emit_loop_sync(
        extraction, loop,
        [Dependency(0, 1, "RAW"), Dependency(1, 0, "WAR", 1)], event_ids=(0, 1))

    def remove(node):
        if (isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call) and
                node.value.op.name == "tl.ascend_wait_flag" and node.value.args[0].value == "V" and
                node.value.args[1].value == "MTE3"):
            return tir.Evaluate(0)
        return None

    mutated = tir.stmt_functor.ir_transform(result, None, remove)
    root = tir.Block([], [], [],
                     "root",
                     tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, mutated),
                     alloc_buffers=[ub])
    function = tir.PrimFunc([output.data],
                            tir.BlockRealize([], True, root),
                            buffer_map={output.data: output})
    with pytest.raises(MemoryHazardError):
        validate_memory_synchronization(build_kernel_program(function, platform="A2"))


@pytest.mark.parametrize("extent", [0, 1, 2])
def test_deleted_drain_fails_runtime_flag_accounting(extent):
    from tilelang.simulator import (build_kernel_program, FunctionalSimulator, SimulatorConfig,
                                    SimulationDeadlockError)

    extraction, loop, ub, output = fixture(extent)
    result = emit_loop_sync(
        extraction, loop,
        [Dependency(0, 1, "RAW"), Dependency(1, 0, "WAR", 1)], event_ids=(0, 1))
    mutated = tir.SeqStmt(list(result.seq)[:-2])
    root = tir.Block([], [], [],
                     "root",
                     tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, mutated),
                     alloc_buffers=[ub])
    function = tir.PrimFunc([output.data],
                            tir.BlockRealize([], True, root),
                            buffer_map={output.data: output})
    with pytest.raises(SimulationDeadlockError, match="FLAG ACCOUNTING"):
        FunctionalSimulator(
            build_kernel_program(function, platform="A2"),
            SimulatorConfig(sync_only=True, flag_balance_check="error")).run()
