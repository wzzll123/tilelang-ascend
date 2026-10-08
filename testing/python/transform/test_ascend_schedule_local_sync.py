# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Native TIR tests for the deliberately flat local synchronization slice."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_local_sync_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "tilelang/transform/ascend_schedule")]
sys.modules[PACKAGE] = package
for name in ("model", "local_sync"):
    spec = importlib.util.spec_from_file_location(
        PACKAGE + "." + name, ROOT / f"tilelang/transform/ascend_schedule/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
model, emitter = sys.modules[PACKAGE + ".model"], sys.modules[PACKAGE + ".local_sync"]


@pytest.fixture
def tir():
    pytest.importorskip("tilelang")  # Registers actual tl flag builtin Ops.
    return pytest.importorskip("tvm").tir


def extraction(tir, pipes):
    tasks = tuple(model.Task(i, pipe, "V") for i, pipe in enumerate(pipes))
    return types.SimpleNamespace(
        tasks=tasks,
        task_nodes={t.id: tir.Evaluate(tir.call_extern("handle", f"task_{t.id}")) for t in tasks},
        loops={t.id: () for t in tasks},
        scalar_bindings=())


def calls(statement):
    return [s.value for s in statement.seq if hasattr(s, "value")]


def test_emit_native_flag_arguments_and_wait_before_consumer(tir):
    ext = extraction(tir, ["PIPE_MTE2", "PIPE_V"])
    result = emitter.emit_local_sync(ext, [model.Dependency(0, 1, "RAW")], event_ids=(3,))
    sequence = calls(result.statement)
    assert [c.op.name for c in sequence
           ] == ["tir.call_extern", "tl.ascend_set_flag", "tl.ascend_wait_flag", "tir.call_extern"]
    assert [arg.value for arg in sequence[1].args] == ["MTE2", "V", 3]
    assert [arg.value for arg in sequence[2].args] == ["MTE2", "V", 3]
    assert result.assignments[0].dependency_kinds == ("RAW",)
    assert result.assignments[0].core == "V"


def test_same_pipe_fifo_needs_no_event(tir):
    ext = extraction(tir, ["PIPE_V", "PIPE_V"])
    result = emitter.emit_local_sync(ext, [model.Dependency(0, 1, "WAR")], event_ids=())
    assert len(result.statement.seq) == 2 and not result.assignments
    assert result.execution_edges[0].kind == "fifo"


def test_deduplicate_hazards_but_never_reuse_event_by_lexical_wait_order(tir):
    ext = extraction(tir, ["PIPE_MTE2", "PIPE_V", "PIPE_MTE2", "PIPE_V"])
    deps = [
        model.Dependency(0, 1, "RAW"),
        model.Dependency(0, 1, "WAW"),
        model.Dependency(2, 3, "WAR")
    ]
    with pytest.raises(emitter.EventCapacityError, match="capacity"):
        emitter.emit_local_sync(ext, deps, event_ids=(0,))
    result = emitter.emit_local_sync(ext, deps, event_ids=(0, 1))
    assert [a.event_id for a in result.assignments] == [0, 1]
    assert result.assignments[0].dependency_kinds == ("RAW", "WAW")
    assert result.assignments[1].reused_after is None


def test_live_events_same_destination_different_sources_cannot_share(tir):
    ext = extraction(tir, ["PIPE_MTE2", "PIPE_MTE3", "PIPE_V", "PIPE_V"])
    deps = [model.Dependency(0, 2, "RAW"), model.Dependency(1, 3, "RAW")]
    with pytest.raises(emitter.EventCapacityError, match="capacity"):
        emitter.emit_local_sync(ext, deps, event_ids=(0,))
    result = emitter.emit_local_sync(ext, deps, event_ids=(0, 1))
    assert [a.event_id for a in result.assignments] == [0, 1]


def test_different_destination_pools_preserve_channel_identity(tir):
    ext = extraction(tir, ["PIPE_MTE2", "PIPE_V", "PIPE_V"])
    deps = [model.Dependency(0, 1, "RAW"), model.Dependency(1, 2, "RAW")]
    ext.tasks = (ext.tasks[0], replace(ext.tasks[1], pipe="PIPE_MTE3"), ext.tasks[2])
    result = emitter.emit_local_sync(ext, deps, event_ids=(0,))
    assert all(a.reused_after is None for a in result.assignments)
    assert [a.destination_pipe for a in result.assignments] == ["MTE3", "V"]


@pytest.mark.parametrize("case", [
    "carried", "guard", "loop", "scalar", "core", "pipe", "missing", "order", "reverse",
    "scalar_pipe", "wrong_domain", "existing_flag"
])
def test_unsupported_slice_rejected(tir, case):
    ext = extraction(tir, ["PIPE_MTE2", "PIPE_V"])
    deps, kwargs = [model.Dependency(0, 1, "RAW")], {}
    if case == "carried":
        deps = [model.Dependency(0, 1, "RAW", 1)]
    elif case == "guard":
        ext.tasks = (replace(ext.tasks[0], guard=("condition",)), ext.tasks[1])
    elif case == "loop":
        ext.loops[0] = (object(),)
    elif case == "scalar":
        ext.scalar_bindings = (object(),)
    elif case == "core":
        ext.tasks = (replace(ext.tasks[0], core="C"), ext.tasks[1])
    elif case == "pipe":
        ext.tasks = (replace(ext.tasks[0], pipe="PIPE_MTE2|PIPE_V"), ext.tasks[1])
    elif case == "missing":
        del ext.task_nodes[0]
    elif case == "order":
        kwargs["order"] = (0, 0)
    elif case == "reverse":
        kwargs["order"] = (1, 0)
    elif case == "scalar_pipe":
        ext.tasks = (replace(ext.tasks[0], pipe="PIPE_S"), ext.tasks[1])
    elif case == "wrong_domain":
        ext.tasks = (replace(ext.tasks[0], pipe="PIPE_M"), ext.tasks[1])
    elif case == "existing_flag":
        ext.task_nodes[0] = tir.Evaluate(
            tir.call_intrin("handle", tir.op.Op.get("tl.ascend_set_flag"), "MTE2", "V", 0))
    with pytest.raises(emitter.LocalSyncError):
        emitter.emit_local_sync(ext, deps, event_ids=(0,), **kwargs)


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_actual_tir_generated_wait_removal_exposes_simulator_hazard(tir, platform):
    from tilelang.simulator import (build_kernel_program, FunctionalSimulator, SimulatorConfig,
                                    MemoryHazardError)
    from tilelang.simulator.sync import validate_memory_synchronization

    ub = tir.decl_buffer((8,), "float32", name="ub", scope="shared.ub")
    output = tir.decl_buffer((8,), "float32", name="output", scope="global")
    producer = tir.Evaluate(
        tir.call_extern("handle", "tl.ascend_fill", ub.access_ptr("w"), tir.FloatImm("float32", 1),
                        8))
    consumer = tir.Evaluate(
        tir.call_extern("handle", "copy_ub_to_gm", ub.access_ptr("r"), output.access_ptr("w"), 8))
    physical = object()
    ext = types.SimpleNamespace(
        tasks=(model.Task(0, "PIPE_V", "V", (model.Access(physical, "W", 0, 32),)),
               model.Task(1, "PIPE_MTE3", "V", (model.Access(physical, "R", 0, 32),))),
        task_nodes={
            0: producer,
            1: consumer
        },
        loops={
            0: (),
            1: ()
        },
        scalar_bindings=())
    result = emitter.emit_local_sync(
        ext, [model.Dependency(0, 1, "RAW", storage=physical)], event_ids=(0,))

    def program(statement):
        body = tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, statement)
        root = tir.Block([], [], [], "root", body, alloc_buffers=[ub])
        function = tir.PrimFunc([output.data],
                                tir.BlockRealize([], True, root),
                                buffer_map={output.data: output})
        return build_kernel_program(function, platform=platform)

    accepted = program(result.statement)
    assert not validate_memory_synchronization(accepted)
    FunctionalSimulator(accepted, SimulatorConfig(platform=platform, sync_only=True)).run()
    mutated = tir.SeqStmt(
        [stmt for stmt in result.statement.seq if stmt.value.op.name != "tl.ascend_wait_flag"])
    rejected = program(mutated)
    with pytest.raises(MemoryHazardError, match="synchronization|fence|hazard"):
        validate_memory_synchronization(rejected)
