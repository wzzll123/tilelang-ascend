# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""All phases share event ownership; textual waits do not free channels."""

from types import SimpleNamespace
import pytest
import tilelang  # noqa: F401
from tvm import tir


def test_flat_and_loop_emitters_do_not_restart_local_event_numbering():
    from tilelang.transform.ascend_schedule.events import EventRegistry
    from tilelang.transform.ascend_schedule.local_sync import emit_local_sync, emit_loop_sync
    from tilelang.transform.ascend_schedule.model import Task, Dependency

    tasks = (Task(0, "PIPE_V", "V"), Task(1, "PIPE_MTE3", "V"))
    nodes = {task.id: tir.Evaluate(tir.call_extern("handle", f"task_{task.id}")) for task in tasks}
    extraction = SimpleNamespace(
        tasks=tasks, task_nodes=nodes, loops={
            0: (),
            1: ()
        }, scalar_bindings=())
    registry = EventRegistry(local_ids=(0, 1, 2), cross_ids=())
    prefix = emit_local_sync(
        extraction, [Dependency(0, 1, "RAW")], event_ids=(0, 1, 2), registry=registry)
    loop = tir.For(
        tir.Var("epoch", "int32"), 0, 2, tir.ForKind.SERIAL, tir.SeqStmt([nodes[0], nodes[1]]))
    emit_loop_sync(
        extraction, loop, [Dependency(0, 1, "RAW")], event_ids=(0, 1, 2), registry=registry)
    suffix = emit_local_sync(
        extraction, [Dependency(0, 1, "RAW")], event_ids=(0, 1, 2), registry=registry)
    assert prefix.assignments[0].event_id == 0
    assert suffix.assignments[0].event_id == 2
    records = registry.records
    assert len({(item.destination_core, item.source_pipe, item.destination_pipe, item.event_id)
                for item in records}) == len(records)


def test_cross_channels_are_shared_across_directions_and_local_pools_are_per_core():
    from tilelang.transform.ascend_schedule.events import EventRegistry
    from tilelang.transform.ascend_schedule.local_sync import EventCapacityError

    registry = EventRegistry(local_ids=(0,), cross_ids=(0, 1))
    assert registry.allocate("C", "FIX", "V", "MTE2").event_id == 0
    assert registry.allocate("V", "MTE3", "C", "MTE2").event_id == 1
    assert registry.allocate("V", "V", "V", "MTE2").event_id == 0
    assert registry.allocate("C", "MTE1", "C", "MTE2").event_id == 0
    with pytest.raises(EventCapacityError, match="capacity"):
        registry.allocate("C", "FIX", "V", "MTE2")
    assert registry.allocate("V", "MTE3", "V", "MTE2").event_id == 0
    with pytest.raises(EventCapacityError, match="capacity"):
        registry.allocate("V", "MTE3", "V", "MTE2")
