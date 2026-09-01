# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Tests for the A2/A3 simulator discrete-event scheduler."""

from typing import Mapping

import pytest

from tilelang.simulator import (
    CoreProgram,
    DiscreteEventScheduler,
    ExecutionRecord,
    FlagBarrierSynchronizationModel,
    KernelProgram,
    Lane,
    Pipe,
    SimulationDeadlockError,
    SimulationLimitError,
    SimulatorConfig,
    SyncDecision,
    Task,
)


def _program(*tasks: Task) -> KernelProgram:
    return KernelProgram("schedule_test", "A2", (CoreProgram(0, tasks),))


def test_dependencies_and_pipe_fifo_determine_start_cycles() -> None:
    load_0 = Task("load-0", "copy", 0, Lane.CUBE, Pipe.MTE2, 4)
    load_1 = Task("load-1", "copy", 0, Lane.CUBE, Pipe.MTE2, 6)
    mma = Task(
        "mma", "mma", 0, Lane.CUBE, Pipe.MATRIX, 10, dependencies=("load-1",)
    )

    result = DiscreteEventScheduler().run(_program(load_0, load_1, mma))
    records = {record.task_id: record for record in result.records}

    assert (records["load-0"].start_cycle, records["load-0"].end_cycle) == (0, 4)
    assert (records["load-1"].start_cycle, records["load-1"].end_cycle) == (4, 10)
    assert (records["mma"].start_cycle, records["mma"].end_cycle) == (10, 20)
    assert result.stats.makespan_cycles == 20


def test_independent_pipes_and_lanes_overlap() -> None:
    cube_load = Task("cube-load", "copy", 0, Lane.CUBE, Pipe.MTE2, 9)
    matrix = Task("matrix", "mma", 0, Lane.CUBE, Pipe.MATRIX, 7)
    vector = Task("vector", "add", 0, Lane.VECTOR_0, Pipe.VECTOR, 5)

    result = DiscreteEventScheduler().run(_program(cube_load, matrix, vector))

    assert {record.start_cycle for record in result.records} == {0}
    assert result.stats.makespan_cycles == 9
    assert result.stats.task_count == 3


def test_dependency_can_cross_core_and_lane() -> None:
    producer = Task("producer", "mma", 0, Lane.CUBE, Pipe.MATRIX, 8)
    consumer = Task(
        "consumer", "add", 1, Lane.VECTOR_0, Pipe.VECTOR, 3,
        dependencies=("producer",),
    )
    program = KernelProgram(
        "cross-core", "A2", (CoreProgram(0, (producer,)), CoreProgram(1, (consumer,)))
    )

    records = {
        record.task_id: record for record in DiscreteEventScheduler().run(program).records
    }

    assert records["consumer"].start_cycle == 8
    assert records["consumer"].end_cycle == 11


def test_max_cycles_fails_at_the_task_that_crosses_limit() -> None:
    task = Task("long", "mma", 0, Lane.CUBE, Pipe.MATRIX, 11)
    config = SimulatorConfig(platform="A2", max_cycles=10)

    with pytest.raises(SimulationLimitError, match="finish at cycle 11"):
        DiscreteEventScheduler(config).run(_program(task))


class _NeverReadySynchronization:
    def reset(self, program: KernelProgram) -> None:
        del program

    def evaluate(
        self, task: Task, completed: Mapping[str, ExecutionRecord]
    ) -> SyncDecision:
        del task, completed
        return SyncDecision(ready_cycle=None, reason="flag", detail="waiting for flag 3")

    def on_scheduled(self, task: Task, record: ExecutionRecord) -> None:
        del task, record


def test_sync_extension_reports_actionable_deadlock() -> None:
    task = Task("wait", "wait_flag", 0, Lane.CONTROL, Pipe.SCALAR, 1)

    with pytest.raises(SimulationDeadlockError, match="wait.*flag 3"):
        DiscreteEventScheduler(synchronization=_NeverReadySynchronization()).run(
            _program(task)
        )


def test_fifo_and_explicit_dependency_cycle_reports_deadlock() -> None:
    first = Task(
        "first", "copy", 0, Lane.CUBE, Pipe.MTE2, 1, dependencies=("second",)
    )
    second = Task("second", "copy", 0, Lane.CUBE, Pipe.MTE2, 1)

    with pytest.raises(SimulationDeadlockError, match="first.*second"):
        DiscreteEventScheduler().run(_program(first, second))


def test_local_flag_wait_starts_when_set_completes_and_consumes_token() -> None:
    set_flag = Task(
        "set", "set_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
        metadata={"src_pipe": "mte2", "dst_pipe": "mte1", "flag_id": 2},
    )
    wait_flag = Task(
        "wait", "wait_flag", 0, Lane.CUBE, Pipe.MTE1, 1,
        metadata={"src_pipe": "mte2", "dst_pipe": "mte1", "flag_id": 2},
    )

    result = DiscreteEventScheduler(
        synchronization=FlagBarrierSynchronizationModel()
    ).run(_program(set_flag, wait_flag))
    records = {record.task_id: record for record in result.records}

    assert records["wait"].start_cycle == records["set"].end_cycle
    assert result.stats.wait_cycles_by_reason == {"local flag": 1}


def test_cross_flag_and_barrier_all_order_independent_pipes() -> None:
    cube = Task("cube", "mma", 0, Lane.CUBE, Pipe.MATRIX, 8)
    vector = Task("vector", "add", 0, Lane.VECTOR_0, Pipe.VECTOR, 5)
    barrier = Task("barrier", "barrier_all", 0, Lane.CONTROL, Pipe.SCALAR, 1)
    set_cross = Task(
        "set-cross", "set_cross_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
        dependencies=("barrier",), metadata={"flag_id": 1, "channel": "c2v"},
    )
    wait_cross = Task(
        "wait-cross", "wait_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
        metadata={"flag_id": 1, "channel": "c2v"},
    )

    result = DiscreteEventScheduler(
        synchronization=FlagBarrierSynchronizationModel()
    ).run(_program(cube, vector, barrier, set_cross, wait_cross))
    records = {record.task_id: record for record in result.records}

    assert records["barrier"].start_cycle == 8
    assert records["wait-cross"].start_cycle == records["set-cross"].end_cycle


def test_wait_without_matching_flag_reports_deadlock() -> None:
    wait_flag = Task(
        "wait", "wait_flag", 0, Lane.CUBE, Pipe.MTE1, 1,
        metadata={"src_pipe": "mte2", "dst_pipe": "mte1", "flag_id": 7},
    )

    with pytest.raises(SimulationDeadlockError, match="local flag.*id=7"):
        DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(wait_flag))


def test_pipe_barrier_all_drains_every_prior_pipe() -> None:
    load = Task("load", "copy", 0, Lane.CUBE, Pipe.MTE2, 4)
    matrix = Task("matrix", "mma", 0, Lane.CUBE, Pipe.MATRIX, 9)
    barrier = Task(
        "barrier", "pipe_barrier", 0, Lane.CONTROL, Pipe.SCALAR, 1,
        metadata={"target_pipe": "all"},
    )

    result = DiscreteEventScheduler(
        synchronization=FlagBarrierSynchronizationModel()
    ).run(_program(load, matrix, barrier))
    records = {record.task_id: record for record in result.records}

    assert records["barrier"].start_cycle == 9
