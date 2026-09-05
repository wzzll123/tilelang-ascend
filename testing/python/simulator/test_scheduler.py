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
    ProgramValidationError,
    SimulationDeadlockError,
    SimulationLimitError,
    SimulatorConfig,
    SyncDecision,
    Task,
)
from tilelang.simulator.errors import MemoryHazardError
from tilelang.simulator.sync import validate_memory_synchronization


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
        dependencies=("barrier",),
        metadata={"flag_id": 1, "src_pipe": "m", "mode": 2},
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


def test_cross_flag_fans_out_to_both_vector_lanes_and_joins_before_cube() -> None:
    tasks = (
        Task(
            "cube-set", "set_cross_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"flag_id": 0, "src_pipe": "fix", "mode": 2},
        ),
        Task(
            "cube-wait", "wait_cross_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"flag_id": 1, "channel": "cv"},
        ),
        Task("cube-consumer", "copy", 0, Lane.CUBE, Pipe.MTE2, 1),
        Task(
            "v0-wait", "wait_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 0, "channel": "cv"},
        ),
        Task("v0-work", "add", 0, Lane.VECTOR_0, Pipe.VECTOR, 4),
        Task(
            "v0-set", "set_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 1, "src_pipe": "v", "mode": 2},
        ),
        Task(
            "v1-wait", "wait_cross_flag", 0, Lane.VECTOR_1, Pipe.SCALAR, 1,
            metadata={"flag_id": 0, "channel": "cv"},
        ),
        Task("v1-work", "add", 0, Lane.VECTOR_1, Pipe.VECTOR, 7),
        Task(
            "v1-set", "set_cross_flag", 0, Lane.VECTOR_1, Pipe.SCALAR, 1,
            metadata={"flag_id": 1, "src_pipe": "v", "mode": 2},
        ),
    )

    result = DiscreteEventScheduler(
        synchronization=FlagBarrierSynchronizationModel()
    ).run(_program(*tasks))
    records = {record.task_id: record for record in result.records}

    assert records["v0-wait"].start_cycle == records["cube-set"].end_cycle
    assert records["v1-wait"].start_cycle == records["cube-set"].end_cycle
    assert records["cube-wait"].start_cycle == max(
        records["v0-set"].end_cycle, records["v1-set"].end_cycle
    )
    assert records["cube-consumer"].start_cycle >= records["cube-wait"].end_cycle


def test_cross_flag_keeps_direction_and_vector_lane_credits_independent() -> None:
    tasks = (
        Task(
            "cube-set", "set_cross_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"flag_id": 5, "src_pipe": "fix", "mode": 2},
        ),
        Task(
            "v0-wait", "wait_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 5},
        ),
        Task(
            "v0-set", "set_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 5, "src_pipe": "v", "mode": 2},
        ),
        Task(
            "v1-set", "set_cross_flag", 0, Lane.VECTOR_1, Pipe.SCALAR, 1,
            metadata={"flag_id": 5, "src_pipe": "v", "mode": 2},
        ),
        Task(
            "cube-wait", "wait_cross_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"flag_id": 5},
        ),
    )

    records = {
        record.task_id: record
        for record in DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks)).records
    }

    assert records["v0-wait"].start_cycle == records["cube-set"].end_cycle
    assert records["cube-wait"].start_cycle == max(
        records["v0-set"].end_cycle, records["v1-set"].end_cycle
    )


def test_cube_cross_wait_requires_one_credit_from_each_vector_lane() -> None:
    tasks = (
        Task(
            "v0-set-0", "set_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 2, "src_pipe": "v", "mode": 2},
        ),
        Task(
            "v0-set-1", "set_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 2, "src_pipe": "v", "mode": 2},
        ),
        Task(
            "cube-wait", "wait_cross_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"flag_id": 2},
        ),
    )

    with pytest.raises(SimulationDeadlockError, match="vector1"):
        DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks))


def test_cube_cross_set_gives_each_vector_lane_one_independent_credit() -> None:
    tasks = (
        Task(
            "cube-set", "set_cross_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"flag_id": 3, "src_pipe": "fix", "mode": 2},
        ),
        Task(
            "v0-wait-0", "wait_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 3},
        ),
        Task(
            "v0-wait-1", "wait_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 3},
        ),
    )

    with pytest.raises(SimulationDeadlockError, match="vector0"):
        DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks))


def test_local_flag_is_a_single_outstanding_latch() -> None:
    tasks = tuple(
        Task(
            f"set-{index}", "set_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"src_pipe": "mte2", "dst_pipe": "mte1", "flag_id": 4},
        )
        for index in range(2)
    )

    with pytest.raises(ProgramValidationError, match="reused an outstanding local flag"):
        DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks))


def test_local_flags_are_isolated_between_cube_and_vector_lanes() -> None:
    tasks = (
        Task(
            "cube-set", "set_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"src_pipe": "mte2", "dst_pipe": "s", "flag_id": 4},
        ),
        Task(
            "vector-set", "set_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"src_pipe": "mte2", "dst_pipe": "s", "flag_id": 4},
        ),
        Task(
            "cube-wait", "wait_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"src_pipe": "mte2", "dst_pipe": "s", "flag_id": 4},
        ),
        Task(
            "vector-wait", "wait_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"src_pipe": "mte2", "dst_pipe": "s", "flag_id": 4},
        ),
    )

    records = DiscreteEventScheduler(
        synchronization=FlagBarrierSynchronizationModel()
    ).run(_program(*tasks)).records

    assert {record.task_id for record in records} == {
        "cube-set", "vector-set", "cube-wait", "vector-wait",
    }


@pytest.mark.parametrize("flag_id", [-1, 8])
def test_local_flag_id_must_fit_a2_a3_event_range(flag_id: int) -> None:
    task = Task(
        "set", "set_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
        metadata={"src_pipe": "mte2", "dst_pipe": "mte1", "flag_id": flag_id},
    )

    with pytest.raises(ProgramValidationError, match=r"flag_id in \[0, 7\]"):
        DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(task))


def test_cross_flag_mode_zero_synchronizes_all_cube_cores() -> None:
    core_0 = CoreProgram(0, (
        Task(
            "c0-set", "set_cross_flag", 0, Lane.CUBE, Pipe.FIX, 1,
            metadata={"src_pipe": "fix", "flag_id": 8, "mode": 0},
        ),
        Task(
            "c0-wait", "wait_cross_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"flag_id": 8},
        ),
    ))
    core_1 = CoreProgram(1, (
        Task("c1-work", "copy", 1, Lane.CUBE, Pipe.FIX, 7),
        Task(
            "c1-set", "set_cross_flag", 1, Lane.CUBE, Pipe.FIX, 1,
            metadata={"src_pipe": "fix", "flag_id": 8, "mode": 0},
        ),
        Task(
            "c1-wait", "wait_cross_flag", 1, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"flag_id": 8},
        ),
    ))

    records = {
        record.task_id: record
        for record in DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(KernelProgram("mode0-cube", "A2", (core_0, core_1))).records
    }

    assert records["c0-wait"].start_cycle == records["c1-set"].end_cycle
    assert records["c1-wait"].start_cycle == records["c1-set"].end_cycle


def test_cross_flag_mode_zero_synchronizes_every_vector_lane_on_every_core() -> None:
    cores = []
    for core_id in range(2):
        tasks = []
        for lane_index, lane in enumerate((Lane.VECTOR_0, Lane.VECTOR_1)):
            duration = 1 + core_id * 4 + lane_index
            tasks.append(Task(
                f"c{core_id}-v{lane_index}-set", "set_cross_flag", core_id,
                lane, Pipe.VECTOR, duration,
                metadata={"src_pipe": "v", "flag_id": 6, "mode": 0},
            ))
        if core_id == 0:
            tasks.append(Task(
                "c0-v0-wait", "wait_cross_flag", 0, Lane.VECTOR_0,
                Pipe.SCALAR, 1, metadata={"flag_id": 6},
            ))
        cores.append(CoreProgram(core_id, tuple(tasks)))

    records = {
        record.task_id: record
        for record in DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(KernelProgram("mode0-vector", "A3", tuple(cores))).records
    }

    assert records["c0-v0-wait"].start_cycle == max(
        record.end_cycle
        for task_id, record in records.items()
        if task_id.endswith("-set")
    )


def test_cross_flag_mode_one_is_scoped_to_two_vector_lanes_in_each_group() -> None:
    core_0 = CoreProgram(0, (
        Task(
            "c0-v0-set", "set_cross_flag", 0, Lane.VECTOR_0, Pipe.VECTOR, 1,
            metadata={"src_pipe": "v", "flag_id": 5, "mode": 1},
        ),
        Task(
            "c0-v1-set", "set_cross_flag", 0, Lane.VECTOR_1, Pipe.VECTOR, 2,
            metadata={"src_pipe": "v", "flag_id": 5, "mode": 1},
        ),
        Task(
            "c0-v0-wait", "wait_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 5},
        ),
    ))
    core_1 = CoreProgram(1, (
        Task("c1-delay", "add", 1, Lane.VECTOR_1, Pipe.VECTOR, 9),
        Task(
            "c1-v0-set", "set_cross_flag", 1, Lane.VECTOR_0, Pipe.VECTOR, 1,
            metadata={"src_pipe": "v", "flag_id": 5, "mode": 1},
        ),
        Task(
            "c1-v1-set", "set_cross_flag", 1, Lane.VECTOR_1, Pipe.VECTOR, 1,
            metadata={"src_pipe": "v", "flag_id": 5, "mode": 1},
        ),
        Task(
            "c1-v0-wait", "wait_cross_flag", 1, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 5},
        ),
    ))

    records = {
        record.task_id: record
        for record in DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(KernelProgram("mode1-vector", "A2", (core_0, core_1))).records
    }

    assert records["c0-v0-wait"].start_cycle == 2
    assert records["c1-v0-wait"].start_cycle == records["c1-v1-set"].end_cycle
    assert records["c0-v0-wait"].start_cycle < records["c1-v0-wait"].start_cycle


def test_cross_flag_mode_one_tracks_each_waiter_phase_independently() -> None:
    tasks = (
        Task(
            "v0-set-0", "set_cross_flag", 0, Lane.VECTOR_0, Pipe.VECTOR, 1,
            metadata={"src_pipe": "v", "flag_id": 4, "mode": 1},
        ),
        Task(
            "v0-set-1", "set_cross_flag", 0, Lane.VECTOR_0, Pipe.VECTOR, 3,
            metadata={"src_pipe": "v", "flag_id": 4, "mode": 1},
        ),
        Task(
            "v1-set-0", "set_cross_flag", 0, Lane.VECTOR_1, Pipe.VECTOR, 2,
            metadata={"src_pipe": "v", "flag_id": 4, "mode": 1},
        ),
        Task(
            "v1-set-1", "set_cross_flag", 0, Lane.VECTOR_1, Pipe.VECTOR, 5,
            metadata={"src_pipe": "v", "flag_id": 4, "mode": 1},
        ),
        Task(
            "v0-wait-0", "wait_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 4},
        ),
        Task(
            "v0-wait-1", "wait_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 4},
        ),
    )

    records = {
        record.task_id: record
        for record in DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks)).records
    }

    assert records["v0-wait-0"].start_cycle == max(
        records["v0-set-0"].end_cycle, records["v1-set-0"].end_cycle
    )
    assert records["v0-wait-1"].start_cycle == max(
        records["v0-set-1"].end_cycle, records["v1-set-1"].end_cycle
    )


def test_cross_flag_mode_zero_keeps_cube_and_vector_namespaces_independent() -> None:
    tasks = (
        Task(
            "cube-set", "set_cross_flag", 0, Lane.CUBE, Pipe.FIX, 1,
            metadata={"src_pipe": "fix", "flag_id": 7, "mode": 0},
        ),
        Task(
            "cube-wait", "wait_cross_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"flag_id": 7},
        ),
        Task(
            "v0-set", "set_cross_flag", 0, Lane.VECTOR_0, Pipe.VECTOR, 2,
            metadata={"src_pipe": "v", "flag_id": 7, "mode": 0},
        ),
        Task(
            "v1-set", "set_cross_flag", 0, Lane.VECTOR_1, Pipe.VECTOR, 3,
            metadata={"src_pipe": "v", "flag_id": 7, "mode": 0},
        ),
        Task(
            "v0-wait", "wait_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 7},
        ),
    )

    records = {
        record.task_id: record
        for record in DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks)).records
    }

    assert records["cube-wait"].start_cycle == records["cube-set"].end_cycle
    assert records["v0-wait"].start_cycle == max(
        records["v0-set"].end_cycle, records["v1-set"].end_cycle
    )


def test_collective_cross_flags_keep_flag_ids_independent() -> None:
    tasks = (
        Task(
            "flag1-v0", "set_cross_flag", 0, Lane.VECTOR_0, Pipe.VECTOR, 1,
            metadata={"src_pipe": "v", "flag_id": 1, "mode": 1},
        ),
        Task(
            "flag1-v1", "set_cross_flag", 0, Lane.VECTOR_1, Pipe.VECTOR, 1,
            metadata={"src_pipe": "v", "flag_id": 1, "mode": 1},
        ),
        Task(
            "flag2-v0", "set_cross_flag", 0, Lane.VECTOR_0, Pipe.VECTOR, 1,
            metadata={"src_pipe": "v", "flag_id": 2, "mode": 1},
        ),
        Task(
            "flag2-wait", "wait_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 2},
        ),
    )

    with pytest.raises(SimulationDeadlockError, match="core=0/lane=vector1"):
        DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks))


def test_cross_flag_mode_zero_reports_the_missing_global_participant() -> None:
    core_0 = CoreProgram(0, (
        Task(
            "c0-set", "set_cross_flag", 0, Lane.CUBE, Pipe.FIX, 1,
            metadata={"src_pipe": "fix", "flag_id": 3, "mode": 0},
        ),
        Task(
            "c0-wait", "wait_cross_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"flag_id": 3},
        ),
    ))
    core_1 = CoreProgram(1, (
        Task("c1-work", "copy", 1, Lane.CUBE, Pipe.MTE2, 1),
    ))

    with pytest.raises(SimulationDeadlockError, match="core=1/lane=cube"):
        DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(KernelProgram("mode0-missing", "A2", (core_0, core_1)))


def test_cross_flag_rejects_ambiguous_modes_for_the_same_wait_namespace() -> None:
    tasks = (
        Task(
            "mode0", "set_cross_flag", 0, Lane.VECTOR_0, Pipe.VECTOR, 1,
            metadata={"src_pipe": "v", "flag_id": 2, "mode": 0},
        ),
        Task(
            "mode1", "set_cross_flag", 0, Lane.VECTOR_1, Pipe.VECTOR, 1,
            metadata={"src_pipe": "v", "flag_id": 2, "mode": 1},
        ),
    )

    with pytest.raises(ProgramValidationError, match="ambiguous modes"):
        DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks))


def test_cross_flag_mode_one_rejects_cube_participants() -> None:
    task = Task(
        "set", "set_cross_flag", 0, Lane.CUBE, Pipe.FIX, 1,
        metadata={"src_pipe": "fix", "flag_id": 1, "mode": 1},
    )

    with pytest.raises(ProgramValidationError, match="mode 1 requires a vector lane"):
        DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(task))


@pytest.mark.parametrize("mode", [-1, 3, True, "2"])
def test_cross_flag_rejects_invalid_modes(mode: object) -> None:
    task = Task(
        "set", "set_cross_flag", 0, Lane.CUBE, Pipe.FIX, 1,
        metadata={"src_pipe": "fix", "flag_id": 1, "mode": mode},
    )

    with pytest.raises(ProgramValidationError, match="requires mode 0, 1, or 2"):
        DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(task))


def test_cross_flag_rejects_more_than_fifteen_outstanding_phases() -> None:
    tasks = tuple(
        Task(
            f"set-{index}", "set_cross_flag", 0, Lane.CUBE, Pipe.FIX, 1,
            metadata={"src_pipe": "fix", "flag_id": 0, "mode": 2},
        )
        for index in range(16)
    )

    with pytest.raises(ProgramValidationError, match="15 outstanding credits"):
        DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks))


def test_collective_cross_flag_rejects_sixteenth_unconsumed_phase() -> None:
    tasks = tuple(
        Task(
            f"set-{index}", "set_cross_flag", 0, Lane.VECTOR_0, Pipe.VECTOR, 1,
            metadata={"src_pipe": "v", "flag_id": 0, "mode": 1},
        )
        for index in range(16)
    )

    with pytest.raises(ProgramValidationError, match="15 outstanding phases"):
        DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks))


def test_local_wait_only_fences_its_destination_pipe() -> None:
    tasks = (
        Task("producer", "copy", 0, Lane.VECTOR_0, Pipe.MTE2, 6),
        Task(
            "set", "set_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"src_pipe": "mte2", "dst_pipe": "v", "flag_id": 0},
        ),
        Task(
            "wait", "wait_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"src_pipe": "mte2", "dst_pipe": "v", "flag_id": 0},
        ),
        Task("vector", "add", 0, Lane.VECTOR_0, Pipe.VECTOR, 2),
        Task("store", "copy", 0, Lane.VECTOR_0, Pipe.MTE3, 9),
    )

    records = {
        record.task_id: record
        for record in DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks)).records
    }

    assert records["store"].start_cycle == 0
    assert records["vector"].start_cycle == records["wait"].end_cycle


def test_wait_on_one_pipe_does_not_block_a_later_set_on_another_pipe() -> None:
    tasks = (
        Task(
            "wait", "wait_flag", 0, Lane.VECTOR_0, Pipe.VECTOR, 1,
            metadata={"src_pipe": "mte2", "dst_pipe": "v", "flag_id": 1},
        ),
        Task(
            "set", "set_flag", 0, Lane.VECTOR_0, Pipe.MTE2, 1,
            metadata={"src_pipe": "mte2", "dst_pipe": "v", "flag_id": 1},
        ),
    )

    records = {
        record.task_id: record
        for record in DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks)).records
    }

    assert records["set"].start_cycle == 0
    assert records["wait"].start_cycle == records["set"].end_cycle


def test_local_wait_to_scalar_fences_later_work_across_the_lane() -> None:
    tasks = (
        Task(
            "set", "set_flag", 0, Lane.VECTOR_0, Pipe.VECTOR, 1,
            metadata={"src_pipe": "v", "dst_pipe": "s", "flag_id": 2},
        ),
        Task(
            "wait", "wait_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"src_pipe": "v", "dst_pipe": "s", "flag_id": 2},
        ),
        Task("load", "copy", 0, Lane.VECTOR_0, Pipe.MTE2, 3),
        Task("store", "copy", 0, Lane.VECTOR_0, Pipe.MTE3, 4),
    )

    records = {
        record.task_id: record
        for record in DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks)).records
    }

    assert records["load"].start_cycle == records["wait"].end_cycle
    assert records["store"].start_cycle == records["wait"].end_cycle


def test_pipe_barrier_only_fences_the_named_pipe() -> None:
    tasks = (
        Task("vector-0", "add", 0, Lane.VECTOR_0, Pipe.VECTOR, 6),
        Task(
            "barrier", "pipe_barrier", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"target_pipe": "PIPE_V"},
        ),
        Task("vector-1", "add", 0, Lane.VECTOR_0, Pipe.VECTOR, 2),
        Task("store", "copy", 0, Lane.VECTOR_0, Pipe.MTE3, 9),
    )

    records = {
        record.task_id: record
        for record in DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks)).records
    }

    assert records["barrier"].start_cycle == records["vector-0"].end_cycle
    assert records["vector-1"].start_cycle == records["barrier"].end_cycle
    assert records["store"].start_cycle == 0


def test_barrier_all_is_lane_local() -> None:
    tasks = (
        Task("cube-work", "mma", 0, Lane.CUBE, Pipe.MATRIX, 3),
        Task("vector-work", "add", 0, Lane.VECTOR_0, Pipe.VECTOR, 10),
        Task("barrier", "barrier_all", 0, Lane.CUBE, Pipe.SCALAR, 1),
        Task("cube-after", "copy", 0, Lane.CUBE, Pipe.MTE2, 2),
    )

    records = {
        record.task_id: record
        for record in DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks)).records
    }

    assert records["barrier"].start_cycle == records["cube-work"].end_cycle
    assert records["barrier"].start_cycle < records["vector-work"].end_cycle
    assert records["cube-after"].start_cycle == records["barrier"].end_cycle


def test_cross_set_waits_only_for_its_named_producer_pipe() -> None:
    tasks = (
        Task("fix", "copy", 0, Lane.CUBE, Pipe.FIX, 4),
        Task("matrix", "mma", 0, Lane.CUBE, Pipe.MATRIX, 10),
        Task(
            "set", "set_cross_flag", 0, Lane.CUBE, Pipe.SCALAR, 1,
            metadata={"src_pipe": "PIPE_FIX", "flag_id": 6, "mode": 2},
        ),
        Task(
            "wait", "wait_cross_flag", 0, Lane.VECTOR_0, Pipe.SCALAR, 1,
            metadata={"flag_id": 6},
        ),
    )

    records = {
        record.task_id: record
        for record in DiscreteEventScheduler(
            synchronization=FlagBarrierSynchronizationModel()
        ).run(_program(*tasks)).records
    }

    assert records["set"].start_cycle == records["fix"].end_cycle
    assert records["set"].start_cycle < records["matrix"].end_cycle
    assert records["wait"].start_cycle == records["set"].end_cycle


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


def test_memory_sync_validation_rejects_unfenced_cross_pipe_edge() -> None:
    producer = Task("load", "copy", 0, Lane.CUBE, Pipe.MTE2, 1)
    consumer = Task(
        "consume", "copy", 0, Lane.CUBE, Pipe.MTE1, 1,
        dependencies=("load",),
        metadata={"memory_dependencies": ("load",)},
    )

    with pytest.raises(MemoryHazardError, match="expected.*flag.*PIPE_ALL"):
        validate_memory_synchronization(_program(producer, consumer))


def test_memory_sync_validation_accepts_matching_local_flag_pair() -> None:
    tasks = (
        Task("load", "copy", 0, Lane.CUBE, Pipe.MTE2, 1),
        Task(
            "set", "auto_set_flag", 0, Lane.CUBE, Pipe.MTE2, 1,
            metadata={"src_pipe": "mte2", "dst_pipe": "mte1", "flag_id": 3},
        ),
        Task(
            "wait", "auto_wait_flag", 0, Lane.CUBE, Pipe.MTE1, 1,
            metadata={"src_pipe": "mte2", "dst_pipe": "mte1", "flag_id": 3},
        ),
        Task(
            "consume", "copy", 0, Lane.CUBE, Pipe.MTE1, 1,
            dependencies=("load",),
            metadata={"memory_dependencies": ("load",)},
        ),
    )

    assert validate_memory_synchronization(_program(*tasks)) == ()


def test_memory_sync_validation_accepts_pipe_all() -> None:
    tasks = (
        Task("load", "copy", 0, Lane.CUBE, Pipe.MTE2, 1),
        Task(
            "barrier", "auto_barrier", 0, Lane.CONTROL, Pipe.SCALAR, 1,
            metadata={"target_pipe": "all"},
        ),
        Task(
            "consume", "copy", 0, Lane.CUBE, Pipe.MTE1, 1,
            dependencies=("load",),
            metadata={"memory_dependencies": ("load",)},
        ),
    )

    assert validate_memory_synchronization(_program(*tasks)) == ()
