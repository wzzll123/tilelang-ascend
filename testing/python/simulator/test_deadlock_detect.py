# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Proof-oriented deadlock and end-of-kernel flag-accounting tests."""

import pytest

from tilelang.simulator import (
    CoreProgram,
    DiscreteEventScheduler,
    FlagBarrierSynchronizationModel,
    KernelProgram,
    Lane,
    Pipe,
    SimulationDeadlockError,
    SimulatorConfig,
    SimulatorConfigError,
    Task,
)


def _program(*tasks: Task) -> KernelProgram:
    return KernelProgram("deadlock", "A2", (CoreProgram(0, tasks),))


def test_deadlock_config_validation() -> None:
    with pytest.raises(SimulatorConfigError, match="deadlock_detect"):
        SimulatorConfig(deadlock_detect=1)
    with pytest.raises(SimulatorConfigError, match="deadlock_history_limit"):
        SimulatorConfig(deadlock_history_limit=0)
    with pytest.raises(SimulatorConfigError, match="flag_balance_check"):
        SimulatorConfig(flag_balance_check="strict")


def test_missing_local_set_reports_proof_and_flag() -> None:
    wait = Task(
        "wait",
        "wait_flag",
        0,
        Lane.CUBE,
        Pipe.MTE1,
        1,
        metadata={"src_pipe": "mte2", "dst_pipe": "mte1", "flag_id": 3},
    )
    with pytest.raises(
        SimulationDeadlockError,
        match=r"DEADLOCK \(global no-progress\).*local flag.*id=3",
    ):
        DiscreteEventScheduler(synchronization=FlagBarrierSynchronizationModel()).run(_program(wait))


def test_dependency_wait_cycle_is_reported() -> None:
    first = Task("first", "copy", 0, Lane.CUBE, Pipe.MTE2, 1, dependencies=("second",))
    second = Task("second", "copy", 0, Lane.CUBE, Pipe.MTE2, 1)
    with pytest.raises(SimulationDeadlockError, match=r"wait-for cycle:.*first.*second"):
        DiscreteEventScheduler().run(_program(first, second))


def test_kernel_end_rejects_outstanding_local_flag_credit() -> None:
    set_flag = Task(
        "set",
        "set_flag",
        0,
        Lane.CUBE,
        Pipe.MTE2,
        1,
        metadata={"src_pipe": "mte2", "dst_pipe": "mte1", "flag_id": 5},
    )
    with pytest.raises(
        SimulationDeadlockError,
        match=r"FLAG ACCOUNTING.*local flag.*id=5.*level=1",
    ):
        DiscreteEventScheduler(SimulatorConfig(), synchronization=FlagBarrierSynchronizationModel()).run(
            _program(set_flag)
        )
