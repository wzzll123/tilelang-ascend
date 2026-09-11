# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Flag credit depth (blocking semantics) tests.

flag_blocking=True models the hardware queue: a SET issued while a flag
already has `depth` outstanding credits stalls the issuing pipe until a WAIT
frees a slot. This turns "level-balanced but depth-unbalanced" kernels into
scheduler-visible deadlocks (detectable via the no-progress detector) instead
of silent passes (idealized unbounded model) or hard validation errors.
"""

import pytest

from tilelang.simulator import (
    BufferRegion,
    BufferSpec,
    CoreProgram,
    FunctionalSimulator,
    KernelProgram,
    Lane,
    MemoryScope,
    Pipe,
    ProgramValidationError,
    SimulationDeadlockError,
    SimulatorConfig,
    SimulatorConfigError,
    Task,
)


def _task(tid, operation, pipe, metadata, core=0, lane=Lane.CUBE):
    return Task(tid, operation, core, lane, pipe, 1, metadata=metadata)


def _local_set(tid, flag_id=0, pipe=Pipe.SCALAR):
    return _task(
        tid,
        "set_flag",
        pipe,
        {"src_pipe": "mte1", "dst_pipe": "mte2", "flag_id": flag_id},
    )


def _local_wait(tid, flag_id=0, pipe=Pipe.SCALAR):
    return _task(
        tid,
        "wait_flag",
        pipe,
        {"src_pipe": "mte1", "dst_pipe": "mte2", "flag_id": flag_id},
    )


def _program(name, tasks):
    return KernelProgram(
        name,
        "A2",
        (CoreProgram(0, tuple(tasks)),),
        buffers=(),
    )


def _run(program, **cfg):
    return FunctionalSimulator(
        program, SimulatorConfig(platform=program.platform, sync_only=True, **cfg)
    ).run()


def test_config_validation() -> None:
    with pytest.raises(SimulatorConfigError, match="flag_blocking must be a boolean"):
        SimulatorConfig(flag_blocking=1)
    with pytest.raises(
        SimulatorConfigError, match="local_flag_depth must be a positive integer"
    ):
        SimulatorConfig(local_flag_depth=0)
    with pytest.raises(
        SimulatorConfigError, match="cross_flag_depth must be a positive integer"
    ):
        SimulatorConfig(cross_flag_depth=-2)


def test_double_set_same_pipe_deadlocks_under_blocking() -> None:
    """set;set;wait on one pipe with depth 1: the second set stalls the pipe,
    the draining wait is queued behind it -> no progress -> deadlock report."""
    program = _program(
        "flag-depth-deadlock",
        [_local_set("s1"), _local_set("s2"), _local_wait("w1")],
    )
    with pytest.raises(SimulationDeadlockError, match=r"local flag depth"):
        _run(program, flag_blocking=True, local_flag_depth=1)


def test_double_set_default_counter_model_consumes_each_credit() -> None:
    """Default PTO-aligned semantics accept two FIFO credits on one ID."""
    program = _program(
        "flag-counter-default",
        [_local_set("s1"), _local_set("s2"), _local_wait("w1"), _local_wait("w2")],
    )
    assert _run(program) is not None


def test_depth_two_absorbs_double_set() -> None:
    """set;set;wait;wait with depth 2: both credits fit, kernel completes."""
    program = _program(
        "flag-depth-ok",
        [
            _local_set("s1"),
            _local_set("s2"),
            _local_wait("w1"),
            _local_wait("w2"),
        ],
    )
    result = _run(program, flag_blocking=True, local_flag_depth=2)
    assert result is not None


def test_blocking_waits_for_drain_before_second_set() -> None:
    """set;wait;set;wait with depth 1: interleaved drain keeps the pipe moving
    (no deadlock, no error) -- the blocking only bites when depth is exceeded."""
    program = _program(
        "flag-depth-interleaved",
        [
            _local_set("s1"),
            _local_wait("w1"),
            _local_set("s2"),
            _local_wait("w2"),
        ],
    )
    result = _run(program, flag_blocking=True, local_flag_depth=1)
    assert result is not None


def test_cross_flag_depth_blocks_and_reports() -> None:
    """cross mode-2 sets beyond cross_flag_depth stall the issuer; with no
    consumer the scheduler must report the depth blockage."""
    sets = [
        _task(
            f"cs{i}",
            "set_cross_flag",
            Pipe.FIX,
            {"src_pipe": "fix", "flag_id": 0, "mode": 2},
        )
        for i in range(3)
    ]
    program = _program("cross-flag-depth", sets)
    with pytest.raises(SimulationDeadlockError, match=r"cross flag depth"):
        _run(program, flag_blocking=True, cross_flag_depth=2)


def test_cross_flag_depth_legacy_overflow_error_unchanged() -> None:
    # legacy mode ignores the configured depth and keeps the hardcoded
    # _MAX_CROSS_FLAG_CREDITS=15 overflow contract.
    sets = [
        _task(
            f"cs{i}",
            "set_cross_flag",
            Pipe.FIX,
            {"src_pipe": "fix", "flag_id": 0, "mode": 2},
        )
        for i in range(16)
    ]
    program = _program("cross-flag-depth-legacy", sets)
    with pytest.raises(ProgramValidationError, match="overflowed"):
        _run(program, cross_flag_depth=2)
