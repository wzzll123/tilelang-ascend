# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""A2 GM visibility-window synchronization regressions."""

import pytest

from tilelang.simulator import (
    BufferRegion,
    CoreProgram,
    KernelProgram,
    Lane,
    MemoryScope,
    Pipe,
    SimulatorConfig,
    Task,
)
from tilelang.simulator.errors import MemoryHazardError
from tilelang.simulator.hazard import SimulatorHazardWarning
from tilelang.simulator.sync import validate_memory_synchronization


GM = BufferRegion("workspace", MemoryScope.WORKSPACE, (8,), "float32")
UB = BufferRegion("ub", MemoryScope.UB, (8,), "float32", core_id=0)


def _program(*tasks: Task, platform="A2") -> KernelProgram:
    by_core = {}
    for task in tasks:
        by_core.setdefault(task.core_id, []).append(task)
    return KernelProgram(
        "gm_visibility",
        platform,
        tuple(CoreProgram(core_id, tuple(core_tasks)) for core_id, core_tasks in sorted(by_core.items())),
    )


def _write(task_id="write", *, core=0, region=GM) -> Task:
    return Task(
        task_id,
        "copy_ub_to_gm",
        core,
        Lane.VECTOR_0,
        Pipe.MTE3,
        1,
        metadata={"dst": region},
    )


def _read(task_id="read", *, core=0, region=GM, dependency="write") -> Task:
    return Task(
        task_id,
        "copy_gm_to_ub",
        core,
        Lane.VECTOR_0,
        Pipe.MTE2,
        1,
        dependencies=(dependency,),
        metadata={"src": region, "memory_dependencies": (dependency,)},
    )


def _local(operation, src, dst, flag_id, *, core=0) -> Task:
    pipe = Pipe(src if operation == "auto_set_flag" else dst)
    return Task(
        f"{operation}-{core}-{flag_id}",
        operation,
        core,
        Lane.VECTOR_0,
        pipe,
        1,
        metadata={"src_pipe": src, "dst_pipe": dst, "flag_id": flag_id},
    )


def _cross(operation, flag_id, *, core=0) -> Task:
    pipe = Pipe.MTE3 if operation == "set_cross_flag" else Pipe.SCALAR
    return Task(
        f"{operation}-{core}-{flag_id}",
        operation,
        core,
        Lane.VECTOR_0,
        pipe,
        1,
        metadata={"src_pipe": "mte3", "flag_id": flag_id, "mode": 0},
    )


def _barrier(target, *, core=0, suffix="") -> Task:
    return Task(
        f"barrier-{target}-{core}{suffix}",
        "pipe_barrier",
        core,
        Lane.VECTOR_0,
        Pipe.MTE2 if target == "mte2" else Pipe.SCALAR,
        1,
        metadata={"target_pipe": target},
    )


def test_gm_visibility_uses_unified_hazard_policy() -> None:
    assert SimulatorConfig(hazard_check="warn").hazard_check == "warn"


def test_d53_gm_raw_rejects_unrelated_pipe_barrier() -> None:
    program = _program(_write(), _barrier("v"), _read())
    with pytest.raises(MemoryHazardError, match="GM visibility"):
        validate_memory_synchronization(program, hazard_check="error")


def test_gm_visibility_policy_and_diagnostic_metadata() -> None:
    program = _program(_write(), _read())
    with pytest.warns(SimulatorHazardWarning, match="GM visibility"):
        diagnostics = validate_memory_synchronization(
            program,
            hazard_check="warn",
        )
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.kind == "gm-visibility-window"
    assert diagnostic.buffer == "workspace"
    assert diagnostic.metadata["producer_task"] == "write"
    assert diagnostic.metadata["consumer_task"] == "read"
    assert (
        validate_memory_synchronization(
            program,
            hazard_check="off",
        )
        == ()
    )


def test_d53_gm_raw_accepts_same_phase_mte3_mte2_flag() -> None:
    program = _program(
        _write(),
        _local("auto_set_flag", "mte3", "mte2", 2),
        _local("auto_wait_flag", "mte3", "mte2", 2),
        _read(),
    )
    assert validate_memory_synchronization(program, hazard_check="error") == ()


def test_d54_collective_alone_does_not_make_gm_raw_visible() -> None:
    program = _program(
        _write(),
        _cross("set_cross_flag", 3),
        _cross("wait_cross_flag", 3),
        _cross("set_cross_flag", 4),
        _cross("wait_cross_flag", 4),
        _read(),
    )
    with pytest.raises(MemoryHazardError, match="GM visibility"):
        validate_memory_synchronization(program, hazard_check="error")


def test_d54_read_side_mte2_barrier_makes_gm_raw_visible() -> None:
    program = _program(
        _write(),
        _cross("set_cross_flag", 3),
        _cross("wait_cross_flag", 3),
        _barrier("mte2"),
        _read(),
    )
    assert validate_memory_synchronization(program, hazard_check="error") == ()


def test_d54_flag_pair_across_collective_is_not_a_visibility_fence() -> None:
    program = _program(
        _write(),
        _local("auto_set_flag", "mte3", "mte2", 7),
        _cross("set_cross_flag", 3),
        _cross("wait_cross_flag", 3),
        _local("auto_wait_flag", "mte3", "mte2", 7),
        _read(),
    )
    with pytest.raises(MemoryHazardError, match="GM visibility"):
        validate_memory_synchronization(program, hazard_check="error")


def test_d54_rule_is_not_extrapolated_to_unverified_a3() -> None:
    program = _program(
        _write(),
        _local("auto_set_flag", "mte3", "mte2", 7),
        _cross("set_cross_flag", 3),
        _cross("wait_cross_flag", 3),
        _local("auto_wait_flag", "mte3", "mte2", 7),
        _read(),
        platform="A3",
    )
    assert validate_memory_synchronization(program, hazard_check="error") == ()


def test_ub_raw_keeps_existing_local_flag_semantics() -> None:
    producer = Task(
        "ub-write",
        "copy_gm_to_ub",
        0,
        Lane.VECTOR_0,
        Pipe.MTE2,
        1,
        metadata={"dst": UB},
    )
    consumer = Task(
        "ub-read",
        "add",
        0,
        Lane.VECTOR_0,
        Pipe.VECTOR,
        1,
        dependencies=("ub-write",),
        metadata={"src": UB, "memory_dependencies": ("ub-write",)},
    )
    program = _program(
        producer,
        _local("auto_set_flag", "mte2", "v", 2),
        _local("auto_wait_flag", "mte2", "v", 2),
        consumer,
    )
    assert validate_memory_synchronization(program, hazard_check="error") == ()


@pytest.mark.parametrize("with_read_drain", [False, True])
def test_cross_core_gm_raw_requires_collective_then_read_drain(with_read_drain) -> None:
    tasks = [
        _write(core=0),
        _cross("set_cross_flag", 5, core=0),
        _cross("wait_cross_flag", 5, core=0),
        _cross("set_cross_flag", 5, core=1),
        _cross("wait_cross_flag", 5, core=1),
    ]
    if with_read_drain:
        tasks.append(_barrier("mte2", core=1))
    tasks.append(_read(core=1))
    program = _program(*tasks)

    if with_read_drain:
        assert validate_memory_synchronization(program, hazard_check="error") == ()
    else:
        with pytest.raises(MemoryHazardError, match="GM visibility"):
            validate_memory_synchronization(program, hazard_check="error")


def _cube_flag(operation, src, dst, flag_id) -> Task:
    return Task(
        f"{operation}-{flag_id}",
        operation,
        0,
        Lane.CUBE,
        Pipe(src if operation == "auto_set_flag" else dst),
        1,
        metadata={"src_pipe": src, "dst_pipe": dst, "flag_id": flag_id},
    )


def _cube_transitive_program(*flags: Task) -> KernelProgram:
    producer = Task("l1-load", "copy", 0, Lane.CUBE, Pipe.MTE2, 1)
    consumer = Task(
        "mma",
        "mma",
        0,
        Lane.CUBE,
        Pipe.MATRIX,
        1,
        dependencies=("l1-load",),
        metadata={"memory_dependencies": ("l1-load",)},
    )
    return _program(producer, *flags, consumer)


def test_ordered_transitive_cube_flags_are_a_valid_fence() -> None:
    program = _cube_transitive_program(
        _cube_flag("auto_set_flag", "mte2", "mte1", 0),
        _cube_flag("auto_wait_flag", "mte2", "mte1", 0),
        _cube_flag("auto_set_flag", "mte1", "m", 1),
        _cube_flag("auto_wait_flag", "mte1", "m", 1),
    )
    assert validate_memory_synchronization(program) == ()


def test_out_of_order_transitive_cube_flags_are_not_a_fence() -> None:
    program = _cube_transitive_program(
        _cube_flag("auto_set_flag", "mte1", "m", 1),
        _cube_flag("auto_wait_flag", "mte1", "m", 1),
        _cube_flag("auto_set_flag", "mte2", "mte1", 0),
        _cube_flag("auto_wait_flag", "mte2", "mte1", 0),
    )
    with pytest.raises(MemoryHazardError, match="missing.*synchronization"):
        validate_memory_synchronization(program)
