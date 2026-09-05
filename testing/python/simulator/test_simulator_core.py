# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CPU-only tests for the A2/A3 simulator foundations."""

import json
from pathlib import Path

import pytest

from tilelang.simulator import (
    BufferSpec,
    ChromeTraceExporter,
    CoreProgram,
    ExecutionRecord,
    KernelProgram,
    Lane,
    MemoryScope,
    Pipe,
    ProgramValidationError,
    SimulationStats,
    SimulatorConfig,
    SimulatorConfigError,
    Task,
    TimingProfile,
    UnsupportedMemoryScopeError,
    UnsupportedSimOpError,
    get_device_profile,
)


@pytest.mark.parametrize("platform", ["A2", "A3", "a2", " a3 "])
def test_config_selects_a2_a3_profile(platform: str) -> None:
    config = SimulatorConfig(platform=platform)

    assert config.platform in {"A2", "A3"}
    assert config.device_profile.cube_core_count == 20
    assert config.device_profile.vector_core_count == 40
    assert config.device_profile.vector_lanes_per_cube == 2
    assert config.timing_profile.calibration == "uncalibrated-unit-cost"


def test_config_rejects_unsupported_platform_and_mismatched_timing() -> None:
    with pytest.raises(SimulatorConfigError, match="supported platforms: A2, A3"):
        SimulatorConfig(platform="A5")

    a3_timing = TimingProfile(platform="A3", operation_cycles={"mma": 7})
    with pytest.raises(SimulatorConfigError, match="does not match"):
        SimulatorConfig(platform="A2", timing_profile=a3_timing)


def test_timing_profile_uses_explicit_cost_and_visible_fallback() -> None:
    profile = TimingProfile(platform="A2", operation_cycles={"mma": 23}, fallback_cycles=2)

    assert profile.estimate_cycles("mma") == 23
    assert profile.estimate_cycles("copy") == 2
    assert profile.calibration == "uncalibrated-unit-cost"
    assert get_device_profile("A3").calibration == "uncalibrated"


@pytest.mark.parametrize("scope", ["shmem", "shared.shmem", "shared_memory"])
def test_physical_shmem_fails_fast(scope: str) -> None:
    with pytest.raises(UnsupportedMemoryScopeError, match="intentionally unsupported"):
        MemoryScope.parse(scope)


def test_ascend_local_shared_scope_alias_is_not_physical_shmem() -> None:
    buffer = BufferSpec(
        name="input_ub", scope=MemoryScope.parse("shared.ub"), shape=(8, 16), dtype="float16"
    )

    assert buffer.scope is MemoryScope.UB
    assert buffer.shape == (8, 16)


@pytest.mark.parametrize(
    "operation",
    ["tl.ascend_shmem_put_nbi", "ascend_shmem_get_nbi", "shmem_ub_put_nbi"],
)
def test_shmem_operations_fail_fast(operation: str) -> None:
    with pytest.raises(UnsupportedSimOpError, match="intentionally unsupported"):
        Task("shmem", operation, 0, Lane.VECTOR_0, Pipe.MTE3, 1)


def test_kernel_program_validates_dependencies_and_core_ownership() -> None:
    load = Task("load", "copy_gm_to_l1", 0, Lane.CUBE, Pipe.MTE2, 4)
    mma = Task("mma", "mma", 0, Lane.CUBE, Pipe.MATRIX, 10, dependencies=("load",))
    program = KernelProgram("gemm", "A2", (CoreProgram(0, (load, mma)),))

    assert program.tasks == (load, mma)

    unknown_dependency = Task(
        "bad", "mma", 0, Lane.CUBE, Pipe.MATRIX, 1, dependencies=("missing",)
    )
    with pytest.raises(ProgramValidationError, match="unknown dependencies"):
        KernelProgram("bad", "A2", (CoreProgram(0, (unknown_dependency,)),))

    with pytest.raises(ProgramValidationError, match="belongs to core"):
        CoreProgram(1, (load,))


def test_kernel_program_rejects_cycles_and_invalid_lane_pipe_pair() -> None:
    first = Task("first", "one", 0, Lane.CUBE, Pipe.MTE2, 1, dependencies=("second",))
    second = Task("second", "two", 0, Lane.CUBE, Pipe.MATRIX, 1,
                  dependencies=("first",))
    with pytest.raises(ProgramValidationError, match="dependency cycle"):
        KernelProgram("cycle", "A2", (CoreProgram(0, (first, second)),))

    with pytest.raises(ProgramValidationError, match="not valid on lane"):
        Task("bad-pipe", "mma", 0, Lane.VECTOR_0, Pipe.MATRIX, 1)


def test_trace_export_and_stats_are_overlap_aware(tmp_path: Path) -> None:
    records = [
        ExecutionRecord(
            "load-0", "copy_gm_to_l1", 0, Lane.CUBE, Pipe.MTE2, 0, 10,
            metadata={"bytes": 4096},
        ),
        ExecutionRecord("load-1", "copy_gm_to_l1", 0, Lane.CUBE, Pipe.MTE2, 5, 15),
        ExecutionRecord("mma", "mma", 0, Lane.CUBE, Pipe.MATRIX, 10, 30),
        ExecutionRecord(
            "wait", "wait_flag", 1, Lane.VECTOR_0, Pipe.SCALAR, 8, 12,
            category="wait", stall_reason="event",
        ),
    ]

    stats = SimulationStats.from_records(records)
    assert stats.makespan_cycles == 30
    assert stats.task_count == 4
    assert stats.busy_cycles_by_resource["core-0/cube/mte2"] == 15
    assert stats.utilization_by_resource["core-0/cube/mte2"] == pytest.approx(0.5)
    assert stats.wait_cycles_by_reason == {"event": 4}
    assert stats.completion_cycle_by_core == {0: 30, 1: 12}

    trace_path = ChromeTraceExporter("A2", "uncalibrated-unit-cost").write(
        tmp_path / "trace.json", records
    )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    complete_events = [event for event in trace["traceEvents"] if event["ph"] == "X"]
    metadata = next(
        event for event in trace["traceEvents"] if event["name"] == "simulator_metadata"
    )

    assert len(complete_events) == 4
    assert complete_events[0]["ts"] == 0
    assert complete_events[0]["dur"] == 10
    assert complete_events[0]["args"]["bytes"] == 4096
    assert metadata["args"]["timestamp_unit"] == "simulator_cycle"
    assert metadata["args"]["calibration"] == "uncalibrated-unit-cost"


def test_empty_stats_are_well_defined() -> None:
    stats = SimulationStats.from_records([])

    assert stats.makespan_cycles == 0
    assert stats.task_count == 0
    assert stats.to_dict()["utilization_by_resource"] == {}
