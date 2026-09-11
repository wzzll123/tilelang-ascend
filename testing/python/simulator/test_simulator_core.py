# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CPU-only tests for the A2/A3 simulator foundations."""

import json
from pathlib import Path

import pytest

from tilelang.simulator import (
    AffineInt,
    BufferRegion,
    BufferSpec,
    ChromeTraceExporter,
    CoreProgram,
    ExecutionRecord,
    HazardDiagnostic,
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
    pto_fallback_timing_profile,
)


@pytest.mark.parametrize("platform", ["A2", "A3", "a2", " a3 "])
def test_config_selects_a2_a3_profile(platform: str) -> None:
    config = SimulatorConfig(platform=platform)

    assert config.platform in {"A2", "A3"}
    assert config.device_profile.cube_core_count == 20
    assert config.device_profile.vector_core_count == 40
    assert config.device_profile.vector_lanes_per_cube == 2
    assert config.timing_profile.calibration == "pto-perf-sim-derived-fallback"


def test_config_rejects_unsupported_platform_and_mismatched_timing() -> None:
    with pytest.raises(SimulatorConfigError, match="supported platforms: A2, A3"):
        SimulatorConfig(platform="A5")

    a3_timing = TimingProfile(platform="A3", operation_cycles={"mma": 7})
    with pytest.raises(SimulatorConfigError, match="does not match"):
        SimulatorConfig(platform="A2", timing_profile=a3_timing)
    assert SimulatorConfig(platform="A3").timing_profile.estimator == "pto-fallback"


def test_timing_profile_uses_explicit_cost_and_visible_fallback() -> None:
    profile = TimingProfile(platform="A2", operation_cycles={"mma": 23}, fallback_cycles=2)

    assert profile.estimate_cycles("mma") == 23
    assert profile.estimate_cycles("copy") == 2
    assert profile.calibration == "uncalibrated-unit-cost"
    assert get_device_profile("A3").calibration == "uncalibrated"


def test_pto_fallback_timing_profile_ports_public_pipe_formulas() -> None:
    profile = pto_fallback_timing_profile("A2")
    fp16_tile = BufferRegion("tile", MemoryScope.UB, (16, 16), "float16", core_id=0)

    assert profile.calibration == "pto-perf-sim-derived-fallback"
    # PTO fallback: GM = 3 + elements * 2 / 64; MTE1 = 1 + elements / 64;
    # Matrix = 4 + elements / 16; Vector = 2 + elements / 32.
    assert profile.estimate_task("copy", pipe="mte2", metadata={"transfer_bytes": 512, "src": fp16_tile}) == 11
    assert profile.estimate_task("copy", pipe="mte1", metadata={"dst": fp16_tile}) == 5
    assert profile.estimate_task("mma", pipe="m", metadata={"dst": fp16_tile}) == 20
    assert profile.estimate_task("binary", pipe="v", metadata={"dst": fp16_tile}) == 10
    assert profile.estimate_task("set_flag", pipe="mte1", metadata={}) == 1


def test_pto_timing_profile_keeps_explicit_operation_override() -> None:
    profile = TimingProfile(
        platform="A2", estimator="pto-fallback", operation_cycles={"mma": 99}
    )
    assert profile.estimate_task("mma", pipe="m", metadata={}) == 99


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
    gm = BufferRegion("input", MemoryScope.GM, (1024,), "float32")
    l1 = BufferRegion("tile", MemoryScope.L1, (1024,), "float32", core_id=0)
    records = [
        ExecutionRecord(
            "load-0", "copy_gm_to_l1", 0, Lane.CUBE, Pipe.MTE2, 0, 10,
            metadata={"bytes": 4096, "src": gm, "dst": l1},
        ),
        ExecutionRecord(
            "load-1", "copy_gm_to_l1", 0, Lane.CUBE, Pipe.MTE2, 5, 15,
            metadata={
                "memory_dependencies": ("load-0",),
                "queue_enter_cycle": 0,
                "src": gm,
                "dst": l1,
            },
        ),
        ExecutionRecord(
            "mma", "mma", 0, Lane.CUBE, Pipe.MATRIX, 10, 30,
            metadata={"memory_dependencies": ("load-1",)},
        ),
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
    assert stats.operation_counts == {
        "copy_gm_to_l1": 2, "mma": 1, "wait_flag": 1,
    }
    assert stats.memory_bytes_by_path == {"gm->l1": 8192}
    assert stats.load_imbalance_cycles == 18

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
    assert metadata["args"]["schema_version"] == "1.0"
    assert trace["schemaVersion"] == "1.0"
    active_core_events = [
        event for event in trace["traceEvents"]
        if event.get("name") == "active_cores"
    ]
    assert [(event["ts"], event["args"]["active_cores"])
            for event in active_core_events] == [(0, 1), (30, 0)]
    queue_events = [
        event for event in trace["traceEvents"]
        if event.get("name") == "queue_depth"
    ]
    assert [(event["ts"], event["args"]["queue_depth"])
            for event in queue_events] == [(0, 1), (5, 0)]
    flows = [
        event for event in trace["traceEvents"]
        if event.get("cat") == "dependency"
    ]
    assert [(event["ph"], event["args"]["from"], event["args"]["to"])
            for event in flows] == [
        ("s", "load-0", "load-1"), ("f", "load-0", "load-1"),
        ("s", "load-1", "mma"), ("f", "load-1", "mma"),
    ]


def test_empty_stats_are_well_defined() -> None:
    stats = SimulationStats.from_records([])

    assert stats.makespan_cycles == 0
    assert stats.task_count == 0
    assert stats.to_dict()["utilization_by_resource"] == {}
    assert stats.to_dict()["operation_counts"] == {}
    assert stats.to_dict()["memory_bytes_by_path"] == {}
    assert stats.to_dict()["hazard_counts"] == {}
    assert stats.to_dict()["peak_local_memory_bytes_by_scope"] == {}
    assert stats.to_dict()["load_imbalance_cycles"] == 0


def test_stats_skip_unresolved_dynamic_memory_bytes() -> None:
    dynamic = AffineInt.variable("count")
    records = (
        ExecutionRecord(
            "dynamic-copy", "copy_gm_to_ub", 0, Lane.VECTOR_0, Pipe.MTE2, 0, 1,
            metadata={
                "src": BufferRegion("input", MemoryScope.GM, (dynamic,), "float16"),
                "dst": BufferRegion(
                    "tile", MemoryScope.UB, (dynamic,), "float16", core_id=0
                ),
            },
        ),
    )

    assert SimulationStats.from_records(records).memory_bytes_by_path == {}
    assert SimulationStats.from_records(
        records, bindings={"count": 7}
    ).memory_bytes_by_path == {"gm->ub": 14}


def test_stats_do_not_count_vector_operand_bytes_as_memory_transfer() -> None:
    source = BufferRegion("lhs", MemoryScope.UB, (64,), "float32", core_id=0)
    destination = BufferRegion("out", MemoryScope.UB, (64,), "float32", core_id=0)
    record = ExecutionRecord(
        "add", "add", 0, Lane.VECTOR_0, Pipe.VECTOR, 0, 1,
        metadata={"src": source, "dst": destination},
    )

    assert SimulationStats.from_records((record,)).memory_bytes_by_path == {}


def test_stats_count_hazard_diagnostics_by_kind() -> None:
    diagnostics = (
        HazardDiagnostic("read-before-write", "first"),
        HazardDiagnostic("read-before-write", "second"),
        HazardDiagnostic("overlapping-allocation", "third"),
    )

    stats = SimulationStats.from_records((), hazard_diagnostics=diagnostics)

    assert stats.hazard_counts == {
        "read-before-write": 2,
        "overlapping-allocation": 1,
    }


def test_trace_exports_matched_flag_flow() -> None:
    records = (
        ExecutionRecord("set", "set_flag", 0, Lane.CUBE, Pipe.MTE2, 0, 3),
        ExecutionRecord(
            "wait", "wait_flag", 0, Lane.CUBE, Pipe.MTE1, 3, 4,
            metadata={"sync_producers": ("set",)},
        ),
    )

    trace = ChromeTraceExporter("A3", "uncalibrated-unit-cost").to_dict(records)
    flows = [
        event for event in trace["traceEvents"]
        if event.get("name") == "flag_dependency"
    ]
    assert [(event["ph"], event["ts"]) for event in flows] == [("s", 3), ("f", 3)]
    assert all(event["args"] == {"from": "set", "to": "wait"} for event in flows)


def test_trace_active_core_counter_deduplicates_pipe_overlap() -> None:
    records = (
        ExecutionRecord("c0-load", "copy", 0, Lane.CUBE, Pipe.MTE2, 0, 8),
        ExecutionRecord("c0-mma", "mma", 0, Lane.CUBE, Pipe.MATRIX, 4, 12),
        ExecutionRecord("c1-op", "add", 1, Lane.VECTOR_0, Pipe.VECTOR, 5, 10),
        ExecutionRecord(
            "c2-wait", "wait", 2, Lane.CUBE, Pipe.SCALAR, 1, 20,
            category="wait", stall_reason="flag",
        ),
    )

    trace = ChromeTraceExporter("A2", "uncalibrated-unit-cost").to_dict(records)
    counters = [
        event for event in trace["traceEvents"]
        if event.get("name") == "active_cores"
    ]
    assert [(event["ts"], event["args"]["active_cores"])
            for event in counters] == [(0, 1), (5, 2), (10, 1), (12, 0)]


def test_trace_exports_live_local_memory_counters() -> None:
    trace = ChromeTraceExporter("A2", "uncalibrated-unit-cost").to_dict(
        (),
        local_memory_timeline=((0, {"ub": 64}), (7, {})),
    )
    counters = [
        event for event in trace["traceEvents"]
        if event.get("name") == "live_local_memory_bytes"
    ]

    assert [(event["ts"], event["args"]) for event in counters] == [
        (0, {"ub": 64}),
        (7, {}),
    ]
