# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Sync-only simulator mode: parity, mutation detection, poison, and speed."""

import time

import numpy as np
import pytest

from tilelang.simulator import (
    AddressRange,
    BufferRegion,
    BufferSpec,
    CoreProgram,
    FunctionalSimulator,
    KernelProgram,
    Lane,
    MemoryScope,
    MemoryRuntime,
    Pipe,
    SimulationDeadlockError,
    SimulatorConfig,
    SimulatorConfigError,
    Task,
    UnsupportedSimOpError,
)
from tilelang.simulator.adapter import _resolve_config
from tilelang.simulator.layout import pack_matrix, storage_elements


def _flag_program() -> KernelProgram:
    tile = BufferRegion("tile", MemoryScope.UB, (32,), "float32")
    tasks = (
        Task(
            "produce",
            "fill",
            0,
            Lane.VECTOR_0,
            Pipe.MTE2,
            2,
            metadata={"dst": tile, "scalar": 1.0},
        ),
        Task(
            "set",
            "set_flag",
            0,
            Lane.VECTOR_0,
            Pipe.SCALAR,
            1,
            metadata={"src_pipe": "mte2", "dst_pipe": "v", "flag_id": 3},
        ),
        Task(
            "wait",
            "wait_flag",
            0,
            Lane.VECTOR_0,
            Pipe.VECTOR,
            1,
            metadata={"src_pipe": "mte2", "dst_pipe": "v", "flag_id": 3},
        ),
    )
    return KernelProgram(
        "sync-only-parity",
        "A2",
        (CoreProgram(0, tasks),),
        buffers=(BufferSpec("tile", MemoryScope.UB, (32,), "float32"),),
    )


def _run(program: KernelProgram, *, sync_only: bool):
    return FunctionalSimulator(
        program,
        SimulatorConfig(platform=program.platform, sync_only=sync_only),
    ).run()


def test_sync_only_config_mapping_and_validation() -> None:
    config = _resolve_config("A2", {"sync_only": True})

    assert config.sync_only is True
    with pytest.raises(SimulatorConfigError, match="sync_only must be a boolean"):
        SimulatorConfig(sync_only=1)


def test_sync_only_keeps_unknown_operations_fail_closed() -> None:
    program = KernelProgram(
        "unknown",
        "A2",
        (CoreProgram(0, (Task("unknown", "unknown_compute", 0, Lane.VECTOR_0, Pipe.VECTOR, 1),)),),
    )

    with pytest.raises(UnsupportedSimOpError, match="sync-only.*unknown_compute"):
        _run(program, sync_only=True)


def test_sync_only_preserves_flag_schedule_and_hazard_conclusion() -> None:
    full = _run(_flag_program(), sync_only=False)
    skeleton = _run(_flag_program(), sync_only=True)

    def signature(result):
        return [
            (
                record.task_id,
                record.start_cycle,
                record.end_cycle,
                record.stall_reason,
                tuple(record.metadata.get("sync_producers", ())),
            )
            for record in result.schedule.records
        ]

    assert signature(skeleton) == signature(full)
    assert skeleton.schedule.stats.hazard_counts == full.schedule.stats.hazard_counts
    assert full.numeric_results_available is True
    assert skeleton.numeric_results_available is False


def test_sync_only_rejects_numeric_output_reads() -> None:
    program = _flag_program()
    simulator = FunctionalSimulator(program, SimulatorConfig(platform="A2", sync_only=True))
    simulator.run()

    with pytest.raises(UnsupportedSimOpError, match="numeric results.*unavailable"):
        simulator.read(BufferRegion("tile", MemoryScope.UB, (32,), "float32"))


@pytest.mark.parametrize("sync_only", [False, True])
def test_sync_only_mutation_keeps_deadlock_flag_location(sync_only: bool) -> None:
    broken = KernelProgram(
        "missing-set",
        "A3",
        (
            CoreProgram(
                0,
                (
                    Task(
                        "wait",
                        "wait_flag",
                        0,
                        Lane.VECTOR_1,
                        Pipe.VECTOR,
                        1,
                        metadata={
                            "src_pipe": "mte2",
                            "dst_pipe": "v",
                            "flag_id": 7,
                        },
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        SimulationDeadlockError,
        match=r"wait.*local flag.*vector1.*id=7",
    ):
        _run(broken, sync_only=sync_only)


def test_sync_only_propagates_poison_without_copying_values() -> None:
    source = BufferRegion("source", MemoryScope.GM, (64,), "float32")
    destination = BufferRegion("destination", MemoryScope.UB, (64,), "float32")
    program = KernelProgram(
        "poison-copy",
        "A2",
        (
            CoreProgram(
                0,
                (
                    Task(
                        "copy",
                        "copy_gm_to_ub",
                        0,
                        Lane.VECTOR_0,
                        Pipe.MTE2,
                        1,
                        metadata={"src": source, "dst": destination},
                    ),
                ),
            ),
        ),
        buffers=(
            BufferSpec("source", MemoryScope.GM, (64,), "float32"),
            BufferSpec("destination", MemoryScope.UB, (64,), "float32"),
        ),
    )
    simulator = FunctionalSimulator(
        program,
        SimulatorConfig(platform="A2", hazard_check="warn", sync_only=True),
    )

    with pytest.warns(RuntimeWarning, match="read-before-write"):
        result = simulator.run()

    allocation = simulator.memory.get("destination", scope=MemoryScope.UB, core_id=0)
    assert not allocation.initialized()
    assert result.schedule.stats.hazard_counts == {"read-before-write": 1}


def test_sync_only_metadata_coalesces_dense_transposed_views() -> None:
    allocation = MemoryRuntime((0,), hazard_check="off").allocate(BufferSpec("matrix", MemoryScope.GM, (2, 3), "float16"))
    transposed = allocation.view(shape=(2, 3), strides_bytes=(2, 4))

    assert len(transposed.address_ranges) == 6
    assert transposed.physical_address_ranges == (AddressRange(0, 12),)
    allocation.mark_initialized(transposed)
    assert allocation.initialized()


def test_sync_only_metadata_preserves_sparse_stride_gaps() -> None:
    allocation = MemoryRuntime((0,), hazard_check="off").allocate(BufferSpec("matrix", MemoryScope.GM, (4, 4), "float16"))
    sparse = allocation.view(byte_offset=2, shape=(2, 2), strides_bytes=(8, 2))

    allocation.mark_initialized(sparse)

    assert sparse.physical_address_ranges == (
        AddressRange(2, 6),
        AddressRange(10, 14),
    )
    assert not allocation.initialized(AddressRange(6, 10))


def _measure_compute_heavy_mma() -> tuple[float, float]:
    rows = cols = inner = 128
    a_elements = storage_elements("l0a", (rows, inner), 2)
    b_elements = storage_elements("l0b", (inner, cols), 2)
    c_elements = storage_elements("l0c", (rows, cols), 4)
    left = BufferRegion("left", MemoryScope.L0A, (a_elements,), "float16")
    right = BufferRegion("right", MemoryScope.L0B, (b_elements,), "float16")
    destination = BufferRegion("destination", MemoryScope.L0C, (c_elements,), "float32")
    tasks = tuple(
        Task(
            f"mma-{index}",
            "mma",
            0,
            Lane.CUBE,
            Pipe.MATRIX,
            1,
            metadata={
                "lhs": left,
                "rhs": right,
                "dst": destination,
                "mma": {
                    "rows": rows,
                    "cols": cols,
                    "inner": inner,
                    "n_actual": cols,
                    "init": True,
                },
            },
        )
        for index in range(64)
    )
    program = KernelProgram(
        "sync-only-speed",
        "A2",
        (CoreProgram(0, tasks),),
        buffers=(
            BufferSpec("left", MemoryScope.L0A, (a_elements,), "float16"),
            BufferSpec("right", MemoryScope.L0B, (b_elements,), "float16"),
            BufferSpec("destination", MemoryScope.L0C, (c_elements,), "float32"),
        ),
    )

    def timed(sync_only: bool) -> float:
        simulator = FunctionalSimulator(program, SimulatorConfig(platform="A2", sync_only=sync_only))
        simulator.write(
            left,
            pack_matrix(np.ones((rows, inner), dtype=np.float16), "l0a"),
        )
        simulator.write(
            right,
            pack_matrix(np.ones((inner, cols), dtype=np.float16), "l0b"),
        )
        started = time.perf_counter()
        simulator.run()
        return time.perf_counter() - started

    full_seconds = timed(False)
    sync_seconds = timed(True)

    return full_seconds, sync_seconds


def test_sync_only_is_at_least_ten_times_faster_for_compute_heavy_mma() -> None:
    full_seconds, sync_seconds = _measure_compute_heavy_mma()

    assert full_seconds / max(sync_seconds, 1e-9) >= 10, (
        f"sync_only speedup was {full_seconds / max(sync_seconds, 1e-9):.2f}x (full={full_seconds:.6f}s, sync_only={sync_seconds:.6f}s)"
    )
