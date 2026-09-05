# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Minimal reproduction for the wqbm singleNLoop=3 simulator deadlock.

Root cause under test: the bridge's memory-dependency overlap check uses a
bounding-box approximation for strided regions.  Two per-core GM output shards
that are actually disjoint (different column strips of one row-major matrix)
have overlapping bounding boxes, so the bridge emits a spurious cross-core
task dependency.  On real hardware cross-core ordering is carried only by
flags/barriers, never by task order; the spurious edge pins one core's vector
lane behind another core and, combined with the mode-0 all-core barrier and
the mode-2 V2C/C2V credit chain, deadlocks the schedule.

These tests are RED until the overlap check compares strided regions exactly.
"""

import pytest

from tilelang.simulator import (
    BufferRegion,
    CoreProgram,
    DiscreteEventScheduler,
    FlagBarrierSynchronizationModel,
    KernelProgram,
    Lane,
    MemoryScope,
    Pipe,
    Task,
)
from tilelang.simulator.bridge import _TirBridge, _region_bounds
from tilelang.simulator.program import BufferSpec


def _gm_shard(byte_offset: int) -> BufferRegion:
    """One core's (8, 128) bf16 column strip of a row-major (8, 7680) GM matrix."""
    return BufferRegion(
        buffer="Y",
        scope=MemoryScope.GM,
        shape=(8, 128),
        dtype="bfloat16",
        byte_offset=byte_offset,
        strides_bytes=(7680 * 2, 2),  # row stride 15360 B, column stride 2 B
        core_id=None,
    )


def _bare_bridge() -> _TirBridge:
    bridge = _TirBridge.__new__(_TirBridge)
    bridge.buffers = {
        "Y": BufferSpec("Y", MemoryScope.GM, (8, 7680), "bfloat16"),
    }
    bridge.active_aliases = {}
    return bridge


def test_strided_gm_shards_with_disjoint_columns_do_not_overlap() -> None:
    """Bounding boxes overlap but the actual byte ranges are disjoint.

    core0 writes columns [5120, 5376) -> byte_offset 10240;
    core1 writes columns [128, 256)  -> byte_offset 256.  Each row occupies a
    256-byte strip, so no byte is shared, yet the bounding boxes
    [10240, 118016) and [256, 108032) intersect.
    """
    core0 = _gm_shard(10240)
    core1 = _gm_shard(256)

    # Exact strided intersection: row r of each shard covers
    # [off + r*15360, off + r*15360 + 256).  These never meet.
    def rows(off):
        return [(off + r * 15360, off + r * 15360 + 256) for r in range(8)]

    assert not any(
        a0 < b1 and b0 < a1
        for a0, a1 in rows(10240)
        for b0, b1 in rows(256)
    ), "test premise broken: shards must be truly disjoint"

    # The bounds helper currently returns the bounding box, which overlaps.
    assert _region_bounds(core0) == (10240, 118016)
    assert _region_bounds(core1) == (256, 108032)

    # The overlap predicate must not confuse the bounding box with real overlap.
    bridge = _bare_bridge()
    assert not bridge._regions_overlap(core0, core1, core_id=1)
    assert not bridge._regions_overlap(core1, core0, core_id=0)


def test_overlapping_strided_gm_shards_still_overlap() -> None:
    """Guard: a genuine overlap must still be detected after the fix."""
    bridge = _bare_bridge()
    # Same column strip, adjacent rows would overlap; here identical offset.
    assert bridge._regions_overlap(_gm_shard(10240), _gm_shard(10240), core_id=0)
    # core0 row 0 covers [10240, 10496); a shard starting at 10300 overlaps it.
    assert bridge._regions_overlap(_gm_shard(10240), _gm_shard(10300), core_id=0)


def _mode0_chain_program(num_cores: int, num_rounds: int) -> KernelProgram:
    """mode-0 vector barrier + mode-2 V2C/C2V credit chain.

    Each round every vector lane (both vector0 and vector1) sets the mode-0
    collective and waits it; vector0 then sets a mode-2 V2C credit.  The cube
    waits V2C, does work, and returns a C2V credit.  This pattern alone (no
    spurious cross-core edge) must schedule.
    """
    cores = []
    for core_id in range(num_cores):
        tasks = []
        for rnd in range(num_rounds):
            for lane in (Lane.VECTOR_0, Lane.VECTOR_1):
                tag = "v0" if lane is Lane.VECTOR_0 else "v1"
                tasks.append(Task(
                    f"c{core_id}-{tag}-set-{rnd}", "set_cross_flag", core_id,
                    lane, Pipe.VECTOR, 1,
                    metadata={"flag_id": 6, "mode": 0, "src_pipe": "v"},
                ))
                tasks.append(Task(
                    f"c{core_id}-{tag}-wait-{rnd}", "wait_cross_flag", core_id,
                    lane, Pipe.VECTOR, 1,
                    metadata={"flag_id": 6},
                ))
            tasks.append(Task(
                f"c{core_id}-v0-v2c-{rnd}", "set_cross_flag", core_id,
                Lane.VECTOR_0, Pipe.VECTOR, 1,
                metadata={"flag_id": 8, "mode": 2, "src_pipe": "v"},
            ))
            tasks.append(Task(
                f"c{core_id}-v1-v2c-{rnd}", "set_cross_flag", core_id,
                Lane.VECTOR_1, Pipe.VECTOR, 1,
                metadata={"flag_id": 8, "mode": 2, "src_pipe": "v"},
            ))
            tasks.append(Task(
                f"c{core_id}-cube-wait-{rnd}", "wait_cross_flag", core_id,
                Lane.CUBE, Pipe.MATRIX, 1,
                metadata={"flag_id": 8},
            ))
            tasks.append(Task(
                f"c{core_id}-cube-c2v-{rnd}", "set_cross_flag", core_id,
                Lane.CUBE, Pipe.MATRIX, 1,
                metadata={"flag_id": 9, "mode": 2, "src_pipe": "m"},
            ))
        cores.append(CoreProgram(core_id, tuple(tasks)))
    return KernelProgram("mode0_chain", "A2", tuple(cores))


def test_mode0_multi_phase_with_mode2_credit_chain_schedules() -> None:
    """The sync pattern itself is sound and must schedule without deadlock."""
    program = _mode0_chain_program(num_cores=4, num_rounds=3)
    result = DiscreteEventScheduler(
        synchronization=FlagBarrierSynchronizationModel()
    ).run(program)
    # Scheduling the full pattern without deadlock is the assertion; the sync
    # model may also emit wait records, so only check the real tasks ran.
    real = [r for r in result.records if r.category != "wait"]
    assert len(real) == 4 * 3 * 8
