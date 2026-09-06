# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Regression: cross-core same-lane GM/workspace accesses must not order by task.

Root cause under test: the bridge's ``_memory_dependencies`` tracked
``last_writes``/``last_reads`` per *lane* (cube/vector0/vector1) but not per
*core*.  Two different cores' accesses to one shared GM/workspace buffer (e.g.
the MSD X-quant reduceMax workspace: core i writes row i before a mode-0
all-vector barrier, core j reads rows [0, M) after it) overlap in bytes, so the
bridge emitted a cross-core source-order dependency.  On real hardware
cross-core visibility of shared GM is carried only by cross-core
flags/barriers, never by task order; the spurious edge runs a later-scheduled
core's pre-barrier write behind another core's post-barrier read and deadlocks
the collective barrier.

Fix: ``_memory_dependencies`` skips GM/WORKSPACE edges between different cores
(``_same_on_chip_owner``); on-chip scopes (UB/L1/L0) keep per-core edges.

These tests are RED until cross-core GM/workspace edges are dropped.
"""

import pytest

from tilelang.simulator import BufferRegion, MemoryScope
from tilelang.simulator.bridge import _TirBridge
from tilelang.simulator.program import BufferSpec


def _ws_region(byte_offset: int) -> BufferRegion:
    """One core's (1, 8) fp32 row shard of a shared (2, M, 8) workspace."""
    return BufferRegion(
        buffer="WS",
        scope=MemoryScope.WORKSPACE,
        shape=(1, 8),
        dtype="float32",
        byte_offset=byte_offset,
        strides_bytes=(8 * 4, 4),
        core_id=None,
    )


def _bare_bridge() -> _TirBridge:
    bridge = _TirBridge.__new__(_TirBridge)
    bridge.buffers = {
        "WS": BufferSpec("WS", MemoryScope.WORKSPACE, (2, 64, 8), "float32"),
    }
    bridge.active_aliases = {}
    return bridge


def test_cross_core_workspace_overlap_is_not_a_dependency() -> None:
    """core1's write overlapping core0's read of shared workspace: no edge."""
    bridge = _bare_bridge()
    # core0 reads rows [0,4) (byte 0..128) after the barrier; core1 writes row 2
    # (byte 64..96) before it.  They overlap in bytes but are on different cores.
    read = _ws_region(0)
    read = BufferRegion(
        buffer="WS", scope=MemoryScope.WORKSPACE, shape=(4, 8), dtype="float32",
        byte_offset=0, strides_bytes=(8 * 4, 4), core_id=None,
    )
    write = _ws_region(64)
    # Genuine byte overlap (write row 2 sits inside read rows [0,4)).
    assert bridge._regions_overlap(read, write, core_id=1)
    # ... but the dependency predicate must drop it (different cores).
    assert not bridge._same_on_chip_owner(write, read, core_id=1, previous_core=0)
    assert not bridge._same_on_chip_owner(read, write, core_id=0, previous_core=1)


def test_same_core_workspace_overlap_keeps_dependency() -> None:
    """Guard: same-core GM/workspace edges are still meaningful (FIFO order)."""
    bridge = _bare_bridge()
    region = _ws_region(64)
    assert bridge._same_on_chip_owner(region, region, core_id=1, previous_core=1)


def test_on_chip_scope_unaffected_by_cross_core_rule() -> None:
    """Guard: on-chip (UB) regions keep their dependency regardless of core."""
    bridge = _bare_bridge()
    ub = BufferRegion(
        buffer="U", scope=MemoryScope.UB, shape=(8,), dtype="float32",
        byte_offset=0, strides_bytes=(4,), core_id=None,
    )
    # UB is per-core; the same-core/cross-core distinction is handled by
    # _regions_overlap (owner check).  _same_on_chip_owner must not drop it.
    assert bridge._same_on_chip_owner(ub, ub, core_id=1, previous_core=0)
