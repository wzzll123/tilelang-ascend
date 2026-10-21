# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""UnitFlag physical-block ownership, including split MMA / combined FIX."""
from dataclasses import replace

import pytest

from tilelang.simulator import (
    BufferRegion,
    BufferSpec,
    CoreProgram,
    DiscreteEventScheduler,
    KernelProgram,
    Lane,
    MemoryScope,
    Pipe,
    Task,
    ProgramValidationError,
)
from tilelang.simulator.bridge import _TirBridge
from tilelang.simulator.profile import default_timing_profile


def make_bridge(operations):
    bridge = _TirBridge(
        tvm=None,
        tir=None,
        analyzer=None,
        platform="A2",
        timing_profile=default_timing_profile("A2"),
        max_unrolled_iterations=65536)
    bridge.buffers["c"] = BufferSpec("c", MemoryScope.L0C, (512,), "float32")
    tasks = []
    for index, (kind, offset, elements) in enumerate(operations):
        region = BufferRegion(
            "c", MemoryScope.L0C, (elements,), "float32", byte_offset=offset, core_id=0)
        metadata = ({
            "dst": region,
            "mma": {
                "unit_flag": 3,
                "n_actual": 16
            }
        } if kind == "m" else {
            "src": region,
            "copy": {
                "unit_flag": 3,
                "destination_shape": (16, 32)
            }
        })
        tasks.append(
            Task(
                str(index),
                "mma" if kind == "m" else "copy_l0c_to_gm",
                0,
                Lane.CUBE,
                Pipe.MATRIX if kind == "m" else Pipe.FIX,
                10,
                metadata=metadata))
    bridge.tasks[0] = tasks
    return bridge


def test_two_half_mmas_feed_one_wide_fix():
    bridge = make_bridge([("m", 0, 256), ("m", 1024, 256), ("f", 0, 512)])
    bridge._validate_unit_flag_pairs()
    assert bridge.tasks[0][2].metadata["unit_flag_pairs"] == ("0", "1")
    assert set(bridge.tasks[0][2].metadata["hardware_dependencies"]) == {"0", "1"}


def test_physical_alias_view_matches_same_unit_blocks():
    bridge = make_bridge([("m", 0, 256), ("m", 1024, 256), ("f", 0, 512)])
    bridge.buffers["c"] = replace(bridge.buffers["c"], address=0)
    bridge.buffers["wide"] = BufferSpec("wide", MemoryScope.L0C, (512,), "float32", address=0)
    fix = bridge.tasks[0][2]
    metadata = dict(fix.metadata)
    metadata["src"] = replace(metadata["src"], buffer="wide")
    bridge.tasks[0][2] = replace(fix, metadata=metadata)
    bridge._validate_unit_flag_pairs()
    assert bridge.tasks[0][2].metadata["unit_flag_pairs"] == ("0", "1")


def test_one_wide_mma_feeds_two_half_fixes_and_reuse_waits():
    bridge = make_bridge([("m", 0, 512), ("f", 0, 256), ("f", 1024, 256), ("m", 0, 512),
                          ("f", 0, 512)])
    bridge._validate_unit_flag_pairs()
    assert set(bridge.tasks[0][3].metadata["hardware_dependencies"]) == {"1", "2"}
    program = KernelProgram("unitflag", "A2", (CoreProgram(0, tuple(bridge.tasks[0])),),
                            tuple(bridge.buffers.values()))
    result = DiscreteEventScheduler().run(program)
    records = {
        record.task_id: record for record in result.records if record.category == "operation"
    }
    assert records["1"].start_cycle >= records["0"].end_cycle
    assert records["2"].start_cycle >= records["0"].end_cycle
    assert records["3"].start_cycle >= records["2"].end_cycle


@pytest.mark.parametrize(
    "operations",
    [
        [("m", 0, 256), ("f", 0, 512)],  # missing right half
        [("m", 0, 512), ("f", 0, 512), ("f", 0, 512)],  # double consume
        [("m", 0, 512), ("m", 0, 512), ("f", 0, 512)],  # overwrite unread data
        [("m", 0, 512), ("f", 0, 256)],  # incomplete drain
    ])
def test_bad_block_protocol_remains_rejected(operations):
    with pytest.raises(ProgramValidationError, match="unitFlag"):
        make_bridge(operations)._validate_unit_flag_pairs()
