# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Regression coverage for 32-byte-aligned rebased zN GM-to-L1 copies."""

import numpy as np
import pytest

from testing.python.simulator.test_tir_bridge import _gm_to_l1_zn_splice_primfunc
from tilelang.simulator import FunctionalSimulator, build_kernel_program
from tilelang.simulator.layout import physical_index


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_zn_gm_to_l1_accepts_32_byte_aligned_rebased_zN_destination(platform) -> None:
    program = build_kernel_program(
        _gm_to_l1_zn_splice_primfunc(
            second_dma=False,
            dst_offset_elements=16,
        ),
        platform=platform,
    )
    (task,) = program.tasks

    assert task.metadata["copy"]["rebased_zN"] is True
    regions = task.metadata["dst_regions"]
    assert len(regions) == 16
    assert [region.byte_offset for region in regions] == [16 * 2 + physical_index("zN", row, 0, (48, 16), 2) * 2 for row in range(16)]

    source = np.arange(16 * 16, dtype=np.float16).reshape(16, 16) + 5
    simulator = FunctionalSimulator(program)
    simulator.write(task.metadata["src"], source)
    simulator.run()
    written = np.concatenate([simulator.read(region) for region in regions]).reshape(16, 16)
    np.testing.assert_array_equal(written, source)
