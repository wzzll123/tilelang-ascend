# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Tests for physical Cube matrix layouts."""

import numpy as np
import pytest

from tilelang.simulator.errors import ProgramValidationError, UnsupportedSimOpError
from tilelang.simulator.layout import (
    pack_matrix,
    physical_index,
    storage_elements,
    unpack_matrix,
)


@pytest.mark.parametrize(
    ("layout", "shape", "expected_elements"),
    [("zN", (17, 9), 512), ("nZ", (9, 17), 512)],
)
def test_fractal_layout_round_trip(layout, shape, expected_elements) -> None:
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    packed = pack_matrix(values, layout)

    assert storage_elements(layout, shape, 4) == expected_elements
    assert packed.size == expected_elements
    np.testing.assert_array_equal(unpack_matrix(packed, layout, shape), values)


def test_zn_and_nz_use_repository_coordinate_formulas() -> None:
    assert physical_index("zN", 16, 8, (17, 9), 4) == 384
    assert physical_index("nZ", 8, 16, (9, 17), 4) == 384


def test_layout_codec_rejects_unknown_layout_and_short_storage() -> None:
    with pytest.raises(UnsupportedSimOpError, match="unsupported.*layout"):
        storage_elements("future-layout", (16, 16), 2)
    with pytest.raises(ProgramValidationError, match="requires 256"):
        unpack_matrix(np.zeros(16, dtype=np.float16), "zN", (16, 16))
