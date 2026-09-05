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
    [
        ("zN", (17, 9), 512),
        ("zZ", (17, 9), 512),
        ("nZ", (9, 17), 512),
        ("l0c", (17, 9), 512),
    ],
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


def test_zz_uses_easyasc_a2_l0a_physical_order() -> None:
    assert physical_index("zZ", 16, 0, (32, 32), 2) == 512
    assert physical_index("zZ", 0, 16, (32, 32), 2) == 256
    assert physical_index("zZ", 1, 0, (32, 32), 2) == 1
    assert physical_index("zZ", 0, 1, (32, 32), 2) == 16

    logical = np.array([[1, 2], [3, 4]], dtype=np.float16)
    packed = pack_matrix(logical, "zZ")
    np.testing.assert_array_equal(packed[[0, 1, 16, 17]], [1, 3, 2, 4])


def test_l0_operand_roles_match_easyasc_zz_and_nz_payloads() -> None:
    logical = np.array([[1, 2], [3, 4]], dtype=np.float16)
    l0a = pack_matrix(logical, "l0a")
    l0b = pack_matrix(logical, "l0b")

    # EasyASC encode_nd_to_zz: [row_block, aligned_col, inner_row].
    np.testing.assert_array_equal(l0a[[0, 1, 16, 17]], [1, 3, 2, 4])
    # EasyASC encode_nd_to_nz: [col_block, aligned_row, C0_col].
    np.testing.assert_array_equal(l0b[[0, 1, 16, 17]], [1, 3, 2, 4])
    np.testing.assert_array_equal(unpack_matrix(l0a, "l0a", (2, 2)), logical)
    np.testing.assert_array_equal(unpack_matrix(l0b, "l0b", (2, 2)), logical)


def test_l0c_uses_fixed_16_by_16_accumulator_fractals() -> None:
    assert physical_index("l0c", 16, 0, (17, 17), 4) == 256
    assert physical_index("l0c", 0, 16, (17, 17), 4) == 512


def test_layout_codec_rejects_unknown_layout_and_short_storage() -> None:
    with pytest.raises(UnsupportedSimOpError, match="unsupported.*layout"):
        storage_elements("future-layout", (16, 16), 2)
    with pytest.raises(ProgramValidationError, match="requires 256"):
        unpack_matrix(np.zeros(16, dtype=np.float16), "zN", (16, 16))
