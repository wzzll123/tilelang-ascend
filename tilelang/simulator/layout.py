# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Physical A2/A3 matrix-layout codecs used by Cube simulation."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .errors import ProgramValidationError, UnsupportedSimOpError


BYTE_PER_C0 = 32
C0_NUM_PER_FRACTAL = 16
BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _round_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def _validate(shape: Tuple[int, int], itemsize: int) -> Tuple[int, int, int]:
    if len(shape) != 2 or any(value < 0 for value in shape):
        raise ProgramValidationError(f"matrix layout requires a non-negative 2D shape, got {shape}")
    if itemsize <= 0 or BYTE_PER_C0 % itemsize:
        raise ProgramValidationError(
            f"matrix layout item size must divide {BYTE_PER_C0} bytes, got {itemsize}"
        )
    return shape[0], shape[1], BYTE_PER_C0 // itemsize


def storage_elements(layout: str, shape: Tuple[int, int], itemsize: int) -> int:
    """Return padded physical element capacity for a logical matrix."""
    rows, cols, elements_per_c0 = _validate(shape, itemsize)
    normalized = layout.strip().lower()
    if normalized == "row_major":
        return rows * cols
    if normalized == "l0c":
        return _round_up(rows, C0_NUM_PER_FRACTAL) * _round_up(
            cols, C0_NUM_PER_FRACTAL
        )
    if normalized in {"zn", "zz"}:
        return _round_up(rows, C0_NUM_PER_FRACTAL) * _round_up(cols, elements_per_c0)
    if normalized == "nz":
        return _round_up(rows, elements_per_c0) * _round_up(cols, C0_NUM_PER_FRACTAL)
    raise UnsupportedSimOpError(f"unsupported A2/A3 matrix layout {layout!r}")


def physical_index(
    layout: str,
    row: int,
    col: int,
    shape: Tuple[int, int],
    itemsize: int,
) -> int:
    """Map one logical matrix coordinate to its physical element offset.

    zN/nZ follow ``tilelang/intrinsics/ascend_layout.py``; zZ and L0C follow
    the Catlass ``layout::zZ::MakeLayout`` and ``tla::MakeLayoutL0C`` strides.
    """
    rows, cols, elements_per_c0 = _validate(shape, itemsize)
    if not (0 <= row < rows and 0 <= col < cols):
        raise ProgramValidationError(
            f"logical coordinate {(row, col)} is outside matrix shape {shape}"
        )
    elements_per_fractal = BYTE_PER_FRACTAL // itemsize
    normalized = layout.strip().lower()
    if normalized == "row_major":
        return row * cols + col
    if normalized == "l0c":
        return (
            row // C0_NUM_PER_FRACTAL
            * C0_NUM_PER_FRACTAL
            * C0_NUM_PER_FRACTAL
            + col // C0_NUM_PER_FRACTAL
            * _round_up(rows, C0_NUM_PER_FRACTAL)
            * C0_NUM_PER_FRACTAL
            + row % C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL
            + col % C0_NUM_PER_FRACTAL
        )
    if normalized == "zn":
        return (
            row // C0_NUM_PER_FRACTAL * elements_per_fractal
            + col // elements_per_c0
            * _round_up(rows, C0_NUM_PER_FRACTAL)
            * elements_per_c0
            + row % C0_NUM_PER_FRACTAL * elements_per_c0
            + col % elements_per_c0
        )
    if normalized == "zz":
        return (
            row // C0_NUM_PER_FRACTAL
            * _round_up(cols, elements_per_c0)
            * C0_NUM_PER_FRACTAL
            + col // elements_per_c0 * elements_per_fractal
            + row % C0_NUM_PER_FRACTAL * elements_per_c0
            + col % elements_per_c0
        )
    if normalized == "nz":
        return (
            row // elements_per_c0
            * _round_up(cols, C0_NUM_PER_FRACTAL)
            * elements_per_c0
            + col // C0_NUM_PER_FRACTAL * elements_per_fractal
            + row % elements_per_c0
            + col % C0_NUM_PER_FRACTAL * elements_per_c0
        )
    raise UnsupportedSimOpError(f"unsupported A2/A3 matrix layout {layout!r}")


def pack_matrix(values: np.ndarray, layout: str) -> np.ndarray:
    """Pack a logical 2D NumPy matrix into padded physical layout storage."""
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ProgramValidationError(f"matrix layout pack requires rank 2, got {matrix.shape}")
    shape = (matrix.shape[0], matrix.shape[1])
    result = np.zeros(storage_elements(layout, shape, matrix.dtype.itemsize), dtype=matrix.dtype)
    for row in range(shape[0]):
        for col in range(shape[1]):
            index = physical_index(layout, row, col, shape, matrix.dtype.itemsize)
            result[index] = matrix[row, col]
    return result


def unpack_matrix(storage: np.ndarray, layout: str, shape: Tuple[int, int]) -> np.ndarray:
    """Read a logical 2D matrix from padded physical layout storage."""
    physical = np.asarray(storage).reshape(-1)
    required = storage_elements(layout, shape, physical.dtype.itemsize)
    if physical.size < required:
        raise ProgramValidationError(
            f"{layout} storage has {physical.size} elements, requires {required} for {shape}"
        )
    result = np.empty(shape, dtype=physical.dtype)
    for row in range(shape[0]):
        for col in range(shape[1]):
            result[row, col] = physical[
                physical_index(layout, row, col, shape, physical.dtype.itemsize)
            ]
    return result
