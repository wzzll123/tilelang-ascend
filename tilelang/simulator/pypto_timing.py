# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Adapted for TileLang-Ascend.
# This program is free software, you can redistribute it and/or modify it under the terms
# and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License. See licenses/CANN-Open-Software-License-2.0.txt.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
"""CANN/PyPTO fast CostModel, not a measured A2/A3 hardware calibration.

Source: https://gitcode.com/cann/pypto, revision
49587f393680c6bc814b00e7ed4b99956a1aa979:
framework/src/cost_model/simulation/arch/PipeSimulatorFast.cpp and
framework/src/cost_model/simulation/arch/A2A3/PostSimulatorA2A3.h.

Keep the upstream coefficients, float arithmetic, minimum operand shape,
parallel ratio and fallback semantics. This does NOT port the optional cache
machine, PV simulator, or scheduler. Units are upstream model cycles.
"""

from collections.abc import Mapping
from math import prod
from struct import pack, unpack
from types import MappingProxyType

OP_LATENCY = MappingProxyType({
    "COPY_IN": (2.613, 0.00240884),
    "L1_COPY_IN": (2.613, 0.00240884),
    "UB_COPY_IN": (2.613, 0.00240884),
    "COPY_OUT": (3.2883, 0.00195),
    "L0C_COPY_OUT": (3.2883, 0.00195),
    "UB_COPY_OUT": (3.2883, 0.00195),
    "ADD": (2.40457, 0.0188),
    "SUB": (1.64044, 0.02152),
    "MUL": (1.63706, 0.01419),
    "DIV": (3.82854, 0.02462),
    "ADDS": (0.76264, 0.0221),
    "SUBS": (0.76264, 0.0221),
    "RANGE": (0.76264, 0.0221),
    "MULS": (0.9527, 0.015),
    "DIVS": (0.9527, 0.015),
    "SQRT": (3.67358, 0.02197),
    "EXP": (3.49358, 0.02597),
    "NEG": (0.9527, 0.015),
    "LN": (3.49358, 0.02597),
    "WHERE_TT": (2.40457, 0.0188),
    "WHERE_TS": (2.40457, 0.0188),
    "WHERE_ST": (2.40457, 0.0188),
    "WHERE_SS": (2.40457, 0.0188),
    "RECIPROCAL": (0.98673, 0.00989),
    "ROWSUMLINE": (14.58731, 0.07288),
    "ROWMAX_SINGLE": (15.58731, 0.00888),
    "ROWMIN_SINGLE": (15.58731, 0.00888),
    "EXPAND": (0.58, 0.014),
    "CAST": (0.91250, 0.015687),
    "TRANSPOSE_MOVEOUT": (0.00025893, 0.03350),
    "L1_TO_L0B": (2.05893, 0.012),
    "L1_TO_L0A": (1.12719, 0.015),
    "L1_TO_L0Bt": (1.868830, 0.015),
    "A_MUL_B": (3.91153, 0.0107),
    "A_MUL_Bt": (3.91153, 0.0110),
    "A_MULACC_B": (3.90376, 0.0103),
    "A_MULACC_Bt": (3.90376, 0.0105),
    "PAIRMAX": (19.28672, 0.0),
    "PAIRSUM": (19.58115, 0.0),
    "MAXIMUM": (308.7260, 0.0),
})

INHERENT_LATENCY = {
    "UB_ADD": {
        "float32": 6,
        "int32": 4,
        "int16": 4
    },
    "UB_SUB": {
        "float16": 6,
        "float32": 6,
        "int32": 4,
        "int16": 4
    },
    "UB_MUL": {
        "float16": 7,
        "float32": 7,
        "int32": 5,
        "int16": 5
    },
    "UB_DIV": {
        "float16": 13,
        "float32": 13
    },
    "UB_EXP": {
        "float16": 12,
        "float32": 12
    },
    "UB_SQRT": {
        "float16": 13,
        "float32": 13
    },
    "UB_RSQRT": {
        "float16": 5,
        "float32": 5
    },
    "UB_ADDS": {
        "float16": 6,
        "float32": 6,
        "int32": 4,
        "int16": 4
    },
    "UB_MULS": {
        "float16": 7,
        "float32": 7,
        "int32": 5,
        "int16": 5
    },
    "UB_PAIRSUM": {
        "float16": 6,
        "float32": 6,
        "int32": 4,
        "int16": 4
    },
    "UB_PAIRMAX": {
        "float16": 4,
        "float32": 4,
        "int32": 4,
        "int16": 4
    },
    "UB_ROWEXPSUM": {
        "float16": 23,
        "float32": 20
    },
    "UB_ROWEXPMAX": {
        "float16": 23,
        "float32": 20
    },
    "UB_MAXIMUM": {
        "float16": 4,
        "float32": 4,
        "int32": 4,
        "int16": 4
    },
    "UB_COMPACT": {
        "float16": 0,
        "float32": 0,
        "int32": 0,
        "int16": 0
    },
    "UB_EXPAND": {
        "float16": 0,
        "float32": 0,
        "int32": 0,
        "int16": 0
    },
    "UB_MOV": {
        "float16": 0,
        "float32": 0,
        "int32": 0,
        "int16": 0
    },
    "UB_RECIPROCAL": {
        "float16": 5,
        "float32": 5
    },
    "UB_ROWSUM": {
        "float16": 23,
        "float32": 20
    },
    "UB_ROWMAX": {
        "float16": 9,
        "float32": 8
    },
    "L0C_COPY_OUT": {
        "float16": 200,
        "float32": 200,
        "int32": 0,
        "int16": 0
    },
    "UB_COPY_IN": {
        "float16": 50,
        "float32": 50,
        "int32": 0,
        "int16": 0
    },
    "L1_COPY_IN": {
        "float16": 200,
        "float32": 200,
        "int32": 0,
        "int16": 0
    },
    "L1_TO_L0A": {
        "float16": 0,
        "float32": 0,
        "int32": 0,
        "int16": 0
    },
    "L1_TO_L0At": {
        "float16": 0,
        "float32": 0,
        "int32": 0,
        "int16": 0
    },
    "L1_TO_L0B": {
        "float16": 0,
        "float32": 0,
        "int32": 0,
        "int16": 0
    },
    "L1_TO_L0Bt": {
        "float16": 0,
        "float32": 0,
        "int32": 0,
        "int16": 0
    },
    "CUBE_A_MUL_B": {
        "float16": 0,
        "float32": 0,
        "int32": 7,
        "int16": 7
    },
    "CUBE_A_MUL_Bt": {
        "float16": 0,
        "float32": 0,
        "int32": 7,
        "int16": 7
    },
    "CUBE_A_MULACC_B": {
        "float16": 0,
        "float32": 0,
        "int32": 7,
        "int16": 7
    },
    "CUBE_A_MULACC_Bt": {
        "float16": 0,
        "float32": 0,
        "int32": 7,
        "int16": 7
    },
}

DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "bfloat16": 2,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
    "bool": 1,
}
DTYPE_ALIASES = {"half": "float16", "float": "float32", "double": "float64"}
_OP_NAMES = {op.lower(): op for op in (*OP_LATENCY, *INHERENT_LATENCY)}


def _f32(value):
    return unpack("f", pack("f", value))[0]


def estimate_cycles(op, shapes, dtype, *, original_opcode=None):
    """Port PostSimulate/Simulate; return None for unavailable operand metadata.

    A rank-one TileLang view is represented as (1, N), since upstream expects
    at least two dimensions. Nonpositive dimensions retain upstream's 256
    placeholder. No empirical corrections or MMA repeat formula are added.
    """
    dtype = DTYPE_ALIASES.get(dtype, dtype)
    if not shapes or dtype not in DTYPE_BYTES:
        return None
    normalized = []
    for shape in shapes:
        if not shape or not all(isinstance(d, int) for d in shape):
            return None
        dims = tuple(d if d > 0 else 256 for d in shape)
        normalized.append((1, *dims) if len(dims) == 1 else dims)
    shape_size = min(prod(shape[-2:]) for shape in normalized)
    shape_count = min(prod(shape[:-2]) for shape in normalized)
    size = DTYPE_BYTES[dtype] * shape_size
    if op not in OP_LATENCY:
        latency = INHERENT_LATENCY.get(op, {}).get(dtype, 10)
        total_bytes = size * shape_count
        # C++ integer division truncates toward zero (also for tiny tiles).
        numerator = total_bytes - 256
        repeats = (abs(numerator) // 256) * (1 if numerator >= 0 else -1) + 1
        return latency + repeats * 2 - 1
    a, b = OP_LATENCY[op]
    ratio = 1 if (dtype == "int32" and op in {"MULS", "MUL"} or
                  dtype == "float32" and op in {"DIV", "EXP", "LN", "SQRT"}) else 2
    # C++: float*uint64*uint64 before promotion by the double literal 1E-6.
    byte_term = _f32(_f32(_f32(a) * size) * ratio)
    latency = shape_count * (int((byte_term * 1e-6 + _f32(b)) * 2000) + 1)
    opcode = original_opcode or op
    if opcode == "COPY_IN":
        latency += 1400
    elif opcode == "COPY_OUT":
        latency += 300
    return latency


def estimate_task(operation, pipe, metadata):
    """Translate TileLang operand views into the upstream TileOp convention.

    Scratch/temporary regions are not TileOp operands. Logical MMA dimensions
    are reconstructed from the bridge ABI rather than flattened storage views.
    An explicit pypto_opcode can represent upstream operations without aliases.
    """
    short = operation.split("<", 1)[0].rsplit("::", 1)[-1].rsplit(".", 1)[-1].lower()
    op = _OP_NAMES.get(short, short.upper())
    shapes = []
    dtype = None
    for key in ("src", "lhs", "rhs", "accumulator", "dst"):
        region = metadata.get(key)
        shape = getattr(region, "shape", None)
        if shape:
            shapes.append(shape)
            if dtype is None:
                dtype = str(region.dtype).lower()
    mma = metadata.get("mma", metadata.get("gemm"))
    if short in {"mma", "gemm_v0"} and isinstance(mma, Mapping):
        m, n, k = (mma.get(key) for key in ("rows", "cols", "inner"))
        if all(isinstance(d, int) for d in (m, n, k)):
            shapes = [(m, k), (k, n), (m, n)]
        op = "A_MUL_B" if mma.get("init", True) else "A_MULACC_B"
        if mma.get("transpose_b", False):
            op += "t"
    aliases = {
        "reduce_sum": "ROWSUMLINE",
        "reduce_max": "ROWMAX_SINGLE",
        "reduce_min": "ROWMIN_SINGLE",
        "broadcast": "EXPAND",
    }
    if short == "reduce":
        short = metadata.get("reduce_kind", short)
    op = aliases.get(short, op)
    src, dst = metadata.get("src"), metadata.get("dst")
    src_scope = getattr(getattr(src, "scope", None), "value", getattr(src, "scope", None))
    dst_scope = getattr(getattr(dst, "scope", None), "value", getattr(dst, "scope", None))
    original_opcode = op
    if "copy" in short and src_scope in {"gm", "workspace"} and dst_scope in {"ub", "l1"}:
        original_opcode = "COPY_IN"
        op = "UB_COPY_IN" if dst_scope == "ub" else "L1_COPY_IN"
    elif "copy" in short and dst_scope in {"gm", "workspace"}:
        original_opcode = "COPY_OUT"
        op = "L0C_COPY_OUT" if src_scope == "l0c" else "UB_COPY_OUT"
    elif "copy" in short and src_scope == "l1" and dst_scope in {"l0a", "l0b"}:
        op = "L1_TO_L0A" if dst_scope == "l0a" else "L1_TO_L0B"
        if metadata.get("transpose", False):
            op += "t"
        original_opcode = op
    if "pypto_opcode" in metadata:
        op = original_opcode = metadata["pypto_opcode"]
    return estimate_cycles(op, shapes, dtype, original_opcode=original_opcode)
