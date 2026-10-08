# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Expose the single-tile FP16 GEMM's physical pipe work in existing TIR.

The numerical copy/MMA recipe follows common.h gemm_v0, without its hidden
events or implicit scratch. Scratch allocation/lifetimes and synchronization
are the caller's responsibility; this helper never claims to emit a pipeline.
"""

import re


def decompose_gemm_v0(call, l0a, l0b):
    """Return MTE1(A), MTE1(B), M tasks using caller-owned L0 buffers.

    First supported family is FP16 input/FP32 accumulation with static aligned
    M/N/K <=128 and one K tile. Runtime n_actual is retained only on transpose-B,
    exactly as in the original helper. Unsupported forms fail rather than leave
    an opaque multi-pipe function in the task graph.
    """
    from tvm import arith, tir

    if not isinstance(call, tir.Call) or call.op.name != "tl.ascend_gemm_v0" or len(call.args) != 6:
        raise ValueError("expected the six-argument gemm_v0 intrinsic")
    if not isinstance(call.args[0], tir.StringImm):
        raise ValueError("gemm_v0 template must be static")
    match = re.fullmatch(
        r"gemm_v0<\s*half\s*,\s*float\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,"
        r"\s*(true|false)\s*,\s*(true|false)\s*,\s*(\d+)\s*>", call.args[0].value)
    if match is None:
        raise ValueError("GEMM decomposition requires static FP16/FP32 template dimensions")
    m, n, k = map(int, match.group(1, 2, 3))
    transpose_a, transpose_b = match.group(4) == "true", match.group(5) == "true"
    tile_k = int(match.group(6))
    if any(dim < 16 or dim > 128 or dim % 16 for dim in (m, n, k)) or tile_k != 128:
        raise ValueError("GEMM decomposition supports aligned M/N/K<=128 with kL0Size=128")

    def pointer(ptr, dtype, scope, required_mask):
        if (not isinstance(ptr, tir.Call) or ptr.op.name != "tir.tvm_access_ptr" or
                len(ptr.args) != 5 or str(ptr.args[0].dtype) != dtype or
                not isinstance(ptr.args[1], tir.Var)):
            raise ValueError("invalid GEMM operand access pointer")
        annotation = ptr.args[1].type_annotation
        if str(getattr(annotation, "storage_scope", "")) != scope:
            raise ValueError(f"GEMM operand requires storage scope {scope}")
        mask = ptr.args[4]
        if not isinstance(mask, tir.IntImm) or not int(mask) & required_mask:
            raise ValueError("GEMM operand access mask contradicts its operation")

    for ptr in call.args[1:3]:
        pointer(ptr, "float16", "shared.l1", 1)
    pointer(call.args[3], "float32", "wmma.accumulator", 2)
    analyzer = arith.Analyzer()
    for ptr, required in zip(call.args[1:4], (m * k, n * k, m * n)):
        extent = analyzer.simplify(ptr.args[3])
        if not isinstance(extent, tir.IntImm) or int(extent) < required:
            raise ValueError("GEMM operand capacity must cover its complete static tile")
    for buffer, scope, count in ((l0a, "wmma.matrix_a", m * k), (l0b, "wmma.matrix_b", n * k)):
        if not isinstance(buffer, tir.Buffer) or str(
                buffer.dtype) != "float16" or buffer.scope() != scope:
            raise ValueError(f"GEMM scratch must be FP16 {scope}")
        if buffer.strides or not isinstance(buffer.elem_offset, tir.IntImm) or int(
                buffer.elem_offset) != 0:
            raise ValueError("GEMM scratch requires a compact allocation base")
        size = 1
        for extent in buffer.shape:
            if not isinstance(extent, tir.IntImm):
                raise ValueError("GEMM scratch capacity must be static")
            size *= int(extent)
        if size < count or size * 2 > 65536:
            raise ValueError("GEMM scratch capacity does not fit the tile or A2/A3 L0 budget")
    if l0a.data.same_as(l0b.data) or any(
            scratch.data.same_as(ptr.args[1]) for scratch in (l0a, l0b) for ptr in call.args[1:4]):
        raise ValueError("GEMM scratch must have independent allocation identities")
    actual_n = call.args[5] if transpose_b else tir.IntImm("int32", n)
    if isinstance(actual_n, tir.IntImm) and (int(actual_n) <= 0 or int(actual_n) > n or
                                             int(actual_n) % 16):
        raise ValueError("GEMM actual N must be positive, <=N, and fractal aligned")

    a_template = f"copy_l1_to_l0a<half,{k},{m},true>" if transpose_a else f"copy_l1_to_l0a<half,{m},{k},false>"
    b_template = f"copy_l1_to_l0b<half,{n},{k},true>" if transpose_b else f"copy_l1_to_l0b<half,{k},{n},false>"
    copy_a = tir.call_extern("handle", "tl::ascend::" + a_template, call.args[1],
                             l0a.access_ptr("w"), m, k)
    copy_b = tir.call_extern("handle", "tl::ascend::" + b_template, call.args[2],
                             l0b.access_ptr("w"), k, actual_n)
    mma = tir.call_intrin("handle", tir.op.Op.get("tl.ascend_mma"), f"mma<half,float,{m},{n}>",
                          l0a.access_ptr("r"), l0b.access_ptr("r"), call.args[3], call.args[4], k,
                          actual_n, 0)
    return tir.SeqStmt([tir.Evaluate(copy_a), tir.Evaluate(copy_b), tir.Evaluate(mma)])
