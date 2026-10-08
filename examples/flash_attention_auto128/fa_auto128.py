# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""FA128 automatic-scheduler ladder entry, based on the developer FA example.

No hand synchronization, physical addresses, or ring-slot indexing. Initial
32x64 tiling is a correctness baseline, not a performance acceptance claim.
"""

from dataclasses import dataclass

import tilelang
from tilelang import language as T

MAX_DIM = 128
pass_configs = {
    "tl.ascend_auto_schedule": True,
    "tl.ascend_auto_cv_combine": True,
    "tl.ascend_auto_cross_core_sync": True,
}


@dataclass(frozen=True)
class FlashAttentionConfig:
    batch: int
    heads: int
    q_seq_len: int
    kv_seq_len: int
    dim: int
    dtype: str = "float16"
    causal: bool = False


@tilelang.jit(out_idx=[3], pass_configs=pass_configs, target="ascendc")
def flash_attention_auto128_fwd(cfg: FlashAttentionConfig):
    if cfg.dim != 128 or cfg.dtype != "float16":
        raise ValueError("auto128 supports only FP16 head dimension 128")
    batch, heads, sq, skv = cfg.batch, cfg.heads, cfg.q_seq_len, cfg.kv_seq_len
    if min(batch, heads, sq, skv) <= 0:
        raise ValueError("batch, heads and sequence lengths must be positive")
    if cfg.causal and sq > skv:
        raise ValueError("causal requires q_seq_len <= kv_seq_len")
    bm, bn, dim = 32, 64, 128
    q_tiles = (sq + bm - 1) // bm
    causal = cfg.causal

    @T.prim_func
    def main(Q: T.Tensor((batch, heads, sq, dim), "float16"), K: T.Tensor(
        (batch, heads, skv, dim), "float16"), V: T.Tensor((batch, heads, skv, dim), "float16"),
             Output: T.Tensor((batch, heads, sq, dim), "float16"), sm_scale: T.float32):
        with T.Kernel(batch * heads * q_tiles, threads=2, is_npu=True) as cid:
            q = T.alloc_shared((bm, dim), "float16")
            k = T.alloc_shared((bn, dim), "float16")
            v = T.alloc_shared((bn, dim), "float16")
            p = T.alloc_shared((bm, bn), "float16")
            score_c = T.alloc_fragment((bm, bn), "float32")
            output_c = T.alloc_fragment((bm, dim), "float32")
            score = T.alloc_shared((bm, bn), "float32")
            score_recv = T.alloc_shared((bm, bn), "float32")
            probability = T.alloc_shared((bm, bn), "float16")
            output_recv = T.alloc_shared((bm, dim), "float32")
            output = T.alloc_shared((bm, dim), "float32")
            output_half = T.alloc_shared((bm, dim), "float16")
            row_broadcast = T.alloc_shared((bm, dim), "float32")
            maximum = T.alloc_shared((bm,), "float32")
            correction = T.alloc_shared((bm,), "float32")
            denominator = T.alloc_shared((bm,), "float32")
            tile_sum = T.alloc_shared((bm,), "float32")
            query_rows = T.alloc_ub((bm,), "float32")
            key_columns = T.alloc_ub((bn,), "float32")
            query_grid = T.alloc_ub((bm, bn), "float32")
            key_grid = T.alloc_ub((bm, bn), "float32")
            allowed = T.alloc_ub((bm, bn // 8), "uint8")
            if causal:
                T.tile.createvecindex(query_rows, T.cast((cid % q_tiles) * bm + skv - sq,
                                                         "float32"))
                T.tile.broadcast(query_grid, query_rows, axis=1)
            T.tile.fill(output, 0)
            T.tile.fill(maximum, -1e30)
            T.tile.fill(denominator, 0)
            T.copy(
                Q[cid // (q_tiles * heads), (cid // q_tiles) % heads,
                  (cid % q_tiles) * bm:((cid % q_tiles) + 1) * bm, :], q)
            for epoch in T.serial(T.ceildiv(skv, bn)):
                T.copy(
                    K[cid // (q_tiles * heads), (cid // q_tiles) % heads,
                      epoch * bn:(epoch + 1) * bn, :], k)
                T.gemm_v0(q, k, score_c, transpose_B=True, init=True)
                T.copy(score_c, score_recv)
                T.copy(score_recv, score)
                T.tile.mul(score, score, sm_scale)
                if causal:
                    T.tile.createvecindex(key_columns, T.cast(epoch * bn, "float32"))
                    T.tile.broadcast(key_grid, key_columns, axis=0)
                    T.tile.compare(allowed, key_grid, query_grid, "LE")
                    T.tile.select(score, allowed, score, -1e30, "VSEL_TENSOR_SCALAR_MODE")
                elif skv % bn:
                    T.tile.createvecindex(key_columns, T.cast(epoch * bn, "float32"))
                    T.tile.broadcast(key_grid, key_columns, axis=0)
                    T.tile.compare(allowed, key_grid, T.float32(skv), "LT")
                    T.tile.select(score, allowed, score, -1e30, "VSEL_TENSOR_SCALAR_MODE")
                T.copy(maximum, correction)
                T.reduce_max(score, maximum, dim=-1)
                T.tile.max(maximum, maximum, correction)
                T.tile.sub(correction, correction, maximum)
                T.tile.exp(correction, correction)
                T.tile.broadcast(score_recv, maximum, axis=1)
                T.tile.sub(score, score, score_recv)
                T.tile.exp(score, score)
                if causal or skv % bn:
                    T.tile.select(score, allowed, score, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_sum(score, tile_sum, dim=-1)
                T.tile.mul(denominator, denominator, correction)
                T.tile.add(denominator, denominator, tile_sum)
                T.copy(score, probability)
                T.copy(probability, p)
                T.copy(
                    V[cid // (q_tiles * heads), (cid // q_tiles) % heads,
                      epoch * bn:(epoch + 1) * bn, :], v)
                T.gemm_v0(p, v, output_c, init=True)
                T.copy(output_c, output_recv)
                T.tile.broadcast(row_broadcast, correction, axis=1)
                T.tile.mul(output, output, row_broadcast)
                T.tile.add(output, output, output_recv)
            T.tile.broadcast(row_broadcast, denominator, axis=1)
            T.tile.div(output, output, row_broadcast)
            T.copy(output, output_half)
            T.copy(
                output_half, Output[cid // (q_tiles * heads), (cid // q_tiles) % heads,
                                    (cid % q_tiles) * bm:((cid % q_tiles) + 1) * bm, :])

    return main
