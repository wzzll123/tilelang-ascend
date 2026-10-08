# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Regular FA128 migration fixture; no launch, hidden helpers or hand sync.

Based on examples/developer_mode/flash_attn_bshd_developer.py. This is a
compiler integration fixture, not the final configurable ladder entry.
"""

from tilelang import language as T


def fa128_auto_fixture(causal=False, q_len=128, kv_len=128):
    block_m, block_n, dim = 32, 64, 128
    if q_len <= 0 or kv_len <= 0:
        raise ValueError("Q and KV lengths must be positive")
    if causal and q_len > kv_len:
        raise ValueError("Existing golden does not define finite fully-masked causal rows")
    q_shape = (1, 1, q_len, dim)
    kv_shape = (1, 1, kv_len, dim)

    @T.prim_func
    def main(Q: T.Tensor(q_shape, "float16"), K: T.Tensor(kv_shape, "float16"),
             V: T.Tensor(kv_shape, "float16"), Output: T.Tensor(q_shape, "float16")):
        with T.Kernel(T.ceildiv(q_len, block_m), threads=2, is_npu=True) as cid:
            q = T.alloc_shared((block_m, dim), "float16")
            k = T.alloc_shared((block_n, dim), "float16")
            v = T.alloc_shared((block_n, dim), "float16")
            p = T.alloc_shared((block_m, block_n), "float16")
            score_c = T.alloc_fragment((block_m, block_n), "float32")
            output_c = T.alloc_fragment((block_m, dim), "float32")
            score = T.alloc_shared((block_m, block_n), "float32")
            score_recv = T.alloc_shared((block_m, block_n), "float32")
            probability = T.alloc_shared((block_m, block_n), "float16")
            output_recv = T.alloc_shared((block_m, dim), "float32")
            output = T.alloc_shared((block_m, dim), "float32")
            output_half = T.alloc_shared((block_m, dim), "float16")
            row_broadcast = T.alloc_shared((block_m, dim), "float32")
            maximum = T.alloc_shared((block_m,), "float32")
            correction = T.alloc_shared((block_m,), "float32")
            denominator = T.alloc_shared((block_m,), "float32")
            tile_sum = T.alloc_shared((block_m,), "float32")
            query_rows = T.alloc_ub((block_m,), "float32")
            key_columns = T.alloc_ub((block_n,), "float32")
            query_grid = T.alloc_ub((block_m, block_n), "float32")
            key_grid = T.alloc_ub((block_m, block_n), "float32")
            allowed = T.alloc_ub((block_m, block_n // 8), "uint8")
            if causal:
                T.tile.createvecindex(query_rows, T.cast(cid * block_m + kv_len - q_len, "float32"))
                T.tile.broadcast(query_grid, query_rows, axis=1)
            T.tile.fill(output, 0)
            T.tile.fill(maximum, -1e30)
            T.tile.fill(denominator, 0)
            T.copy(Q[0, 0, cid * block_m:(cid + 1) * block_m, :], q)
            for epoch in T.serial(T.ceildiv(kv_len, block_n)):
                T.copy(K[0, 0, epoch * block_n:(epoch + 1) * block_n, :], k)
                T.gemm_v0(q, k, score_c, transpose_B=True, init=True)
                T.copy(score_c, score_recv)
                T.copy(score_recv, score)
                T.tile.mul(score, score, (1.0 / dim)**0.5)
                if causal:
                    T.tile.createvecindex(key_columns, T.cast(epoch * block_n, "float32"))
                    T.tile.broadcast(key_grid, key_columns, axis=0)
                    T.tile.compare(allowed, key_grid, query_grid, "LE")
                    T.tile.select(score, allowed, score, -1e30, "VSEL_TENSOR_SCALAR_MODE")
                elif kv_len % block_n:
                    T.tile.createvecindex(key_columns, T.cast(epoch * block_n, "float32"))
                    T.tile.broadcast(key_grid, key_columns, axis=0)
                    T.tile.compare(allowed, key_grid, T.float32(kv_len), "LT")
                    T.tile.select(score, allowed, score, -1e30, "VSEL_TENSOR_SCALAR_MODE")
                T.copy(maximum, correction)
                T.reduce_max(score, maximum, dim=-1)
                T.tile.max(maximum, maximum, correction)
                T.tile.sub(correction, correction, maximum)
                T.tile.exp(correction, correction)
                T.tile.broadcast(score_recv, maximum, axis=1)
                T.tile.sub(score, score, score_recv)
                T.tile.exp(score, score)
                if causal or kv_len % block_n:
                    T.tile.select(score, allowed, score, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                T.reduce_sum(score, tile_sum, dim=-1)
                T.tile.mul(denominator, denominator, correction)
                T.tile.add(denominator, denominator, tile_sum)
                T.copy(score, probability)
                T.copy(probability, p)
                T.copy(V[0, 0, epoch * block_n:(epoch + 1) * block_n, :], v)
                T.gemm_v0(p, v, output_c, init=True)
                T.copy(output_c, output_recv)
                T.tile.broadcast(row_broadcast, correction, axis=1)
                T.tile.mul(output, output, row_broadcast)
                T.tile.add(output, output, output_recv)
            T.tile.broadcast(row_broadcast, denominator, axis=1)
            T.tile.div(output, output, row_broadcast)
            T.copy(output, output_half)
            T.copy(output_half, Output[0, 0, cid * block_m:(cid + 1) * block_m, :])

    return main
