"""Regression coverage for FA runtime scalars and row broadcasts."""

import numpy as np

import tilelang
import tilelang.language as T


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def _row_broadcast_kernel(platform):
    @tilelang.jit(
        out_idx=[1],
        simulator=True,
        platform=platform,
        pass_configs=PASS_CONFIGS,
    )
    def kernel():
        @T.prim_func
        def main(
            source: T.Tensor([2, 8], "float32"),
            output: T.Tensor([2, 8], "float32"),
            scale: T.float32,
        ):
            with T.Kernel(1, is_npu=True):
                source_ub = T.alloc_ub([2, 8], "float32")
                row_max = T.alloc_ub([2, 1], "float32")
                expanded = T.alloc_ub([2, 8], "float32")
                with T.Scope("V"):
                    T.copy(source, source_ub)
                    T.reduce_max(source_ub, row_max, dim=-1)
                    T.tile.broadcast(expanded, row_max)
                    T.tile.mul(expanded, expanded, scale)
                    T.copy(expanded, output)

        return main

    return kernel()


def test_fa_row_broadcast_preserves_rows_and_runtime_float_scalar() -> None:
    source = np.array([
        [1, 4, 2, 3, -1, 0, 2, 1],
        [9, 3, 5, 7, 2, 8, 0, 4],
    ], dtype=np.float32)
    expected = np.array([[2] * 8, [4.5] * 8], dtype=np.float32)

    for platform in ("A2", "A3"):
        output = _row_broadcast_kernel(platform)(source, 0.5)
        np.testing.assert_array_equal(output, expected)
