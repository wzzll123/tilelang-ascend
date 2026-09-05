# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Regression: L1->L0 copies from a multi-slot (double-buffered) L1 tile.

A handwritten pipelined cube kernel allocates L1 as ``[S1, R, C]`` and slices
``a_l1[slot, :, k0:k1]`` for the L1->L0A/B load.  The bridge's
``_l1_to_l0_metadata`` mapped the slice ``byte_offset`` to a logical tile by
searching a single ``(R, C)`` zN/nZ grid, so any slot > 0 carried a slot base
(``slot * per_slot_capacity``) that fell outside the grid and raised
``cannot map its source offset to a logical tile``.  The fix reduces the offset
modulo the per-slot padded capacity for the origin search and re-adds the slot
base to the emitted source regions.

This kernel uses ``T.copy`` (direct ``copy_l1_to_l0a/b``) + ``T.mma`` with a
two-slot L1 buffer so slot 1 is exercised.
"""

import numpy as np

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}


def _double_slot_l1_kernel(platform):
    @tilelang.jit(
        out_idx=[2],
        simulator=True,
        platform=platform,
        pass_configs=PASS_CONFIGS,
    )
    def kernel():
        @T.prim_func
        def main(
            left: T.Tensor([16, 64], "float16"),
            right: T.Tensor([64, 16], "float16"),
            output: T.Tensor([16, 16], "float32"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _vid):
                a_l1 = T.alloc_L1([2, 16, 32], "float16")
                b_l1 = T.alloc_L1([2, 32, 16], "float16")
                T.annotate_layout({
                    a_l1: make_zn_layout(a_l1),
                    b_l1: make_zn_layout(b_l1),
                })
                a_l0 = T.alloc_L0A([16, 32], "float16")
                b_l0 = T.alloc_L0B([32, 16], "float16")
                acc = T.alloc_L0C([16, 16], "float32")
                with T.Scope("C"):
                    T.set_flag("mte1", "mte2", 0)
                    T.set_flag("mte1", "mte2", 1)
                    for k in T.serial(2):
                        T.wait_flag("mte1", "mte2", k % 2)
                        T.copy(left[:, k * 32:(k + 1) * 32], a_l1[k % 2, :, :])
                        T.copy(right[k * 32:(k + 1) * 32, :], b_l1[k % 2, :, :])
                        T.set_flag("mte2", "mte1", k % 2)
                        T.wait_flag("mte2", "mte1", k % 2)
                        # Direct L1 -> L0 slice from slot ``k % 2``.
                        T.copy(a_l1[k % 2, :, :], a_l0)
                        T.copy(b_l1[k % 2, :, :], b_l0)
                        T.set_flag("mte1", "m", 0)
                        T.wait_flag("mte1", "m", 0)
                        T.mma(a_l0, b_l0, acc, init=(k == 0))
                        T.set_flag("m", "mte1", 0)
                        T.wait_flag("m", "mte1", 0)
                        T.set_flag("mte1", "mte2", k % 2)
                    T.copy(acc, output)
                    T.wait_flag("mte1", "mte2", 0)
                    T.wait_flag("mte1", "mte2", 1)

        return main

    return kernel


def test_double_slot_l1_slice(platform="A2"):
    ker = _double_slot_l1_kernel(platform)()
    left = np.random.randn(16, 64).astype(np.float16)
    right = np.random.randn(64, 16).astype(np.float16)
    out = ker(left, right)
    golden = left.astype(np.float32) @ right.astype(np.float32)
    np.testing.assert_allclose(out, golden, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_double_slot_l1_slice()
    print("PASS")
