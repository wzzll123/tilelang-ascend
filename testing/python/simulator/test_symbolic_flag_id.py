"""Regression: symbolic local flag ids (e.g. ``k % 2``) must be evaluated.

The simulator bridge unrolls ``tir.For`` loops and tracks the loop variable in
the per-iteration ``environment``.  ``_sync_metadata`` previously extracted the
flag id with ``_literal`` (no environment substitution), so a handwritten
double-buffered kernel using ``set_flag(..., k % S1)`` reached the scheduler
with the *string* ``"k % 2"`` as its flag id and failed validation with
``requires flag_id in [0, 7]``.  Handwritten pipelined kernels (matmul, wqbm
B1) depend on symbolic slot ids, so the bridge must resolve them through the
unrolled-loop environment like every other integer operand.
"""

import numpy as np

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}


def _symbolic_flag_kernel(platform):
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
                acc = T.alloc_L0C([16, 16], "float32")
                with T.Scope("C"):
                    # Pre-set both L1 slots so the first loads may proceed.
                    T.set_flag("mte1", "mte2", 0)
                    T.set_flag("mte1", "mte2", 1)
                    for k in T.serial(2):
                        # Symbolic slot id ``k % 2`` — the bridge must fold it.
                        T.wait_flag("mte1", "mte2", k % 2)
                        T.copy(left[:, k * 32:(k + 1) * 32], a_l1[k % 2, :, :])
                        T.copy(right[k * 32:(k + 1) * 32, :], b_l1[k % 2, :, :])
                        T.set_flag("mte2", "mte1", k % 2)
                        T.wait_flag("mte2", "mte1", k % 2)
                        T.gemm_v0(a_l1[k % 2, :, :], b_l1[k % 2, :, :], acc,
                                  init=(k == 0))
                        T.set_flag("mte1", "mte2", k % 2)
                    T.copy(acc, output)
                    # Drain the pre-set slot flags.
                    T.wait_flag("mte1", "mte2", 0)
                    T.wait_flag("mte1", "mte2", 1)

        return main

    return kernel


def test_symbolic_local_flag_id_resolves(platform="A2"):
    ker = _symbolic_flag_kernel(platform)()
    left = np.random.randn(16, 64).astype(np.float16)
    right = np.random.randn(64, 16).astype(np.float16)
    out = ker(left, right)
    golden = left.astype(np.float32) @ right.astype(np.float32)
    np.testing.assert_allclose(out, golden, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_symbolic_local_flag_id_resolves()
    print("PASS")
