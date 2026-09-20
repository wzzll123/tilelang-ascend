"""Regression tests for manual-pipeline memory-hazard validation."""

import pytest

import tilelang
from tilelang import language as T


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}


def _kernel(fenced: bool):
    @T.prim_func
    def main(a: T.Tensor([256], "float16"), b: T.Tensor([256], "float16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub = T.alloc_ub([256], "float16")
            with T.Scope("V"):
                T.copy(a[0:256], ub)  # MTE2 writes UB.
                if fenced:
                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)
                T.tile.mul(ub, ub, 2.0)  # V reads UB.
                T.set_flag("v", "mte3", 0)
                T.wait_flag("v", "mte3", 0)
                T.copy(ub, b[0:256])
    return main


def _compile(fenced: bool):
    return tilelang.compile(
        _kernel(fenced), out_idx=[1], pass_configs=_PASS_CONFIGS,
        target="ascendc", simulator=True, platform="A2",
        sim_config={"hazard_check": "error", "sync_only": True},
    )


def test_hand_sync_mte2_to_vector_requires_fence():
    with pytest.raises(Exception, match="synchronization|hazard|fence"):
        _compile(False)


def test_hand_sync_mte2_to_vector_fence_is_accepted():
    _compile(True)
