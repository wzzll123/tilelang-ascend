"""Regression tests for manual-pipeline memory-hazard validation."""

import pytest

import tilelang
from tilelang import language as T


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}


def _kernel(
    *, mte2_to_v: bool, v_to_mte3: bool, mte2_to_v_pipe_all: bool = False
):
    """PTO's canonical MTE2 -> V -> MTE3 UB pipeline.

    PTO's A2/A3 ``taddplus`` kernels use this exact local-event sequence:
    DMA fills UB, Vector consumes it, then DMA stores the result.  Keep the
    hand-synchronized version here so validation is exercised through the
    complete TileLang lowering path, rather than only with synthetic Tasks.
    """
    @T.prim_func
    def main(a: T.Tensor([256], "float16"), b: T.Tensor([256], "float16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub = T.alloc_ub([256], "float16")
            with T.Scope("V"):
                T.copy(a[0:256], ub)  # MTE2 writes UB.
                if mte2_to_v:
                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)
                elif mte2_to_v_pipe_all:
                    T.pipe_barrier("ALL")
                T.tile.mul(ub, ub, 2.0)  # V reads UB.
                if v_to_mte3:
                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                T.copy(ub, b[0:256])
    return main


def _compile(*, mte2_to_v: bool, v_to_mte3: bool, mte2_to_v_pipe_all: bool = False):
    return tilelang.compile(
        _kernel(
            mte2_to_v=mte2_to_v,
            v_to_mte3=v_to_mte3,
            mte2_to_v_pipe_all=mte2_to_v_pipe_all,
        ),
        out_idx=[1], pass_configs=_PASS_CONFIGS,
        target="ascendc", simulator=True, platform="A2",
        sim_config={"hazard_check": "error", "sync_only": True},
    )


@pytest.mark.parametrize(
    ("mte2_to_v", "v_to_mte3"),
    [(False, True), (True, False)],
    ids=["missing-mte2-to-v", "missing-v-to-mte3"],
)
def test_hand_sync_pto_ub_pipeline_requires_every_fence(
    mte2_to_v: bool, v_to_mte3: bool
):
    """Removing either PTO event must remain a compile-time simulator error."""
    with pytest.raises(Exception, match="synchronization|hazard|fence"):
        _compile(mte2_to_v=mte2_to_v, v_to_mte3=v_to_mte3)


def test_hand_sync_pto_ub_pipeline_fences_are_accepted():
    _compile(mte2_to_v=True, v_to_mte3=True)


def test_hand_sync_pipe_all_fences_pto_mte2_to_vector_edge():
    """PIPE_ALL is a valid, deliberately broad substitute for the event."""
    _compile(mte2_to_v=False, mte2_to_v_pipe_all=True, v_to_mte3=True)
