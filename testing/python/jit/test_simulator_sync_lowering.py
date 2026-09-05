"""Regression coverage for mainline's conservative synchronization around if."""

import numpy as np

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def _conditional_cube_kernel(platform):
    @tilelang.jit(
        out_idx=[2],
        simulator=True,
        platform=platform,
        pass_configs=PASS_CONFIGS,
    )
    def kernel():
        @T.prim_func
        def main(
            left: T.Tensor([32, 16], "float16"),
            right: T.Tensor([16, 16], "float16"),
            output: T.Tensor([32, 16], "float32"),
        ):
            with T.Kernel(2, is_npu=True) as (cid, _vid):
                left_l1 = T.alloc_L1([16, 16], "float16")
                right_l1 = T.alloc_L1([16, 16], "float16")
                T.annotate_layout({
                    left_l1: make_zn_layout(left_l1),
                    right_l1: make_zn_layout(right_l1),
                })
                accumulator = T.alloc_L0C([16, 16], "float32")
                with T.Scope("C"):
                    # Keep a real two-branch IfThenElse in the sync pass.  The
                    # simulator later specializes cid for each concrete core.
                    if cid == 0:
                        T.copy(left[0:16, :], left_l1)
                        T.copy(right, right_l1)
                        T.gemm_v0(left_l1, right_l1, accumulator, init=True)
                        T.copy(accumulator, output[0:16, :])
                    else:
                        T.copy(left[16:32, :], left_l1)
                        T.copy(right, right_l1)
                        T.gemm_v0(left_l1, right_l1, accumulator, init=True)
                        T.copy(accumulator, output[16:32, :])

        return main

    return kernel()


def test_reverted_if_sync_barriers_are_preserved_and_executed() -> None:
    for platform in ("A2", "A3"):
        kernel = _conditional_cube_kernel(platform)
        tasks = list(kernel.adapter.program.tasks)
        barriers = [
            task for task in tasks
            if task.operation == "auto_barrier"
            and task.metadata.get("target_pipe") == "all"
        ]
        assert len(barriers) >= 4
        for core_id in (0, 1):
            core_tasks = [task for task in tasks if task.core_id == core_id]
            core_barriers = [task for task in barriers if task.core_id == core_id]
            assert len(core_barriers) >= 2
            assert core_tasks.index(core_barriers[0]) < next(
                index for index, task in enumerate(core_tasks)
                if task.operation == "copy_gm_to_l1"
            )
            assert core_tasks.index(core_barriers[-1]) > max(
                index for index, task in enumerate(core_tasks)
                if task.operation == "copy_l0c_to_gm"
            )

        rng = np.random.default_rng(0)
        left = rng.normal(size=(32, 16)).astype(np.float16)
        right = rng.normal(size=(16, 16)).astype(np.float16)
        output = kernel(left, right)
        np.testing.assert_allclose(
            output,
            left.astype(np.float32) @ right.astype(np.float32),
            rtol=1e-5,
            atol=1e-5,
        )
        records = kernel.adapter.last_schedule.records
        barrier_ids = {barrier.task_id for barrier in barriers}
        assert barrier_ids.issubset({record.task_id for record in records})
