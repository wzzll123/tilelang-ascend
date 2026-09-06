# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""End-to-end coverage for persistent work distribution and termination."""

import numpy as np
import pytest

import tilelang
import tilelang.language as T


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def _persistent_copy_kernel(platform):
    rows, cols, core_num = 10, 32, 4

    @tilelang.jit(
        out_idx=[1], simulator=True, platform=platform,
        pass_configs=PASS_CONFIGS,
    )
    def kernel():
        @T.prim_func
        def main(
            source: T.Tensor([rows, cols], "float32"),
            output: T.Tensor([rows, cols], "float32"),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, _):
                tile = T.alloc_ub([cols], "float32")
                with T.Scope("V"):
                    for row_group, row_in_group in T.Persistent(
                        [5, 2], core_num, cid
                    ):
                        row = row_group * 2 + row_in_group
                        T.copy(source[row, :], tile)
                        T.copy(tile, output[row, :])

        return main

    return kernel()


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_persistent_distributes_tail_work_and_terminates(platform):
    kernel = _persistent_copy_kernel(platform)
    source = np.arange(10 * 32, dtype=np.float32).reshape(10, 32)

    np.testing.assert_array_equal(kernel(source), source)
    # The C220 final TIR contains one persistent loop per vector sub-core.
    assert kernel.adapter.program.metadata["loop_break_count"] == 4
    task_counts = [len(core.tasks) for core in kernel.adapter.program.cores]
    assert task_counts[0] == task_counts[1] > task_counts[2] == task_counts[3]
