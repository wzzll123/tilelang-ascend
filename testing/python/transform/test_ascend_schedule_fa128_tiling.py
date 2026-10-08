# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Capacity/synchronization admission for larger FA tiles; no timing claim."""

import pytest
import tvm

from test_ascend_schedule_score_pipeline import prepared


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("causal", [False, True])
def test_64x128_tiles_keep_complete_workspace_wave_and_native_generation(platform, causal):
    from tilelang.engine.phase import OptimizeForTarget
    from tilelang.transform.ascend_schedule.entry import schedule_module
    from examples.flash_attention_auto128.fa_auto128 import pass_configs
    import json
    function, _, target = prepared(platform, causal, 130, 130, block_m=64, block_n=128)
    scheduled = schedule_module(tvm.IRModule({"main": function}), "ascendc", pass_configs)
    report = json.loads(str(scheduled["main"].attrs["tl.ascend.schedule_report"]))
    assert report["pipeline_kind"] == "cube_prefix_workspace_overlap"
    assert not report["schedule"]["fallback"]
    assert report["workspace_version_pitch_bytes"] == 3 * 64 * 128 * 4
    with tvm.transform.PassContext(config=pass_configs):
        native = OptimizeForTarget(scheduled, target, platform)
        source = tvm.get_global_func("target.build.tilelang_ascend")(native, target,
                                                                     platform).get_source()
    assert "epoch_wave" in source and "CrossCoreSetFlag<0x2, PIPE_FIX>" in source
    assert "PipeBarrier<PIPE_ALL>" not in source
    import re
    from tilelang.transform.ascend_schedule.resources import allocation_budgets
    allocations = re.findall(r"ascend_ub\.GetWithOffset<([\w:]+)>\((\d+),\s*(\d+)\)", source)
    assert allocations
    width = {"float": 4, "half": 2, "uint8_t": 1}
    assert {dtype for dtype, _, _ in allocations} <= width.keys()
    intervals = sorted((int(offset), int(offset) + int(count) * width[dtype])
                       for dtype, count, offset in allocations)
    assert intervals[-1][1] <= allocation_budgets(platform)["UB"]
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))


@pytest.mark.parametrize("m,n", [(0, 64), (16, 64), (True, 64), (32, 0), (32, 24), (32, 64.0)])
def test_invalid_tile_geometry_is_rejected_before_lowering(m, n):
    from examples.flash_attention_auto128.fa_auto128 import FlashAttentionConfig, flash_attention_auto128_fwd
    cfg = FlashAttentionConfig(1, 1, 128, 128, 128, block_m=m, block_n=n)
    with pytest.raises(ValueError, match="block_m|block_n"):
        flash_attention_auto128_fwd.__wrapped__(cfg)
