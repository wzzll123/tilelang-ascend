# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""A measured single-duration profile reaches the actual compiler bridge."""

import numpy as np
import pytest

T = pytest.importorskip("tilelang.language")
import tilelang

from tilelang.simulator.linear_timing import fit_linear_cost, measured_timing_profile


@pytest.mark.parametrize("use_default", [False, True])
def test_compiled_add_uses_measured_tile_duration(use_default):
    families = []
    for op, slope in (("add", 0.005), ("copy_gm_ub", 0.01), ("copy_ub_gm", 0.02)):
        cost = fit_linear_cost([(8192, 8192 * slope + 10), (65536, 65536 * slope + 10)],
                               feature="bytes",
                               evidence="synthetic-integration-test")
        families.append({
            "operation": op,
            "dtype": "float32",
            "geometry": "contiguous",
            "measurement_mode": "two_slot",
            "status": "fitted",
            "cost": cost.to_dict()
        })
    profile = measured_timing_profile({
        "schema": "tilelang-linear-cost-calibration-v1",
        "unit": "ns",
        "platform_measured": "A2",
        "families": families
    })
    if use_default:
        from tilelang.simulator.profile import default_timing_profile
        profile = default_timing_profile("A2")

    @T.prim_func
    def kernel(a: T.Tensor((2, 16, 512), "float32"), b: T.Tensor((2, 16, 512), "float32"),
               c: T.Tensor((2, 16, 512), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            x = T.alloc_ub((16, 512), "float32")
            y = T.alloc_ub((16, 512), "float32")
            z = T.alloc_ub((16, 512), "float32")
            with T.Scope("V"):
                T.copy(a[vid, :, :], x)
                T.copy(b[vid, :, :], y)
                T.set_flag("MTE2", "V", 0)
                T.wait_flag("MTE2", "V", 0)
                T.tile.add(z, x, y)
                T.set_flag("V", "MTE3", 0)
                T.wait_flag("V", "MTE3", 0)
                T.copy(z, c[vid, :, :])

    compiled = tilelang.compile(
        kernel.with_attr("global_symbol", "main"),
        target="ascendc",
        platform="A2",
        simulator=True,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False
        },
        sim_config={
            "timing_profile": profile,
            "sync_only": True
        })
    add_tasks = [task for task in compiled.get_simulator_ir().tasks if task.operation == "add"]
    assert len(add_tasks) == 2
    assert all(task.duration_cycles == (326 if use_default else 348) for task in add_tasks)
    assert all(task.metadata["timing_source"] == "measured" for task in add_tasks)
    assert all(task.metadata["timing_source"] in {"measured", "bookkeeping"}
               for task in compiled.get_simulator_ir().tasks)
    inputs = np.ones((2, 16, 512), dtype=np.float32)
    compiled(inputs, inputs)
    assert compiled.adapter.last_stats.makespan_cycles > 0
