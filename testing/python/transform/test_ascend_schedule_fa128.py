# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Real regular FA lowering, numerical execution and missing-sync mutations."""

import json
import numpy as np
import pytest
import tvm
from tvm import tir
from fa128_auto_fixture import fa128_auto_fixture


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_regular_fa128_has_values_sync_and_native_generation(platform):
    from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
    from tilelang.simulator import (build_kernel_program, FunctionalSimulator, SimulatorConfig,
                                    BufferRegion, MemoryScope, SimulationDeadlockError,
                                    UninitializedMemoryError)

    function = fa128_auto_fixture().with_attr("npu_platform", platform)
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    config = {
        "tl.ascend_auto_schedule": True,
        "tl.ascend_auto_cv_combine": True,
        "tl.ascend_auto_cross_core_sync": True
    }
    with tvm.transform.PassContext(config=config):
        mod = LowerAndLegalize(tvm.IRModule({"main": function}), target)
        native = OptimizeForTarget(mod, target, platform)
        source = tvm.get_global_func("target.build.tilelang_ascend")(native, target,
                                                                     platform).get_source()
    assert "gemm_v0<" not in source and "mma<half,float," in source
    assert "CrossCoreSetFlag<0x2," in source and "CrossCoreWaitFlag(" in source
    assert "PipeBarrier<PIPE_ALL>" not in source
    scheduled = mod["main"]
    report = json.loads(str(scheduled.attrs["tl.ascend.schedule_report"]))
    assert any(edge["kind"] == "WAR" and edge["distance"] == 1 for edge in report["dependencies"])
    inputs = [
        np.random.default_rng(seed).normal(0, .3, (1, 1, 128, 128)).astype("float16")
        for seed in (7, 8, 9)
    ]
    q, k, v = [values.astype("float64") for values in inputs]
    score = q @ k.swapaxes(-1, -2) / 128**.5
    probability = np.exp(score - score.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)
    golden = probability @ v

    def simulator(candidate, sync_only):
        sim = FunctionalSimulator(
            build_kernel_program(candidate, platform=platform),
            SimulatorConfig(
                platform=platform,
                sync_only=sync_only,
                execution_timeout_s=60,
                flag_balance_check="error"))
        for name, values in zip(("Q", "K", "V"), inputs):
            sim.write(BufferRegion(name, MemoryScope.GM, values.shape, "float16"), values)
        return sim

    for sync_only in (False, True):
        sim = simulator(scheduled, sync_only)
        sim.run()
        if not sync_only:
            actual = sim.read(BufferRegion("Output", MemoryScope.GM, golden.shape, "float16"))
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, golden, atol=5e-4, rtol=5e-3)

    for deleted in ("set", "wait"):

        def remove(node, deleted=deleted):
            if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call) and \
                    node.value.op.name == "tl.ascend_" + deleted + "_cross_flag":
                return tir.Evaluate(0)
            return None

        bad = scheduled.with_body(tir.stmt_functor.ir_transform(scheduled.body, None, remove))
        # This GM-hop mutation is diagnosed by actual execution, not assumed to
        # be covered by the static local-memory checker.
        error = SimulationDeadlockError if deleted == "set" else UninitializedMemoryError
        with pytest.raises(error):
            simulator(bad, True).run()
