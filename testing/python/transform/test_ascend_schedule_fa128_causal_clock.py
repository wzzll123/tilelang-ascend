# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Check the actual ladder factory, not an independent FA-like fixture."""

import pytest
import tvm
from tvm import tir, arith
from examples.flash_attention_auto128.fa_auto128 import (FlashAttentionConfig,
                                                         flash_attention_auto128_fwd, pass_configs)
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("causal", [False, True])
def test_production_causal_clock_is_group_dependent_and_native(platform, causal):
    cfg = FlashAttentionConfig(1, 1, 128, 128, 128, causal=causal)
    function = flash_attention_auto128_fwd.__wrapped__(cfg).with_attr("npu_platform", platform)
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    with tvm.transform.PassContext(config=pass_configs):
        mod = LowerAndLegalize(tvm.IRModule({"main": function}), target)
        owners, groups = [], []

        def collect(node):
            if isinstance(node, tir.For):
                owners.append(node)
            if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent" and isinstance(
                    node.node, tir.IterVar) and node.node.thread_tag == "blockIdx.x":
                groups.append(node.node.var)

        tir.stmt_functor.post_order_visit(mod["main"].body, collect)
        assert len(owners) == len(groups) == 1
        extent = owners[0].extent
        assert isinstance(extent, tir.IntImm) != causal
        counts = [
            int(arith.Analyzer().simplify(
                tir.stmt_functor.substitute(extent, {groups[0]: tir.IntImm("int32", group)})))
            for group in range(4)
        ]
        pipelined = int(mod["main"].attrs.get("tl.ascend.prefetch_materialized", 0))
        assert [count - pipelined for count in counts] == ([1, 1, 2, 2] if causal else [2, 2, 2, 2])
        native = OptimizeForTarget(mod, target, platform)
        source = tvm.get_global_func("target.build.tilelang_ascend")(native, target,
                                                                     platform).get_source()
    assert "mma<" in source
    assert "CrossCoreSetFlag<0x2," in source
    assert "CrossCoreWaitFlag(" in source
    assert "PipeBarrier<PIPE_ALL>" not in source
