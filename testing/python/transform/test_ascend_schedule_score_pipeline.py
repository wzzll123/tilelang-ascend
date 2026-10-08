# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Actual FA pipeline protocol generation; not a numerical/device acceptance test."""

from unittest.mock import patch
import pytest
import tvm
import tilelang
from tvm import tir


def prepared(platform, causal, kv, sq=128, block_m=32, block_n=64):
    from examples.flash_attention_auto128.fa_auto128 import (FlashAttentionConfig,
                                                             flash_attention_auto128_fwd,
                                                             pass_configs)
    from tilelang.engine.phase import LowerAndLegalize
    from tilelang.transform.ascend_schedule.conditions import normalize_conditions
    from tilelang.transform.ascend_schedule.core_scopes import normalize_core_scopes
    from tilelang.transform.ascend_schedule.gemm_decomposition import decompose_function_gemms
    from tilelang.transform.ascend_schedule.workspace_tasks import prepare_cross_core_copies
    from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks
    function = flash_attention_auto128_fwd.__wrapped__(
        FlashAttentionConfig(1, 1, sq, kv, 128, causal=causal, block_m=block_m,
                             block_n=block_n)).with_attr("npu_platform", platform)
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    with patch("tilelang.transform.ascend_schedule.entry.schedule_module", lambda mod, *_: mod), \
            tvm.transform.PassContext(config=pass_configs):
        mod = prepare_cross_core_copies(LowerAndLegalize(tvm.IRModule({"main": function}), target))
    function = normalize_conditions(decompose_function_gemms(mod["main"], root_scratch=True))
    function = normalize_core_scopes(function, extract_tasks(function))
    return function, extract_tasks(function), target


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv", [192, 130])
@pytest.mark.parametrize("sq", [128, 130])
def test_actual_fa_score_pipeline_emits_phased_mode2_and_native_waves(platform, causal, kv, sq):
    import json
    from tilelang.transform.ascend_schedule.score_pipeline import prepare_score_pipeline
    from tilelang.transform.ascend_schedule.materializer import materialize_flat_function
    function, extraction, target = prepared(platform, causal, kv, sq)
    staged, report = prepare_score_pipeline(function, extraction, platform)
    assert staged is not None
    assert set(staged.stages.values()) == {0, 1}
    assert report["host_workspace"]["selected_bytes"] > 0
    assert not any(a.full_overwrite for t in staged.extraction.tasks for a in t.accesses)
    ring = [a for t in staged.extraction.tasks for a in t.accesses if a.ring_size == 2]
    assert ring and all(a.iteration_stride == report["workspace_version_pitch_bytes"] for a in ring)
    result = materialize_flat_function(
        staged.function, staged.extraction, platform, prefetch=staged, prefetch_report=report)
    records = json.loads(str(result.attrs["tl.ascend.phase_events"]))
    assert all(0 <= record["event_id"] < 8 for record in records)
    assert any(record["source_core"] != record["destination_core"] for record in records)
    native = tvm.IRModule({"main": result.with_attr("global_symbol", "main")})
    native = tilelang.transform.AscendLowerOpaqueBlock()(native)
    native = tilelang.transform.Flatten2DBuffer()(native)
    native = tilelang.transform.FlattenBuffer()(native)
    native = tilelang.transform.AscendMemoryPlanning()(native)
    source = tvm.get_global_func("target.build.tilelang_ascend")(native, target,
                                                                 platform).get_source()
    assert "epoch_wave" in source and "epoch_wave - 1" in source
    assert "CrossCoreSetFlag<0x2, PIPE_FIX>" in source
    assert "CrossCoreWaitFlag" in source
    assert "copy_l0c_to_gm" in source and "copy_gm_to_ub" in source


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sq", [128, 130])
def test_actual_auto_schedule_entry_selects_workspace_overlap(platform, causal, sq):
    import json
    from tilelang.transform.ascend_schedule.entry import schedule_module
    from examples.flash_attention_auto128.fa_auto128 import pass_configs
    function, _, _ = prepared(platform, causal, 130, sq)
    result = schedule_module(tvm.IRModule({"main": function}), "ascendc", pass_configs)["main"]
    report = json.loads(str(result.attrs["tl.ascend.schedule_report"]))
    assert report["pipeline_kind"] == "cube_prefix_workspace_overlap"
    assert not report["schedule"]["fallback"]
    assert set(report["schedule"]["stages"]) == {0, 1}
    assert any(storage["scope"] == "GM" and storage["selected_versions"] == 2
               for storage in report["storage"])


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_real_target_pipeline_keeps_score_ub_allocations_disjoint(platform):
    import re
    from tilelang.engine.phase import OptimizeForTarget
    from tilelang.transform.ascend_schedule.entry import schedule_module
    from examples.flash_attention_auto128.fa_auto128 import pass_configs
    function, _, target = prepared(platform, False, 130, 130)
    scheduled = schedule_module(tvm.IRModule({"main": function}), "ascendc", pass_configs)
    # Even an explicit request for lexical-lifetime reuse must not override
    # the asynchronous allocation policy of the automatic scheduling path.
    config = dict(pass_configs, **{"tl.ascend_memory_planning": True})
    with tvm.transform.PassContext(config=config):
        native = OptimizeForTarget(scheduled, target, platform)
        source = tvm.get_global_func("target.build.tilelang_ascend")(native, target,
                                                                     platform).get_source()
    addresses = {
        str(var): int(offset) for var, offset in native["main"].attrs["address_map"].items()
    }
    intervals = []
    for name in ("score", "score_recv"):
        offset = int(addresses[name])
        allocation = re.search(
            rf"\b{name}\s*=\s*ascend_ub\.GetWithOffset<float>\((\d+),\s*{offset}\s*\)", source)
        assert allocation, source
        # Use the physical allocation after Vector subblock partitioning,
        # not the logical tile shape (each lane owns only half its rows).
        intervals.append((offset, offset + int(allocation.group(1)) * 4))
    a, b = intervals
    assert a[1] <= b[0] or b[1] <= a[0]


@pytest.mark.parametrize("trips", [0, 1])
def test_zero_and_one_trip_owner_keeps_seed_and_drain_outside_wave(trips):
    from tilelang.transform.ascend_schedule.score_pipeline import prepare_score_pipeline
    from tilelang.transform.ascend_schedule.materializer import materialize_flat_function
    from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks
    function, extraction, _ = prepared("A2", False, 192)
    owner = next(context[0] for context in extraction.loops.values() if context)
    changed = tir.For(
        owner.loop_var, owner.min, trips, owner.kind, owner.body, annotations=owner.annotations)
    function = function.with_body(
        tir.stmt_functor.ir_transform(function.body, None, lambda node: changed
                                      if node.same_as(owner) else None))
    staged, report = prepare_score_pipeline(function, extract_tasks(function), "A2")
    result = materialize_flat_function(
        staged.function, staged.extraction, "A2", prefetch=staged, prefetch_report=report)
    owners = []
    tir.stmt_functor.post_order_visit(
        result.body, lambda node: owners.append(node) if isinstance(node, tir.For) else None)
    assert len(owners) == 1 and int(owners[0].extent) == trips + 1
    flags_all, flags_inside = [], []

    def flags(node, out):
        if isinstance(node, tir.Call) and getattr(
                node.op, "name", "") in ("tl.ascend_set_cross_flag", "tl.ascend_wait_cross_flag"):
            out.append(node)

    tir.stmt_functor.post_order_visit(result.body, lambda node: flags(node, flags_all))
    tir.stmt_functor.post_order_visit(owners[0].body, lambda node: flags(node, flags_inside))
    assert len(flags_all) > len(flags_inside)


@pytest.mark.parametrize("bad", ["drop_carried", "wrong_slots"])
def test_protocol_mutations_do_not_bypass_complete_dependency_or_slot_checks(bad):
    from dataclasses import replace
    from tilelang.transform.ascend_schedule.score_pipeline import prepare_score_pipeline
    from tilelang.transform.ascend_schedule.materializer import materialize_flat_function
    function, extraction, _ = prepared("A2", False, 192)
    staged, report = prepare_score_pipeline(function, extraction, "A2")
    edge = next(
        edge for edge in staged.dependencies
        if edge.distance == 2 and (edge.source, edge.target, edge.distance) in staged.slot_counts)
    if bad == "drop_carried":
        staged = replace(
            staged, dependencies=tuple(item for item in staged.dependencies if item != edge))
    else:
        slots = dict(staged.slot_counts)
        slots[(edge.source, edge.target, edge.distance)] = 1
        staged = replace(staged, slot_counts=slots)
    with pytest.raises(ValueError, match="missing dependency|slot count"):
        materialize_flat_function(
            staged.function, staged.extraction, "A2", prefetch=staged, prefetch_report=report)


def test_additional_carried_war_at_same_wave_is_not_ignored():
    from tilelang.transform.ascend_schedule.score_pipeline import prepare_score_pipeline
    from tilelang.transform.ascend_schedule.dependency import analyze_dependencies
    from tilelang.transform.ascend_schedule.model import Dependency
    from tilelang.transform.ascend_schedule.solver import ScheduleInfeasible
    function, extraction, _ = prepared("A2", False, 192)
    body = [task for task in extraction.tasks if extraction.loops[task.id]]
    source = next(task for task in reversed(body) if task.core == "C" and task.pipe == "PIPE_M")
    target = next(task for task in body if task.core == "C" and task.pipe == "PIPE_M")

    # Model an additional conservative overlap supplied by dependency analysis.
    # This tests the ordering gate, not a claim the current FA aliases these tiles.
    def extra_edge(tasks, **kwargs):
        return [*analyze_dependencies(tasks, **kwargs), Dependency(source.id, target.id, "WAR", 1)]

    with patch("tilelang.transform.ascend_schedule.dependency.analyze_dependencies", extra_edge):
        with pytest.raises(ScheduleInfeasible, match="same-wave dependency"):
            prepare_score_pipeline(function, extraction, "A2")


def test_infeasible_score_schedule_keeps_diagnosed_sequential_path(monkeypatch):
    from tilelang.transform.ascend_schedule import score_pipeline
    from tilelang.transform.ascend_schedule.entry import schedule_module
    from tilelang.transform.ascend_schedule.solver import ScheduleInfeasible
    from examples.flash_attention_auto128.fa_auto128 import pass_configs
    function, _, _ = prepared("A2", False, 192)

    def fail(*_):
        raise ScheduleInfeasible("test score schedule unavailable")

    monkeypatch.setattr(score_pipeline, "prepare_score_pipeline", fail)
    result = schedule_module(tvm.IRModule({"main": function}), "ascendc", pass_configs)["main"]
    assert "test score schedule unavailable" in str(result.attrs["tl.ascend.prefetch_diagnostic"])
    assert not result.attrs.get("tl.ascend.iteration_versions", {})
    assert not result.attrs.get("tl.ascend.prefetch_materialized", 0)
