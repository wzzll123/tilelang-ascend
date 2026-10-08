# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Composition preflight only; not production entry or real numerical acceptance."""
from dataclasses import replace
import math
import tvm
from tvm import tir, DataType
from test_ascend_schedule_score_pipeline import prepared
from tilelang.transform.ascend_schedule.score_pipeline import prepare_score_pipeline
from tilelang.transform.ascend_schedule.l1_prefetch import discover_l1_prefetch, materialize_l1_prefetch
from tilelang.transform.ascend_schedule.ownership import OwnerDescriptor, plan_storage_epochs
from tilelang.transform.ascend_schedule.workspace_proof import VERSION_ATTR, certify_workspace_versions
from tilelang.transform.ascend_schedule.dependency import analyze_dependencies
from tilelang.transform.ascend_schedule.phases import emit_three_phase_sync
from tilelang.transform.ascend_schedule.materializer import MARKER
from tilelang.transform.ascend_schedule.resources import allocation_budgets
from tilelang.engine.phase import OptimizeForTarget
from examples.flash_attention_auto128.fa_auto128 import pass_configs

import pytest


@pytest.mark.parametrize('platform', ['A2', 'A3'])
@pytest.mark.parametrize('causal', [False, True])
def test_combined_input_and_score_rings_generate_three_stage_protocol(platform, causal):
    f, e, target = prepared(platform, causal, 256, 130, block_m=64, block_n=128)
    score, report = prepare_score_pipeline(f, e, platform)
    d = discover_l1_prefetch(score.function, score.extraction)
    assert d.candidates, d.diagnostics
    stages = {key: value + 1 for key, value in score.stages.items()}
    for c in d.candidates:
        stages[c.writer_id] = 0
    candidates = []
    for c in d.candidates:
        descriptor = OwnerDescriptor(
            c.owner, d.tasks, stages, full_footprints=((c.storage, ((0, c.size_bytes),)),))
        plan = plan_storage_epochs(c.storage, (descriptor,))
        assert plan.eligible and plan.owners[0].mode == 'iteration', plan
        candidates.append(replace(c, plan=plan))
    d = replace(d, candidates=tuple(candidates), stages=stages)
    counts = {f'prefetch_{i}': c.plan.count for i, c in enumerate(candidates)}
    combined = materialize_l1_prefetch(score.function, d, counts)
    records = dict(combined.function.attrs[VERSION_ATTR].items())
    gm = {
        v: info for v, info in records.items() if any(
            b.data.same_as(v) for b in combined.function.buffer_map.values())
    }
    fresh, diagnostics = certify_workspace_versions(
        combined.function.with_attr(VERSION_ATTR, gm), combined.extraction)
    assert not diagnostics, diagnostics
    body = [t for t in fresh.tasks if t.id in stages]
    deps = tuple(
        analyze_dependencies(body, max_distance=max(int(info[0]) for info in records.values())))
    order = {t.id: i for i, t in enumerate(sorted(body, key=lambda t: stages[t.id]))}
    for edge in deps:
        source, destination = stages[edge.source], stages[edge.target] + edge.distance
        assert source < destination or (source == destination and
                                        order[edge.source] < order[edge.target]), edge
    slots = {}
    for edge in deps:
        for var, info in records.items():
            if edge.storage is not None and edge.storage.var.same_as(var):
                key = edge.source, edge.target, edge.distance
                count = int(info[0])
                assert key not in slots or slots[key] == count
                slots[key] = count
    combined = replace(combined, extraction=fresh, dependencies=deps, slot_counts=slots)
    buffers = []
    tir.stmt_functor.post_order_visit(
        combined.function.body, lambda n: buffers.extend(n.alloc_buffers)
        if isinstance(n, tir.Block) else None)
    scopes = {
        'shared.ub': 'UB',
        'shared.l1': 'L1',
        'wmma.matrix_a': 'L0A',
        'wmma.matrix_b': 'L0B',
        'wmma.accumulator': 'L0C'
    }
    usage = {scope: 0 for scope in scopes.values()}
    seen = set()
    for b in buffers:
        assert b.data not in seen
        seen.add(b.data)
        dims = [int(x) for x in b.shape]
        if b.scope() != 'shared.ub' and len(dims) == 2:
            dims = [(x + 15) // 16 * 16 for x in dims]
        dtype = DataType(b.dtype)
        size = math.prod(dims) * dtype.bits * dtype.lanes // 8
        usage[scopes[b.scope()]] += (size + 31) // 32 * 32
    assert all(size <= allocation_budgets(platform)[scope] for scope, size in usage.items()), usage
    result = emit_three_phase_sync(combined.function, fresh, prefetch=combined).with_attr(MARKER, 1)
    with tvm.transform.PassContext(config=pass_configs):
        native = OptimizeForTarget(tvm.IRModule({'main': result}), target, platform)
        source = tvm.get_global_func('target.build.tilelang_ascend')(native, target,
                                                                     platform).get_source()
    assert 'epoch_wave' in source and 'CrossCoreSetFlag<0x2,' in source
    assert sorted(counts.values()) == [2, 3]
    assert sorted(set(stages.values())) == [0, 1, 2]
    assert len(deps) > 0


def test_partial_kv_inputs_are_not_misreported_as_full_slot_writes():
    f, e, _ = prepared('A2', False, 130, 130, block_m=64, block_n=128)
    score, _ = prepare_score_pipeline(f, e, 'A2')
    discovery = discover_l1_prefetch(score.function, score.extraction)
    assert not discovery.candidates
    assert sum('full rows' in diagnostic for diagnostic in discovery.diagnostics) == 2
