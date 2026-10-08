# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Discover prefetch storage by ownership and source provenance, not names."""

import pytest
import tilelang  # noqa: F401
from tvm import tir
from tilelang.transform.ascend_schedule.tir_adapter import extract_tasks


def fixture(tail=64, prefix=False, mutable_source=False):
    gm = tir.decl_buffer((128, 128), "float16", name="input")
    tiles = [
        tir.decl_buffer((8192,), "float16", name=name, scope="shared.l1")
        for name in ("arbitrary_a", "arbitrary_b")
    ]
    l0 = tir.decl_buffer((8192,), "float16", name="l0", scope="wmma.matrix_b")
    ub = tir.decl_buffer((8192,), "float16", name="ub", scope="shared.ub")
    epoch = tir.Var("epoch", "int32")
    writes = [
        tir.Evaluate(
            tir.call_extern("handle", "copy_gm_to_l1<half,64,128>", gm.access_ptr("r"),
                            tile.access_ptr("w"), 128, tail if index == 0 else 64, 128, 64, 128))
        for index, tile in enumerate(tiles)
    ]
    reads = [
        tir.Evaluate(
            tir.call_extern("handle", "copy_l1_to_l0b<half,64,128,false>", tile.access_ptr("r"),
                            l0.access_ptr("w"), 64, 128)) for tile in tiles
    ]
    statements = [writes[0], reads[0], writes[1], reads[1]]
    if mutable_source:
        statements.insert(
            0,
            tir.Evaluate(
                tir.call_extern("handle", "copy_ub_to_gm<half,128,64>", ub.access_ptr("r"),
                                gm.access_ptr("w"), 128, 64, 128)))
    owner = tir.For(epoch, 0, 2, tir.ForKind.SERIAL, tir.SeqStmt(statements))
    body = tir.SeqStmt([writes[0], owner]) if prefix else owner
    root = tir.Block([], [], [], "root", body, alloc_buffers=[*tiles, l0, ub])
    function = tir.PrimFunc([gm.data], tir.BlockRealize([], True, root), buffer_map={gm.data: gm})
    return function, owner, tiles


def test_joint_candidates_use_readonly_input_and_full_epoch_ownership():
    from tilelang.transform.ascend_schedule.l1_prefetch import discover_l1_prefetch
    function, owner, tiles = fixture()
    result = discover_l1_prefetch(function, extract_tasks(function))
    assert len(result.candidates) == 2
    assert not result.diagnostics
    assert [candidate.buffer.name for candidate in result.candidates
           ] == [tile.name for tile in tiles]
    assert all(
        candidate.owner.same_as(owner) and candidate.plan.count == 2
        for candidate in result.candidates)
    assert len(result.tasks) == 4
    assert tuple(result.stages.values()) == (0, 1, 0, 1)
    assert all(candidate.size_bytes == 16384 for candidate in result.candidates)


@pytest.mark.parametrize("case", ["tail", "prefix", "mutable_source"])
def test_ineligible_storage_does_not_gain_prefetch_versions(case):
    from tilelang.transform.ascend_schedule.l1_prefetch import discover_l1_prefetch
    function, _, _ = fixture(
        tail=32 if case == "tail" else 64,
        prefix=case == "prefix",
        mutable_source=case == "mutable_source")
    result = discover_l1_prefetch(function, extract_tasks(function))
    assert len(result.candidates) == (0 if case == "mutable_source" else 1)
    assert result.diagnostics


def test_joint_solver_selects_versions_with_persistent_storage_fixed():
    pytest.importorskip("z3")
    from tilelang.transform.ascend_schedule.l1_prefetch import discover_l1_prefetch, solve_l1_prefetch
    function, _, _ = fixture()
    discovery = discover_l1_prefetch(function, extract_tasks(function))
    report = solve_l1_prefetch(function, discovery, "A2")
    counts = report["schedule"]["buffer_versions"]
    assert counts["prefetch_0"] == counts["prefetch_1"] == 2
    assert all(count == 1 for name, count in counts.items() if name.startswith("fixed_"))


def test_joint_materialization_refreshes_owner_and_models_real_reuse_distance():
    from tilelang.transform.ascend_schedule.l1_prefetch import (discover_l1_prefetch,
                                                                materialize_l1_prefetch)
    function, _, tiles = fixture()
    discovery = discover_l1_prefetch(function, extract_tasks(function))
    result = materialize_l1_prefetch(function, discovery, {"prefetch_0": 2, "prefetch_1": 2})
    selected = [
        buffer for buffer in result.function.body.block.alloc_buffers if any(
            buffer.data.same_as(tile.data) for tile in tiles)
    ]
    assert len(selected) == 2
    assert all(tuple(map(int, buffer.shape)) == (16384,) for buffer in selected)
    candidate_edges = [
        edge for edge in result.dependencies
        if edge.storage is not None and any(edge.storage.var.same_as(tile.data) for tile in tiles)
    ]
    assert {edge.distance for edge in candidate_edges} == {0, 2}
    assert all(count == 2 for count in result.slot_counts.values())
    assert all(context[0].same_as(result.owner) for context in result.extraction.loops.values())
    # L0B is shared by the two consumers and remains one physical version.
    assert any(edge.distance == 1 for edge in result.dependencies if edge.kind == "WAW")


def test_joint_materialization_cannot_accept_unproven_single_slot_selection():
    from tilelang.transform.ascend_schedule.l1_prefetch import (discover_l1_prefetch,
                                                                materialize_l1_prefetch)
    function, _, _ = fixture()
    discovery = discover_l1_prefetch(function, extract_tasks(function))
    with pytest.raises(ValueError, match="lifecycle"):
        materialize_l1_prefetch(function, discovery, {"prefetch_0": 1, "prefetch_1": 2})


def test_joint_ring_dependencies_feed_actual_event_bundles_and_stage_waves():
    from dataclasses import replace
    from tilelang.transform.ascend_schedule.l1_prefetch import (discover_l1_prefetch,
                                                                materialize_l1_prefetch)
    from tilelang.transform.ascend_schedule.cross_sync import emit_mode2_loop_sync
    from tilelang.transform.ascend_schedule.stage_skew import materialize_staged_owner
    function, owner, _ = fixture()
    scoped = tir.For(
        owner.loop_var, owner.min, owner.extent, owner.kind,
        tir.SeqStmt([
            tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, node)
            for node in owner.body.seq
        ]))
    function = function.with_body(
        tir.stmt_functor.ir_transform(function.body, None, lambda node: scoped
                                      if node.same_as(owner) else None))
    vid = tir.IterVar(None, tir.Var("vid", "int32"), tir.IterVar.ThreadIndex, "threadIdx.x")
    function = function.with_body(tir.AttrStmt(vid, "thread_extent", 2, function.body))
    discovery = discover_l1_prefetch(function, extract_tasks(function))
    physical = materialize_l1_prefetch(function, discovery, {"prefetch_0": 2, "prefetch_1": 2})
    body = replace(physical.extraction, loops={task.id: () for task in physical.extraction.tasks})
    synchronized = emit_mode2_loop_sync(
        physical.function,
        physical.owner,
        body,
        physical.dependencies,
        local_ids=tuple(range(8)),
        cross_ids=tuple(range(8)),
        slot_counts=physical.slot_counts)
    owners = []
    tir.stmt_functor.post_order_visit(
        synchronized.body, lambda node: owners.append(node) if isinstance(node, tir.For) else None)
    assert len(owners) == 1
    staged = materialize_staged_owner(synchronized, owners[0], body.task_nodes, physical.stages,
                                      physical.dependencies)
    waves = []
    tir.stmt_functor.post_order_visit(
        staged.body, lambda node: waves.append(node) if isinstance(node, tir.For) else None)
    assert len(waves) == 1 and int(waves[0].extent) == 3
    assert all(isinstance(node, tir.IfThenElse) for node in waves[0].body.seq)


def test_joint_capacity_not_two_independent_per_buffer_checks(monkeypatch):
    pytest.importorskip("z3")
    from tilelang.transform.ascend_schedule.l1_prefetch import discover_l1_prefetch, solve_l1_prefetch
    from tilelang.transform.ascend_schedule import resources
    original = resources.allocation_budgets
    monkeypatch.setattr(resources, "allocation_budgets", lambda platform: {
        **original(platform), "L1": 49152
    })
    function, _, _ = fixture()
    discovery = discover_l1_prefetch(function, extract_tasks(function))
    with pytest.raises(ValueError):
        solve_l1_prefetch(function, discovery, "A2")


def test_unaccounted_physical_allocation_cannot_bypass_joint_capacity():
    from tilelang.transform.ascend_schedule.l1_prefetch import discover_l1_prefetch
    function, _, _ = fixture()
    hidden = tir.Var("hidden", "handle")
    function = function.with_body(
        tir.Allocate(hidden, "float16", [262144], tir.const(True, "bool"), function.body))
    with pytest.raises(ValueError, match="allocation accounting"):
        discover_l1_prefetch(function, extract_tasks(function))


@pytest.mark.parametrize("solver_failure", [False, True])
def test_real_compiler_entry_selects_fa_prefetch_without_hand_hints(solver_failure, monkeypatch):
    pytest.importorskip("z3")
    import tvm
    from examples.flash_attention_auto128.fa_auto128 import (FlashAttentionConfig,
                                                             flash_attention_auto128_fwd,
                                                             pass_configs)
    from tilelang.engine.phase import LowerAndLegalize
    if solver_failure:
        from tilelang.transform.ascend_schedule import l1_prefetch
        from tilelang.transform.ascend_schedule.solver import ScheduleInfeasible

        def fail_solve(*_):
            raise ScheduleInfeasible("test solver failure")

        monkeypatch.setattr(l1_prefetch, "solve_l1_prefetch", fail_solve)
    function = flash_attention_auto128_fwd.__wrapped__(FlashAttentionConfig(
        1, 1, 128, 192, 128)).with_attr("npu_platform", "A2")
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    with tvm.transform.PassContext(config=pass_configs):
        lowered = LowerAndLegalize(tvm.IRModule({"main": function}), target)["main"]
    assert int(lowered.attrs.get("tl.ascend.prefetch_materialized",
                                 0)) == int(not solver_failure), str(
                                     lowered.attrs.get("tl.ascend.prefetch_diagnostic",
                                                       "no optimization diagnostic"))
    if solver_failure:
        assert "test solver failure" in str(lowered.attrs["tl.ascend.prefetch_diagnostic"])
        assert not lowered.attrs.get("tl.ascend.iteration_versions", {})
    else:
        assert len(lowered.attrs["tl.ascend.iteration_versions"]) == 2


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("causal", [False, True])
def test_production_fa_discovers_kv_but_not_q_or_softmax_probability(platform, causal, monkeypatch):
    import tvm
    from examples.flash_attention_auto128.fa_auto128 import (FlashAttentionConfig,
                                                             flash_attention_auto128_fwd,
                                                             pass_configs)
    from tilelang.engine.phase import LowerAndLegalize
    from tilelang.transform.ascend_schedule.gemm_decomposition import decompose_function_gemms
    from tilelang.transform.ascend_schedule.workspace_tasks import prepare_cross_core_copies
    from tilelang.transform.ascend_schedule.l1_prefetch import discover_l1_prefetch
    from tilelang.transform.ascend_schedule import entry
    # Observe the real scheduler input before generated synchronization.
    monkeypatch.setattr(entry, "schedule_module", lambda mod, *_: mod)
    function = flash_attention_auto128_fwd.__wrapped__(
        FlashAttentionConfig(1, 1, 128, 192, 128,
                             causal=causal)).with_attr("npu_platform", platform)
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    with tvm.transform.PassContext(config=pass_configs):
        mod = prepare_cross_core_copies(LowerAndLegalize(tvm.IRModule({"main": function}), target))
        function = decompose_function_gemms(mod["main"], root_scratch=True)
    extraction = extract_tasks(function)
    discovery = discover_l1_prefetch(function, extraction)
    assert sorted(candidate.buffer.name for candidate in discovery.candidates) == ["k", "v"]
    assert sum(stage == 0 for stage in discovery.stages.values()) == 2
    assert any(message.startswith("q:") for message in discovery.diagnostics)
    assert any(
        message.startswith("p:") and "readonly" in message for message in discovery.diagnostics)
    from tilelang.transform.ascend_schedule.l1_prefetch import materialize_l1_prefetch
    from tilelang.transform.ascend_schedule.core_scopes import normalize_core_scopes
    scoped = normalize_core_scopes(function, extraction)
    scoped_extraction = extract_tasks(scoped)
    scoped_discovery = discover_l1_prefetch(scoped, scoped_extraction)
    physical = materialize_l1_prefetch(scoped, scoped_discovery, {"prefetch_0": 2, "prefetch_1": 2})
    candidate_vars = [candidate.buffer.data for candidate in discovery.candidates]
    edges = [
        edge for edge in physical.dependencies if edge.storage is not None and any(
            edge.storage.var.same_as(var) for var in candidate_vars)
    ]
    assert {edge.distance for edge in edges} == {0, 2}
    assert any(edge.kind == "WAR" and edge.distance == 2 for edge in edges)
    records = physical.function.attrs["tl.ascend.iteration_versions"]
    assert len(records) == 2
    assert all([int(record[0]), int(record[1])] == [2, 8192] for record in records.values())
    from tilelang.transform.ascend_schedule.phases import emit_three_phase_sync
    from dataclasses import replace
    from tilelang.transform.ascend_schedule.local_sync import LocalSyncError
    broken = replace(
        physical,
        slot_counts={
            key: count for key, count in physical.slot_counts.items() if key[2] != 0
        })
    with pytest.raises(LocalSyncError, match="event slot count"):
        emit_three_phase_sync(broken.function, broken.extraction, prefetch=broken)
    synchronized = emit_three_phase_sync(physical.function, physical.extraction, prefetch=physical)
    waves = []
    tir.stmt_functor.post_order_visit(
        synchronized.body, lambda node: waves.append(node) if isinstance(node, tir.For) else None)
    assert len(waves) == 1
    assert waves[0].loop_var.name.endswith("_wave")
    assert len(waves[0].body.seq) == len(physical.stages)
    from tilelang.engine.phase import OptimizeForTarget
    synchronized = synchronized.with_attr("tl.ascend.schedule_materialized", 1)
    with tvm.transform.PassContext(config=pass_configs):
        native = OptimizeForTarget(tvm.IRModule({"main": synchronized}), target, platform)
        source = tvm.get_global_func("target.build.tilelang_ascend")(native, target,
                                                                     platform).get_source()
    assert "_wave" in source and "8192" in source
    assert "CrossCoreSetFlag<0x2," in source
    assert "PipeBarrier<PIPE_ALL>" not in source
