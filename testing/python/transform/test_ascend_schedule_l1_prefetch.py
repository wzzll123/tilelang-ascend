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
