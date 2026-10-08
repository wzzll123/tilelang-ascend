"""Physical alias contracts require explicit execution proofs, never costs."""

import importlib.util
from pathlib import Path
import sys

import pytest

_PATH = Path(__file__).resolve().parents[3] / "tilelang/transform/ascend_schedule/async_alias.py"
_SPEC = importlib.util.spec_from_file_location("ascend_async_alias_test", _PATH)
alias = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = alias
_SPEC.loader.exec_module(alias)


def alloc(name, task, size=16, core="vector", scope="UB", **kwargs):
    return alias.Allocation(name, size, 16, scope, core, (task,), **kwargs)


def test_cross_pipe_without_flag_cannot_reuse():
    tasks = [alias.ExecutionTask("read"), alias.ExecutionTask("write")]
    layout = alias.plan_safe_aliases([alloc("a", "read"), alloc("b", "write")], tasks, [],
                                     {("vector", "UB"): 32})
    assert layout.offsets == {"a": 0, "b": 16}
    assert not layout.proofs


def test_explicit_event_and_async_completion_allow_reuse():
    tasks = [
        alias.ExecutionTask("issue", asynchronous=True, completion_task="done"),
        alias.ExecutionTask("done"),
        alias.ExecutionTask("consume")
    ]
    edges = [
        alias.ExecutionEdge("issue", "done", "fifo"),
        alias.ExecutionEdge("done", "consume", "sync")
    ]
    layout = alias.plan_safe_aliases(
        [alloc("a", "issue"), alloc("b", "consume")], tasks, edges, {("vector", "UB"): 16})
    assert layout.offsets == {"a": 0, "b": 0}
    proof = layout.proofs[0]
    assert proof.before == "a" and proof.after == "b"
    assert ("done", "consume") in proof.ordered_access_pairs
    assert proof.paths


def test_inferred_data_dependency_is_not_execution_proof():
    tasks = [alias.ExecutionTask("a"), alias.ExecutionTask("b")]
    edges = [alias.ExecutionEdge("a", "b", "memory_dependency")]
    layout = alias.plan_safe_aliases([alloc("a", "a"), alloc("b", "b")], tasks, edges,
                                     {("vector", "UB"): 32})
    assert layout.offsets["a"] != layout.offsets["b"]


def test_chained_sync_and_all_accesses_requirement():
    tasks = [alias.ExecutionTask(t) for t in ("a1", "a2", "event", "b")]
    edges = [alias.ExecutionEdge("a1", "event"), alias.ExecutionEdge("event", "b")]
    allocations = [alias.Allocation("a", 16, 16, "UB", "vector", ("a1", "a2")), alloc("b", "b")]
    unsafe = alias.plan_safe_aliases(allocations, tasks, edges, {("vector", "UB"): 32})
    assert unsafe.offsets == {"a": 0, "b": 16}
    edges.append(alias.ExecutionEdge("a2", "event"))
    safe = alias.plan_safe_aliases(allocations, tasks, edges, {("vector", "UB"): 16})
    assert safe.offsets == {"a": 0, "b": 0}


def test_domains_have_separate_capacities_and_no_alias_proof():
    tasks = [alias.ExecutionTask("a"), alias.ExecutionTask("b")]
    layout = alias.plan_safe_aliases(
        [alloc("a", "a"), alloc("b", "b", core="cube", scope="L1")], tasks, [], {
            ("vector", "UB"): 16,
            ("cube", "L1"): 16
        })
    assert layout.offsets == {"a": 0, "b": 0}
    assert layout.domains["a"] != layout.domains["b"]
    assert not layout.proofs


def test_unknown_guards_lifetimes_and_no_access_fail_closed():
    tasks = [alias.ExecutionTask("a", guard_known=False), alias.ExecutionTask("b")]
    edges = [alias.ExecutionEdge("a", "b")]
    layout = alias.plan_safe_aliases([alloc("a", "a"), alloc("b", "b")], tasks, edges,
                                     {("vector", "UB"): 32})
    assert layout.offsets == {"a": 0, "b": 16}
    for accesses in ((), ("a",)):
        a = alias.Allocation("a", 16, 16, "UB", "vector", accesses, lifetime_known=False)
        layout = alias.plan_safe_aliases([a, alloc("b", "b")], tasks, edges, {("vector", "UB"): 32})
        assert not layout.proofs


def test_async_issue_order_without_completion_order_is_insufficient():
    tasks = [
        alias.ExecutionTask("a", asynchronous=True, completion_task="done"),
        alias.ExecutionTask("done"),
        alias.ExecutionTask("b")
    ]
    edges = [alias.ExecutionEdge("a", "done", "fifo"), alias.ExecutionEdge("a", "b")]
    layout = alias.plan_safe_aliases([alloc("a", "a"), alloc("b", "b")], tasks, edges,
                                     {("vector", "UB"): 32})
    assert not layout.proofs


def test_cycles_backedges_unknown_ids_and_unbounded_instances_rejected():
    tasks = [alias.ExecutionTask("a"), alias.ExecutionTask("b")]
    allocations = [alloc("a", "a"), alloc("b", "b")]
    kwargs = dict(allocations=allocations, tasks=tasks, capacities={("vector", "UB"): 32})
    with pytest.raises(ValueError, match="cycle"):
        alias.plan_safe_aliases(
            **kwargs,
            execution_edges=[
                alias.ExecutionEdge("a", "b"),
                alias.ExecutionEdge("b", "a", "memory_dependency")
            ])
    with pytest.raises(ValueError, match="single-instance"):
        alias.plan_safe_aliases(
            **kwargs, execution_edges=[alias.ExecutionEdge("a", "b", distance=1)])
    with pytest.raises(ValueError, match="unknown task"):
        alias.plan_safe_aliases(**kwargs, execution_edges=[alias.ExecutionEdge("a", "missing")])
    with pytest.raises(ValueError, match="single-instance"):
        alias.plan_safe_aliases(**kwargs, execution_edges=[], bounded_single_instance=False)


def test_alignment_and_capacity_are_checked():
    tasks = [alias.ExecutionTask("a"), alias.ExecutionTask("b")]
    allocations = [
        alias.Allocation("a", 17, 16, "UB", "vector", ("a",)),
        alias.Allocation("b", 8, 32, "UB", "vector", ("b",))
    ]
    layout = alias.plan_safe_aliases(allocations, tasks, [], {("vector", "UB"): 40})
    assert layout.offsets == {"a": 0, "b": 32}
    assert layout.footprints[("vector", "UB")] == 40
    with pytest.raises(alias.CapacityExceeded):
        alias.plan_safe_aliases(allocations, tasks, [], {("vector", "UB"): 39})


def test_every_overlapping_allocation_requires_pairwise_proof():
    tasks = [alias.ExecutionTask(t) for t in ("a", "b", "c")]
    allocations = [alloc(t, t) for t in ("a", "b", "c")]
    edges = [alias.ExecutionEdge("a", "b"), alias.ExecutionEdge("a", "c")]
    layout = alias.plan_safe_aliases(allocations, tasks, edges, {("vector", "UB"): 32})
    assert layout.offsets == {"a": 0, "b": 0, "c": 16}
    edges.append(alias.ExecutionEdge("b", "c"))
    layout = alias.plan_safe_aliases(allocations, tasks, edges, {("vector", "UB"): 16})
    assert layout.offsets == {"a": 0, "b": 0, "c": 0}
    assert len(layout.proofs) == 3
    with pytest.raises(TypeError):
        layout.offsets["a"] = 100


def test_reverse_execution_order_is_valid_proof():
    tasks = [alias.ExecutionTask("a"), alias.ExecutionTask("b")]
    layout = alias.plan_safe_aliases([alloc("a", "a"), alloc("b", "b")], tasks,
                                     [alias.ExecutionEdge("b", "a")], {("vector", "UB"): 16})
    assert layout.proofs[0].before == "b" and layout.proofs[0].after == "a"


def test_unknown_completion_and_guarded_intermediate_cannot_prove_order():
    tasks = [alias.ExecutionTask("a", asynchronous=True), alias.ExecutionTask("b")]
    layout = alias.plan_safe_aliases([alloc("a", "a"), alloc("b", "b")], tasks,
                                     [alias.ExecutionEdge("a", "b")], {("vector", "UB"): 32})
    assert not layout.proofs
    tasks = [
        alias.ExecutionTask("a"),
        alias.ExecutionTask("event", guard_known=False),
        alias.ExecutionTask("b")
    ]
    layout = alias.plan_safe_aliases(
        [alloc("a", "a"), alloc("b", "b")], tasks,
        [alias.ExecutionEdge("a", "event"),
         alias.ExecutionEdge("event", "b")], {("vector", "UB"): 32})
    assert not layout.proofs
