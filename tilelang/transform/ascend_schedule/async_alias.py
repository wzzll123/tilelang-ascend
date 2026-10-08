# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Independent, bounded-instance physical alias contracts.

This is a pure contract/layout model, not a TIR allocation planner. Lexical
positions, data hazards, estimated costs, and solver starts cannot establish
completion. Reuse requires explicit sync/FIFO paths ordering every access and
asynchronous completion in one allocation before every access/completion in the
other. Missing facts retain disjoint storage; cyclic or symbolic instances are
rejected. Allocation domains are separate ``(core, scope)`` address spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping


@dataclass(frozen=True)
class Allocation:
    id: str | int
    size_bytes: int
    alignment: int
    scope: str
    core: str
    access_tasks: tuple[str | int, ...] = ()
    lifetime_known: bool = True

    def __post_init__(self):
        object.__setattr__(self, "access_tasks", tuple(self.access_tasks))


@dataclass(frozen=True)
class ExecutionTask:
    id: str | int
    guard_known: bool = True
    asynchronous: bool = False
    completion_task: str | int | None = None


@dataclass(frozen=True)
class ExecutionEdge:
    source: str | int
    target: str | int
    kind: str = "sync"
    distance: int = 0


@dataclass(frozen=True)
class AliasProof:
    before: str | int
    after: str | int
    domain: tuple[str, str]
    offset: int
    ordered_access_pairs: tuple[tuple[str | int, str | int], ...]
    paths: tuple[tuple[ExecutionEdge, ...], ...]


@dataclass(frozen=True)
class AliasLayout:
    offsets: Mapping[str | int, int]
    domains: Mapping[str | int, tuple[str, str]]
    footprints: Mapping[tuple[str, str], int]
    proofs: tuple[AliasProof, ...]


class CapacityExceeded(ValueError):
    """A conservative physical layout does not fit explicit capacity."""


def plan_safe_aliases(
    allocations,
    tasks,
    execution_edges,
    capacities,
    *,
    bounded_single_instance=True,
) -> AliasLayout:
    """Produce first-fit exact-slot contracts from supplied execution evidence.

    Edges of kinds ``sync`` and ``fifo`` are admissible happens-before proof.
    Other edge kinds are checked for contradictory graph geometry but never
    prove order. ``fifo`` is explicit evidence supplied by the caller: task
    positions or matching pipe names are never used to infer such edges.
    ``guard_known`` certifies an unconditional execution instance. An unknown
    guard invalidates proof paths through that task. Iteration distances and
    unbounded/symbolic instances are unsupported and rejected.
    """
    if bounded_single_instance is not True:
        raise ValueError("only an explicitly bounded single-instance lifetime is supported")
    allocations, tasks, execution_edges = tuple(allocations), tuple(tasks), tuple(execution_edges)
    capacities = dict(capacities)
    by_id = {}
    for task in tasks:
        if type(task.id) not in (str, int):
            raise ValueError("single-instance task ids must be concrete strings or integers")
        if type(task.asynchronous) is not bool:
            raise ValueError("asynchronous lifetime must be explicitly true or false")
        if task.id in by_id:
            raise ValueError("task ids must be unique")
        by_id[task.id] = task
    for task in tasks:
        if task.completion_task is not None and task.completion_task not in by_id:
            raise ValueError("asynchronous completion references unknown task")
    seen_allocations = set()
    for allocation in allocations:
        if type(allocation.id) not in (str, int) or allocation.id in seen_allocations:
            raise ValueError("allocation ids must be concrete and unique")
        seen_allocations.add(allocation.id)
        if (type(allocation.size_bytes) is not int or allocation.size_bytes < 1 or
                type(allocation.alignment) is not int or allocation.alignment < 1):
            raise ValueError("allocation size and alignment must be explicit positive integers")
        if type(allocation.scope) is not str or type(allocation.core) is not str:
            raise ValueError("allocation core/scope domains must be explicit strings")
        if any(task not in by_id for task in allocation.access_tasks):
            raise ValueError("allocation access references unknown task")
        domain = (allocation.core, allocation.scope)
        if domain not in capacities:
            raise ValueError(f"explicit capacity required for allocation domain {domain}")
    if any(type(cap) is not int or cap < 0 for cap in capacities.values()):
        raise ValueError("capacities must be explicit nonnegative integer byte counts")

    full_graph = {task.id: [] for task in tasks}
    proof_graph = {task.id: [] for task in tasks}
    for edge in execution_edges:
        if type(edge.distance) is not int or edge.distance != 0:
            raise ValueError("backedges are outside bounded single-instance lifetime analysis")
        if edge.source not in by_id or edge.target not in by_id:
            raise ValueError("execution edge references unknown task")
        full_graph[edge.source].append(edge.target)
        if (edge.kind in ("sync", "fifo") and by_id[edge.source].guard_known is True and
                by_id[edge.target].guard_known is True):
            proof_graph[edge.source].append(edge)
    visited, active = set(), set()

    def check_cycles(node):
        if node in active:
            raise ValueError("execution/dependency graph contains a cycle")
        if node in visited:
            return
        active.add(node)
        for target in full_graph[node]:
            check_cycles(target)
        active.remove(node)
        visited.add(node)

    for task in by_id:
        check_cycles(task)

    def path(source, target):
        # Equality is never an ordering proof: one access cannot precede itself.
        if source == target:
            return None
        pending = [(source, ())]
        seen = {source}
        while pending:
            current, prefix = pending.pop(0)
            for edge in proof_graph[current]:
                candidate = prefix + (edge,)
                if edge.target == target:
                    return candidate
                if edge.target not in seen:
                    seen.add(edge.target)
                    pending.append((edge.target, candidate))
        return None

    def lifetime(allocation):
        if allocation.lifetime_known is not True or not allocation.access_tasks:
            return None
        expanded, stack = set(), set()

        def include(task_id):
            if task_id in stack:
                raise ValueError("asynchronous completion lifetime contains a cycle")
            if task_id in expanded:
                return True
            task = by_id[task_id]
            if task.guard_known is not True:
                return False
            stack.add(task_id)
            if task.asynchronous:
                completion = task.completion_task
                if completion is None or path(task_id, completion) is None:
                    return False
                if not include(completion):
                    return False
            stack.remove(task_id)
            expanded.add(task_id)
            return True

        for task_id in allocation.access_tasks:
            if not include(task_id):
                return None
        return tuple(task.id for task in tasks if task.id in expanded)

    lifetimes = {allocation.id: lifetime(allocation) for allocation in allocations}

    def proof(a, b, offset):
        left, right = lifetimes[a.id], lifetimes[b.id]
        if left is None or right is None:
            return None
        for before, after, sources, targets in ((a, b, left, right), (b, a, right, left)):
            pairs, paths = [], []
            for source in sources:
                for target in targets:
                    found = path(source, target)
                    if found is None:
                        break
                    pairs.append((source, target))
                    paths.append(found)
                else:
                    continue
                break
            else:
                return AliasProof(before.id, after.id, (a.core, a.scope), offset, tuple(pairs),
                                  tuple(paths))
        return None

    placed, offsets, domains, footprints, proofs = [], {}, {}, {}, []
    for allocation in allocations:
        domain = (allocation.core, allocation.scope)
        peers = [a for a in placed if domains[a.id] == domain]

        def align(value, alignment=allocation.alignment):
            units = (value + alignment - 1) // alignment
            return units * alignment

        candidates = {0}
        for peer in peers:
            candidates.add(align(offsets[peer.id]))
            candidates.add(align(offsets[peer.id] + peer.size_bytes))
        chosen, chosen_proofs = None, []
        for candidate in sorted(candidates):
            if candidate + allocation.size_bytes > capacities[domain]:
                continue
            candidate_proofs = []
            for peer in peers:
                offset = offsets[peer.id]
                overlap = (
                    candidate < offset + peer.size_bytes and
                    offset < candidate + allocation.size_bytes)
                if overlap:
                    # Only exact starting-slot reuse is represented by this contract.
                    evidence = proof(peer, allocation, candidate) if candidate == offset else None
                    if evidence is None:
                        break
                    candidate_proofs.append(evidence)
            else:
                chosen, chosen_proofs = candidate, candidate_proofs
                break
        if chosen is None:
            raise CapacityExceeded(f"safe layout exceeds capacity for {domain}: {allocation.id}")
        offsets[allocation.id], domains[allocation.id] = chosen, domain
        footprints[domain] = max(footprints.get(domain, 0), chosen + allocation.size_bytes)
        proofs.extend(chosen_proofs)
        placed.append(allocation)
    return AliasLayout(
        MappingProxyType(offsets), MappingProxyType(domains), MappingProxyType(footprints),
        tuple(proofs))
