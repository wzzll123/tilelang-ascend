# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Conservative RAW, WAR, and WAW dependencies on physical byte storage.

This models temporal order, not initial-value validity: a read before a first
write stays a WAR edge at distance zero, and a future ring producer cannot
initialize that read. The caller must separately prove initialization or retain
the original semantics. No edge is removed based on owner/full_overwrite hints.
"""

from __future__ import annotations

from math import gcd
from collections.abc import Iterable

from .model import Access, Dependency, Task


def accesses_conflict(a: Access, b: Access, distance: int = 0) -> bool:
    """Whether bytes can overlap between epoch i and epoch i + distance."""
    if a.storage is not b.storage:
        return False
    if not a.writes and not b.writes:
        return False
    if a.footprint() is None or b.footprint() is None:
        return True
    # Unequal unbounded translations or a translated footprint paired with a
    # ring cannot be ruled out by examining just the first epoch.
    a_linear = a.iteration_stride if a.ring_size is None else 0
    b_linear = b.iteration_stride if b.ring_size is None else 0
    if a_linear != b_linear:
        return True
    period_a, period_b = a.ring_size or 1, b.ring_size or 1
    period = period_a * period_b // gcd(period_a, period_b)
    # Large declared rings must not make conservative analysis impractical.
    if period > 4096:
        return True
    for epoch in range(period):
        left, right = a.footprint(epoch), b.footprint(epoch + distance)
        if any(start_a < end_b and start_b < end_a
               for start_a, end_a in left
               for start_b, end_b in right
               if start_a < end_a and start_b < end_b):
            return True
    return False


def _hazards(a: Access, b: Access):
    if a.writes and b.reads:
        yield "RAW"
    if a.reads and b.writes:
        yield "WAR"
    if a.writes and b.writes:
        yield "WAW"


def analyze_dependencies(tasks: Iterable[Task],
                         max_distance: int | None = None) -> list[Dependency]:
    """Return direct hazards, preserving program order within each iteration.

    Positive distances compare every source task in an earlier iteration with
    every target task in a later one, including task self-dependencies. The
    default horizon expands to cover declared ring periods; pass zero to request
    only same-iteration analysis, or an explicit larger horizon as needed.
    Opaque guards are deliberately never compared across iterations.
    """
    tasks = tuple(tasks)
    task_ids = [task.id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task ids must be unique")
    if max_distance is not None and max_distance < 0:
        raise ValueError("max_distance must be nonnegative")
    if max_distance is None:
        max_distance = max((access.ring_size or 1 for task in tasks for access in task.accesses),
                           default=1)
    result, seen = [], set()
    for distance in range(max_distance + 1):
        for source_index, source in enumerate(tasks):
            targets = tasks[source_index + 1:] if distance == 0 else tasks
            for target in targets:
                for a in source.accesses:
                    for b in target.accesses:
                        if not accesses_conflict(a, b, distance):
                            continue
                        for kind in _hazards(a, b):
                            edge = Dependency(source.id, target.id, kind, distance, a.storage)
                            if edge.key not in seen:
                                result.append(edge)
                                seen.add(edge.key)
    return result


def validate_dependencies(tasks: Iterable[Task],
                          dependencies: Iterable[Dependency],
                          max_distance: int | None = None) -> None:
    """Require every modeled hazard; used to catch dropped-edge mutations.

    This validates the direct dependency graph, not lowered flags or a runtime
    schedule. Additional conservative edges are permitted.
    """
    actual = {edge.key for edge in dependencies}
    missing = [edge for edge in analyze_dependencies(tasks, max_distance) if edge.key not in actual]
    if missing:
        raise ValueError(f"missing dependency: {missing[0]!r}")
