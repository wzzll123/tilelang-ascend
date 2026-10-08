# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Standalone modulo scheduling constraints for explicit Ascend cost models.

No TileLang or TVM imports are needed. Install the optional ``z3-solver`` package
to optimize schedules; missing Z3 or an incomplete solve produces a diagnosed
sequential schedule in automatic mode. Hardware capacities are supplied by the
caller rather than inferred here. Resource constraints follow the cyclic issue
window model in tilelang/ascend/transform/z3_scheduler.py (revision 47976f2).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class TaskCost:
    latency: int
    ii: int
    pipe: str | int


@dataclass(frozen=True)
class Dependency:
    src: int
    dst: int
    distance: int = 0
    latency: int | None = None


@dataclass(frozen=True)
class StorageCost:
    size_bytes: int
    scope: str
    max_versions: int = 2
    fixed_versions: int | None = None


@dataclass(frozen=True)
class ReuseDependency:
    """A storage anti-dependence whose iteration distance is its version count."""

    src: int
    dst: int
    storage: str
    latency: int | None = None


@dataclass(frozen=True)
class ScheduleResult:
    start_times: tuple[int, ...]
    stages: tuple[int, ...]
    order: tuple[int, ...]
    buffer_versions: dict[str, int]
    ii: int
    fallback: bool = False
    diagnostics: tuple[str, ...] = ()


class ScheduleInfeasible(ValueError):
    """No valid schedule can be returned for the requested constraints."""


def _load_z3():
    try:
        import z3
    except ImportError:
        return None
    return z3


def _shares_pipe(a: TaskCost, b: TaskCost) -> bool:
    if isinstance(a.pipe, int) and isinstance(b.pipe, int):
        return bool(a.pipe & b.pipe)
    return bool(a.pipe) and a.pipe == b.pipe


def solve_schedule(
    tasks: Sequence[TaskCost],
    dependencies: Sequence[Dependency],
    *,
    storage: Mapping[str, StorageCost] | None = None,
    reuse_dependencies: Sequence[ReuseDependency] = (),
    capacities: Mapping[str, int] | None = None,
    manual_stages: Sequence[int] | None = None,
    max_ii: int | None = None,
    timeout_ms: int = 1000,
) -> ScheduleResult:
    """Minimize II, then storage versions and starts within a bounded search.

    True dependencies always retain their supplied iteration distance. Only
    ``ReuseDependency`` uses a storage version variable. Pipe issue windows are
    disjoint both inside an II and across its wrap-around boundary. Manual stage
    constraints never silently fall back. An ``unknown`` solver status is
    reported distinctly from proven infeasibility.
    """
    tasks, dependencies = tuple(tasks), tuple(dependencies)
    storage, capacities = dict(storage or {}), dict(capacities or {})
    n = len(tasks)
    if not n:
        raise ValueError("at least one task is required")
    if any(t.latency < 0 or t.ii < 1 for t in tasks):
        raise ValueError("task latency must be nonnegative and task ii positive")
    if timeout_ms < 0 or (max_ii is not None and max_ii < 1):
        raise ValueError("timeout_ms must be nonnegative and max_ii positive")
    if manual_stages is not None:
        manual_stages = tuple(manual_stages)
        if len(manual_stages) != n or any(s < 0 for s in manual_stages):
            raise ValueError("manual stages must supply one nonnegative stage per task")
    for dep in (*dependencies, *reuse_dependencies):
        if not 0 <= dep.src < n or not 0 <= dep.dst < n:
            raise ValueError("dependency task index out of range")
        if dep.latency is not None and dep.latency < 0:
            raise ValueError("dependency latency must be nonnegative")
    if any(d.distance < 0 for d in dependencies):
        raise ValueError("data dependency distance must be nonnegative")
    if any(d.storage not in storage for d in reuse_dependencies):
        raise ValueError("reuse dependency references unknown storage")
    versions = {}
    for name, cost in storage.items():
        if cost.size_bytes < 0 or cost.max_versions < 1:
            raise ValueError("storage size must be nonnegative and max_versions positive")
        if cost.fixed_versions is not None and not 1 <= cost.fixed_versions <= cost.max_versions:
            raise ValueError("fixed_versions must be within [1, max_versions]")
        if cost.scope not in capacities:
            raise ValueError(f"explicit capacity required for storage scope {cost.scope}")
        versions[name] = cost.fixed_versions or 1
    for scope, cap in capacities.items():
        if cap < 0:
            raise ValueError("capacity must be nonnegative")
        minimum = sum(c.size_bytes * versions[k] for k, c in storage.items() if c.scope == scope)
        if minimum > cap:
            raise ScheduleInfeasible(f"minimum storage exceeds {scope} capacity: {minimum} > {cap}")

    def latency(dep):
        return tasks[dep.src].latency if dep.latency is None else dep.latency

    # The zero-distance graph also supplies a safe sequential fallback order.
    pending = [set() for _ in tasks]
    for dep in dependencies:
        if dep.distance == 0:
            pending[dep.dst].add(dep.src)
    order = []
    while len(order) < n:
        ready = next((i for i in range(n) if i not in order and pending[i] <= set(order)), None)
        if ready is None:
            raise ScheduleInfeasible("zero-distance dependency cycle has no sequential schedule")
        order.append(ready)
    sequential = [0] * n
    cursor = 0
    for i in order:
        sequential[i] = max([cursor] + [
            sequential[d.src] + latency(d) for d in dependencies if d.dst == i and d.distance == 0
        ])
        cursor = sequential[i] + max(tasks[i].latency, tasks[i].ii)
    sequential_ii = max(cursor, 1)
    for dep in dependencies:
        if dep.distance:
            required = latency(dep) + sequential[dep.src] - sequential[dep.dst]
            sequential_ii = max(sequential_ii, (required + dep.distance - 1) // dep.distance)
    for dep in reuse_dependencies:
        count = versions[dep.storage]
        required = latency(dep) + sequential[dep.src] - sequential[dep.dst]
        sequential_ii = max(sequential_ii, (required + count - 1) // count)

    def fallback(reason):
        if manual_stages is not None:
            raise ScheduleInfeasible(f"manual schedule unavailable: {reason}")
        return ScheduleResult(
            tuple(sequential), (0,) * n, tuple(order), versions, sequential_ii, True,
            (reason, "using sequential schedule"))

    z3 = _load_z3()
    if z3 is None:
        return fallback("optional z3-solver package missing; install with pip install z3-solver")
    if timeout_ms == 0:
        return fallback("solver unknown: timeout budget exhausted")
    deadline = monotonic() + timeout_ms / 1000
    upper = max_ii or sequential_ii
    lower = max(t.ii for t in tasks)
    ctx = z3.Context()

    def attempt(ii):
        remaining = int((deadline - monotonic()) * 1000)
        if remaining <= 0:
            return None, "solver unknown: timeout budget exhausted"
        opt = z3.Optimize(ctx=ctx)
        opt.set(timeout=max(remaining, 1))
        starts = [z3.Int(f"start_{i}", ctx) for i in range(n)]
        phase = [s % ii for s in starts]
        buf = {name: z3.Int(f"buffer_{i}", ctx) for i, name in enumerate(storage)}
        for i, start in enumerate(starts):
            opt.add(start >= 0)
            if manual_stages is not None:
                opt.add(start / ii == manual_stages[i])
        for name, cost in storage.items():
            opt.add(buf[name] >= 1, buf[name] <= cost.max_versions)
            if cost.fixed_versions is not None:
                opt.add(buf[name] == cost.fixed_versions)
        for scope, cap in capacities.items():
            terms = [buf[k] * c.size_bytes for k, c in storage.items() if c.scope == scope]
            if terms:
                opt.add(z3.Sum(terms) <= cap)
        for d in dependencies:
            opt.add(starts[d.dst] - starts[d.src] >= latency(d) - d.distance * ii)
        for d in reuse_dependencies:
            opt.add(starts[d.dst] - starts[d.src] >= latency(d) - buf[d.storage] * ii)
        for i in range(n):
            for j in range(i + 1, n):
                if _shares_pipe(tasks[i], tasks[j]):
                    opt.add(
                        z3.Or(
                            z3.And(phase[j] - phase[i] >= tasks[i].ii, phase[i] - phase[j] + ii
                                   >= tasks[j].ii),
                            z3.And(phase[i] - phase[j] >= tasks[j].ii, phase[j] - phase[i] + ii
                                   >= tasks[i].ii)))
        if buf:
            opt.minimize(z3.Sum(list(buf.values())))
        opt.minimize(z3.Sum(starts))
        for start in starts:
            opt.minimize(start)
        status = opt.check()
        if status == z3.unknown:
            return None, f"solver unknown: {opt.reason_unknown()}"
        if status == z3.unsat:
            return None, "infeasible"
        model = opt.model()
        times = tuple(model.eval(s).as_long() for s in starts)
        counts = {name: model.eval(var).as_long() for name, var in buf.items()}
        return ScheduleResult(times, tuple(t // ii for t in times),
                              tuple(sorted(range(n), key=lambda i: (times[i], i))), counts,
                              ii), None

    # Fixed stages can make feasibility non-monotone in II, so examine each II.
    if manual_stages is not None:
        for ii in range(lower, upper + 1):
            result, reason = attempt(ii)
            if result is not None:
                return result
            if reason != "infeasible":
                return fallback(reason)
        return fallback(f"constraints infeasible for searched II range [{lower}, {upper}]")
    best = None
    while lower <= upper:
        ii = (lower + upper) // 2
        result, reason = attempt(ii)
        if result is not None:
            best, upper = result, ii - 1
        elif reason == "infeasible":
            lower = ii + 1
        else:
            return fallback(reason)
    if best is None:
        return fallback("constraints infeasible within requested II search bound")
    return best
