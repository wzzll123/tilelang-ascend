# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""JSON-safe scheduling reports shared by compiler and analysis tools.

All supplied RAW/WAR/WAW edges survive index conversion. This module does not
discover or insert memory order: callers must provide the compiler's analyzed
dependencies. Opaque symbolic objects are identified locally by identity, never
serialized with ``repr`` or treated as known predicates/byte footprints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .solver import Dependency, ScheduleResult, TaskCost, solve_schedule


class ReportValidationError(ValueError):
    """A selected schedule violates the supplied constraint model."""


def _metadata(value):
    if value is None or type(value) in (str, int, bool):
        return value
    return {"status": "unknown_symbolic", "type": type(value).__name__}


def validate_schedule_result(
        tasks: Sequence[TaskCost],
        dependencies: Sequence[Dependency],
        result: ScheduleResult,
        *,
        storage=None,
        capacities=None,
        host_workspace_budget_bytes=None,
        reuse_dependencies=(),
        manual_stages=None,
        dependency_kinds=None,
) -> None:
    """Check actual dependency, cyclic resource, stage, and memory inequalities."""
    storage, capacities = dict(storage or {}), dict(capacities or {})
    if "GM" in capacities:
        raise ReportValidationError("GM workspace budget is not a local hardware capacity")
    if host_workspace_budget_bytes is not None and (type(host_workspace_budget_bytes) is not int or
                                                    host_workspace_budget_bytes < 0):
        raise ReportValidationError(
            "host workspace budget must be a nonnegative integer byte count")
    n, ii = len(tasks), result.ii
    if type(ii) is not int or ii < 1:
        raise ReportValidationError("selected II must be a positive integer")
    if len(result.start_times) != n or len(result.stages) != n:
        raise ReportValidationError("schedule must contain every task")
    if any(type(t) is not int or t < 0 for t in result.start_times):
        raise ReportValidationError("start times must be nonnegative integers")
    if tuple(result.stages) != tuple(t // ii for t in result.start_times):
        raise ReportValidationError("stage does not match start time / II")
    expected_order = tuple(sorted(range(n), key=lambda i: (result.start_times[i], i)))
    if tuple(result.order) != expected_order:
        raise ReportValidationError("order must include every task sorted by start time")
    if manual_stages is not None and tuple(result.stages) != tuple(manual_stages):
        raise ReportValidationError("selected stages violate manual stages")
    for i, task in enumerate(tasks):
        if task.latency < 0 or task.ii < 1 or ii < task.ii:
            raise ReportValidationError(f"invalid resource initiation interval for task {i}")
    for index, dep in enumerate(dependencies):
        if not 0 <= dep.src < n or not 0 <= dep.dst < n or dep.distance < 0:
            raise ReportValidationError("invalid dependency endpoint or distance")
        latency = tasks[dep.src].latency if dep.latency is None else dep.latency
        if result.start_times[dep.dst] - result.start_times[dep.src] < latency - dep.distance * ii:
            kind = dependency_kinds[index] if dependency_kinds is not None else "data"
            raise ReportValidationError(f"{kind} dependency {dep.src}->{dep.dst} violated")
    if set(result.buffer_versions) != set(storage):
        raise ReportValidationError("selected storage versions must cover every allocation")
    for name, cost in storage.items():
        count = result.buffer_versions[name]
        if type(count) is not int or not 1 <= count <= cost.max_versions:
            raise ReportValidationError(f"storage version bound violated for {name}")
        if cost.fixed_versions is not None and count != cost.fixed_versions:
            raise ReportValidationError(f"fixed storage versions violated for {name}")
        if cost.scope != "GM" and cost.scope not in capacities:
            raise ReportValidationError(f"explicit capacity missing for {cost.scope}")
        if type(cost.size_bytes) is not int or cost.size_bytes < 0:
            raise ReportValidationError("storage byte size must be a nonnegative integer")
    workspace_bytes = sum(cost.size_bytes * result.buffer_versions[name]
                          for name, cost in storage.items()
                          if cost.scope == "GM")
    if host_workspace_budget_bytes is not None and workspace_bytes > host_workspace_budget_bytes:
        raise ReportValidationError("selected host workspace exceeds its allocation budget")
    for scope, cap in capacities.items():
        footprint = sum(cost.size_bytes * result.buffer_versions[name]
                        for name, cost in storage.items()
                        if cost.scope == scope)
        if cap < 0 or footprint > cap:
            raise ReportValidationError(f"storage exceeds {scope} capacity")
    for dep in reuse_dependencies:
        if dep.storage not in storage or not 0 <= dep.src < n or not 0 <= dep.dst < n:
            raise ReportValidationError("invalid reuse dependency")
        latency = tasks[dep.src].latency if dep.latency is None else dep.latency
        distance = result.buffer_versions[dep.storage]
        if result.start_times[dep.dst] - result.start_times[dep.src] < latency - distance * ii:
            raise ReportValidationError(f"storage reuse dependency violated for {dep.storage}")
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tasks[i], tasks[j]
            shared = bool(a.pipe & b.pipe) if isinstance(a.pipe, int) and isinstance(b.pipe, int) \
                else bool(a.pipe) and a.pipe == b.pipe
            if shared:
                forward = (result.start_times[j] - result.start_times[i]) % ii
                reverse = (result.start_times[i] - result.start_times[j]) % ii
                if forward < a.ii or reverse < b.ii:
                    raise ReportValidationError(
                        f"cyclic resource conflict between tasks {i} and {j}")


def build_schedule_report(
        tasks_or_extraction,
        dependencies,
        costs: Mapping[object, TaskCost],
        *,
        stages=None,
        storage=None,
        capacities=None,
        host_workspace_budget_bytes=None,
        resource_model="A2",
        reuse_dependencies=(),
        result=None,
        **solver_options,
):
    """Solve and report an extraction or task sequence with explicit costs.

    ``stages`` and ``costs`` are keyed by original task ids. Allocation costs and
    capacities retain the solver's explicit scope/name mapping. Physical access
    labels are a separate local identity table, since logical allocation names
    cannot prove physical aliasing. An injected result undergoes the same checks
    as a solved result and is useful for compiler/simulator agreement tests.
    """
    if resource_model not in ("A2", "A3"):
        raise ValueError("resource_model must be A2 or A3 with explicit capacities")
    tasks = tuple(getattr(tasks_or_extraction, "tasks", tasks_or_extraction))
    dependencies = tuple(dependencies)
    indexes = {task.id: index for index, task in enumerate(tasks)}
    if len(indexes) != len(tasks):
        raise ValueError("task ids must be unique")
    if set(costs) != set(indexes):
        raise ValueError("cost model must supply exactly every task id")
    task_costs = tuple(costs[task.id] for task in tasks)
    if any(type(c.latency) is not int or type(c.ii) is not int for c in task_costs):
        raise ValueError("cost latency and initiation interval must be explicit integers")
    mapped = []
    for dep in dependencies:
        if dep.source not in indexes or dep.target not in indexes:
            raise ValueError("dependency references unknown task id")
        if dep.kind not in ("RAW", "WAR", "WAW"):
            raise ValueError("dependency kind must be RAW, WAR, or WAW")
        mapped.append(Dependency(indexes[dep.source], indexes[dep.target], dep.distance))
    manual_stages = None
    if stages is not None:
        if set(stages) != set(indexes):
            raise ValueError("manual stage map must cover exactly every task id")
        manual_stages = tuple(stages[task.id] for task in tasks)
    storage, capacities = dict(storage or {}), dict(capacities or {})
    if (any(type(name) is not str for name in storage) or
            any(type(s) is not str for s in capacities)):
        raise ValueError("allocation names and capacity scopes must be strings")
    if any(type(cap) is not int for cap in capacities.values()):
        raise ValueError("capacities must be explicit integer byte counts")
    for cost in storage.values():
        if (type(cost.scope) is not str or type(cost.size_bytes) is not int or
                type(cost.max_versions) is not int or
            (cost.fixed_versions is not None and type(cost.fixed_versions) is not int)):
            raise ValueError("storage costs must contain explicit integer byte/version counts")
    if result is None:
        result = solve_schedule(
            task_costs,
            mapped,
            storage=storage,
            capacities=capacities,
            host_workspace_budget_bytes=host_workspace_budget_bytes,
            reuse_dependencies=reuse_dependencies,
            manual_stages=manual_stages,
            **solver_options)
    validate_schedule_result(
        task_costs,
        mapped,
        result,
        storage=storage,
        capacities=capacities,
        host_workspace_budget_bytes=host_workspace_budget_bytes,
        reuse_dependencies=reuse_dependencies,
        manual_stages=manual_stages,
        dependency_kinds=[d.kind for d in dependencies])

    identities = []

    def storage_label(value):
        if value is None:
            return None
        for index, existing in enumerate(identities):
            if existing is value:
                return f"storage_{index}"
        identities.append(value)
        return f"storage_{len(identities) - 1}"

    task_records = []
    for i, task in enumerate(tasks):
        accesses = []
        for access in task.accesses:
            intervals = access.intervals
            if intervals is None and type(access.offset) is int and type(access.extent) is int:
                intervals = ((access.offset, access.offset + access.extent),)
            known = intervals is not None and all(
                type(a) is int and type(b) is int for a, b in intervals)
            footprint = {"status": "known", "intervals": [list(p) for p in intervals]} \
                if known else {"status": "unknown"}
            accesses.append({
                "storage": storage_label(access.storage),
                "mode": access.mode,
                "footprint": footprint,
                "iteration_stride": _metadata(access.iteration_stride),
                "ring_size": _metadata(access.ring_size)
            })
        guard = {"status": "unconditional"} if task.guard is None else \
            {"status": "constant", "value": task.guard} if type(task.guard) is bool else \
            {"status": "unknown_symbolic", "type": type(task.guard).__name__}
        task_records.append({
            "index": i,
            "id": _metadata(task.id),
            "core": _metadata(task.core),
            "pipe": _metadata(task.pipe),
            "guard": guard,
            "accesses": accesses,
            "cost": {
                "latency": task_costs[i].latency,
                "ii": task_costs[i].ii,
                "pipe": _metadata(task_costs[i].pipe)
            },
            "start_time": result.start_times[i],
            "stage": result.stages[i]
        })
    dependency_records = [{
        "src": edge.src,
        "dst": edge.dst,
        "source": _metadata(dep.source),
        "target": _metadata(dep.target),
        "kind": dep.kind,
        "distance": dep.distance,
        "latency": task_costs[edge.src].latency,
        "storage": storage_label(dep.storage)
    } for dep, edge in zip(dependencies, mapped)]
    source_diagnostics = getattr(tasks_or_extraction, "diagnostics", ())
    return {
        "schema_version": 1,
        "resource_model": resource_model,
        "capacities": capacities,
        "host_workspace": {
            "selected_bytes":
                sum(cost.size_bytes * result.buffer_versions[name]
                    for name, cost in storage.items()
                    if cost.scope == "GM"),
            "budget_bytes":
                host_workspace_budget_bytes,
            "device_available_memory_verified":
                False,
        },
        "tasks": task_records,
        "dependencies": dependency_records,
        "reuse_dependencies": [{
            "src": dep.src,
            "dst": dep.dst,
            "storage": dep.storage,
            "distance": result.buffer_versions[dep.storage],
            "latency": task_costs[dep.src].latency if dep.latency is None else dep.latency
        } for dep in reuse_dependencies],
        "storage": [{
            "name": name,
            "scope": cost.scope,
            "size_bytes": cost.size_bytes,
            "max_versions": cost.max_versions,
            "fixed_versions": cost.fixed_versions,
            "selected_versions": result.buffer_versions[name],
            "footprint_bytes": cost.size_bytes * result.buffer_versions[name]
        } for name, cost in storage.items()],
        "schedule": {
            "start_times": list(result.start_times),
            "stages": list(result.stages),
            "order": list(result.order),
            "selected_ii": result.ii,
            "buffer_versions": dict(result.buffer_versions),
            "fallback": result.fallback
        },
        "diagnostics": [_metadata(d) for d in (*source_diagnostics, *result.diagnostics)],
    }
