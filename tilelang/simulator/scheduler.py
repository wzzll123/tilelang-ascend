# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Deterministic discrete-event scheduler for A2/A3 simulator tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping

from .config import SimulatorConfig
from .errors import (
    SimulationDeadlockError,
    SimulationLimitError,
    SimulatorConfigError,
    UnknownDependencyError,
)
from .program import KernelProgram, Task
from .stats import SimulationStats
from .sync import NoOpSynchronizationModel, SynchronizationModel, readonly_records
from .trace import ExecutionRecord


ResourceKey = tuple[int, str, str]


@dataclass(frozen=True)
class ScheduleResult:
    """Immutable records and derived statistics from one scheduling run."""

    records: tuple[ExecutionRecord, ...]
    stats: SimulationStats


class DiscreteEventScheduler:
    """Schedule tasks using dependencies, pipe FIFO, and synchronization constraints.

    FIFO order is scoped to ``(core_id, lane, pipe)``.  Consequently, independent tasks on
    distinct pipes start concurrently, while tasks sharing a physical simulator resource retain
    their stable ``KernelProgram`` order.
    """

    def __init__(
        self,
        config: SimulatorConfig | None = None,
        synchronization: SynchronizationModel | None = None,
    ) -> None:
        self.config = config
        self.synchronization = synchronization or NoOpSynchronizationModel()

    def run(
        self,
        program: KernelProgram,
        *,
        bindings: Mapping[str, int | float] | None = None,
    ) -> ScheduleResult:
        """Schedule ``program`` and return trace-ready records plus summary statistics."""
        config = self.config or SimulatorConfig(platform=program.platform)
        if config.platform != program.platform:
            raise SimulatorConfigError(
                f"scheduler config platform does not match program platform: {config.platform} != {program.platform}"
            )

        tasks = program.tasks
        task_by_id = {task.task_id: task for task in tasks}
        self._validate_dependencies(tasks, task_by_id)
        execution_dependencies = {
            task.task_id: self._execution_dependencies(task) for task in tasks
        }
        fifo_predecessor = self._fifo_predecessors(tasks)
        self.synchronization.reset(program)

        pending: set[str] = set(task_by_id)
        completed: dict[str, ExecutionRecord] = {}
        records: list[ExecutionRecord] = []
        started_at = time.monotonic()

        while pending:
            self._check_wall_timeout(started_at, config.execution_timeout_s, pending)
            made_progress = False
            blocked_details: dict[str, str] = {}

            # Stable source order makes equal-cycle schedules reproducible.
            for task in tasks:
                if task.task_id not in pending:
                    continue
                dependency_required = set(execution_dependencies[task.task_id])
                required = set(dependency_required)
                predecessor = fifo_predecessor.get(task.task_id)
                if predecessor is not None:
                    required.add(predecessor)
                missing = sorted(required - completed.keys())
                if missing:
                    blocked_details[task.task_id] = "waiting for " + ", ".join(missing)
                    continue

                decision = self.synchronization.evaluate(task, readonly_records(completed))
                if decision.blocked:
                    reason = decision.reason or "synchronization"
                    suffix = f": {decision.detail}" if decision.detail else ""
                    blocked_details[task.task_id] = f"blocked by {reason}{suffix}"
                    continue

                dependency_cycle = max(
                    (completed[task_id].end_cycle for task_id in dependency_required),
                    default=0,
                )
                resource_cycle = completed[predecessor].end_cycle if predecessor is not None else 0
                synchronization_cycle = decision.ready_cycle or 0
                ready_cycle = max(dependency_cycle, synchronization_cycle)
                start_cycle = max(ready_cycle, resource_cycle)
                if synchronization_cycle > max(dependency_cycle, resource_cycle):
                    records.append(
                        ExecutionRecord(
                            task_id=f"{task.task_id}#wait",
                            operation="wait",
                            core_id=task.core_id,
                            lane=task.lane,
                            pipe=task.pipe,
                            start_cycle=max(dependency_cycle, resource_cycle),
                            end_cycle=synchronization_cycle,
                            category="wait",
                            stall_reason=decision.reason or "synchronization",
                            metadata={
                                "blocked_task": task.task_id,
                                "detail": decision.detail,
                            },
                        )
                    )
                end_cycle = start_cycle + task.duration_cycles
                if config.max_cycles is not None and end_cycle > config.max_cycles:
                    raise SimulationLimitError(
                        f"task {task.task_id!r} would finish at cycle {end_cycle}, exceeding max_cycles={config.max_cycles}"
                    )

                trace_metadata = {}
                if decision.producer_task_ids:
                    trace_metadata["sync_producers"] = decision.producer_task_ids
                if start_cycle > ready_cycle:
                    trace_metadata["queue_enter_cycle"] = ready_cycle
                record = ExecutionRecord.from_task(
                    task,
                    start_cycle,
                    end_cycle,
                    metadata=trace_metadata or None,
                )
                completed[task.task_id] = record
                records.append(record)
                pending.remove(task.task_id)
                self.synchronization.on_scheduled(task, record)
                made_progress = True

            if not made_progress:
                if not config.deadlock_detect:
                    continue
                details = "; ".join(f"{task_id}: {blocked_details.get(task_id, 'blocked')}" for task_id in sorted(pending))
                cycle = self._wait_cycle(
                    pending,
                    tasks,
                    fifo_predecessor,
                    execution_dependencies,
                    completed,
                )
                cycle_detail = (
                    "; wait-for cycle: " + " -> ".join(cycle)
                    if cycle
                    else "; wait-for graph has no cycle; all remaining tasks are blocked and no future event can fire"
                )
                history = self._recent_history(records, config.deadlock_history_limit)
                raise SimulationDeadlockError(
                    f"DEADLOCK (global no-progress): {len(pending)} blocked task(s): {details}{cycle_detail}; recent events: {history}"
                )

        if config.flag_balance_check != "off":
            audit = getattr(self.synchronization, "audit_flag_balance", None)
            if callable(audit):
                audit(config.flag_balance_check)

        ordered_records = tuple(
            sorted(
                records,
                key=lambda record: (record.start_cycle, record.end_cycle, record.task_id),
            )
        )
        return ScheduleResult(
            records=ordered_records,
            stats=SimulationStats.from_records(ordered_records, bindings=bindings),
        )

    @staticmethod
    def _wait_cycle(
        pending: set[str],
        tasks: tuple[Task, ...],
        fifo_predecessor: Mapping[str, str],
        execution_dependencies: Mapping[str, tuple[str, ...]],
        completed: Mapping[str, ExecutionRecord],
    ) -> tuple[str, ...]:
        edges: dict[str, tuple[str, ...]] = {}
        for task in tasks:
            if task.task_id not in pending:
                continue
            required = list(execution_dependencies[task.task_id])
            predecessor = fifo_predecessor.get(task.task_id)
            if predecessor is not None:
                required.append(predecessor)
            edges[task.task_id] = tuple(dependency for dependency in required if dependency not in completed)
        visiting: list[str] = []
        visited: set[str] = set()

        def visit(task_id: str) -> tuple[str, ...]:
            if task_id in visiting:
                start = visiting.index(task_id)
                return tuple(visiting[start:] + [task_id])
            if task_id in visited:
                return ()
            visiting.append(task_id)
            for dependency in edges.get(task_id, ()):
                cycle = visit(dependency)
                if cycle:
                    return cycle
            visiting.pop()
            visited.add(task_id)
            return ()

        for task_id in sorted(pending):
            cycle = visit(task_id)
            if cycle:
                return cycle
        return ()

    @staticmethod
    def _execution_dependencies(task: Task) -> tuple[str, ...]:
        """Return actual execution edges, excluding inferred memory hazards.

        ``memory_dependencies`` are retained in metadata for static hazard
        validation and diagnostics.  They are not hardware synchronization:
        A2/A3 orders distinct pipes only through their FIFO queues and explicit
        flag/barrier instructions.  PTO perf-sim follows the same rule and
        intentionally has no structural auto-dependency edges.
        """
        memory_dependencies = task.metadata.get("memory_dependencies", ())
        diagnostic_edges = (
            {str(dependency) for dependency in memory_dependencies}
            if isinstance(memory_dependencies, (tuple, list))
            else set()
        )
        hardware_dependencies = task.metadata.get("hardware_dependencies", ())
        hardware_edges = (
            {str(dependency) for dependency in hardware_dependencies}
            if isinstance(hardware_dependencies, (tuple, list))
            else set()
        )
        return tuple(
            dependency
            for dependency in task.dependencies
            if dependency not in diagnostic_edges or dependency in hardware_edges
        )

    @staticmethod
    def _recent_history(records: list[ExecutionRecord], limit: int) -> str:
        recent = records[-limit:]
        if not recent:
            return "none"
        return ", ".join(f"{record.task_id}@{record.end_cycle}" for record in recent)

    @staticmethod
    def _validate_dependencies(tasks: tuple[Task, ...], task_by_id: Mapping[str, Task]) -> None:
        known = set(task_by_id)
        for task in tasks:
            missing = sorted(set(task.dependencies) - known)
            if missing:
                raise UnknownDependencyError(f"task {task.task_id!r} has unknown dependencies: {', '.join(missing)}")

    @staticmethod
    def _fifo_predecessors(tasks: tuple[Task, ...]) -> Mapping[str, str]:
        previous_by_resource: dict[ResourceKey, str] = {}
        predecessors: dict[str, str] = {}
        for task in tasks:
            resource = (task.core_id, task.lane.value, task.pipe.value)
            previous = previous_by_resource.get(resource)
            if previous is not None:
                predecessors[task.task_id] = previous
            previous_by_resource[resource] = task.task_id
        return MappingProxyType(predecessors)

    @staticmethod
    def _check_wall_timeout(started_at: float, timeout_s: float, pending: set[str]) -> None:
        if time.monotonic() - started_at > timeout_s:
            names = ", ".join(sorted(pending))
            raise SimulationLimitError(f"simulation exceeded execution_timeout_s={timeout_s}; pending tasks: {names}")
