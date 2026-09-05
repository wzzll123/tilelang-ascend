# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Deterministic discrete-event scheduler for A2/A3 simulator tasks."""

import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Set, Tuple

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


ResourceKey = Tuple[int, str, str]


@dataclass(frozen=True)
class ScheduleResult:
    """Immutable records and derived statistics from one scheduling run."""

    records: Tuple[ExecutionRecord, ...]
    stats: SimulationStats


class DiscreteEventScheduler:
    """Schedule tasks using dependencies, pipe FIFO, and synchronization constraints.

    FIFO order is scoped to ``(core_id, lane, pipe)``.  Consequently, independent tasks on
    distinct pipes start concurrently, while tasks sharing a physical simulator resource retain
    their stable ``KernelProgram`` order.
    """

    def __init__(
        self,
        config: Optional[SimulatorConfig] = None,
        synchronization: Optional[SynchronizationModel] = None,
    ) -> None:
        self.config = config
        self.synchronization = synchronization or NoOpSynchronizationModel()

    def run(self, program: KernelProgram) -> ScheduleResult:
        """Schedule ``program`` and return trace-ready records plus summary statistics."""
        config = self.config or SimulatorConfig(platform=program.platform)
        if config.platform != program.platform:
            raise SimulatorConfigError(
                "scheduler config platform does not match program platform: "
                f"{config.platform} != {program.platform}"
            )

        tasks = program.tasks
        task_by_id = {task.task_id: task for task in tasks}
        self._validate_dependencies(tasks, task_by_id)
        fifo_predecessor = self._fifo_predecessors(tasks)
        self.synchronization.reset(program)

        pending: Set[str] = set(task_by_id)
        completed: Dict[str, ExecutionRecord] = {}
        records: List[ExecutionRecord] = []
        started_at = time.monotonic()

        while pending:
            self._check_wall_timeout(started_at, config.execution_timeout_s, pending)
            made_progress = False
            blocked_details: Dict[str, str] = {}

            # Stable source order makes equal-cycle schedules reproducible.
            for task in tasks:
                if task.task_id not in pending:
                    continue
                required = set(task.dependencies)
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
                    (completed[task_id].end_cycle for task_id in required), default=0
                )
                start_cycle = max(dependency_cycle, decision.ready_cycle or 0)
                if start_cycle > dependency_cycle:
                    records.append(ExecutionRecord(
                        task_id=f"{task.task_id}#wait",
                        operation="wait",
                        core_id=task.core_id,
                        lane=task.lane,
                        pipe=task.pipe,
                        start_cycle=dependency_cycle,
                        end_cycle=start_cycle,
                        category="wait",
                        stall_reason=decision.reason or "synchronization",
                        metadata={
                            "blocked_task": task.task_id,
                            "detail": decision.detail,
                        },
                    ))
                end_cycle = start_cycle + task.duration_cycles
                if config.max_cycles is not None and end_cycle > config.max_cycles:
                    raise SimulationLimitError(
                        f"task {task.task_id!r} would finish at cycle {end_cycle}, "
                        f"exceeding max_cycles={config.max_cycles}"
                    )

                record = ExecutionRecord.from_task(task, start_cycle, end_cycle)
                completed[task.task_id] = record
                records.append(record)
                pending.remove(task.task_id)
                self.synchronization.on_scheduled(task, record)
                made_progress = True

            if not made_progress:
                details = "; ".join(
                    f"{task_id}: {blocked_details.get(task_id, 'blocked')}"
                    for task_id in sorted(pending)
                )
                raise SimulationDeadlockError(
                    f"simulation deadlock with {len(pending)} blocked task(s): {details}"
                )

        ordered_records = tuple(
            sorted(
                records,
                key=lambda record: (record.start_cycle, record.end_cycle, record.task_id),
            )
        )
        return ScheduleResult(
            records=ordered_records,
            stats=SimulationStats.from_records(ordered_records),
        )

    @staticmethod
    def _validate_dependencies(
        tasks: Tuple[Task, ...], task_by_id: Mapping[str, Task]
    ) -> None:
        known = set(task_by_id)
        for task in tasks:
            missing = sorted(set(task.dependencies) - known)
            if missing:
                raise UnknownDependencyError(
                    f"task {task.task_id!r} has unknown dependencies: {', '.join(missing)}"
                )

    @staticmethod
    def _fifo_predecessors(tasks: Tuple[Task, ...]) -> Mapping[str, str]:
        previous_by_resource: Dict[ResourceKey, str] = {}
        predecessors: Dict[str, str] = {}
        for task in tasks:
            resource = (task.core_id, task.lane.value, task.pipe.value)
            previous = previous_by_resource.get(resource)
            if previous is not None:
                predecessors[task.task_id] = previous
            previous_by_resource[resource] = task.task_id
        return MappingProxyType(predecessors)

    @staticmethod
    def _check_wall_timeout(
        started_at: float, timeout_s: float, pending: Set[str]
    ) -> None:
        if time.monotonic() - started_at > timeout_s:
            names = ", ".join(sorted(pending))
            raise SimulationLimitError(
                f"simulation exceeded execution_timeout_s={timeout_s}; pending tasks: {names}"
            )
