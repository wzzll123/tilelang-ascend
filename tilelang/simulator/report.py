# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Compact, machine-readable performance summaries for simulator schedules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .config import SimulatorConfig
from .stats import SimulationStats
from .trace import ExecutionRecord

if TYPE_CHECKING:
    from .scheduler import ScheduleResult


REPORT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class PerformanceReport:
    """A compact schedule summary intended for people and automated consumers.

    The report deliberately summarizes an already-produced schedule.  It does
    not claim measured hardware timing when the selected profile is an
    analytical PTO fallback. Its critical chain and resource pressure fields
    are explicitly schedule-derived.
    """

    platform: str
    calibration: str
    timing_estimator: str
    sync_only: bool
    trace_path: str | None
    stats: SimulationStats
    critical_path: Mapping[str, Any]
    schedule_records: tuple[ExecutionRecord, ...] = ()

    @classmethod
    def from_stats(
        cls,
        stats: SimulationStats,
        config: SimulatorConfig,
        *,
        trace_path: Path | None = None,
    ) -> "PerformanceReport":
        return cls(
            platform=config.platform,
            calibration=config.timing_profile.calibration,
            timing_estimator=config.timing_profile.estimator,
            sync_only=config.sync_only,
            trace_path=None if trace_path is None else str(trace_path),
            stats=stats,
            critical_path={
                "status": "unavailable",
                "reason": "a schedule result is required for critical-chain extraction",
            },
        )

    @classmethod
    def from_schedule(
        cls,
        result: "ScheduleResult",
        config: SimulatorConfig,
        *,
        trace_path: Path | None = None,
    ) -> "PerformanceReport":
        """Build a report with a critical chain from actual scheduler edges.

        This is deliberately a simulator-schedule chain, not a claim about a
        hardware microarchitectural critical path.  It contains only explicit
        execution dependencies, same-resource FIFO order, and matched flag or
        barrier producers recorded by the synchronization model.
        """
        return cls(
            platform=config.platform,
            calibration=config.timing_profile.calibration,
            timing_estimator=config.timing_profile.estimator,
            sync_only=config.sync_only,
            trace_path=None if trace_path is None else str(trace_path),
            stats=result.stats,
            critical_path=cls._critical_chain(result.records),
            schedule_records=result.records,
        )

    @staticmethod
    def _critical_chain(records: tuple[ExecutionRecord, ...]) -> dict[str, Any]:
        operations = {
            record.task_id: record
            for record in records
            if record.category == "operation"
        }
        if not operations:
            return {
                "status": "unavailable",
                "reason": "schedule contains no operation records",
            }

        current = max(
            operations.values(),
            key=lambda record: (record.end_cycle, record.start_cycle, record.task_id),
        )
        reverse_chain: list[tuple[ExecutionRecord, str | None]] = [(current, None)]
        visited = {current.task_id}
        while True:
            raw_predecessors = current.metadata.get("critical_predecessors", {})
            if not isinstance(raw_predecessors, Mapping):
                break
            candidates: list[tuple[int, str, str, ExecutionRecord]] = []
            for kind, task_ids in raw_predecessors.items():
                if not isinstance(task_ids, (tuple, list)):
                    continue
                for task_id in task_ids:
                    predecessor = operations.get(str(task_id))
                    if (
                        predecessor is not None
                        and predecessor.task_id not in visited
                        and predecessor.end_cycle <= current.start_cycle
                    ):
                        candidates.append(
                            (predecessor.end_cycle, str(kind), predecessor.task_id, predecessor)
                        )
            if not candidates:
                break
            _, kind, _, predecessor = max(candidates)
            reverse_chain.append((predecessor, kind))
            visited.add(predecessor.task_id)
            current = predecessor

        chain = list(reversed(reverse_chain))
        steps = []
        for index, (record, relationship_to_successor) in enumerate(chain):
            steps.append(
                {
                    "task_id": record.task_id,
                    "operation": record.operation,
                    "resource": f"core-{record.core_id}/{record.resource}",
                    "start_cycle": record.start_cycle,
                    "end_cycle": record.end_cycle,
                    "duration_cycles": record.duration_cycles,
                    "to_successor": relationship_to_successor if index + 1 < len(chain) else None,
                }
            )
        first = chain[0][0]
        terminal = chain[-1][0]
        return {
            "status": "schedule-derived",
            "scope": (
                "explicit execution dependencies, same-resource FIFO order, and "
                "matched synchronization producers only; not a hardware timing proof"
            ),
            "terminal_task_id": terminal.task_id,
            "span_cycles": terminal.end_cycle - first.start_cycle,
            "operation_cycles": sum(record.duration_cycles for record, _ in chain),
            "steps": steps,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-ready report without trace-sized payloads."""
        utilization = self.stats.utilization_by_resource
        busiest = sorted(
            utilization,
            key=lambda resource: (-utilization[resource], resource),
        )[:5]
        waits = self.stats.wait_cycles_by_reason
        major_stalls = sorted(waits, key=lambda reason: (-waits[reason], reason))[:5]
        measured = self.calibration.startswith("measured-")
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "platform": self.platform,
            "simulation_mode": "sync_only" if self.sync_only else "functional",
            "timing": {
                "calibration": self.calibration,
                "estimator": self.timing_estimator,
                "measured": measured,
                "interpretation": (
                    "measured hardware timing"
                    if measured
                    else "analytical schedule estimate; not measured hardware latency"
                ),
            },
            "schedule": self.stats.to_dict(),
            "top_resources": [
                {
                    "resource": resource,
                    "utilization": utilization[resource],
                    "busy_cycles": self.stats.busy_cycles_by_resource[resource],
                }
                for resource in busiest
            ],
            "major_stalls": [
                {"reason": reason, "cycles": waits[reason]}
                for reason in major_stalls
            ],
            "critical_path": dict(self.critical_path),
            "top_inactive_intervals": self._top_inactive_intervals(),
            "copy_compute_overlap": self._copy_compute_overlap(),
            "resource_pressure": self._resource_pressure(),
            "trace_path": self.trace_path,
        }

    def _operation_records(self) -> tuple[ExecutionRecord, ...]:
        """Return records retained by ``from_schedule`` when available."""
        return tuple(
            record for record in self.schedule_records if record.category == "operation"
        )

    def _top_inactive_intervals(self) -> list[dict[str, int]]:
        """Return largest core-level intervals without any operation running.

        A core can still be blocked on a flag during such an interval, so this
        intentionally says *inactive*, not "hardware idle".  Flag-blocking
        time remains separately available in ``major_stalls``.
        """
        by_core: dict[int, list[tuple[int, int]]] = {}
        for record in self._operation_records():
            if record.duration_cycles:
                by_core.setdefault(record.core_id, []).append(
                    (record.start_cycle, record.end_cycle)
                )
        makespan = self.stats.makespan_cycles
        intervals: list[dict[str, int]] = []
        for core_id, raw_intervals in by_core.items():
            cursor = 0
            for start, end in self._merge_intervals(raw_intervals):
                if start > cursor:
                    intervals.append(
                        {
                            "core_id": core_id,
                            "start_cycle": cursor,
                            "end_cycle": start,
                            "duration_cycles": start - cursor,
                        }
                    )
                cursor = max(cursor, end)
            if cursor < makespan:
                intervals.append(
                    {
                        "core_id": core_id,
                        "start_cycle": cursor,
                        "end_cycle": makespan,
                        "duration_cycles": makespan - cursor,
                    }
                )
        return sorted(
            intervals,
            key=lambda interval: (
                -interval["duration_cycles"], interval["core_id"], interval["start_cycle"]
            ),
        )[:5]

    def _copy_compute_overlap(self) -> dict[str, Any]:
        """Measure per-core overlap of copy pipes with compute pipes.

        This is a union-of-intervals calculation, so overlapping work on two
        copy pipes is counted once.  It is schedule information, not a memory
        bandwidth or hardware-throughput estimate.
        """
        copies: dict[int, list[tuple[int, int]]] = {}
        compute: dict[int, list[tuple[int, int]]] = {}
        for record in self._operation_records():
            if not record.duration_cycles:
                continue
            interval = (record.start_cycle, record.end_cycle)
            if self._is_copy(record):
                copies.setdefault(record.core_id, []).append(interval)
            elif record.pipe.value in {"m", "v", "fix"}:
                compute.setdefault(record.core_id, []).append(interval)

        by_core = []
        for core_id in sorted(set(copies) & set(compute)):
            copy_intervals = self._merge_intervals(copies[core_id])
            compute_intervals = self._merge_intervals(compute[core_id])
            overlap = self._intersection_length(copy_intervals, compute_intervals)
            if overlap:
                by_core.append({"core_id": core_id, "cycles": overlap})
        return {
            "scope": "per-core union of copy and compute intervals; not a bandwidth estimate",
            "total_per_core_cycles": sum(item["cycles"] for item in by_core),
            "by_core": by_core[:5],
        }

    def _resource_pressure(self) -> dict[str, Any]:
        """Rank resources by FIFO queueing first, then busy utilization.

        Queue delay is summed per task (not unioned): it measures how much task
        delay this resource contributed in this schedule. It is not a claim
        about hardware queue depth or physical saturation.
        """
        pressure: dict[str, dict[str, int]] = {}
        for record in self._operation_records():
            resource = f"core-{record.core_id}/{record.resource}"
            entry = pressure.setdefault(
                resource, {"operation_count": 0, "fifo_queued_task_cycles": 0}
            )
            entry["operation_count"] += 1
            queued_at = record.metadata.get("queue_enter_cycle")
            if isinstance(queued_at, int) and queued_at < record.start_cycle:
                entry["fifo_queued_task_cycles"] += record.start_cycle - queued_at
        resources = [
            {
                "resource": resource,
                "busy_cycles": self.stats.busy_cycles_by_resource.get(resource, 0),
                "utilization": self.stats.utilization_by_resource.get(resource, 0.0),
                "operation_count": entry["operation_count"],
                "fifo_queued_task_cycles": entry["fifo_queued_task_cycles"],
            }
            for resource, entry in pressure.items()
        ]
        resources.sort(
            key=lambda item: (
                -item["fifo_queued_task_cycles"],
                -item["utilization"],
                item["resource"],
            )
        )
        return {
            "scope": (
                "schedule-derived FIFO queued task-cycles then utilization; "
                "not physical queue depth or hardware saturation"
            ),
            "top_resources": resources[:5],
        }

    @staticmethod
    def compare(
        baseline: "PerformanceReport", candidate: "PerformanceReport"
    ) -> dict[str, Any]:
        """Return a compact, schedule-derived comparison of two reports.

        ``candidate - baseline`` is used consistently for deltas. Both reports
        must describe the same platform: comparing A2 to A3 schedules has no
        stable interpretation without a separate normalization model.
        """
        if baseline.platform != candidate.platform:
            raise ValueError(
                "cannot compare simulator reports from different platforms: "
                f"{baseline.platform} != {candidate.platform}"
            )
        baseline_document = baseline.to_dict()
        candidate_document = candidate.to_dict()
        baseline_schedule = baseline_document["schedule"]
        candidate_schedule = candidate_document["schedule"]
        baseline_makespan = int(baseline_schedule["makespan_cycles"])
        candidate_makespan = int(candidate_schedule["makespan_cycles"])
        makespan_delta = candidate_makespan - baseline_makespan

        baseline_utilization = baseline_schedule["utilization_by_resource"]
        candidate_utilization = candidate_schedule["utilization_by_resource"]
        resource_changes = []
        for resource in sorted(set(baseline_utilization) | set(candidate_utilization)):
            before = float(baseline_utilization.get(resource, 0.0))
            after = float(candidate_utilization.get(resource, 0.0))
            resource_changes.append(
                {
                    "resource": resource,
                    "baseline_utilization": before,
                    "candidate_utilization": after,
                    "delta": after - before,
                }
            )
        resource_changes.sort(
            key=lambda item: (-abs(item["delta"]), item["resource"])
        )

        baseline_waits = baseline_schedule["wait_cycles_by_reason"]
        candidate_waits = candidate_schedule["wait_cycles_by_reason"]
        wait_changes = []
        for reason in sorted(set(baseline_waits) | set(candidate_waits)):
            before = int(baseline_waits.get(reason, 0))
            after = int(candidate_waits.get(reason, 0))
            wait_changes.append(
                {
                    "reason": reason,
                    "baseline_cycles": before,
                    "candidate_cycles": after,
                    "delta_cycles": after - before,
                }
            )
        wait_changes.sort(key=lambda item: (-abs(item["delta_cycles"]), item["reason"]))

        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "scope": (
                "comparison of simulator schedules; candidate minus baseline; "
                "not a measured hardware-performance comparison"
            ),
            "platform": candidate.platform,
            "timing_comparable": (
                baseline.calibration == candidate.calibration
                and baseline.timing_estimator == candidate.timing_estimator
            ),
            "makespan_cycles": {
                "baseline": baseline_makespan,
                "candidate": candidate_makespan,
                "delta": makespan_delta,
                "ratio": (
                    None if baseline_makespan == 0 else candidate_makespan / baseline_makespan
                ),
            },
            "critical_chain_span_cycles": {
                "baseline": baseline_document["critical_path"].get("span_cycles"),
                "candidate": candidate_document["critical_path"].get("span_cycles"),
            },
            "copy_compute_overlap_per_core_cycles": {
                "baseline": baseline_document["copy_compute_overlap"]["total_per_core_cycles"],
                "candidate": candidate_document["copy_compute_overlap"]["total_per_core_cycles"],
                "delta": (
                    candidate_document["copy_compute_overlap"]["total_per_core_cycles"]
                    - baseline_document["copy_compute_overlap"]["total_per_core_cycles"]
                ),
            },
            "top_resource_utilization_changes": resource_changes[:5],
            "top_wait_cycle_changes": wait_changes[:5],
        }

    @staticmethod
    def _is_copy(record: ExecutionRecord) -> bool:
        operation = record.operation.lower()
        return (
            operation.startswith("copy")
            or "datacopy" in operation
            or "data_copy" in operation
        )

    @staticmethod
    def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _intersection_length(
        left: list[tuple[int, int]], right: list[tuple[int, int]]
    ) -> int:
        total = 0
        left_index = right_index = 0
        while left_index < len(left) and right_index < len(right):
            start = max(left[left_index][0], right[right_index][0])
            end = min(left[left_index][1], right[right_index][1])
            total += max(0, end - start)
            if left[left_index][1] <= right[right_index][1]:
                left_index += 1
            else:
                right_index += 1
        return total

    def to_text(self) -> str:
        """Return a short, stable human-readable summary."""
        document = self.to_dict()
        schedule = document["schedule"]
        timing = document["timing"]
        lines = [
            (
                f"Simulator performance report: {self.platform} "
                f"({document['simulation_mode']})"
            ),
            (
                f"makespan={schedule['makespan_cycles']} cycles, "
                f"tasks={schedule['task_count']}, "
                f"load_imbalance={schedule['load_imbalance_cycles']} cycles"
            ),
            f"timing={timing['calibration']}: {timing['interpretation']}",
        ]
        resources = document["top_resources"]
        if resources:
            lines.append(
                "top_resources=" + ", ".join(
                    f"{item['resource']} ({item['utilization']:.1%})"
                    for item in resources[:3]
                )
            )
        stalls = document["major_stalls"]
        if stalls:
            lines.append(
                "major_stalls=" + ", ".join(
                    f"{item['reason']} ({item['cycles']} cycles)"
                    for item in stalls[:3]
                )
            )
        return "\n".join(lines)

    def write_json(self, path: str | Path) -> Path:
        """Write the compact report and return its resolved path."""
        output = Path(path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output
