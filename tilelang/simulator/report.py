# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Compact, machine-readable performance summaries for simulator schedules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SimulatorConfig
from .stats import SimulationStats


REPORT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class PerformanceReport:
    """A compact schedule summary intended for people and automated consumers.

    The report deliberately summarizes an already-produced schedule.  It does
    not claim a critical path or measured hardware timing when the selected
    profile is an analytical PTO fallback.
    """

    platform: str
    calibration: str
    timing_estimator: str
    sync_only: bool
    trace_path: str | None
    stats: SimulationStats

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
        )

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
            "critical_path": {
                "status": "unavailable",
                "reason": "critical-path extraction is not implemented",
            },
            "trace_path": self.trace_path,
        }

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
