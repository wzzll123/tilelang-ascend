# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Chrome/Perfetto trace records and exporter."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from .errors import ProgramValidationError
from .program import Lane, Pipe, Task


@dataclass(frozen=True)
class ExecutionRecord:
    """A scheduled operation interval measured in simulator cycles."""

    task_id: str
    operation: str
    core_id: int
    lane: Lane
    pipe: Pipe
    start_cycle: int
    end_cycle: int
    category: str = "operation"
    stall_reason: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or not self.operation:
            raise ProgramValidationError("execution record identifiers must not be empty")
        if self.core_id < 0 or self.start_cycle < 0:
            raise ProgramValidationError("execution record core and cycles must not be negative")
        if self.end_cycle < self.start_cycle:
            raise ProgramValidationError("execution record end_cycle precedes start_cycle")
        if not isinstance(self.lane, Lane):
            object.__setattr__(self, "lane", Lane(self.lane))
        if not isinstance(self.pipe, Pipe):
            object.__setattr__(self, "pipe", Pipe(self.pipe))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_task(cls, task: Task, start_cycle: int, end_cycle: int,
                  category: str = "operation", stall_reason: Optional[str] = None,
                  metadata: Optional[Mapping[str, Any]] = None) -> "ExecutionRecord":
        """Build a record while preserving a task's trace metadata."""
        merged_metadata = dict(task.metadata)
        if metadata:
            merged_metadata.update(metadata)
        if task.stage is not None:
            merged_metadata.setdefault("stage", task.stage)
        return cls(
            task_id=task.task_id,
            operation=task.operation,
            core_id=task.core_id,
            lane=task.lane,
            pipe=task.pipe,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
            category=category,
            stall_reason=stall_reason,
            metadata=merged_metadata,
        )

    @property
    def duration_cycles(self) -> int:
        """Return the interval duration in simulator cycles."""
        return self.end_cycle - self.start_cycle

    @property
    def resource(self) -> str:
        """Return the stable trace lane identifier."""
        return f"{self.lane.value}/{self.pipe.value}"


class ChromeTraceExporter:
    """Export execution records using the Chrome Trace Event Format."""

    def __init__(self, platform: str, calibration: str) -> None:
        self.platform = platform
        self.calibration = calibration

    def to_dict(self, records: Iterable[ExecutionRecord]) -> Dict[str, Any]:
        """Convert records to a JSON-serializable trace document."""
        record_list = list(records)
        events: List[Dict[str, Any]] = [
            {
                "name": "process_name",
                "ph": "M",
                "pid": "simulator",
                "tid": 0,
                "args": {"name": f"TileLang Ascend {self.platform} simulator"},
            },
            {
                "name": "simulator_metadata",
                "ph": "M",
                "pid": "simulator",
                "tid": 0,
                "args": {
                    "platform": self.platform,
                    "timestamp_unit": "simulator_cycle",
                    "calibration": self.calibration,
                },
            },
        ]
        resources = sorted({(record.core_id, record.resource) for record in record_list})
        for core_id, resource in resources:
            events.append({
                "name": "thread_name",
                "ph": "M",
                "pid": f"core-{core_id}",
                "tid": resource,
                "args": {"name": resource},
            })
        for record in record_list:
            args = dict(record.metadata)
            args["task_id"] = record.task_id
            args["cycle_begin"] = record.start_cycle
            args["cycle_end"] = record.end_cycle
            if record.stall_reason is not None:
                args["stall_reason"] = record.stall_reason
            events.append({
                "name": record.operation,
                "cat": record.category,
                "ph": "X",
                "ts": record.start_cycle,
                "dur": record.duration_cycles,
                "pid": f"core-{record.core_id}",
                "tid": record.resource,
                "args": args,
            })
        return {"traceEvents": events, "displayTimeUnit": "ns"}

    def write(self, path: Union[str, Path], records: Iterable[ExecutionRecord]) -> Path:
        """Write a trace document and return its resolved output path."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(records), indent=2), encoding="utf-8")
        return output.resolve()
