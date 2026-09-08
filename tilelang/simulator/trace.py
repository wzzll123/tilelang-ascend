# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Chrome/Perfetto trace records and exporter."""

import json
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from .errors import ProgramValidationError
from .program import Lane, Pipe, Task


TRACE_SCHEMA_VERSION = "1.0"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


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
                    "schema_version": TRACE_SCHEMA_VERSION,
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
        events.extend(self._active_core_events(record_list))
        events.extend(self._queue_depth_events(record_list))
        for record in record_list:
            args = _json_safe(record.metadata)
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
        records_by_id = {
            record.task_id: record
            for record in record_list
            if record.category == "operation"
        }
        for consumer in records_by_id.values():
            dependency_groups = (
                ("memory_dependency", consumer.metadata.get("memory_dependencies", ())),
                ("flag_dependency", consumer.metadata.get("sync_producers", ())),
            )
            for flow_name, producer_ids in dependency_groups:
                for producer_id in producer_ids:
                    producer = records_by_id.get(str(producer_id))
                    if producer is None:
                        continue
                    flow_id = f"{flow_name}:{producer.task_id}:{consumer.task_id}"
                    events.extend((
                        {
                            "name": flow_name,
                            "cat": "dependency",
                            "ph": "s",
                            "id": flow_id,
                            "ts": producer.end_cycle,
                            "pid": f"core-{producer.core_id}",
                            "tid": producer.resource,
                            "args": {"from": producer.task_id, "to": consumer.task_id},
                        },
                        {
                            "name": flow_name,
                            "cat": "dependency",
                            "ph": "f",
                            "bp": "e",
                            "id": flow_id,
                            "ts": consumer.start_cycle,
                            "pid": f"core-{consumer.core_id}",
                            "tid": consumer.resource,
                            "args": {"from": producer.task_id, "to": consumer.task_id},
                        },
                    ))
        return {
            "schemaVersion": TRACE_SCHEMA_VERSION,
            "traceEvents": events,
            "displayTimeUnit": "ns",
        }

    @staticmethod
    def _active_core_events(records: Iterable[ExecutionRecord]) -> List[Dict[str, Any]]:
        """Emit a global counter after unioning overlapping pipes on each core."""
        intervals_by_core: Dict[int, List[tuple[int, int]]] = {}
        for record in records:
            if record.category != "operation" or record.duration_cycles == 0:
                continue
            intervals_by_core.setdefault(record.core_id, []).append(
                (record.start_cycle, record.end_cycle)
            )

        transitions: Dict[int, int] = {}
        for intervals in intervals_by_core.values():
            merged: List[tuple[int, int]] = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            for start, end in merged:
                transitions[start] = transitions.get(start, 0) + 1
                transitions[end] = transitions.get(end, 0) - 1

        active = 0
        events = []
        for cycle, delta in sorted(transitions.items()):
            active += delta
            events.append({
                "name": "active_cores",
                "cat": "counter",
                "ph": "C",
                "ts": cycle,
                "pid": "simulator",
                "tid": "counters",
                "args": {"active_cores": active},
            })
        return events

    @staticmethod
    def _queue_depth_events(records: Iterable[ExecutionRecord]) -> List[Dict[str, Any]]:
        """Emit per-resource counters for tasks ready but waiting behind FIFO work."""
        transitions: Dict[tuple[int, str, int], int] = {}
        for record in records:
            if record.category != "operation":
                continue
            queued_at = record.metadata.get("queue_enter_cycle")
            if not isinstance(queued_at, int) or queued_at >= record.start_cycle:
                continue
            resource = (record.core_id, record.resource)
            transitions[(resource[0], resource[1], queued_at)] = (
                transitions.get((resource[0], resource[1], queued_at), 0) + 1
            )
            transitions[(resource[0], resource[1], record.start_cycle)] = (
                transitions.get((resource[0], resource[1], record.start_cycle), 0) - 1
            )

        depths: Dict[tuple[int, str], int] = {}
        events: List[Dict[str, Any]] = []
        for (core_id, resource, cycle), delta in sorted(transitions.items()):
            key = (core_id, resource)
            depths[key] = depths.get(key, 0) + delta
            events.append({
                "name": "queue_depth",
                "cat": "counter",
                "ph": "C",
                "ts": cycle,
                "pid": f"core-{core_id}",
                "tid": resource,
                "args": {"queue_depth": depths[key]},
            })
        return events

    def write(self, path: Union[str, Path], records: Iterable[ExecutionRecord]) -> Path:
        """Write a trace document and return its resolved output path."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(records), indent=2), encoding="utf-8")
        return output.resolve()
