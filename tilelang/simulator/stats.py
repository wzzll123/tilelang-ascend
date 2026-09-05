# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Basic schedule statistics derived from simulator execution records."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .trace import ExecutionRecord


def _union_length(intervals: Sequence[Tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


@dataclass(frozen=True)
class SimulationStats:
    """Summary metrics for comparing simulator schedules."""

    makespan_cycles: int
    task_count: int
    busy_cycles_by_resource: Mapping[str, int]
    utilization_by_resource: Mapping[str, float]
    wait_cycles_by_reason: Mapping[str, int]
    completion_cycle_by_core: Mapping[int, int]

    @classmethod
    def from_records(cls, records: Iterable[ExecutionRecord]) -> "SimulationStats":
        """Compute overlap-aware resource utilization and simple stall totals."""
        record_list = list(records)
        if not record_list:
            empty = MappingProxyType({})
            return cls(0, 0, empty, empty, empty, empty)

        makespan = max(record.end_cycle for record in record_list)
        intervals: Dict[str, List[Tuple[int, int]]] = {}
        waits: Dict[str, int] = {}
        completion: Dict[int, int] = {}
        for record in record_list:
            resource = f"core-{record.core_id}/{record.resource}"
            intervals.setdefault(resource, []).append((record.start_cycle, record.end_cycle))
            completion[record.core_id] = max(
                completion.get(record.core_id, 0), record.end_cycle
            )
            if record.stall_reason is not None:
                waits[record.stall_reason] = (
                    waits.get(record.stall_reason, 0) + record.duration_cycles
                )

        busy = {resource: _union_length(values) for resource, values in intervals.items()}
        utilization = {
            resource: cycles / makespan if makespan else 0.0
            for resource, cycles in busy.items()
        }
        return cls(
            makespan_cycles=makespan,
            task_count=len(record_list),
            busy_cycles_by_resource=MappingProxyType(busy),
            utilization_by_resource=MappingProxyType(utilization),
            wait_cycles_by_reason=MappingProxyType(waits),
            completion_cycle_by_core=MappingProxyType(completion),
        )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable summary."""
        return {
            "makespan_cycles": self.makespan_cycles,
            "task_count": self.task_count,
            "busy_cycles_by_resource": dict(self.busy_cycles_by_resource),
            "utilization_by_resource": dict(self.utilization_by_resource),
            "wait_cycles_by_reason": dict(self.wait_cycles_by_reason),
            "completion_cycle_by_core": dict(self.completion_cycle_by_core),
        }
