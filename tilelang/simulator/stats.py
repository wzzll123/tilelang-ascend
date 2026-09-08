# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Basic schedule statistics derived from simulator execution records."""

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .errors import ProgramValidationError
from .memory import dtype_size_bytes
from .program import AffineInt, BufferRegion, SymbolicInt
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


def _runtime_int(value: Any, bindings: Mapping[str, int | float]) -> Optional[int]:
    if isinstance(value, (AffineInt, SymbolicInt)):
        try:
            return value.evaluate(bindings)
        except ProgramValidationError:
            return None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    return None


def _region_nbytes(
    region: BufferRegion, bindings: Mapping[str, int | float]
) -> Optional[int]:
    """Return logical payload bytes when all region extents can be resolved."""
    elements = 1
    for extent in region.shape:
        resolved = _runtime_int(extent, bindings)
        if resolved is None:
            return None
        elements *= resolved
    return elements * dtype_size_bytes(region.dtype)


def _memory_path(metadata: Mapping[str, Any]) -> Optional[str]:
    source = metadata.get("src")
    destination = metadata.get("dst")
    if not isinstance(source, BufferRegion) or not isinstance(destination, BufferRegion):
        return None
    return f"{source.scope.value}->{destination.scope.value}"


def _transfer_bytes(
    metadata: Mapping[str, Any], bindings: Mapping[str, int | float]
) -> Optional[int]:
    """Resolve a task's logical transfer size without guessing dynamic values."""
    explicit = metadata.get("transfer_bytes", metadata.get("bytes"))
    resolved = _runtime_int(explicit, bindings)
    if resolved is not None:
        return resolved
    destination = metadata.get("dst")
    if isinstance(destination, BufferRegion):
        return _region_nbytes(destination, bindings)
    return None


@dataclass(frozen=True)
class SimulationStats:
    """Summary metrics for comparing simulator schedules."""

    makespan_cycles: int
    task_count: int
    busy_cycles_by_resource: Mapping[str, int]
    utilization_by_resource: Mapping[str, float]
    wait_cycles_by_reason: Mapping[str, int]
    completion_cycle_by_core: Mapping[int, int]
    operation_counts: Mapping[str, int]
    memory_bytes_by_path: Mapping[str, int]
    load_imbalance_cycles: int

    @classmethod
    def from_records(
        cls,
        records: Iterable[ExecutionRecord],
        *,
        bindings: Optional[Mapping[str, int | float]] = None,
    ) -> "SimulationStats":
        """Compute overlap-aware resource utilization and simple stall totals."""
        runtime_bindings = bindings or {}
        record_list = list(records)
        if not record_list:
            empty = MappingProxyType({})
            return cls(0, 0, empty, empty, empty, empty, empty, empty, 0)

        makespan = max(record.end_cycle for record in record_list)
        intervals: Dict[str, List[Tuple[int, int]]] = {}
        waits: Dict[str, int] = {}
        completion: Dict[int, int] = {}
        operation_counts: Dict[str, int] = {}
        memory_bytes: Dict[str, int] = {}
        for record in record_list:
            resource = f"core-{record.core_id}/{record.resource}"
            intervals.setdefault(resource, []).append((record.start_cycle, record.end_cycle))
            completion[record.core_id] = max(
                completion.get(record.core_id, 0), record.end_cycle
            )
            operation_counts[record.operation] = (
                operation_counts.get(record.operation, 0) + 1
            )
            path = (
                _memory_path(record.metadata)
                if record.operation.startswith("copy_") else None
            )
            transferred = (
                _transfer_bytes(record.metadata, runtime_bindings)
                if path is not None else None
            )
            if path is not None and transferred is not None:
                memory_bytes[path] = memory_bytes.get(path, 0) + transferred
            if record.stall_reason is not None:
                waits[record.stall_reason] = (
                    waits.get(record.stall_reason, 0) + record.duration_cycles
                )

        busy = {resource: _union_length(values) for resource, values in intervals.items()}
        utilization = {
            resource: cycles / makespan if makespan else 0.0
            for resource, cycles in busy.items()
        }
        completion_values = tuple(completion.values())
        return cls(
            makespan_cycles=makespan,
            task_count=len(record_list),
            busy_cycles_by_resource=MappingProxyType(busy),
            utilization_by_resource=MappingProxyType(utilization),
            wait_cycles_by_reason=MappingProxyType(waits),
            completion_cycle_by_core=MappingProxyType(completion),
            operation_counts=MappingProxyType(operation_counts),
            memory_bytes_by_path=MappingProxyType(memory_bytes),
            load_imbalance_cycles=(
                max(completion_values) - min(completion_values)
                if completion_values else 0
            ),
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
            "operation_counts": dict(self.operation_counts),
            "memory_bytes_by_path": dict(self.memory_bytes_by_path),
            "load_imbalance_cycles": self.load_imbalance_cycles,
        }
