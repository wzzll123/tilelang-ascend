# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Functional execution of concrete simulator tasks on CPU memory."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from .config import SimulatorConfig
from .errors import ProgramValidationError, UnsupportedSimOpError
from .memory import MemoryRuntime, MemoryView
from .program import BufferRegion, KernelProgram, MemoryScope, Task
from .scheduler import DiscreteEventScheduler, ScheduleResult
from .sync import FlagBarrierSynchronizationModel


_BINARY_OPERATIONS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "max": np.maximum,
    "min": np.minimum,
}


@dataclass(frozen=True)
class FunctionalExecutionResult:
    """Schedule and final memory state produced by functional execution."""

    schedule: ScheduleResult
    memory: MemoryRuntime


class FunctionalSimulator:
    """Execute explicit copy and vector tasks using byte-accurate simulator memory."""

    def __init__(
        self,
        program: KernelProgram,
        config: Optional[SimulatorConfig] = None,
    ) -> None:
        self.program = program
        self.config = config or SimulatorConfig(platform=program.platform)
        if self.config.platform != program.platform:
            raise ProgramValidationError(
                "functional simulator config platform does not match the program"
            )
        self.memory = MemoryRuntime.from_program(
            program, hazard_check=self.config.hazard_check
        )

    def write(self, region: BufferRegion, values: Any, *, task_core_id: int = 0) -> None:
        """Initialize a concrete region from an array-like CPU value."""
        view = self._resolve(region, task_core_id)
        array = np.asarray(values, dtype=_numpy_dtype(region.dtype))
        if array.shape != region.shape:
            raise ProgramValidationError(
                f"input for {region.buffer!r} has shape {array.shape}, expected {region.shape}"
            )
        view.allocation.write(view, np.ascontiguousarray(array).tobytes(order="C"))

    def read(self, region: BufferRegion, *, task_core_id: int = 0) -> np.ndarray:
        """Read a concrete region into an independent NumPy array."""
        view = self._resolve(region, task_core_id)
        payload = view.allocation.read(view)
        return np.frombuffer(payload, dtype=_numpy_dtype(region.dtype)).reshape(
            region.shape
        ).copy()

    def run(self) -> FunctionalExecutionResult:
        """Schedule the program, then apply operations in deterministic event order."""
        schedule = DiscreteEventScheduler(
            self.config,
            synchronization=FlagBarrierSynchronizationModel(),
        ).run(self.program)
        task_by_id = {task.task_id: task for task in self.program.tasks}
        for record in schedule.records:
            if record.category == "wait":
                continue
            self._execute(task_by_id[record.task_id])
        return FunctionalExecutionResult(schedule=schedule, memory=self.memory)

    def _execute(self, task: Task) -> None:
        operation = task.operation.lower()
        if "copy" in operation or "datacopy" in operation or "data_copy" in operation:
            self._copy(task)
            return
        if operation.startswith("tail_"):
            operation = operation[len("tail_"):]
        if operation in _BINARY_OPERATIONS:
            self._binary(task, operation)
            return
        if operation in {"set_flag", "wait_flag", "auto_set_flag", "auto_wait_flag",
                         "set_cross_flag", "wait_cross_flag", "auto_set_cross_flag",
                         "auto_wait_cross_flag", "barrier_all", "pipe_barrier",
                         "auto_barrier"}:
            return
        raise UnsupportedSimOpError(
            f"functional execution is not implemented for {task.operation!r} "
            f"(task {task.task_id!r})"
        )

    def _copy(self, task: Task) -> None:
        source = _operand(task, "src")
        destination = _operand(task, "dst")
        values = self.read(source, task_core_id=task.core_id)
        self.write(destination, values, task_core_id=task.core_id)

    def _binary(self, task: Task, operation: str) -> None:
        left = _operand(task, "lhs")
        destination = _operand(task, "dst")
        left_values = self.read(left, task_core_id=task.core_id)
        right = task.metadata.get("rhs")
        if isinstance(right, BufferRegion):
            right_values: Any = self.read(right, task_core_id=task.core_id)
        elif "scalar" in task.metadata:
            right_values = task.metadata["scalar"]
        else:
            raise ProgramValidationError(
                f"task {task.task_id!r} requires a BufferRegion 'rhs' or scalar"
            )
        result = _BINARY_OPERATIONS[operation](left_values, right_values)
        self.write(destination, result, task_core_id=task.core_id)

    def _resolve(self, region: BufferRegion, task_core_id: int) -> MemoryView:
        owner = region.core_id if region.core_id is not None else task_core_id
        allocation = self.memory.get(
            region.buffer,
            scope=region.scope,
            core_id=None if region.scope in {MemoryScope.GM, MemoryScope.WORKSPACE} else owner,
        )
        return allocation.view(
            byte_offset=region.byte_offset,
            shape=region.shape,
            dtype=region.dtype,
            strides_bytes=region.strides_bytes,
        )


def _operand(task: Task, name: str) -> BufferRegion:
    value = task.metadata.get(name)
    if not isinstance(value, BufferRegion):
        raise ProgramValidationError(
            f"task {task.task_id!r} requires BufferRegion metadata {name!r}"
        )
    return value


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    normalized = dtype.strip().lower()
    if "x" in normalized:
        raise UnsupportedSimOpError(f"vector-packed dtype is not executable yet: {dtype!r}")
    try:
        return np.dtype(normalized)
    except TypeError as error:
        raise UnsupportedSimOpError(
            f"NumPy cannot represent simulator dtype {dtype!r}"
        ) from error
