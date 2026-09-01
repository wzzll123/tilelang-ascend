# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Synchronization extension points for the discrete-event scheduler.

The first simulator milestone only needs task dependencies and pipe FIFO.  This module keeps
flag and barrier state out of the scheduler so those semantics can be added without changing
the scheduling API.
"""

from dataclasses import dataclass
from types import MappingProxyType
from collections import defaultdict, deque
from typing import Deque, Dict, Mapping, Optional, Protocol, Tuple

from .errors import ProgramValidationError
from .program import KernelProgram, Task
from .trace import ExecutionRecord


@dataclass(frozen=True)
class SyncDecision:
    """Result of evaluating synchronization state for one otherwise-ready task.

    ``ready_cycle`` is the earliest cycle allowed by synchronization.  ``None`` means the task
    is blocked until another task updates the synchronization model.  ``reason`` and ``detail``
    are intended for deadlock diagnostics and future trace wait records.
    """

    ready_cycle: Optional[int] = 0
    reason: Optional[str] = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.ready_cycle is not None and self.ready_cycle < 0:
            raise ValueError("synchronization ready_cycle must not be negative")

    @property
    def blocked(self) -> bool:
        """Return whether synchronization currently prevents the task from running."""
        return self.ready_cycle is None


class SynchronizationModel(Protocol):
    """Protocol implemented by future flag and barrier state machines."""

    def reset(self, program: KernelProgram) -> None:
        """Reset state before scheduling a program."""

    def evaluate(
        self, task: Task, completed: Mapping[str, ExecutionRecord]
    ) -> SyncDecision:
        """Return the current synchronization constraint for ``task``."""

    def on_scheduled(self, task: Task, record: ExecutionRecord) -> None:
        """Observe a newly scheduled task and update synchronization state."""


class NoOpSynchronizationModel:
    """Synchronization model used until a task represents a flag or barrier operation."""

    def reset(self, program: KernelProgram) -> None:
        del program

    def evaluate(
        self, task: Task, completed: Mapping[str, ExecutionRecord]
    ) -> SyncDecision:
        del task, completed
        return SyncDecision()

    def on_scheduled(self, task: Task, record: ExecutionRecord) -> None:
        del task, record


FlagKey = Tuple[str, int, str, str, int]


class FlagBarrierSynchronizationModel:
    """Model A2/A3 local flags, C/V flags, and pipe barriers.

    Synchronization operands are carried in ``Task.metadata`` so the TIR bridge can
    preserve the exact lowered identifiers without coupling the scheduler to TVM.
    Local flag tasks require ``src_pipe``, ``dst_pipe``, and ``flag_id``. Cross-C/V
    tasks require ``flag_id`` and may provide ``channel`` to distinguish protocols.
    ``pipe_barrier`` may provide ``target_pipe``; otherwise it drains the task's pipe.
    """

    _SET_LOCAL = frozenset({
        "set_flag", "auto_set_flag", "tl.ascend_set_flag", "tl.ascend_auto_set_flag",
    })
    _WAIT_LOCAL = frozenset({
        "wait_flag", "auto_wait_flag", "tl.ascend_wait_flag", "tl.ascend_auto_wait_flag",
    })
    _SET_CROSS = frozenset({
        "set_cross_flag", "auto_set_cross_flag", "tl.ascend_set_cross_flag",
        "tl.ascend_auto_set_cross_flag",
    })
    _WAIT_CROSS = frozenset({
        "wait_cross_flag", "auto_wait_cross_flag", "tl.ascend_wait_cross_flag",
        "tl.ascend_auto_wait_cross_flag",
    })
    _BARRIER_ALL = frozenset({"barrier_all", "tl.ascend_barrier_all"})
    _PIPE_BARRIER = frozenset({
        "pipe_barrier", "auto_barrier", "tl.ascend_pipe_barrier",
        "tl.ascend_auto_barrier",
    })

    def __init__(self) -> None:
        self._tokens: Dict[FlagKey, Deque[int]] = defaultdict(deque)
        self._barrier_dependencies: Dict[str, Tuple[str, ...]] = {}

    @staticmethod
    def _operation(task: Task) -> str:
        return task.operation.strip().lower()

    def reset(self, program: KernelProgram) -> None:
        self._tokens.clear()
        self._barrier_dependencies.clear()
        prior_by_core: Dict[int, list[Task]] = defaultdict(list)
        for task in program.tasks:
            operation = self._operation(task)
            if operation in self._BARRIER_ALL:
                self._barrier_dependencies[task.task_id] = tuple(
                    prior.task_id for prior in prior_by_core[task.core_id]
                )
            elif operation in self._PIPE_BARRIER:
                target_pipe = str(task.metadata.get("target_pipe", task.pipe.value)).lower()
                if target_pipe == "all":
                    self._barrier_dependencies[task.task_id] = tuple(
                        prior.task_id for prior in prior_by_core[task.core_id]
                    )
                else:
                    target_lane = str(
                        task.metadata.get("target_lane", task.lane.value)
                    ).lower()
                    self._barrier_dependencies[task.task_id] = tuple(
                        prior.task_id
                        for prior in prior_by_core[task.core_id]
                        if prior.pipe.value == target_pipe
                        and prior.lane.value == target_lane
                    )
            prior_by_core[task.core_id].append(task)

    def evaluate(
        self, task: Task, completed: Mapping[str, ExecutionRecord]
    ) -> SyncDecision:
        operation = self._operation(task)
        if operation in self._WAIT_LOCAL or operation in self._WAIT_CROSS:
            key = self._flag_key(task, is_cross=operation in self._WAIT_CROSS)
            tokens = self._tokens.get(key)
            if not tokens:
                return SyncDecision(
                    ready_cycle=None,
                    reason="cross flag" if operation in self._WAIT_CROSS else "local flag",
                    detail=self._format_flag_key(key),
                )
            return SyncDecision(
                ready_cycle=tokens[0],
                reason="cross flag" if operation in self._WAIT_CROSS else "local flag",
                detail=self._format_flag_key(key),
            )

        if operation in self._BARRIER_ALL or operation in self._PIPE_BARRIER:
            required = self._barrier_dependencies.get(task.task_id, ())
            missing = tuple(task_id for task_id in required if task_id not in completed)
            if missing:
                return SyncDecision(
                    ready_cycle=None,
                    reason="barrier",
                    detail="waiting for " + ", ".join(missing),
                )
            ready_cycle = max(
                (completed[task_id].end_cycle for task_id in required), default=0
            )
            return SyncDecision(ready_cycle=ready_cycle, reason="barrier")

        return SyncDecision()

    def on_scheduled(self, task: Task, record: ExecutionRecord) -> None:
        operation = self._operation(task)
        if operation in self._SET_LOCAL or operation in self._SET_CROSS:
            key = self._flag_key(task, is_cross=operation in self._SET_CROSS)
            self._tokens[key].append(record.end_cycle)
        elif operation in self._WAIT_LOCAL or operation in self._WAIT_CROSS:
            key = self._flag_key(task, is_cross=operation in self._WAIT_CROSS)
            tokens = self._tokens.get(key)
            if not tokens:
                raise ProgramValidationError(
                    f"wait task {task.task_id!r} consumed a missing {self._format_flag_key(key)}"
                )
            tokens.popleft()

    def _flag_key(self, task: Task, *, is_cross: bool) -> FlagKey:
        flag_id = task.metadata.get("flag_id")
        if isinstance(flag_id, bool) or not isinstance(flag_id, int) or flag_id < 0:
            raise ProgramValidationError(
                f"synchronization task {task.task_id!r} requires a non-negative integer flag_id"
            )
        if is_cross:
            channel = str(task.metadata.get("channel", "cv")).lower()
            return ("cross", task.core_id, channel, channel, flag_id)

        src_pipe = task.metadata.get("src_pipe")
        dst_pipe = task.metadata.get("dst_pipe")
        if not isinstance(src_pipe, str) or not src_pipe:
            raise ProgramValidationError(
                f"local synchronization task {task.task_id!r} requires src_pipe"
            )
        if not isinstance(dst_pipe, str) or not dst_pipe:
            raise ProgramValidationError(
                f"local synchronization task {task.task_id!r} requires dst_pipe"
            )
        return ("local", task.core_id, src_pipe.lower(), dst_pipe.lower(), flag_id)

    @staticmethod
    def _format_flag_key(key: FlagKey) -> str:
        family, core_id, src, dst, flag_id = key
        return f"{family} flag core={core_id} {src}->{dst} id={flag_id}"


def readonly_records(
    records: Mapping[str, ExecutionRecord]
) -> Mapping[str, ExecutionRecord]:
    """Expose completed records to synchronization models without allowing mutation."""
    return MappingProxyType(dict(records))
