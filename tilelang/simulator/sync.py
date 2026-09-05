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
from .hazard import HazardDiagnostic, HazardReporter
from .program import BufferRegion, KernelProgram, Task
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


FlagKey = Tuple[object, ...]
Participant = Tuple[int, str]
CollectiveKey = Tuple[int, str, int, int]
_VECTOR_LANE_NAMES = ("vector0", "vector1")
_MAX_CROSS_FLAG_CREDITS = 15
_PIPE_NAMES = frozenset({"mte2", "mte1", "m", "fix", "v", "mte3", "s"})
_PIPES_BY_LANE = {
    "cube": frozenset({"mte2", "mte1", "m", "fix", "s"}),
    "vector0": frozenset({"mte2", "v", "mte3", "s"}),
    "vector1": frozenset({"mte2", "v", "mte3", "s"}),
    "control": frozenset({"s"}),
}


def validate_memory_synchronization(
    program: KernelProgram, *, hazard_check: str = "error"
) -> tuple[HazardDiagnostic, ...]:
    """Verify that inferred cross-pipe memory edges have hardware fences.

    The bridge's memory dependencies are an execution aid, not an Ascend
    synchronization primitive.  Accept an edge only when pipe FIFO suffices,
    a full ``PIPE_ALL`` lies between the accesses, or a matching local
    set/wait flag pair connects the producer and consumer pipes.
    """
    reporter = HazardReporter(hazard_check)
    for core in program.cores:
        tasks = core.tasks
        index_by_id = {task.task_id: index for index, task in enumerate(tasks)}
        task_by_id = {task.task_id: task for task in tasks}
        for consumer_index, consumer in enumerate(tasks):
            dependencies = consumer.metadata.get("memory_dependencies", ())
            if not isinstance(dependencies, (tuple, list)):
                continue
            for dependency_id in dependencies:
                producer = task_by_id.get(str(dependency_id))
                if (
                    producer is None
                    or producer.lane != consumer.lane
                    or producer.pipe == consumer.pipe
                ):
                    continue
                producer_index = index_by_id[producer.task_id]
                if producer_index >= consumer_index:
                    continue
                between = tasks[producer_index + 1:consumer_index]
                if _has_full_barrier(between) or _has_local_flag_fence(
                    between, producer, consumer
                ):
                    continue
                buffer_name = _shared_buffer_name(producer, consumer)
                buffer_detail = f" for buffer {buffer_name!r}" if buffer_name else ""
                reporter.report(HazardDiagnostic(
                    kind="missing-pipe-synchronization",
                    message=(
                        "missing A2/A3 pipe synchronization for memory dependency "
                        f"{producer.task_id} ({producer.operation}/{producer.pipe.value}) -> "
                        f"{consumer.task_id} ({consumer.operation}/{consumer.pipe.value}) "
                        f"on core {consumer.core_id} lane {consumer.lane.value}"
                        f"{buffer_detail}; "
                        "expected a matching set/wait flag pair or PIPE_ALL"
                    ),
                    buffer=buffer_name,
                    core_id=consumer.core_id,
                    metadata={
                        "producer_task": producer.task_id,
                        "consumer_task": consumer.task_id,
                        "producer_pipe": producer.pipe.value,
                        "consumer_pipe": consumer.pipe.value,
                    },
                ))
    return reporter.diagnostics


def _shared_buffer_name(producer: Task, consumer: Task) -> Optional[str]:
    keys = (
        "src_regions", "src", "lhs", "rhs", "mask", "accumulator",
        "scalar_src", "offsets", "bias", "dst", "dst_regions", "pad_dst",
        "scratch", "output_scratch",
    )

    def names(task: Task) -> set[str]:
        result = set()
        for key in keys:
            value = task.metadata.get(key)
            values = value if isinstance(value, (tuple, list)) else (value,)
            result.update(
                region.buffer for region in values if isinstance(region, BufferRegion)
            )
        return result

    shared = names(producer).intersection(names(consumer))
    return sorted(shared)[0] if shared else None


def _has_full_barrier(tasks: Tuple[Task, ...]) -> bool:
    for task in tasks:
        operation = task.operation.strip().lower()
        if operation in FlagBarrierSynchronizationModel._BARRIER_ALL:
            return True
        if operation in FlagBarrierSynchronizationModel._PIPE_BARRIER:
            target = str(task.metadata.get("target_pipe", task.pipe.value)).lower()
            if target.startswith("pipe_"):
                target = target[5:]
            if target == "all":
                return True
    return False


def _has_local_flag_fence(
    tasks: Tuple[Task, ...], producer: Task, consumer: Task
) -> bool:
    pending = set()
    for task in tasks:
        if task.core_id != consumer.core_id or task.lane != consumer.lane:
            continue
        operation = task.operation.strip().lower()
        source_pipe = str(task.metadata.get("src_pipe", "")).lower()
        destination_pipe = str(task.metadata.get("dst_pipe", "")).lower()
        if source_pipe.startswith("pipe_"):
            source_pipe = source_pipe[5:]
        if destination_pipe.startswith("pipe_"):
            destination_pipe = destination_pipe[5:]
        signature = (
            source_pipe,
            destination_pipe,
            task.metadata.get("flag_id"),
        )
        if operation in FlagBarrierSynchronizationModel._SET_LOCAL:
            if source_pipe == producer.pipe.value:
                pending.add(signature)
        elif operation in FlagBarrierSynchronizationModel._WAIT_LOCAL:
            if destination_pipe == consumer.pipe.value and signature in pending:
                return True
    return False


class FlagBarrierSynchronizationModel:
    """Model A2/A3 local flags, C/V flags, and pipe barriers.

    Synchronization operands are carried in ``Task.metadata`` so the TIR bridge can
    preserve the exact lowered identifiers without coupling the scheduler to TVM.
    Local flag tasks require ``src_pipe``, ``dst_pipe``, and ``flag_id``. Cross-C/V
    set tasks additionally require ``src_pipe`` and mode 0, 1, or 2. ``pipe_barrier`` may
    provide ``target_pipe``; otherwise it drains the task's pipe.
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
        self._collective_ready: Dict[
            CollectiveKey, Dict[Participant, list[int]]
        ] = defaultdict(lambda: defaultdict(list))
        self._collective_wait_phase: Dict[
            Tuple[CollectiveKey, Participant], int
        ] = defaultdict(int)
        self._cross_wait_modes: Dict[Tuple[int, str], int] = {}
        self._core_ids: Tuple[int, ...] = ()
        self._barrier_dependencies: Dict[str, Tuple[str, ...]] = {}
        self._ordering_dependencies: Dict[str, Tuple[str, ...]] = {}

    @staticmethod
    def _operation(task: Task) -> str:
        return task.operation.strip().lower()

    def reset(self, program: KernelProgram) -> None:
        self._tokens.clear()
        self._collective_ready.clear()
        self._collective_wait_phase.clear()
        self._cross_wait_modes.clear()
        self._core_ids = tuple(sorted(core.core_id for core in program.cores))
        self._barrier_dependencies.clear()
        self._ordering_dependencies.clear()
        wait_modes: Dict[Tuple[int, str], set[int]] = defaultdict(set)
        for task in program.tasks:
            operation = self._operation(task)
            self._validate_sync_task(task, operation)
            if operation not in self._SET_CROSS:
                continue
            mode = self._cross_mode(task)
            target_kind = self._cross_target_kind(mode, task)
            wait_modes[(self._flag_id(task), target_kind)].add(mode)
        for signature, modes in wait_modes.items():
            if len(modes) != 1:
                flag_id, target_kind = signature
                raise ProgramValidationError(
                    f"cross flag id={flag_id} targeting {target_kind} has ambiguous "
                    f"modes {sorted(modes)}"
                )
            self._cross_wait_modes[signature] = next(iter(modes))
        prior_by_core: Dict[int, list[Task]] = defaultdict(list)
        prior_by_lane: Dict[Tuple[int, str], list[Task]] = defaultdict(list)
        last_fence: Dict[Tuple[int, str], str] = {}
        last_pipe_fence: Dict[Tuple[int, str, str], str] = {}
        for task in program.tasks:
            operation = self._operation(task)
            self._validate_sync_task(task, operation)
            lane_key = (task.core_id, task.lane.value)
            ordering = []
            if lane_key in last_fence:
                ordering.append(last_fence[lane_key])
            pipe_key = (task.core_id, task.lane.value, task.pipe.value)
            if pipe_key in last_pipe_fence:
                ordering.append(last_pipe_fence[pipe_key])
            # EasyASC enqueues a set marker on the named producer pipe.  Match
            # that behavior instead of draining unrelated pipes in the lane.
            if operation in self._SET_LOCAL or operation in self._SET_CROSS:
                source_pipe = self._pipe_name(task.metadata.get("src_pipe"))
                ordering.extend(
                    prior.task_id
                    for prior in prior_by_lane[lane_key]
                    if prior.pipe.value == source_pipe
                )
            if ordering:
                self._ordering_dependencies[task.task_id] = tuple(dict.fromkeys(ordering))
            if operation in self._BARRIER_ALL:
                self._barrier_dependencies[task.task_id] = tuple(
                    prior.task_id
                    for prior in prior_by_core[task.core_id]
                    if task.lane.value == "control" or prior.lane == task.lane
                )
            elif operation in self._PIPE_BARRIER:
                target_pipe = self._pipe_name(
                    task.metadata.get("target_pipe", task.pipe.value)
                )
                if target_pipe == "all":
                    self._barrier_dependencies[task.task_id] = tuple(
                        prior.task_id
                        for prior in prior_by_core[task.core_id]
                        if task.lane.value == "control" or prior.lane == task.lane
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
            prior_by_lane[lane_key].append(task)
            if operation in self._WAIT_CROSS or operation in self._BARRIER_ALL:
                last_fence[lane_key] = task.task_id
            elif operation in self._WAIT_LOCAL:
                destination_pipe = self._pipe_name(task.metadata.get("dst_pipe"))
                if destination_pipe == "s":
                    last_fence[lane_key] = task.task_id
                else:
                    last_pipe_fence[
                        (task.core_id, task.lane.value, destination_pipe)
                    ] = task.task_id
            elif operation in self._PIPE_BARRIER:
                target_pipe = self._pipe_name(
                    task.metadata.get("target_pipe", task.pipe.value)
                )
                if target_pipe == "all":
                    last_fence[lane_key] = task.task_id
                elif target_pipe:
                    last_pipe_fence[
                        (task.core_id, task.lane.value, target_pipe)
                    ] = task.task_id
            elif operation in self._SET_LOCAL or operation in self._SET_CROSS:
                source_pipe = self._pipe_name(task.metadata.get("src_pipe"))
                if source_pipe:
                    last_pipe_fence[
                        (task.core_id, task.lane.value, source_pipe)
                    ] = task.task_id

    def evaluate(
        self, task: Task, completed: Mapping[str, ExecutionRecord]
    ) -> SyncDecision:
        operation = self._operation(task)
        ordering = self._ordering_dependencies.get(task.task_id, ())
        missing_ordering = tuple(
            task_id for task_id in ordering if task_id not in completed
        )
        if missing_ordering:
            return SyncDecision(
                ready_cycle=None,
                reason="synchronization ordering",
                detail="waiting for " + ", ".join(missing_ordering),
            )
        ordering_cycle = max(
            (completed[task_id].end_cycle for task_id in ordering), default=0
        )
        if operation in self._WAIT_LOCAL:
            key = self._local_flag_key(task)
            tokens = self._tokens.get(key)
            if not tokens:
                return SyncDecision(
                    ready_cycle=None,
                    reason="local flag",
                    detail=self._format_flag_key(key),
                )
            return SyncDecision(
                ready_cycle=max(ordering_cycle, tokens[0]),
                reason="local flag",
                detail=self._format_flag_key(key),
            )

        if operation in self._WAIT_CROSS:
            mode = self._cross_wait_mode(task)
            if mode is None:
                return SyncDecision(
                    ready_cycle=None,
                    reason="cross flag",
                    detail=(
                        f"no set_cross_flag mode declaration for core={task.core_id} "
                        f"lane={task.lane.value} id={self._flag_id(task)}"
                    ),
                )
            if mode in {0, 1}:
                return self._evaluate_collective_wait(task, mode, ordering_cycle)
            keys = self._cross_wait_keys_mode2(task)
            missing = tuple(key for key in keys if not self._tokens.get(key))
            if missing:
                return SyncDecision(
                    ready_cycle=None,
                    reason="cross flag",
                    detail="; ".join(self._format_flag_key(key) for key in missing),
                )
            return SyncDecision(
                ready_cycle=max(
                    ordering_cycle,
                    max(self._tokens[key][0] for key in keys),
                ),
                reason="cross flag",
                detail="; ".join(self._format_flag_key(key) for key in keys),
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
            return SyncDecision(
                ready_cycle=max(ready_cycle, ordering_cycle), reason="barrier"
            )

        return SyncDecision(ready_cycle=ordering_cycle)

    def on_scheduled(self, task: Task, record: ExecutionRecord) -> None:
        operation = self._operation(task)
        if operation in self._SET_LOCAL:
            key = self._local_flag_key(task)
            tokens = self._tokens.get(key)
            if tokens:
                raise ProgramValidationError(
                    f"set task {task.task_id!r} reused an outstanding "
                    f"{self._format_flag_key(key)}"
                )
            self._tokens[key].append(record.end_cycle)
        elif operation in self._SET_CROSS:
            mode = self._cross_mode(task)
            if mode in {0, 1}:
                self._on_collective_set(task, mode, record.end_cycle)
            else:
                for key in self._cross_set_keys_mode2(task):
                    tokens = self._tokens[key]
                    if len(tokens) >= _MAX_CROSS_FLAG_CREDITS:
                        raise ProgramValidationError(
                            f"set task {task.task_id!r} overflowed "
                            f"{self._format_flag_key(key)} at "
                            f"{_MAX_CROSS_FLAG_CREDITS} outstanding credits"
                        )
                    tokens.append(record.end_cycle)
        elif operation in self._WAIT_LOCAL:
            key = self._local_flag_key(task)
            tokens = self._tokens.get(key)
            if not tokens:
                raise ProgramValidationError(
                    f"wait task {task.task_id!r} consumed a missing {self._format_flag_key(key)}"
                )
            tokens.popleft()
        elif operation in self._WAIT_CROSS:
            mode = self._cross_wait_mode(task)
            if mode is None:
                raise ProgramValidationError(
                    f"wait task {task.task_id!r} has no cross flag mode declaration"
                )
            if mode in {0, 1}:
                key = self._collective_key(task, mode)
                participant = self._participant(task)
                self._collective_wait_phase[(key, participant)] += 1
            else:
                for key in self._cross_wait_keys_mode2(task):
                    tokens = self._tokens.get(key)
                    if not tokens:
                        raise ProgramValidationError(
                            f"wait task {task.task_id!r} consumed a missing "
                            f"{self._format_flag_key(key)}"
                        )
                    tokens.popleft()

    def _local_flag_key(self, task: Task) -> FlagKey:
        flag_id = task.metadata.get("flag_id")
        if isinstance(flag_id, bool) or not isinstance(flag_id, int) or flag_id < 0:
            raise ProgramValidationError(
                f"synchronization task {task.task_id!r} requires a non-negative integer flag_id"
            )
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
        return (
            "local", task.core_id, task.lane.value,
            self._pipe_name(src_pipe), self._pipe_name(dst_pipe), flag_id,
        )

    def _cross_set_keys_mode2(self, task: Task) -> Tuple[FlagKey, ...]:
        flag_id = self._flag_id(task)
        if task.lane.value == "cube":
            return tuple(
                ("cross", task.core_id, "c2v", lane, flag_id)
                for lane in _VECTOR_LANE_NAMES
            )
        if task.lane.value in _VECTOR_LANE_NAMES:
            return (("cross", task.core_id, "v2c", task.lane.value, flag_id),)
        raise ProgramValidationError(
            f"cross flag set task {task.task_id!r} must run on cube or vector lane"
        )

    def _cross_wait_keys_mode2(self, task: Task) -> Tuple[FlagKey, ...]:
        flag_id = self._flag_id(task)
        if task.lane.value == "cube":
            return tuple(
                ("cross", task.core_id, "v2c", lane, flag_id)
                for lane in _VECTOR_LANE_NAMES
            )
        if task.lane.value in _VECTOR_LANE_NAMES:
            return (("cross", task.core_id, "c2v", task.lane.value, flag_id),)
        raise ProgramValidationError(
            f"cross flag wait task {task.task_id!r} must run on cube or vector lane"
        )

    def _evaluate_collective_wait(
        self, task: Task, mode: int, ordering_cycle: int
    ) -> SyncDecision:
        key = self._collective_key(task, mode)
        waiter = self._participant(task)
        phase = self._collective_wait_phase[(key, waiter)]
        expected = self._collective_participants(key)
        ready = self._collective_ready[key]
        missing = tuple(
            participant
            for participant in expected
            if len(ready.get(participant, ())) <= phase
        )
        if missing:
            return SyncDecision(
                ready_cycle=None,
                reason="cross flag",
                detail=(
                    f"mode={mode} id={self._flag_id(task)} phase={phase} waiting for "
                    + ", ".join(self._format_participant(item) for item in missing)
                ),
            )
        return SyncDecision(
            ready_cycle=max(
                ordering_cycle,
                max(ready[participant][phase] for participant in expected),
            ),
            reason="cross flag",
            detail=f"mode={mode} id={self._flag_id(task)} phase={phase}",
        )

    def _on_collective_set(self, task: Task, mode: int, end_cycle: int) -> None:
        key = self._collective_key(task, mode)
        producer = self._participant(task)
        expected = self._collective_participants(key)
        if producer not in expected:
            raise ProgramValidationError(
                f"cross flag set task {task.task_id!r} is not a mode-{mode} participant"
            )
        ready = self._collective_ready[key][producer]
        minimum_wait_phase = min(
            self._collective_wait_phase[(key, participant)]
            for participant in expected
        )
        if len(ready) - minimum_wait_phase >= _MAX_CROSS_FLAG_CREDITS:
            raise ProgramValidationError(
                f"set task {task.task_id!r} overflowed mode={mode} cross flag "
                f"id={self._flag_id(task)} for {self._format_participant(producer)} "
                f"at {_MAX_CROSS_FLAG_CREDITS} outstanding phases"
            )
        ready.append(end_cycle)

    def _collective_key(self, task: Task, mode: int) -> CollectiveKey:
        lane_kind = self._lane_kind(task)
        flag_id = self._flag_id(task)
        if mode == 0:
            return (mode, lane_kind, -1, flag_id)
        if mode == 1:
            return (mode, "vector", task.core_id, flag_id)
        raise ProgramValidationError(f"mode {mode} is not a collective cross flag mode")

    def _collective_participants(self, key: CollectiveKey) -> Tuple[Participant, ...]:
        mode, lane_kind, group_id, _flag_id = key
        if mode == 0 and lane_kind == "cube":
            return tuple((core_id, "cube") for core_id in self._core_ids)
        if mode == 0 and lane_kind == "vector":
            return tuple(
                (core_id, lane)
                for core_id in self._core_ids
                for lane in _VECTOR_LANE_NAMES
            )
        if mode == 1 and lane_kind == "vector":
            return tuple((group_id, lane) for lane in _VECTOR_LANE_NAMES)
        raise ProgramValidationError(f"invalid collective cross flag key {key!r}")

    def _cross_wait_mode(self, task: Task) -> Optional[int]:
        return self._cross_wait_modes.get((self._flag_id(task), self._lane_kind(task)))

    @classmethod
    def _cross_target_kind(cls, mode: int, task: Task) -> str:
        lane_kind = cls._lane_kind(task)
        if mode == 2:
            return "vector" if lane_kind == "cube" else "cube"
        return lane_kind

    @staticmethod
    def _lane_kind(task: Task) -> str:
        if task.lane.value == "cube":
            return "cube"
        if task.lane.value in _VECTOR_LANE_NAMES:
            return "vector"
        raise ProgramValidationError(
            f"cross flag task {task.task_id!r} must run on cube or vector lane"
        )

    @staticmethod
    def _participant(task: Task) -> Participant:
        return (task.core_id, task.lane.value)

    @staticmethod
    def _format_participant(participant: Participant) -> str:
        core_id, lane = participant
        return f"core={core_id}/lane={lane}"

    @staticmethod
    def _cross_mode(task: Task) -> int:
        mode = task.metadata.get("mode", 2)
        if isinstance(mode, bool) or not isinstance(mode, int) or mode not in {0, 1, 2}:
            raise ProgramValidationError(
                f"cross flag set task {task.task_id!r} requires mode 0, 1, or 2, "
                f"got {mode!r}"
            )
        return mode

    @staticmethod
    def _flag_id(task: Task) -> int:
        flag_id = task.metadata.get("flag_id")
        if isinstance(flag_id, bool) or not isinstance(flag_id, int) or flag_id < 0:
            raise ProgramValidationError(
                f"synchronization task {task.task_id!r} requires a non-negative integer flag_id"
            )
        return flag_id

    @classmethod
    def _validate_sync_task(cls, task: Task, operation: str) -> None:
        if operation in cls._SET_LOCAL or operation in cls._WAIT_LOCAL:
            flag_id = task.metadata.get("flag_id")
            if (
                isinstance(flag_id, bool)
                or not isinstance(flag_id, int)
                or not 0 <= flag_id <= 7
            ):
                raise ProgramValidationError(
                    f"local flag task {task.task_id!r} requires flag_id in [0, 7], "
                    f"got {flag_id}"
                )
            src_pipe = cls._pipe_name(task.metadata.get("src_pipe"))
            dst_pipe = cls._pipe_name(task.metadata.get("dst_pipe"))
            if src_pipe not in _PIPE_NAMES or src_pipe == "s":
                raise ProgramValidationError(
                    f"local flag task {task.task_id!r} requires a non-scalar "
                    f"source pipe, got {src_pipe!r}"
                )
            if dst_pipe not in _PIPE_NAMES:
                raise ProgramValidationError(
                    f"local flag task {task.task_id!r} requires a valid destination "
                    f"pipe, got {dst_pipe!r}"
                )
            cls._validate_lane_pipe(task, src_pipe, "source")
            cls._validate_lane_pipe(task, dst_pipe, "destination")
        if operation in cls._SET_CROSS:
            cls._flag_id(task)
            mode = cls._cross_mode(task)
            lane_kind = cls._lane_kind(task)
            if mode == 1 and lane_kind != "vector":
                raise ProgramValidationError(
                    f"cross flag mode 1 requires a vector lane, got "
                    f"{task.lane.value!r} in task {task.task_id!r}"
                )
            source_pipe = cls._pipe_name(task.metadata.get("src_pipe"))
            if source_pipe not in _PIPE_NAMES or source_pipe == "s":
                raise ProgramValidationError(
                    f"cross flag set task {task.task_id!r} requires a non-scalar "
                    f"src_pipe, got {source_pipe!r}"
                )
            cls._validate_lane_pipe(task, source_pipe, "source")
        elif operation in cls._WAIT_CROSS:
            cls._flag_id(task)
            cls._lane_kind(task)
        if operation in cls._PIPE_BARRIER:
            target_pipe = cls._pipe_name(
                task.metadata.get("target_pipe", task.pipe.value)
            )
            if target_pipe != "all" and target_pipe not in _PIPE_NAMES:
                raise ProgramValidationError(
                    f"pipe barrier task {task.task_id!r} names unknown pipe "
                    f"{target_pipe!r}"
                )

    @staticmethod
    def _validate_lane_pipe(task: Task, pipe: str, role: str) -> None:
        if pipe not in _PIPES_BY_LANE[task.lane.value]:
            raise ProgramValidationError(
                f"local/cross flag task {task.task_id!r} names {role} pipe "
                f"{pipe!r}, which is unavailable on lane {task.lane.value!r}"
            )

    @staticmethod
    def _pipe_name(value: object) -> str:
        normalized = str(value or "").strip().lower()
        return normalized[5:] if normalized.startswith("pipe_") else normalized

    @staticmethod
    def _format_flag_key(key: FlagKey) -> str:
        if key[0] == "local":
            family, core_id, lane, src, dst, flag_id = key
            return (
                f"{family} flag core={core_id} lane={lane} "
                f"{src}->{dst} id={flag_id}"
            )
        family, core_id, direction, lane, flag_id = key
        return (
            f"{family} flag core={core_id} direction={direction} "
            f"lane={lane} id={flag_id}"
        )


def readonly_records(
    records: Mapping[str, ExecutionRecord]
) -> Mapping[str, ExecutionRecord]:
    """Expose completed records to synchronization models without allowing mutation."""
    return MappingProxyType(dict(records))
