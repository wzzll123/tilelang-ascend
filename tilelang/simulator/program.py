# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Backend-neutral program and task data structures for simulation."""

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Tuple

from .errors import (
    ProgramValidationError,
    UnsupportedMemoryScopeError,
    reject_shmem,
    reject_shmem_operation,
)
from .profile import normalize_platform


class Lane(str, Enum):
    """Logical C220 execution lanes."""

    CUBE = "cube"
    VECTOR_0 = "vector0"
    VECTOR_1 = "vector1"
    CONTROL = "control"


class Pipe(str, Enum):
    """Logical C220 pipeline resources."""

    MTE2 = "mte2"
    MTE1 = "mte1"
    MATRIX = "m"
    FIX = "fix"
    VECTOR = "v"
    MTE3 = "mte3"
    SCALAR = "s"


class MemoryScope(str, Enum):
    """Memory scopes planned for A2/A3, excluding physical shmem."""

    GM = "gm"
    WORKSPACE = "workspace"
    L1 = "l1"
    L0A = "l0a"
    L0B = "l0b"
    L0C = "l0c"
    UB = "ub"
    BT = "bt"
    LOCAL = "local"

    @classmethod
    def parse(cls, scope: str) -> "MemoryScope":
        """Convert lowered TIR scope spelling to a simulator scope.

        ``shared.ub`` and the other ``shared.*`` spellings are local Ascend memory and are
        accepted.  Physical ``shmem`` is a separate feature and always fails fast.
        """
        reject_shmem(scope)
        aliases = {
            "global": cls.GM,
            "shared.l1": cls.L1,
            "shared.l0a": cls.L0A,
            "shared.l0b": cls.L0B,
            "shared.l0c": cls.L0C,
            "shared.ub": cls.UB,
            "shared.bt": cls.BT,
            "local.var": cls.LOCAL,
            "wmma.matrix_a": cls.L0A,
            "wmma.matrix_b": cls.L0B,
            "wmma.accumulator": cls.L0C,
        }
        normalized = scope.strip().lower()
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as error:
            raise UnsupportedMemoryScopeError(
                f"unsupported A2/A3 simulator memory scope: {scope!r}"
            ) from error


@dataclass(frozen=True)
class BufferSpec:
    """Logical buffer declaration consumed by the simulator memory runtime.

    ``address`` preserves the byte address assigned by the final TIR storage
    rewrite.  ``lifetime`` is a half-open interval in bridge-defined program
    points and lets the memory runtime distinguish legal storage reuse from
    simultaneously-live overlap.  An intentional, simultaneously-live alias
    can instead declare ``metadata={"alias_of": "other_buffer"}``.
    """

    name: str
    scope: MemoryScope
    shape: Tuple[Any, ...]
    dtype: str
    size_bytes: Optional[int] = None
    address: Optional[int] = None
    lifetime: Optional[Tuple[int, int]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ProgramValidationError("buffer name must not be empty")
        if not isinstance(self.scope, MemoryScope):
            object.__setattr__(self, "scope", MemoryScope.parse(str(self.scope)))
        shape = tuple(self.shape)
        if any(isinstance(extent, Integral) and extent < 0 for extent in shape):
            raise ProgramValidationError(f"buffer {self.name!r} has a negative extent")
        if not self.dtype:
            raise ProgramValidationError(f"buffer {self.name!r} has no dtype")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ProgramValidationError(f"buffer {self.name!r} has a negative byte size")
        if self.address is not None and self.address < 0:
            raise ProgramValidationError(f"buffer {self.name!r} has a negative address")
        if self.lifetime is not None:
            lifetime = tuple(self.lifetime)
            if (len(lifetime) != 2 or any(not isinstance(point, Integral) for point in lifetime)
                    or lifetime[0] < 0 or lifetime[1] < lifetime[0]):
                raise ProgramValidationError(
                    f"buffer {self.name!r} lifetime must be a non-negative half-open interval"
                )
            object.__setattr__(self, "lifetime", (int(lifetime[0]), int(lifetime[1])))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class Task:
    """One schedulable simulator operation."""

    task_id: str
    operation: str
    core_id: int
    lane: Lane
    pipe: Pipe
    duration_cycles: int
    dependencies: Tuple[str, ...] = ()
    stage: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ProgramValidationError("task_id must not be empty")
        if not self.operation:
            raise ProgramValidationError(f"task {self.task_id!r} has no operation")
        reject_shmem_operation(self.operation)
        if self.core_id < 0:
            raise ProgramValidationError(f"task {self.task_id!r} has a negative core_id")
        if self.duration_cycles <= 0:
            raise ProgramValidationError(
                f"task {self.task_id!r} duration_cycles must be positive"
            )
        if not isinstance(self.lane, Lane):
            object.__setattr__(self, "lane", Lane(self.lane))
        if not isinstance(self.pipe, Pipe):
            object.__setattr__(self, "pipe", Pipe(self.pipe))
        allowed_pipes = {
            Lane.CUBE: {Pipe.MTE2, Pipe.MTE1, Pipe.MATRIX, Pipe.FIX, Pipe.SCALAR},
            Lane.VECTOR_0: {Pipe.MTE2, Pipe.VECTOR, Pipe.MTE3, Pipe.SCALAR},
            Lane.VECTOR_1: {Pipe.MTE2, Pipe.VECTOR, Pipe.MTE3, Pipe.SCALAR},
            Lane.CONTROL: {Pipe.SCALAR},
        }
        if self.pipe not in allowed_pipes[self.lane]:
            raise ProgramValidationError(
                f"pipe {self.pipe.value!r} is not valid on lane {self.lane.value!r}"
            )
        dependencies = tuple(self.dependencies)
        if self.task_id in dependencies:
            raise ProgramValidationError(f"task {self.task_id!r} cannot depend on itself")
        if len(set(dependencies)) != len(dependencies):
            raise ProgramValidationError(f"task {self.task_id!r} has duplicate dependencies")
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class CoreProgram:
    """Ordered tasks belonging to one logical core group."""

    core_id: int
    tasks: Tuple[Task, ...] = ()

    def __post_init__(self) -> None:
        if self.core_id < 0:
            raise ProgramValidationError("core_id must not be negative")
        tasks = tuple(self.tasks)
        for task in tasks:
            if task.core_id != self.core_id:
                raise ProgramValidationError(
                    f"task {task.task_id!r} belongs to core {task.core_id}, "
                    f"not core {self.core_id}"
                )
        object.__setattr__(self, "tasks", tasks)


@dataclass(frozen=True)
class KernelProgram:
    """Validated simulator program produced by a future TIR bridge."""

    name: str
    platform: str
    cores: Tuple[CoreProgram, ...]
    buffers: Tuple[BufferSpec, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ProgramValidationError("kernel program name must not be empty")
        object.__setattr__(self, "platform", normalize_platform(self.platform))
        cores = tuple(self.cores)
        buffers = tuple(self.buffers)
        self._validate_unique("core", (str(core.core_id) for core in cores))
        self._validate_unique("buffer", (buffer.name for buffer in buffers))

        tasks = tuple(task for core in cores for task in core.tasks)
        self._validate_unique("task", (task.task_id for task in tasks))
        task_ids = {task.task_id for task in tasks}
        for task in tasks:
            missing = set(task.dependencies) - task_ids
            if missing:
                names = ", ".join(sorted(missing))
                raise ProgramValidationError(
                    f"task {task.task_id!r} has unknown dependencies: {names}"
                )
        self._validate_acyclic(tasks)
        object.__setattr__(self, "cores", cores)
        object.__setattr__(self, "buffers", buffers)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @staticmethod
    def _validate_unique(kind: str, names: Iterable[str]) -> None:
        seen = set()
        for name in names:
            if name in seen:
                raise ProgramValidationError(f"duplicate {kind} identifier: {name!r}")
            seen.add(name)

    @staticmethod
    def _validate_acyclic(tasks: Iterable[Task]) -> None:
        dependencies = {task.task_id: task.dependencies for task in tasks}
        visiting = set()
        visited = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ProgramValidationError(
                    f"dependency cycle detected at task {task_id!r}"
                )
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in dependencies[task_id]:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in dependencies:
            visit(task_id)

    @property
    def tasks(self) -> Tuple[Task, ...]:
        """Return all tasks in stable core/program order."""
        return tuple(task for core in self.cores for task in core.tasks)
