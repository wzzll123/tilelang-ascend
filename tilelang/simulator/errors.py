# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Errors raised by the CPU-only Ascend simulator infrastructure."""


class SimulatorError(RuntimeError):
    """Base class for simulator failures."""


class SimulatorConfigError(SimulatorError, ValueError):
    """Raised when a simulator configuration is invalid."""


class ProgramValidationError(SimulatorError, ValueError):
    """Raised when a simulator program is malformed."""


class UnsupportedSimOpError(SimulatorError, NotImplementedError):
    """Raised when an operation has no simulator implementation."""


class UnsupportedMemoryScopeError(UnsupportedSimOpError):
    """Raised when a memory scope is intentionally unsupported."""


class MemoryAccessError(SimulatorError):
    """Base class for invalid simulator memory accesses."""


class MemoryBoundsError(MemoryAccessError, IndexError):
    """Raised when an access escapes its allocation or address space."""


class MemoryCapacityError(MemoryAccessError):
    """Raised when local-memory allocations exceed an A2/A3 scope capacity."""


class MemoryHazardError(MemoryAccessError):
    """Raised when hazard checking is configured to fail on a diagnostic."""


class UninitializedMemoryError(MemoryHazardError):
    """Raised for a read from bytes which have never been written."""


class SchedulerError(SimulatorError):
    """Base class for discrete-event scheduling failures."""


class UnknownDependencyError(SchedulerError):
    """Raised when a task names a dependency absent from the program."""


class SimulationDeadlockError(SchedulerError):
    """Raised when no pending task can make progress."""


class SimulationLimitError(SchedulerError):
    """Raised when a configured simulation limit is exceeded."""


def reject_shmem(scope: str) -> None:
    """Fail fast for physical cross-core shmem, which is outside simulator scope."""
    normalized = scope.strip().lower()
    if normalized in {"shmem", "shared.shmem", "shared_memory"}:
        raise UnsupportedMemoryScopeError(
            "Ascend shmem is intentionally unsupported by the A2/A3 simulator"
        )


def reject_shmem_operation(operation: str) -> None:
    """Fail fast for the four cross-PE shmem intrinsics."""
    normalized = operation.strip().lower()
    if "shmem_" in normalized or "ascend_shmem" in normalized:
        raise UnsupportedSimOpError(
            f"Ascend shmem operation {operation!r} is intentionally unsupported by the "
            "A2/A3 simulator"
        )
