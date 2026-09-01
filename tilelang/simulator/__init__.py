# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CPU-only simulator foundations for TileLang Ascend A2/A3."""

from .config import SimulatorConfig
from .bridge import build_kernel_program, classify_operation
from .errors import (
    MemoryAccessError,
    MemoryBoundsError,
    MemoryCapacityError,
    MemoryHazardError,
    ProgramValidationError,
    SchedulerError,
    SimulationDeadlockError,
    SimulationLimitError,
    SimulatorConfigError,
    SimulatorError,
    UnknownDependencyError,
    UnsupportedMemoryScopeError,
    UnsupportedSimOpError,
    UninitializedMemoryError,
)
from .hazard import HazardDiagnostic, HazardReporter, SimulatorHazardWarning
from .memory import (
    A2_A3_LOCAL_CAPACITIES,
    AddressRange,
    MemoryAllocation,
    MemoryRuntime,
    MemoryView,
    contiguous_strides_bytes,
    dtype_size_bytes,
)
from .profile import DeviceProfile, TimingProfile, default_timing_profile, get_device_profile
from .program import BufferSpec, CoreProgram, KernelProgram, Lane, MemoryScope, Pipe, Task
from .scheduler import DiscreteEventScheduler, ScheduleResult
from .stats import SimulationStats
from .sync import (
    FlagBarrierSynchronizationModel,
    NoOpSynchronizationModel,
    SyncDecision,
    SynchronizationModel,
)
from .trace import ChromeTraceExporter, ExecutionRecord

__all__ = [
    "A2_A3_LOCAL_CAPACITIES",
    "AddressRange",
    "BufferSpec",
    "build_kernel_program",
    "ChromeTraceExporter",
    "CoreProgram",
    "DeviceProfile",
    "DiscreteEventScheduler",
    "ExecutionRecord",
    "FlagBarrierSynchronizationModel",
    "KernelProgram",
    "Lane",
    "MemoryAccessError",
    "MemoryAllocation",
    "MemoryBoundsError",
    "MemoryCapacityError",
    "MemoryHazardError",
    "MemoryRuntime",
    "MemoryScope",
    "MemoryView",
    "NoOpSynchronizationModel",
    "Pipe",
    "ProgramValidationError",
    "ScheduleResult",
    "SchedulerError",
    "SimulationDeadlockError",
    "SimulationLimitError",
    "SimulationStats",
    "SimulatorConfig",
    "SimulatorConfigError",
    "SimulatorError",
    "SyncDecision",
    "SynchronizationModel",
    "Task",
    "TimingProfile",
    "UnsupportedMemoryScopeError",
    "UnsupportedSimOpError",
    "UnknownDependencyError",
    "UninitializedMemoryError",
    "HazardDiagnostic",
    "HazardReporter",
    "SimulatorHazardWarning",
    "contiguous_strides_bytes",
    "classify_operation",
    "default_timing_profile",
    "dtype_size_bytes",
    "get_device_profile",
]
