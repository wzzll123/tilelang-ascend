# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""A2/A3 topology and explicitly uncalibrated timing profiles."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from .errors import SimulatorConfigError


@dataclass(frozen=True)
class DeviceProfile:
    """Static topology used by the simulator.

    Timing is deliberately kept separate from topology.  The A3 core counts mirror the
    repository's current fallback table and must be revised when measured data is available.
    """

    platform: str
    cube_core_count: int
    vector_core_count: int
    vector_lanes_per_cube: int
    cube_pipes: Tuple[str, ...] = ("mte2", "mte1", "m", "fix")
    vector_pipes: Tuple[str, ...] = ("mte2", "v", "mte3")
    calibration: str = "uncalibrated"


@dataclass(frozen=True)
class TimingProfile:
    """Parameter table for a discrete-event scheduler.

    Empty operation costs are intentional: callers must either provide calibrated values or
    accept ``fallback_cycles``.  This prevents the scaffold from presenting invented hardware
    latencies as measurements.
    """

    platform: str
    operation_cycles: Mapping[str, int] = field(default_factory=dict)
    fallback_cycles: int = 1
    calibration: str = "uncalibrated-unit-cost"
    estimator: str = "fixed"

    def __post_init__(self) -> None:
        get_device_profile(self.platform)
        if self.fallback_cycles <= 0:
            raise SimulatorConfigError("fallback_cycles must be positive")
        normalized = {}
        for name, cycles in self.operation_cycles.items():
            if not name:
                raise SimulatorConfigError("operation name must not be empty")
            if cycles <= 0:
                raise SimulatorConfigError(f"operation cycle count must be positive: {name}")
            normalized[str(name)] = int(cycles)
        object.__setattr__(self, "operation_cycles", MappingProxyType(normalized))
        object.__setattr__(self, "platform", normalize_platform(self.platform))
        if self.estimator not in {"fixed", "pto-fallback"}:
            raise SimulatorConfigError("timing estimator must be one of: fixed, pto-fallback")

    def estimate_cycles(self, operation: str) -> int:
        """Return a configured cost or the visibly uncalibrated fallback cost."""
        return self.operation_cycles.get(operation, self.fallback_cycles)

    def estimate_task(
        self,
        operation: str,
        *,
        pipe: str,
        metadata: Mapping[str, Any],
    ) -> int:
        """Estimate one bridged task, preserving explicit per-operation overrides.

        ``pto-fallback`` ports the public PTO perf-sim fallback formulas for
        instructions not covered by its lightweight cost model.  It is a
        relative-performance model, not an A2/A3 hardware calibration.
        """
        configured = self.operation_cycles.get(operation)
        if configured is not None:
            return configured
        if self.estimator != "pto-fallback":
            return self.fallback_cycles

        # PTO's fallback formulas consume a logical rows*cols element count.
        # The TIR bridge retains transfer bytes and regions instead, so derive
        # that count only when dtype and shape information are concrete.
        elements = _timing_elements(metadata)
        normalized_pipe = pipe.strip().lower()
        normalized_operation = operation.strip().lower()
        if normalized_operation in {
            "set_flag", "wait_flag", "auto_set_flag", "auto_wait_flag",
            "set_cross_flag", "wait_cross_flag", "auto_set_cross_flag",
            "auto_wait_cross_flag", "pipe_barrier", "barrier_all",
        }:
            return 1
        if elements is None:
            return self.fallback_cycles
        if normalized_pipe == "m":
            return 4 + elements // 16
        if normalized_pipe in {"mte2", "mte3", "fix"}:
            return 3 + elements * 2 // 64
        if normalized_pipe == "mte1":
            return 1 + elements // 64
        if normalized_pipe == "s":
            return 1
        return 2 + elements // 32


def _timing_elements(metadata: Mapping[str, Any]) -> int | None:
    """Recover PTO's logical element count from bridge metadata when possible."""
    transfer_bytes = metadata.get("transfer_bytes")
    if isinstance(transfer_bytes, int) and transfer_bytes >= 0:
        for key in ("src", "dst"):
            dtype = getattr(metadata.get(key), "dtype", None)
            itemsize = _DTYPE_BYTES.get(str(dtype).lower())
            if itemsize is not None:
                return transfer_bytes // itemsize
    for key in ("dst", "src"):
        shape = getattr(metadata.get(key), "shape", None)
        if isinstance(shape, tuple) and all(isinstance(extent, int) and extent >= 0 for extent in shape):
            elements = 1
            for extent in shape:
                elements *= extent
            return elements
    return None


_DTYPE_BYTES = {
    "float16": 2, "half": 2, "bfloat16": 2, "float32": 4,
    "float": 4, "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
    "int32": 4, "uint32": 4,
}


_DEVICE_PROFILES = {
    "A2": DeviceProfile("A2", cube_core_count=20, vector_core_count=40,
                        vector_lanes_per_cube=2),
    "A3": DeviceProfile("A3", cube_core_count=20, vector_core_count=40,
                        vector_lanes_per_cube=2),
}


def normalize_platform(platform: str) -> str:
    """Normalize and validate a simulator platform name."""
    normalized = platform.strip().upper()
    if normalized not in _DEVICE_PROFILES:
        supported = ", ".join(sorted(_DEVICE_PROFILES))
        raise SimulatorConfigError(
            f"unsupported simulator platform {platform!r}; supported platforms: {supported}"
        )
    return normalized


def get_device_profile(platform: str) -> DeviceProfile:
    """Return the immutable device topology for A2 or A3."""
    return _DEVICE_PROFILES[normalize_platform(platform)]


def default_timing_profile(platform: str) -> TimingProfile:
    """Return a unit-cost profile clearly marked as uncalibrated."""
    return TimingProfile(platform=normalize_platform(platform))


def pto_fallback_timing_profile(platform: str) -> TimingProfile:
    """Return the PTO perf-sim-derived relative timing profile.

    The formulas are ported from ``pto/costmodel/perf_sim/costmodel_provider.hpp``.
    They are deliberately labeled as derived rather than measured hardware data.
    """
    return TimingProfile(
        platform=normalize_platform(platform),
        calibration="pto-perf-sim-derived-fallback",
        estimator="pto-fallback",
    )
