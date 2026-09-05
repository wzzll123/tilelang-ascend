# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""A2/A3 topology and explicitly uncalibrated timing profiles."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Tuple

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

    def estimate_cycles(self, operation: str) -> int:
        """Return a configured cost or the visibly uncalibrated fallback cost."""
        return self.operation_cycles.get(operation, self.fallback_cycles)


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
