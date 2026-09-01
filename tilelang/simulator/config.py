# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Configuration for the A2/A3 CPU simulator."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .errors import SimulatorConfigError
from .profile import (
    DeviceProfile,
    TimingProfile,
    default_timing_profile,
    get_device_profile,
    normalize_platform,
)


@dataclass(frozen=True)
class SimulatorConfig:
    """User-visible simulator settings independent of JIT integration."""

    platform: str = "A2"
    trace_path: Optional[Union[str, Path]] = None
    hazard_check: str = "error"
    execution_timeout_s: float = 120.0
    max_cycles: Optional[int] = None
    timing_profile: Optional[TimingProfile] = None

    def __post_init__(self) -> None:
        platform = normalize_platform(self.platform)
        object.__setattr__(self, "platform", platform)
        if self.hazard_check not in {"off", "warn", "error"}:
            raise SimulatorConfigError("hazard_check must be one of: off, warn, error")
        if self.execution_timeout_s <= 0:
            raise SimulatorConfigError("execution_timeout_s must be positive")
        if self.max_cycles is not None and self.max_cycles <= 0:
            raise SimulatorConfigError("max_cycles must be positive when specified")
        if self.trace_path is not None:
            object.__setattr__(self, "trace_path", Path(self.trace_path))
        timing_profile = self.timing_profile or default_timing_profile(platform)
        if timing_profile.platform != platform:
            raise SimulatorConfigError(
                "timing profile platform does not match simulator platform: "
                f"{timing_profile.platform} != {platform}"
            )
        object.__setattr__(self, "timing_profile", timing_profile)

    @property
    def device_profile(self) -> DeviceProfile:
        """Return the selected immutable device profile."""
        return get_device_profile(self.platform)
