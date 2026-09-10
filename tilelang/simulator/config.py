# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Configuration for the A2/A3 CPU simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    trace_path: str | Path | None = None
    hazard_check: str = "error"
    sync_only: bool = False
    dynamic_if: str = "error"
    deadlock_detect: bool = True
    deadlock_history_limit: int = 8
    flag_balance_check: str = "off"
    execution_timeout_s: float = 120.0
    max_cycles: int | None = None
    timing_profile: TimingProfile | None = None

    def __post_init__(self) -> None:
        platform = normalize_platform(self.platform)
        object.__setattr__(self, "platform", platform)
        if self.hazard_check not in {"off", "warn", "error"}:
            raise SimulatorConfigError("hazard_check must be one of: off, warn, error")
        if not isinstance(self.sync_only, bool):
            raise SimulatorConfigError("sync_only must be a boolean")
        if self.dynamic_if not in {"error", "then", "else"}:
            raise SimulatorConfigError("dynamic_if must be one of: error, then, else")
        if not isinstance(self.deadlock_detect, bool):
            raise SimulatorConfigError("deadlock_detect must be a boolean")
        if (
            not isinstance(self.deadlock_history_limit, int)
            or isinstance(self.deadlock_history_limit, bool)
            or self.deadlock_history_limit <= 0
        ):
            raise SimulatorConfigError("deadlock_history_limit must be a positive integer")
        if self.flag_balance_check not in {"off", "warn", "error"}:
            raise SimulatorConfigError("flag_balance_check must be one of: off, warn, error")
        if self.execution_timeout_s <= 0:
            raise SimulatorConfigError("execution_timeout_s must be positive")
        if self.max_cycles is not None and self.max_cycles <= 0:
            raise SimulatorConfigError("max_cycles must be positive when specified")
        if self.trace_path is not None:
            object.__setattr__(self, "trace_path", Path(self.trace_path))
        timing_profile = self.timing_profile or default_timing_profile(platform)
        if timing_profile.platform != platform:
            raise SimulatorConfigError(
                f"timing profile platform does not match simulator platform: {timing_profile.platform} != {platform}"
            )
        object.__setattr__(self, "timing_profile", timing_profile)

    @property
    def device_profile(self) -> DeviceProfile:
        """Return the selected immutable device profile."""
        return get_device_profile(self.platform)
