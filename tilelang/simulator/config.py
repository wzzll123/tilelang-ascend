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
    # Flag credit depth fidelity (HS29 motivation): on real silicon a flag's
    # outstanding SET credits occupy a bounded hardware queue; a SET issued at
    # full depth STALLS the issuing pipe until a WAIT frees a slot (blocking
    # semantics), which can deadlock a kernel whose accounting is level-balanced
    # but depth-unbalanced. The default (flag_blocking=False) keeps the legacy
    # idealized semantics (local: error on double-set; cross: error at 15
    # outstanding credits). flag_blocking=True turns overflow into a blocking
    # wait so the scheduler's deadlock detector can report the cycle.
    flag_blocking: bool = False
    local_flag_depth: int = 1
    cross_flag_depth: int = 15
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
        if not isinstance(self.flag_blocking, bool):
            raise SimulatorConfigError("flag_blocking must be a boolean")
        for name in ("local_flag_depth", "cross_flag_depth"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SimulatorConfigError(f"{name} must be a positive integer")
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
