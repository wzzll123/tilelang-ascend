# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Configurable diagnostics for the simulator memory model."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, List, Mapping, Optional
import warnings

from .errors import MemoryHazardError


class SimulatorHazardWarning(RuntimeWarning):
    """Warning emitted when simulator hazard checking is set to ``warn``."""


@dataclass(frozen=True)
class HazardDiagnostic:
    """One deterministic, machine-readable correctness diagnostic."""

    kind: str
    message: str
    buffer: Optional[str] = None
    core_id: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class HazardReporter:
    """Apply an ``off``/``warn``/``error`` policy and retain diagnostics."""

    _VALID_MODES = frozenset({"off", "warn", "error"})

    def __init__(self, mode: str = "error") -> None:
        if mode not in self._VALID_MODES:
            raise ValueError("hazard mode must be one of: off, warn, error")
        self.mode = mode
        self._diagnostics: List[HazardDiagnostic] = []

    @property
    def diagnostics(self) -> tuple[HazardDiagnostic, ...]:
        """Return diagnostics in report order."""
        return tuple(self._diagnostics)

    def report(
        self,
        diagnostic: HazardDiagnostic,
        *,
        error_type: type[MemoryHazardError] = MemoryHazardError,
    ) -> None:
        """Handle ``diagnostic`` according to the configured policy."""
        if self.mode == "off":
            return
        self._diagnostics.append(diagnostic)
        if self.mode == "warn":
            warnings.warn(diagnostic.message, SimulatorHazardWarning, stacklevel=2)
            return
        raise error_type(diagnostic.message)
