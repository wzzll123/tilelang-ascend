# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""TVM-independent task and physical byte-footprint model.

The adaptor must intern physical storage identities (for TIR, ``buffer.data``).
Logical buffer names and aliases never identify distinct physical allocations.
Unknown footprints use ``None``; intervals are half-open byte ranges. Guards
and owners are opaque metadata, not evidence that conflicting accesses cannot
execute or that a partial write initializes the whole allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ByteInterval = tuple[int, int]


@dataclass(frozen=True)
class Access:
    storage: Any
    mode: str
    offset: int | None = None
    extent: int | None = None
    intervals: tuple[ByteInterval, ...] | None = None
    alias: Any = None
    iteration_stride: int = 0
    ring_size: int | None = None
    full_overwrite: bool = False
    owner: Any = None

    def __post_init__(self):
        if self.mode not in ("R", "W", "RW"):
            raise ValueError("access mode must be R, W, or RW")
        if self.extent is not None and self.extent < 0:
            raise ValueError("byte extent must be nonnegative")
        if (self.offset is None) != (self.extent is None):
            raise ValueError("offset and extent must be supplied together")
        if self.ring_size is not None and self.ring_size < 1:
            raise ValueError("ring_size must be positive")
        if self.intervals is not None:
            normalized = tuple(tuple(interval) for interval in self.intervals)
            if any(len(interval) != 2 or interval[1] < interval[0] for interval in normalized):
                raise ValueError("byte intervals must be ordered half-open ranges")
            object.__setattr__(self, "intervals", normalized)
        if self.intervals is not None and self.offset is not None:
            raise ValueError("use intervals or offset/extent, not both")

    @property
    def reads(self) -> bool:
        return "R" in self.mode

    @property
    def writes(self) -> bool:
        return "W" in self.mode

    def footprint(self, iteration: int = 0) -> tuple[ByteInterval, ...] | None:
        """Physical bytes for an epoch; unknown aliases remain unknown.

        Ring storage uses ``(iteration % ring_size) * iteration_stride``.
        The adaptor may supply exact alias intervals; an alias name alone
        provides no footprint proof.
        """
        intervals = self.intervals
        if intervals is None:
            if self.offset is None:
                return None
            intervals = ((self.offset, self.offset + self.extent),)
        epoch = iteration if self.ring_size is None else iteration % self.ring_size
        shift = epoch * self.iteration_stride
        return tuple((start + shift, end + shift) for start, end in intervals)


@dataclass(frozen=True)
class Task:
    id: Any
    pipe: str
    core: str
    accesses: tuple[Access, ...] = field(default_factory=tuple)
    guard: Any = None

    def __post_init__(self):
        object.__setattr__(self, "accesses", tuple(self.accesses))


@dataclass(frozen=True, eq=False)
class Dependency:
    source: Any
    target: Any
    kind: str
    distance: int = 0
    storage: Any = field(default=None, compare=False, repr=False)

    @property
    def key(self):
        """Identity-aware key used for deduplication and validation."""
        return self.source, self.target, self.kind, self.distance, id(self.storage)

    def __eq__(self, other):
        if not isinstance(other, Dependency):
            return NotImplemented
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)
