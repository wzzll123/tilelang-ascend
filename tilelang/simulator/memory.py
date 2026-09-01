# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Byte-addressed A2/A3 functional memory model."""

from dataclasses import dataclass
from itertools import product
from numbers import Integral
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .errors import (
    MemoryAccessError,
    MemoryBoundsError,
    MemoryCapacityError,
    ProgramValidationError,
    UninitializedMemoryError,
)
from .hazard import HazardDiagnostic, HazardReporter
from .program import BufferSpec, KernelProgram, MemoryScope


# Values used by the A2/A3 host codegen and memory-planning pass.
A2_A3_LOCAL_CAPACITIES: Mapping[MemoryScope, int] = {
    MemoryScope.L1: 524032,
    MemoryScope.L0A: 65536,
    MemoryScope.L0B: 65536,
    MemoryScope.L0C: 131072,
    MemoryScope.UB: 196352,
    MemoryScope.BT: 1024,
}
_SHARED_SCOPES = frozenset({MemoryScope.GM, MemoryScope.WORKSPACE})
_DTYPE_PATTERN = re.compile(r"^(?:u?int|float|bfloat)(\d+)(?:x(\d+))?$")


def dtype_size_bytes(dtype: str) -> int:
    """Return the byte width of one scalar/vector element."""
    normalized = dtype.strip().lower()
    if normalized == "bool":
        return 1
    match = _DTYPE_PATTERN.fullmatch(normalized)
    if match is None:
        raise ProgramValidationError(f"unsupported simulator dtype: {dtype!r}")
    bits, lanes = int(match.group(1)), int(match.group(2) or 1)
    if bits <= 0 or bits % 8:
        raise ProgramValidationError(f"dtype must use a positive whole-byte width: {dtype!r}")
    return bits // 8 * lanes


def contiguous_strides_bytes(shape: Sequence[int], itemsize: int) -> Tuple[int, ...]:
    """Return C-contiguous byte strides for ``shape``."""
    stride = itemsize
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


@dataclass(frozen=True, order=True)
class AddressRange:
    """A half-open byte interval in one simulator address space."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise MemoryBoundsError(f"invalid address range [{self.start}, {self.end})")

    @property
    def size(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "AddressRange") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class MemoryView:
    """A typed, strided view into a named allocation."""

    allocation: "MemoryAllocation"
    byte_offset: int
    shape: Tuple[int, ...]
    dtype: str
    strides_bytes: Tuple[int, ...]

    @property
    def itemsize(self) -> int:
        return dtype_size_bytes(self.dtype)

    @property
    def byte_range(self) -> AddressRange:
        """Return the bounding physical span (including any stride gaps)."""
        start = self.allocation.address + self.byte_offset
        if any(extent == 0 for extent in self.shape):
            return AddressRange(start, start)
        last = sum((extent - 1) * stride
                   for extent, stride in zip(self.shape, self.strides_bytes))
        return AddressRange(start, start + last + self.itemsize)

    @property
    def nbytes(self) -> int:
        """Return logical payload bytes, excluding stride gaps."""
        elements = 1
        for extent in self.shape:
            elements *= extent
        return elements * self.itemsize

    @property
    def address_ranges(self) -> Tuple[AddressRange, ...]:
        """Return exact merged physical ranges touched by the view."""
        if any(extent == 0 for extent in self.shape):
            return ()
        start = self.allocation.address + self.byte_offset
        ranges = []
        for indices in product(*(range(extent) for extent in self.shape)):
            element_start = start + sum(index * stride
                                        for index, stride in zip(indices, self.strides_bytes))
            current = AddressRange(element_start, element_start + self.itemsize)
            if ranges and ranges[-1].end == current.start:
                ranges[-1] = AddressRange(ranges[-1].start, current.end)
            else:
                ranges.append(current)
        return tuple(ranges)


class _AddressSpace:
    """Shared byte backing for one ``(scope, core)`` physical address space."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.initialized = bytearray()

    def ensure_size(self, size_bytes: int) -> None:
        missing = size_bytes - len(self.data)
        if missing > 0:
            self.data.extend(b"\xff" * missing)
            self.initialized.extend(b"\x00" * missing)


class MemoryAllocation:
    """A named view over a shared address space with poison tracking."""

    def __init__(
        self,
        spec: BufferSpec,
        size_bytes: int,
        address: int,
        core_id: Optional[int],
        reporter: HazardReporter,
        backing: _AddressSpace,
    ) -> None:
        self.spec = spec
        self.size_bytes = size_bytes
        self.address = address
        self.core_id = core_id
        self._reporter = reporter
        self._backing = backing
        self._backing.ensure_size(address + size_bytes)

    @property
    def address_range(self) -> AddressRange:
        return AddressRange(self.address, self.address + self.size_bytes)

    def view(
        self,
        *,
        byte_offset: int = 0,
        shape: Optional[Sequence[int]] = None,
        dtype: Optional[str] = None,
        strides_bytes: Optional[Sequence[int]] = None,
    ) -> MemoryView:
        """Create a view after validating rank and physical byte bounds."""
        view_dtype = dtype or self.spec.dtype
        view_shape = _concrete_shape(shape if shape is not None else self.spec.shape)
        strides = (contiguous_strides_bytes(view_shape, dtype_size_bytes(view_dtype))
                   if strides_bytes is None else tuple(strides_bytes))
        if len(strides) != len(view_shape):
            raise MemoryBoundsError("view shape and strides must have the same rank")
        if byte_offset < 0 or any(stride < 0 for stride in strides):
            raise MemoryBoundsError("negative view offsets and strides are unsupported")
        view = MemoryView(self, byte_offset, view_shape, view_dtype, strides)
        if view.byte_range.end > self.address + self.size_bytes:
            relative_end = view.byte_range.end - self.address
            raise MemoryBoundsError(
                f"view of buffer {self.spec.name!r} reaches byte {relative_end}, "
                f"allocation size is {self.size_bytes}"
            )
        return view

    def read(self, target: Union[MemoryView, AddressRange]) -> bytes:
        """Read a contiguous physical range and report uninitialized bytes."""
        ranges = self._resolve_ranges(target)
        missing = [i for interval in ranges for i in range(interval.start, interval.end)
                   if not self._backing.initialized[i]]
        if missing:
            first, end = missing[0], missing[-1] + 1
            self._reporter.report(
                HazardDiagnostic(
                    "read-before-write",
                    f"read-before-write in buffer {self.spec.name!r}, bytes [{first}, {end})",
                    self.spec.name,
                    self.core_id,
                    first,
                    end,
                ),
                error_type=UninitializedMemoryError,
            )
        return b"".join(
            bytes(self._backing.data[interval.start:interval.end]) for interval in ranges
        )

    def write(self, target: Union[MemoryView, AddressRange], data: bytes) -> None:
        """Write a contiguous physical range and mark it initialized."""
        ranges = self._resolve_ranges(target)
        payload = bytes(data)
        expected = sum(interval.size for interval in ranges)
        if len(payload) != expected:
            raise MemoryBoundsError(
                f"write to buffer {self.spec.name!r} expects {expected} bytes, "
                f"got {len(payload)}"
            )
        offset = 0
        for interval in ranges:
            next_offset = offset + interval.size
            self._backing.data[interval.start:interval.end] = payload[offset:next_offset]
            self._backing.initialized[interval.start:interval.end] = b"\x01" * interval.size
            offset = next_offset

    def initialized(self, target: Optional[Union[MemoryView, AddressRange]] = None) -> bool:
        """Return whether all bytes in ``target`` have been written."""
        ranges = ((self.address_range,) if target is None
                  else self._resolve_ranges(target))
        return all(
            all(self._backing.initialized[interval.start:interval.end]) for interval in ranges
        )

    def _resolve_ranges(
        self, target: Union[MemoryView, AddressRange]
    ) -> Tuple[AddressRange, ...]:
        absolute_ranges = target.address_ranges if isinstance(target, MemoryView) else (target,)
        result = []
        for absolute in absolute_ranges:
            if absolute.start < self.address or absolute.end > self.address + self.size_bytes:
                raise MemoryBoundsError(
                    f"access [{absolute.start}, {absolute.end}) escapes buffer "
                    f"{self.spec.name!r} at [{self.address}, {self.address + self.size_bytes})"
                )
            result.append(absolute)
        return tuple(result)


class MemoryRuntime:
    """Own shared GM/workspace and per-core local address spaces."""

    def __init__(
        self,
        core_ids: Iterable[int],
        *,
        hazard_check: str = "error",
        local_capacities: Optional[Mapping[MemoryScope, int]] = None,
    ) -> None:
        ids = tuple(sorted(set(core_ids)))
        if any(core_id < 0 for core_id in ids):
            raise ProgramValidationError("core IDs must not be negative")
        self.core_ids = ids
        self.reporter = HazardReporter(hazard_check)
        self.local_capacities = dict(local_capacities or A2_A3_LOCAL_CAPACITIES)
        self._allocations: Dict[Tuple[MemoryScope, Optional[int], str], MemoryAllocation] = {}
        self._next_address: Dict[Tuple[MemoryScope, Optional[int]], int] = {}
        self._address_spaces: Dict[Tuple[MemoryScope, Optional[int]], _AddressSpace] = {}

    @classmethod
    def from_program(
        cls, program: KernelProgram, *, hazard_check: str = "error"
    ) -> "MemoryRuntime":
        """Instantiate program buffers according to their sharing scope."""
        runtime = cls((core.core_id for core in program.cores), hazard_check=hazard_check)
        for spec in program.buffers:
            if spec.scope in _SHARED_SCOPES:
                runtime.allocate(spec)
            else:
                for core_id in runtime.core_ids:
                    runtime.allocate(spec, core_id=core_id)
        return runtime

    def allocate(
        self,
        spec: BufferSpec,
        *,
        core_id: Optional[int] = None,
        address: Optional[int] = None,
    ) -> MemoryAllocation:
        """Allocate a buffer, enforcing local capacity and overlap policy."""
        owner = self._normalize_owner(spec.scope, core_id)
        key = (spec.scope, owner, spec.name)
        if key in self._allocations:
            raise ProgramValidationError(f"duplicate simulator allocation: {spec.name!r}")
        size_bytes = _buffer_size_bytes(spec)
        space = (spec.scope, owner)
        if address is not None and spec.address is not None and address != spec.address:
            raise ProgramValidationError(
                f"allocation address {address} disagrees with BufferSpec address "
                f"{spec.address} for {spec.name!r}"
            )
        explicit_address = spec.address if address is None else address
        base = self._next_address.get(space, 0) if explicit_address is None else explicit_address
        interval = AddressRange(base, base + size_bytes)
        capacity = self.local_capacities.get(spec.scope)
        if capacity is not None and interval.end > capacity:
            raise MemoryCapacityError(
                f"{spec.scope.value} capacity exceeded on core {owner}: allocation "
                f"{spec.name!r} ends at {interval.end}, capacity is {capacity} bytes"
            )
        for (scope, existing_owner, _), existing in self._allocations.items():
            if (scope == spec.scope and existing_owner == owner
                    and interval.overlaps(existing.address_range)
                    and not _overlap_is_declared_reuse(spec, existing.spec)):
                self.reporter.report(HazardDiagnostic(
                    "overlapping-allocation",
                    f"allocation {spec.name!r} overlaps {existing.spec.name!r} in "
                    f"{spec.scope.value} on core {owner}",
                    spec.name,
                    owner,
                    interval.start,
                    interval.end,
                ))
        backing = self._address_spaces.setdefault(space, _AddressSpace())
        allocation = MemoryAllocation(spec, size_bytes, base, owner, self.reporter, backing)
        self._allocations[key] = allocation
        self._next_address[space] = max(self._next_address.get(space, 0), interval.end)
        return allocation

    def get(
        self, name: str, *, scope: MemoryScope, core_id: Optional[int] = None
    ) -> MemoryAllocation:
        """Resolve an allocation by name and address-space owner."""
        owner = self._normalize_owner(scope, core_id)
        try:
            return self._allocations[(scope, owner, name)]
        except KeyError as error:
            raise MemoryAccessError(
                f"unknown simulator allocation {name!r} in {scope.value} on core {owner}"
            ) from error

    def _normalize_owner(
        self, scope: MemoryScope, core_id: Optional[int]
    ) -> Optional[int]:
        if not isinstance(scope, MemoryScope):
            scope = MemoryScope.parse(str(scope))
        if scope in _SHARED_SCOPES:
            return None
        if core_id is None:
            raise ProgramValidationError(f"core_id is required for local scope {scope.value}")
        if core_id < 0:
            raise ProgramValidationError("core_id must not be negative")
        if core_id not in self.core_ids:
            raise ProgramValidationError(f"core_id {core_id} is not part of this runtime")
        return core_id


def _concrete_shape(shape: Sequence[object]) -> Tuple[int, ...]:
    if any(not isinstance(extent, Integral) for extent in shape):
        raise ProgramValidationError("memory allocation/view shape must be concrete")
    result = tuple(int(extent) for extent in shape)
    if any(extent < 0 for extent in result):
        raise ProgramValidationError("memory allocation/view shape has a negative extent")
    return result


def _buffer_size_bytes(spec: BufferSpec) -> int:
    if spec.size_bytes is not None:
        return spec.size_bytes
    size = dtype_size_bytes(spec.dtype)
    for extent in _concrete_shape(spec.shape):
        size *= extent
    return size


def _overlap_is_declared_reuse(left: BufferSpec, right: BufferSpec) -> bool:
    """Return whether two overlapping planned allocations may share storage."""
    left_alias = left.metadata.get("alias_of")
    right_alias = right.metadata.get("alias_of")
    if left_alias == right.name or right_alias == left.name:
        return True
    if left.lifetime is None or right.lifetime is None:
        return False
    left_start, left_end = left.lifetime
    right_start, right_end = right.lifetime
    return left_end <= right_start or right_end <= left_start
