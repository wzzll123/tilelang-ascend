# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Conservative lifecycle planning for explicitly described buffer owners.

No loop discovery or TIR rewrite is performed here. A descriptor certifies its
ordered direct epoch tasks and full physical footprint. A guard_relation token
certifies equivalent active-epoch clocks, including mutable predicate snapshots
and iteration coordinates; it must come from external proof, not guard text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping

from .model import ByteInterval, Task


@dataclass(frozen=True)
class OwnerDescriptor:
    loop: Any
    tasks: tuple[Task, ...]
    stages: Mapping[Any, int]
    regular: bool = True
    conditional: bool = False
    full_footprints: tuple[tuple[Any, tuple[ByteInterval, ...]], ...] = ()
    guard_relation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(self.tasks))
        if any(task.id not in self.stages for task in self.tasks):
            raise ValueError("every direct task must have an explicit stage")
        if any(not isinstance(stage, int) or stage < 0 for stage in self.stages.values()):
            raise ValueError("stages must be nonnegative integers")


@dataclass(frozen=True)
class OwnerEpochPlan:
    loop: Any
    mode: str
    clock: Any = field(compare=False, repr=False)
    diagnostic: str = ""

    def slot(self, iteration: int, active_epoch: int, count: int) -> int:
        """Counter advances once per active owner epoch, never per buffer access."""
        if count < 1:
            raise ValueError("version count must be positive")
        coordinate = active_epoch if self.mode == "counter" else iteration
        return coordinate % count


@dataclass(frozen=True)
class EpochPlan:
    storage: Any = field(compare=False, repr=False)
    eligible: bool
    count: int
    owners: tuple[OwnerEpochPlan, ...]
    diagnostics: tuple[str, ...] = ()


def _annotation(value):
    if isinstance(value, str):
        count, mode, disabled = None, value, False
    elif isinstance(value, int) and not isinstance(value, bool):
        count, mode, disabled = value, "auto", value == 1
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        count, mode = value
        disabled = False
    else:
        raise ValueError("buffer annotation must be count, mode, or (count, mode)")
    if (not isinstance(value, str) and
        (not isinstance(count, int) or isinstance(count, bool) or count < 1)):
        raise ValueError("version count must be a positive integer")
    if mode not in ("auto", "iteration", "counter"):
        raise ValueError("version mode must be auto, iteration, or counter")
    return count, mode, disabled


def _covered(required, written):
    """Byte coverage proof; unknown required/written footprints fail closed."""
    if required is None or written is None:
        return False
    intervals = sorted((start, end) for start, end in written if start < end)
    for start, end in required:
        if end < start:
            raise ValueError("full footprint must contain ordered half-open intervals")
        cursor = start
        for left, right in intervals:
            if left > cursor:
                break
            if right > cursor:
                cursor = right
            if cursor >= end:
                break
        if cursor < end:
            return False
    return True


def _epoch_proof(storage, owner):
    touching = [
        (task, tuple(a for a in task.accesses if a.storage is storage)) for task in owner.tasks
    ]
    touching = [(task, accesses) for task, accesses in touching if accesses]
    if not touching:
        return False, "owner has no accesses to storage", ()
    first, accesses = touching[0]
    stages = tuple(owner.stages[task.id] for task, _ in touching)
    if any(access.mode != "W" for access in accesses):
        return False, "initial read or RW carries a previous epoch", stages
    if any(not access.full_overwrite for access in accesses):
        return False, "first write lacks explicit full overwrite proof", stages
    footprints = [footprint for token, footprint in owner.full_footprints if token is storage]
    if len(footprints) != 1:
        return False, "owner needs one known full physical footprint", stages
    if any(access.footprint() is None for access in accesses):
        return False, "initial write has unknown physical footprint", stages
    written = tuple(interval for access in accesses for interval in access.footprint())
    if not _covered(footprints[0], written):
        return False, "partial write does not initialize the full epoch footprint", stages
    for task, task_accesses in touching:
        if (first.guard is not None and task.guard is not first.guard and
                owner.guard_relation is None):
            return False, "conditional writer does not provably initialize every read", stages
        for access in task_accesses:
            if access.reads and not _covered(access.footprint(), written):
                return False, "read footprint is not covered by the initial overwrite", stages
    return True, "", stages


def plan_storage_epochs(storage, owners, annotations=()) -> EpochPlan:
    """Plan version clocks only after complete first-write proof for every owner.

    Alias annotation values must agree in explicit count, mode, and integer-one
    opt-out semantics; mode-only strings infer count or share an explicit count
    from an alias. Automatic conditional/irregular clocks use active epoch counters
    when accesses share one stage. Cross-stage automatic fallback is supported
    for a single owner; multiple such owners conservatively retain one version.
    """
    owners = tuple(owners)
    parsed = tuple(_annotation(value) for value in annotations)
    counts = {count for count, _, _ in parsed if count is not None}
    modes = {mode for _, mode, _ in parsed}
    disabled_values = {disabled for _, _, disabled in parsed}
    if len(counts) > 1 or len(modes) > 1 or len(disabled_values) > 1:
        raise ValueError("inconsistent buffer count/mode annotations across aliases")
    count = next(iter(counts), None)
    requested = next(iter(modes), "auto")
    disabled = next(iter(disabled_values), False)
    diagnostics, plans, proofs = [], [], []
    clocks = []
    maximum_span = 0
    for owner in owners:
        eligible, reason, stages = _epoch_proof(storage, owner)
        proofs.append(eligible)
        if reason:
            diagnostics.append(reason)
        cross_stage = len(set(stages)) > 1
        maximum_span = max(maximum_span, max(stages, default=0) - min(stages, default=0))
        if requested == "counter" and cross_stage:
            raise ValueError("counter mode requires all owner accesses at the same stage")
        mode = requested
        diagnostic = ""
        if mode == "auto":
            mode = "counter" if owner.conditional or not owner.regular else "iteration"
            if mode == "counter" and cross_stage:
                mode = "iteration"
                if len(owners) == 1:
                    diagnostic = "automatic counter fallback to iteration for cross-stage owner"
                else:
                    proofs[-1] = False
                    diagnostic = "cross-stage multiple owner clocks need external lifecycle proof"
                diagnostics.append(diagnostic)
        # Clock equivalence comes exclusively from the externally verified token.
        clock = object()
        if owner.guard_relation is not None:
            for relation, prior_mode, prior_clock in clocks:
                if relation is owner.guard_relation and mode == prior_mode:
                    clock = prior_clock
                    break
            clocks.append((owner.guard_relation, mode, clock))
        plans.append(OwnerEpochPlan(owner.loop, mode, clock, diagnostic))
    eligible = bool(owners) and all(proofs) and not disabled
    if disabled:
        diagnostics.append("integer buffer annotation 1 disables automatic versioning")
    if not owners:
        diagnostics.append("no explicit storage owner")
    return EpochPlan(storage, eligible, (count or max(2, maximum_span + 1)) if eligible else 1,
                     tuple(plans), tuple(diagnostics))
