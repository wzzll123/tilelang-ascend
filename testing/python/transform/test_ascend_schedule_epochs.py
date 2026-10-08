# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Explicit owner lifecycle planning, without TVM or a device runtime."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_ascend_schedule_epoch_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "tilelang/transform/ascend_schedule")]
sys.modules[PACKAGE] = package


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.{name}", ROOT / f"tilelang/transform/ascend_schedule/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


model = load_module("model")
ownership = load_module("ownership")
Access, Task = model.Access, model.Task
OwnerDescriptor = ownership.OwnerDescriptor
plan = ownership.plan_storage_epochs


def owner(storage, **kwargs):
    tasks = (Task("write", "MTE2", "vector",
                  (Access(storage, "W", offset=0, extent=16, full_overwrite=True),)),
             Task("read", "V", "vector", (Access(storage, "R", offset=0, extent=16),)))
    stages = kwargs.pop("stages", {"write": 0, "read": 0})
    return OwnerDescriptor(
        object(), tasks, stages, full_footprints=((storage, ((0, 16),)),), **kwargs)


def test_regular_and_conditional_skipped_epochs():
    storage = object()
    assert plan(storage, (owner(storage),)).owners[0].mode == "iteration"
    result = plan(storage, (owner(storage, conditional=True),), ((3, "auto"),))
    epoch = result.owners[0]
    assert result.eligible and epoch.mode == "counter"
    counter, slots = 0, []
    for active in [True, False, False, True, True, False, True]:
        if active:
            slots.append(epoch.slot(iteration=99, active_epoch=counter, count=result.count))
            counter += 1
    assert slots == [0, 1, 2, 0]


def test_two_owners_require_explicit_clock_equivalence():
    storage = object()
    a, b = owner(storage, conditional=True), owner(storage, conditional=True)
    result = plan(storage, (a, b))
    assert result.owners[0].clock is not result.owners[1].clock
    relation = object()
    result = plan(storage, (owner(storage, conditional=True, guard_relation=relation),
                            owner(storage, conditional=True, guard_relation=relation)))
    assert result.owners[0].clock is result.owners[1].clock


def test_opaque_guards_not_coerced_or_compared():

    class Opaque:

        def __bool__(self):
            raise AssertionError("opaque guard truthiness")

        def __eq__(self, other):
            raise AssertionError("opaque guard equality")

        def __repr__(self):
            raise AssertionError("opaque guard representation")

    storage = object()
    result = plan(
        storage, (owner(storage, guard_relation=Opaque()), owner(storage, guard_relation=Opaque())))
    assert result.owners[0].clock is not result.owners[1].clock


def test_cross_stage_counter_rejects_and_auto_falls_back():
    storage = object()
    descriptor = owner(storage, conditional=True, stages={"write": 0, "read": 1})
    with pytest.raises(ValueError, match="counter.*stage"):
        plan(storage, (descriptor,), ((2, "counter"),))
    result = plan(storage, (descriptor,))
    assert result.eligible and result.owners[0].mode == "iteration"
    assert "fallback" in result.owners[0].diagnostic


def test_multiple_cross_stage_conditional_owners_conservatively_opt_out():
    storage = object()
    descriptors = tuple(
        owner(storage, conditional=True, stages={
            "write": 0,
            "read": 1
        }) for _ in range(2))
    result = plan(storage, descriptors)
    assert not result.eligible and result.count == 1


def test_changed_writer_predicate_needs_external_initialization_proof():
    storage = object()
    writer_guard, reader_guard = object(), object()
    tasks = (Task("write", "MTE2", "vector",
                  (Access(storage, "W", offset=0, extent=16, full_overwrite=True),), writer_guard),
             Task("read", "V", "vector", (Access(storage, "R", offset=0, extent=16),),
                  reader_guard))
    descriptor = OwnerDescriptor(
        object(),
        tasks, {
            "write": 0,
            "read": 0
        },
        conditional=True,
        full_footprints=((storage, ((0, 16),)),))
    assert not plan(storage, (descriptor,)).eligible


@pytest.mark.parametrize("first_mode", ["R", "RW"])
def test_initial_carry_opts_out(first_mode):
    storage = object()
    task = Task("carry", "V", "vector",
                (Access(storage, first_mode, offset=0, extent=16, full_overwrite=True),))
    descriptor = OwnerDescriptor(
        object(), (task,), {"carry": 0}, full_footprints=((storage, ((0, 16),)),))
    assert not plan(storage, (descriptor,)).eligible


def test_partial_write_cannot_claim_full_initialization():
    storage = object()
    tasks = (Task("partial", "MTE2", "vector",
                  (Access(storage, "W", offset=0, extent=4, full_overwrite=True),)),
             Task("read", "V", "vector", (Access(storage, "R", offset=8, extent=8),)))
    descriptor = OwnerDescriptor(
        object(), tasks, {
            "partial": 0,
            "read": 0
        }, full_footprints=((storage, ((0, 16),)),))
    assert not plan(storage, (descriptor,)).eligible


def test_annotations_alias_consistency_and_one_opt_out():
    storage = object()
    descriptors = (owner(storage),)
    assert not plan(storage, descriptors, (1,)).eligible
    assert plan(storage, descriptors, ((1, "auto"),)).eligible
    with pytest.raises(ValueError, match="inconsistent"):
        plan(storage, descriptors, (1, (1, "auto")))
    with pytest.raises(ValueError, match="inconsistent"):
        plan(storage, descriptors, ((2, "auto"), (3, "auto")))
    with pytest.raises(ValueError, match="inconsistent"):
        plan(storage, descriptors, ((2, "counter"), (2, "iteration")))


@pytest.mark.parametrize("mode", ["auto", "iteration", "counter"])
def test_mode_only_annotations_infer_count(mode):
    storage = object()
    result = plan(storage, (owner(storage),), (mode,))
    assert result.eligible and result.count == 2
    assert result.owners[0].mode == ("iteration" if mode == "auto" else mode)


def test_mode_only_alias_can_share_explicit_matching_count():
    storage = object()
    result = plan(storage, (owner(storage),), ("counter", (3, "counter")))
    assert result.eligible and result.count == 3
    with pytest.raises(ValueError, match="inconsistent"):
        plan(storage, (owner(storage),), ("counter", (3, "iteration")))
    with pytest.raises(ValueError, match="mode"):
        plan(storage, (owner(storage),), ("unsupported",))


def test_irregular_owner_uses_counter_and_unknown_footprint_opts_out():
    storage = object()
    assert plan(storage, (owner(storage, regular=False),)).owners[0].mode == "counter"
    descriptor = OwnerDescriptor(object(), (Task("write", "MTE2", "vector",
                                                 (Access(storage, "W", full_overwrite=True),)),),
                                 {"write": 0})
    assert not plan(storage, (descriptor,)).eligible
