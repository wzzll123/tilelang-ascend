# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Dependency model tests run without loading TileLang or TVM."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_ascend_schedule_dependency_tests"
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
dependency = load_module("dependency")
Access, Task = model.Access, model.Task
analyze = dependency.analyze_dependencies


def task(name, storage, mode, **kwargs):
    return Task(name, "MTE2", "vector", (Access(storage, mode, **kwargs),))


def edges(tasks, **kwargs):
    return {(d.source, d.target, d.kind, d.distance) for d in analyze(tasks, **kwargs)}


@pytest.mark.parametrize("first,second,kind", [("W", "R", "RAW"), ("R", "W", "WAR"),
                                               ("W", "W", "WAW")])
def test_same_iteration_hazards(first, second, kind):
    storage = object()
    assert edges([task("a", storage, first), task("b", storage, second)],
                 max_distance=0) == {("a", "b", kind, 0)}


def test_read_read_and_disjoint_bytes_have_no_dependency():
    storage = object()
    assert not edges([task("a", storage, "R"), task("b", storage, "R")])
    assert not edges([
        task("a", storage, "W", offset=0, extent=16),
        task("b", storage, "R", offset=16, extent=16)
    ],
                     max_distance=0)


def test_rw_overlap_emits_all_hazards_and_storage_keys_stay_distinct():
    first, second = object(), object()
    tasks = [
        Task("a", "V", "vector", (Access(first, "RW"), Access(second, "W"))),
        Task("b", "MTE2", "vector", (Access(first, "RW"), Access(second, "R")))
    ]
    result = analyze(tasks, max_distance=0)
    assert len(result) == 4
    assert len(set(result)) == 4
    assert {edge.kind for edge in result if edge.storage is first} == {"RAW", "WAR", "WAW"}


def test_partial_write_keeps_prior_producer_edges():
    storage = object()
    tasks = [
        task("full", storage, "W", offset=0, extent=16, owner="outer"),
        task("partial", storage, "W", offset=0, extent=4, owner="inner"),
        task("read", storage, "R", offset=0, extent=16)
    ]
    result = edges(tasks, max_distance=0)
    assert ("full", "read", "RAW", 0) in result
    assert ("partial", "read", "RAW", 0) in result
    assert not tasks[1].accesses[0].full_overwrite


def test_unknown_mismatched_linear_stride_is_conservative():
    storage = object()
    assert ("a", "b", "RAW", 1) in edges([
        task("a", storage, "W", offset=100, extent=4, iteration_stride=4),
        task("b", storage, "R", offset=0, extent=4, iteration_stride=8),
    ])


def test_physical_identity_not_equal_names():

    class Storage:

        def __eq__(self, other):
            return True

    assert not edges([task("a", Storage(), "W"), task("b", Storage(), "R")], max_distance=0)
    storage = Storage()
    assert ("a", "b", "RAW", 0) in edges([task("a", storage, "W"), task("b", storage, "R")])


def test_alias_unknown_and_interval_union():
    storage = object()
    assert ("a", "b", "RAW", 0) in edges(
        [task("a", storage, "W", alias="view"),
         task("b", storage, "R", offset=99, extent=1)])
    assert not edges([
        task("a", storage, "W", intervals=((0, 4), (8, 12))),
        task("b", storage, "R", offset=4, extent=4)
    ],
                     max_distance=0)


def test_opposite_guards_keep_cross_iteration_reuse():
    storage = object()
    tasks = [
        Task("read", "V", "vector", (Access(storage, "R"),), guard="p"),
        Task("write", "MTE2", "vector", (Access(storage, "W"),), guard="!p")
    ]
    assert ("read", "write", "WAR", 0) in edges(tasks)
    assert ("write", "read", "RAW", 1) in edges(tasks)


def test_ring_reuse_distance_and_initial_read():
    storage = object()
    tasks = [
        task("read", storage, "R", offset=0, extent=16, iteration_stride=16, ring_size=3),
        task("write", storage, "W", offset=0, extent=16, iteration_stride=16, ring_size=3)
    ]
    result = edges(tasks, max_distance=3)
    assert ("read", "write", "WAR", 0) in result
    assert ("write", "read", "RAW", 3) in result
    assert not any(distance in (1, 2) for _, _, _, distance in result)
    assert ("write", "read", "RAW", 3) in edges(tasks)


def test_nonring_translation_and_partial_overwrite():
    storage = object()
    tasks = [
        task("a", storage, "W", offset=0, extent=4, iteration_stride=8),
        task("b", storage, "R", offset=8, extent=4, iteration_stride=8)
    ]
    assert ("b", "a", "WAR", 1) in edges(tasks)
    assert not tasks[0].accesses[0].full_overwrite


def test_edge_removal_mutation_is_detected():
    storage = object()
    tasks = [task("a", storage, "W"), task("b", storage, "R")]
    dependencies = analyze(tasks)
    dependency.validate_dependencies(tasks, dependencies)
    with pytest.raises(ValueError, match="missing dependency"):
        dependency.validate_dependencies(tasks, dependencies[1:])


@pytest.mark.parametrize("kwargs", [{
    "mode": "invalid"
}, {
    "mode": "R",
    "extent": -1
}, {
    "mode": "W",
    "ring_size": 0
}, {
    "mode": "R",
    "intervals": ((4, 2),)
}])
def test_reject_invalid_access(kwargs):
    with pytest.raises(ValueError):
        Access(object(), **kwargs)
