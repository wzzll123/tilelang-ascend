"""Shared compiler/simulator report contract, without TVM initialization."""

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

_ROOT = Path(__file__).resolve().parents[3] / "tilelang/transform/ascend_schedule"
_PACKAGE = "ascend_schedule_report_test"
package = types.ModuleType(_PACKAGE)
package.__path__ = [str(_ROOT)]
sys.modules[_PACKAGE] = package
for name in ("model", "solver", "analysis_report"):
    spec = importlib.util.spec_from_file_location(f"{_PACKAGE}.{name}", _ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
model = sys.modules[f"{_PACKAGE}.model"]
solver = sys.modules[f"{_PACKAGE}.solver"]
report = sys.modules[f"{_PACKAGE}.analysis_report"]


def example():
    physical = object()
    tasks = [
        model.Task("read", "V", "vector", (model.Access(physical, "R", 0, 16),)),
        model.Task("write", "MTE2", "vector", (model.Access(physical, "W", 0, 16),))
    ]
    edges = [
        model.Dependency("read", "write", "WAR", 0, physical),
        model.Dependency("write", "read", "RAW", 1, physical),
        model.Dependency("write", "write", "WAW", 1, physical)
    ]
    costs = {"read": solver.TaskCost(2, 1, "V"), "write": solver.TaskCost(3, 1, "MTE2")}
    return tasks, edges, costs


def test_report_distinguishes_workspace_bytes_from_hardware_capacities():
    tasks, edges, costs = example()
    result = report.build_schedule_report(
        tasks,
        edges,
        costs,
        storage={"workspace": solver.StorageCost(4096, "GM", 2, 2)},
        host_workspace_budget_bytes=8192)
    assert result["capacities"] == {}
    assert result["host_workspace"] == {
        "selected_bytes": 8192,
        "budget_bytes": 8192,
        "device_available_memory_verified": False
    }


def test_injected_schedule_cannot_bypass_host_workspace_budget():
    tasks = [model.Task("t", "V", "V")]
    selected = solver.ScheduleResult((0,), (0,), (0,), {"workspace": 2}, 1)
    with pytest.raises(report.ReportValidationError, match="workspace"):
        report.build_schedule_report(
            tasks, [], {"t": solver.TaskCost(1, 1, "V")},
            storage={"workspace": solver.StorageCost(4096, "GM", 2)},
            result=selected,
            host_workspace_budget_bytes=4096)


@pytest.mark.parametrize("size", [float("nan"), 1.5, True])
def test_injected_validator_requires_finite_integer_workspace_bytes(size):
    selected = solver.ScheduleResult((0,), (0,), (0,), {"workspace": 1}, 1)
    with pytest.raises(report.ReportValidationError, match="byte"):
        report.validate_schedule_result([solver.TaskCost(1, 1, "V")], [],
                                        selected,
                                        storage={"workspace": solver.StorageCost(size, "GM")},
                                        host_workspace_budget_bytes=10)


@pytest.mark.parametrize("resource_model", ["A2", "A3"])
def test_preserve_all_dependency_kinds_and_distances(resource_model):
    tasks, edges, costs = example()
    result = report.build_schedule_report(tasks, edges, costs, resource_model=resource_model)
    assert [edge["kind"] for edge in result["dependencies"]] == ["WAR", "RAW", "WAW"]
    assert [edge["distance"] for edge in result["dependencies"]] == [0, 1, 1]
    assert [(edge["src"], edge["dst"]) for edge in result["dependencies"]] == [(0, 1), (1, 0),
                                                                               (1, 1)]
    assert result["schedule"]["selected_ii"] >= 5
    assert result["tasks"][0]["accesses"][0]["storage"] == "storage_0"
    json.dumps(result, allow_nan=False)


def test_dependency_violating_mutation_rejected():
    tasks, edges, costs = example()
    bad = solver.ScheduleResult((0, 0), (0, 0), (0, 1), {}, 5)
    with pytest.raises(report.ReportValidationError, match="WAR"):
        report.build_schedule_report(tasks, edges, costs, result=bad)


def test_symbolic_metadata_is_unknown_and_physical_identity_preserved():

    class Symbolic:

        def __repr__(self):
            raise AssertionError("symbolic repr must not be used as identity")

    a, b = Symbolic(), Symbolic()
    tasks = [
        model.Task("a", "V", "vector", (model.Access(a, "R"),), guard=a),
        model.Task("b", "CUBE", "cube", (model.Access(b, "R"),), guard=b)
    ]
    result = report.build_schedule_report(tasks, [], {
        "a": solver.TaskCost(1, 1, "V"),
        "b": solver.TaskCost(1, 1, "CUBE")
    })
    assert result["tasks"][0]["guard"]["status"] == "unknown_symbolic"
    accesses = [task["accesses"][0] for task in result["tasks"]]
    assert accesses[0]["storage"] != accesses[1]["storage"]
    assert accesses[0]["footprint"]["status"] == "unknown"
    json.dumps(result)


def test_storage_capacity_and_fixed_version_mutation_rejected():
    tasks, edges, costs = example()
    storage = {"x": solver.StorageCost(16, "UB", 2, fixed_versions=2)}
    result = report.build_schedule_report(
        tasks, edges, costs, storage=storage, capacities={"UB": 32})
    assert result["storage"][0]["footprint_bytes"] == 32
    with pytest.raises(ValueError, match="capacity"):
        report.build_schedule_report(tasks, edges, costs, storage=storage)
    bad = solver.ScheduleResult((0, 2), (0, 0), (0, 1), {"x": 1}, 5)
    with pytest.raises(report.ReportValidationError, match="fixed"):
        report.build_schedule_report(
            tasks, edges, costs, result=bad, storage=storage, capacities={"UB": 32})


def test_extraction_input_manual_stages_and_fallback_diagnostic():
    tasks, _, costs = example()
    extraction = types.SimpleNamespace(tasks=tasks, diagnostics=("source note",))
    result = report.build_schedule_report(extraction, [], costs, timeout_ms=0)
    assert result["schedule"]["fallback"]
    assert "unknown" in " ".join(result["diagnostics"])
    assert "source note" in result["diagnostics"]
    manual = report.build_schedule_report(tasks, [], costs, stages={"read": 0, "write": 1})
    assert manual["schedule"]["stages"] == [0, 1]


def test_resource_conflict_mutation_rejected():
    tasks = [model.Task("a", "V", "vector"), model.Task("b", "V", "vector")]
    costs = {name: solver.TaskCost(1, 1, "V") for name in ("a", "b")}
    bad = solver.ScheduleResult((0, 0), (0, 0), (0, 1), {}, 2)
    with pytest.raises(report.ReportValidationError, match="resource"):
        report.build_schedule_report(tasks, [], costs, result=bad)


def test_unrecognized_resource_model_rejected():
    tasks, edges, costs = example()
    with pytest.raises(ValueError, match="A2 or A3"):
        report.build_schedule_report(tasks, edges, costs, resource_model="guess")


def test_reuse_report_preserves_true_dependency_distance():
    tasks, edges, costs = example()
    result = report.build_schedule_report(
        tasks,
        edges,
        costs,
        storage={"x": solver.StorageCost(8, "L1", 2)},
        capacities={"L1": 16},
        reuse_dependencies=[solver.ReuseDependency(1, 0, "x")])
    assert result["dependencies"][1]["distance"] == 1
    assert result["reuse_dependencies"][0]["storage"] == "x"
    assert result["reuse_dependencies"][0]["distance"] == \
        result["schedule"]["buffer_versions"]["x"]


# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
