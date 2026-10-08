# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CPU-only scheduling tests; importing this file does not initialize TileLang."""

import importlib.util
from pathlib import Path
import sys

import pytest

_PATH = Path(__file__).resolve().parents[3] / "tilelang/transform/ascend_schedule/solver.py"
_SPEC = importlib.util.spec_from_file_location("ascend_schedule_solver_under_test", _PATH)
solver = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = solver
_SPEC.loader.exec_module(solver)


def test_straight_pipeline():
    result = solver.solve_schedule(
        [solver.TaskCost(4, 1, "MTE2"),
         solver.TaskCost(5, 2, "CUBE")],
        [solver.Dependency(0, 1)],
    )
    assert not result.fallback
    assert result.ii == 2
    assert result.start_times[1] - result.start_times[0] >= 4
    assert result.stages == tuple(t // result.ii for t in result.start_times)


def test_carried_dependency_is_not_rewritten_by_buffer_versions():
    result = solver.solve_schedule(
        [solver.TaskCost(4, 1, "CUBE")],
        [solver.Dependency(0, 0, 1)],
        storage={"x": solver.StorageCost(16, "L1", 4)},
        capacities={"L1": 64},
    )
    assert result.ii == 4
    assert result.buffer_versions["x"] == 1


def test_reuse_versions_and_memory_capacity():
    kwargs = dict(
        tasks=[solver.TaskCost(4, 1, "MTE2"),
               solver.TaskCost(4, 1, "CUBE")],
        dependencies=[solver.Dependency(0, 1)],
        storage={"x": solver.StorageCost(16, "L1", 4)},
        reuse_dependencies=[solver.ReuseDependency(1, 0, "x")],
    )
    wide = solver.solve_schedule(**kwargs, capacities={"L1": 64})
    narrow = solver.solve_schedule(**kwargs, capacities={"L1": 16})
    assert wide.ii == 2 and wide.buffer_versions["x"] == 4
    assert narrow.ii == 8 and narrow.buffer_versions["x"] == 1
    assert wide.start_times[1] >= wide.start_times[0] + 4


def test_fixed_versions_preserved():
    result = solver.solve_schedule(
        [solver.TaskCost(3, 1, "V")],
        [],
        storage={"x": solver.StorageCost(16, "UB", 4, fixed_versions=2)},
        reuse_dependencies=[solver.ReuseDependency(0, 0, "x")],
        capacities={"UB": 32},
    )
    assert result.buffer_versions == {"x": 2}
    assert result.ii == 2


def test_resources_protect_wraparound():
    tasks = [solver.TaskCost(2, 2, "V"), solver.TaskCost(3, 3, "V")]
    result = solver.solve_schedule(tasks, [])
    assert result.ii == 5
    phases = [time % result.ii for time in result.start_times]
    assert ((phases[1] - phases[0]) % result.ii) >= 2
    assert ((phases[0] - phases[1]) % result.ii) >= 3


def test_manual_infeasibility_is_not_fallback():
    with pytest.raises(solver.ScheduleInfeasible, match="manual"):
        solver.solve_schedule(
            [solver.TaskCost(4, 1, "MTE2"),
             solver.TaskCost(1, 1, "CUBE")],
            [solver.Dependency(0, 1)],
            manual_stages=[1, 0],
            max_ii=10,
        )


def test_automatic_search_limit_has_explicit_sequential_fallback():
    result = solver.solve_schedule(
        [solver.TaskCost(4, 1, "CUBE")],
        [solver.Dependency(0, 0, 1)],
        max_ii=1,
    )
    assert result.fallback and result.ii >= 4
    assert "infeasible" in " ".join(result.diagnostics)


def test_capacity_lower_bound_and_missing_capacity_rejected():
    storage = {"x": solver.StorageCost(16, "UB", 2, fixed_versions=2)}
    with pytest.raises(solver.ScheduleInfeasible, match="capacity"):
        solver.solve_schedule([solver.TaskCost(1, 1, "V")], [],
                              storage=storage,
                              capacities={"UB": 16})
    with pytest.raises(ValueError, match="capacity"):
        solver.solve_schedule([solver.TaskCost(1, 1, "V")], [], storage=storage)


def test_missing_z3_has_actionable_fallback_and_manual_error(monkeypatch):
    monkeypatch.setattr(solver, "_load_z3", lambda: None)
    result = solver.solve_schedule([solver.TaskCost(2, 1, "V")], [])
    assert result.fallback and "z3-solver" in " ".join(result.diagnostics)
    with pytest.raises(solver.ScheduleInfeasible, match="z3-solver"):
        solver.solve_schedule([solver.TaskCost(2, 1, "V")], [], manual_stages=[0])


def test_timeout_is_unknown_not_infeasible():
    result = solver.solve_schedule([solver.TaskCost(2, 1, "V")], [], timeout_ms=0)
    assert result.fallback
    assert "unknown" in " ".join(result.diagnostics)
    assert "infeasible" not in " ".join(result.diagnostics)
    with pytest.raises(solver.ScheduleInfeasible, match="unknown"):
        solver.solve_schedule([solver.TaskCost(2, 1, "V")], [], timeout_ms=0, manual_stages=[0])


def test_valid_manual_stages_are_preserved():
    result = solver.solve_schedule(
        [solver.TaskCost(4, 1, "MTE2"),
         solver.TaskCost(1, 1, "CUBE")],
        [solver.Dependency(0, 1)],
        manual_stages=[0, 1],
    )
    assert not result.fallback and result.stages == (0, 1)
    assert result.start_times[1] - result.start_times[0] >= 4


@pytest.mark.parametrize("scope", ["L1", "UB", "L0A", "L0B", "L0C"])
def test_capacity_is_per_explicit_scope(scope):
    result = solver.solve_schedule(
        [solver.TaskCost(1, 1, 16)],
        [],
        storage={"x": solver.StorageCost(8, scope, 2)},
        capacities={scope: 8},
    )
    assert result.buffer_versions == {"x": 1}
