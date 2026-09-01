# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CPU-only tests for the static simulator adapter."""

from pathlib import Path

import pytest

from tilelang.simulator import CoreProgram, KernelProgram, Lane, Pipe, SimulatorConfig, Task
from tilelang.simulator.adapter import SimulatorKernelAdapter
from tilelang.simulator.errors import UnsupportedSimOpError


class _FakeModule:
    @staticmethod
    def script() -> str:
        return "optimized tir"


class _FakeParam:
    shape = ()


def test_static_adapter_schedules_and_exports_trace(tmp_path: Path) -> None:
    load = Task("load", "copy_gm_to_ub", 0, Lane.VECTOR_0, Pipe.MTE2, 4)
    add = Task(
        "add", "add", 0, Lane.VECTOR_0, Pipe.VECTOR, 5, dependencies=("load",)
    )
    program = KernelProgram("add", "A2", (CoreProgram(0, (load, add)),))
    adapter = SimulatorKernelAdapter(
        optimized_mod=_FakeModule(),
        params=[_FakeParam(), _FakeParam()],
        result_idx=-1,
        workspace_idx=None,
        config=SimulatorConfig(platform="A2", trace_path=tmp_path / "trace.json"),
        program=program,
    )

    result = adapter.schedule()

    assert result.stats.makespan_cycles == 9
    assert adapter.last_stats is result.stats
    assert adapter.last_trace == (tmp_path / "trace.json").resolve()
    assert adapter.get_kernel_source() == "optimized tir"
    assert adapter.get_simulator_ir() is program


def test_static_adapter_fails_closed_for_functional_execution() -> None:
    adapter = SimulatorKernelAdapter(
        optimized_mod=_FakeModule(),
        params=[_FakeParam()],
        result_idx=0,
        workspace_idx=None,
        config=SimulatorConfig(platform="A3"),
        program=KernelProgram("empty", "A3", (CoreProgram(0),)),
    )

    with pytest.raises(UnsupportedSimOpError, match="functional tensor execution"):
        adapter.func(object())
