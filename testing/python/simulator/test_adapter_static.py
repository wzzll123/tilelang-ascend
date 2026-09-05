# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CPU-only tests for the static simulator adapter."""

from pathlib import Path

import numpy as np
import pytest

from tilelang.simulator import (
    CoreProgram,
    KernelProgram,
    Lane,
    Pipe,
    SimulatorConfig,
    Task,
    build_kernel_program,
)
from tilelang.simulator.adapter import SimulatorKernelAdapter
from tilelang.simulator.errors import UnsupportedSimOpError

tvm = pytest.importorskip("tvm")


class _FakeModule:
    @staticmethod
    def script() -> str:
        return "optimized tir"


class _FakeParam:
    def __init__(self, shape=()):
        self.shape = shape


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


def test_static_adapter_requires_final_primfunc_for_functional_execution() -> None:
    adapter = SimulatorKernelAdapter(
        optimized_mod=_FakeModule(),
        params=[_FakeParam()],
        result_idx=0,
        workspace_idx=None,
        config=SimulatorConfig(platform="A3"),
        program=KernelProgram("empty", "A3", (CoreProgram(0),)),
    )

    with pytest.raises(UnsupportedSimOpError, match="PrimFunc parameter metadata"):
        adapter.func()


def _adapter_add_primfunc():
    left = tvm.tir.decl_buffer((8,), "float32", name="left", scope="global")
    right = tvm.tir.decl_buffer((8,), "float32", name="right", scope="global")
    output = tvm.tir.decl_buffer((8,), "float32", name="output", scope="global")
    ub_left = tvm.tir.decl_buffer((8,), "float32", name="ub_left", scope="shared.ub")
    ub_right = tvm.tir.decl_buffer((8,), "float32", name="ub_right", scope="shared.ub")
    ub_output = tvm.tir.decl_buffer((8,), "float32", name="ub_output", scope="shared.ub")

    def copy(name, source, destination):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source.access_ptr("r"),
            destination.access_ptr("w"), 8,
        ))

    body = tvm.tir.SeqStmt([
        copy("copy_gm_to_ub", left, ub_left),
        copy("copy_gm_to_ub", right, ub_right),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_add", ub_output.access_ptr("w"),
            ub_left.access_ptr("r"), ub_right.access_ptr("r"), 8,
        )),
        copy("copy_ub_to_gm", ub_output, output),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body,
        alloc_buffers=[ub_left, ub_right, ub_output],
    )
    return tvm.tir.PrimFunc(
        [left.data, right.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={left.data: left, right.data: right, output.data: output},
    )


def _functional_adapter(tmp_path: Path | None = None) -> SimulatorKernelAdapter:
    function = _adapter_add_primfunc()
    config = SimulatorConfig(
        platform="A2",
        trace_path=None if tmp_path is None else tmp_path / "functional.json",
    )
    return SimulatorKernelAdapter(
        optimized_mod=function,
        params=[_FakeParam((8,)), _FakeParam((8,)), _FakeParam((8,))],
        result_idx=2,
        workspace_idx=None,
        config=config,
        program=build_kernel_program(function, platform="A2"),
    )


def test_functional_adapter_executes_numpy_inputs_and_returns_output(
    tmp_path: Path,
) -> None:
    adapter = _functional_adapter(tmp_path)
    left = np.arange(8, dtype=np.float32)
    right = np.arange(8, dtype=np.float32)[::-1].copy()

    output = adapter.func(left, right)

    assert isinstance(output, np.ndarray)
    np.testing.assert_array_equal(output, left + right)
    assert adapter.last_execution is not None
    assert adapter.last_schedule is adapter.last_execution.schedule
    assert adapter.last_trace == (tmp_path / "functional.json").resolve()


def test_functional_adapter_preserves_cpu_torch_interface() -> None:
    torch = pytest.importorskip("torch")
    adapter = _functional_adapter()
    left = torch.arange(8, dtype=torch.float32)
    right = torch.arange(8, dtype=torch.float32).flip(0)

    output = adapter.func(left, right)

    assert isinstance(output, torch.Tensor)
    assert output.device.type == "cpu"
    torch.testing.assert_close(output, left + right)
