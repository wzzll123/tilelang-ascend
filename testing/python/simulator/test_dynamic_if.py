# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Data-dependent scalar control-flow coverage for the A2/A3 simulator."""

import numpy as np
import pytest

tvm = pytest.importorskip("tvm")
from tvm import tir  # noqa: E402

from tilelang.simulator import (  # noqa: E402
    BufferRegion,
    FunctionalSimulator,
    MemoryScope,
    SimulatorConfig,
    SimulatorConfigError,
    UnsupportedSimOpError,
    build_kernel_program,
)


def _dynamic_if_program():
    source = tir.decl_buffer((1,), "float32", name="source")
    output = tir.decl_buffer((1,), "float32", name="output")
    ub_var = tir.Var(
        "ub",
        tvm.ir.PointerType(tvm.ir.PrimType("float32"), "shared.ub"),
    )
    ub = tir.decl_buffer((1,), "float32", data=ub_var, scope="shared.ub", name="ub")

    def call(name, *arguments):
        return tir.Evaluate(tir.call_extern("int32", name, *arguments))

    copy_in = call("copy_gm_to_ub", source.access_ptr("r"), ub.access_ptr("w"), 1)
    then_fill = call("fill", ub.access_ptr("w"), tir.FloatImm("float32", 7), 1)
    else_fill = call("fill", ub.access_ptr("w"), tir.FloatImm("float32", 9), 1)
    copy_out = call("copy_ub_to_gm", ub.access_ptr("r"), output.access_ptr("w"), 1)
    body = tir.SeqStmt([copy_in, tir.IfThenElse(ub[0] < tir.FloatImm("float32", 0), then_fill, else_fill), copy_out])
    body = tir.Allocate(ub_var, "float32", [1], tir.const(True, "bool"), body)
    return tir.PrimFunc([source.data, output.data], body, buffer_map={source.data: source, output.data: output})


def _run(value: float, config: SimulatorConfig):
    program = build_kernel_program(_dynamic_if_program(), platform="A2")
    simulator = FunctionalSimulator(program, config)
    simulator.write(BufferRegion("source", MemoryScope.GM, (1,), "float32"), np.array([value], dtype=np.float32))
    result = simulator.run()
    return simulator, result


def test_full_mode_evaluates_ub_scalar_condition() -> None:
    negative, _ = _run(-1, SimulatorConfig(platform="A2"))
    positive, _ = _run(1, SimulatorConfig(platform="A2"))
    output = BufferRegion("output", MemoryScope.GM, (1,), "float32")
    assert negative.read(output).item() == 7
    assert positive.read(output).item() == 9


@pytest.mark.parametrize("choice", ["then", "else"])
def test_sync_only_can_force_each_dynamic_branch(choice: str) -> None:
    _, result = _run(1, SimulatorConfig(platform="A2", sync_only=True, dynamic_if=choice))
    assert result.numeric_results_available is False


def test_sync_only_dynamic_if_defaults_to_fail_closed() -> None:
    with pytest.raises(UnsupportedSimOpError, match="dynamic_if='then' or 'else'"):
        _run(1, SimulatorConfig(platform="A2", sync_only=True))


def test_dynamic_if_config_validation() -> None:
    with pytest.raises(SimulatorConfigError, match="dynamic_if"):
        SimulatorConfig(dynamic_if="both")
