# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Regression coverage for representation-only numeric Let casts."""

import pytest

tvm = pytest.importorskip("tvm")

from tilelang.simulator import build_kernel_program  # noqa: E402


def test_bridge_accepts_float_cast_of_resolved_integer_let() -> None:
    """Final Ascend TIR may bind a float Cast of a loop-derived index."""
    h_i = tvm.tir.Var("h_i", "int32")
    cast_index = tvm.tir.Var("cast_index", "float32")
    body = tvm.tir.LetStmt(
        h_i,
        tvm.tir.IntImm("int32", 3),
        tvm.tir.LetStmt(
            cast_index,
            tvm.tir.Cast("float32", h_i % 2 + 318),
            tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0)),
        ),
    )
    program = build_kernel_program(tvm.tir.PrimFunc([], body), platform="A3")

    assert program.tasks == ()
