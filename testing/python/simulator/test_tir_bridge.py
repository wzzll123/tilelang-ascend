# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Tests that exercise the bridge against real TVM TIR nodes."""

import pytest
import numpy as np

tvm = pytest.importorskip("tvm")
from tvm.script import tir as T  # noqa: E402

from tilelang.simulator import (  # noqa: E402
    BufferRegion,
    FunctionalSimulator,
    Lane,
    MemoryScope,
    Pipe,
    UnsupportedSimOpError,
    build_kernel_program,
)


@T.prim_func
def copy_to_ub(a: T.Buffer((16,), "float32")):
    ub = T.alloc_buffer((16,), "float32", scope="shared.ub")
    T.evaluate(T.call_extern("int32", "copy_gm_to_ub", a.data, ub.data, 16))


@T.prim_func
def rejected_shmem(a: T.Buffer((4,), "float32")):
    T.evaluate(T.call_extern("int32", "ascend_shmem_put_nbi", a.data, 4))


def test_real_tir_primfunc_builds_program_and_local_buffer() -> None:
    program = build_kernel_program(copy_to_ub, platform="A2")

    assert program.platform == "A2"
    assert [(buffer.name, buffer.scope, buffer.shape) for buffer in program.buffers] == [
        ("a", MemoryScope.GM, (16,)),
        ("ub", MemoryScope.UB, (16,)),
    ]
    assert len(program.tasks) == 1
    task = program.tasks[0]
    assert (task.operation, task.lane, task.pipe) == (
        "copy_gm_to_ub",
        Lane.CUBE,
        Pipe.MTE2,
    )
    assert task.metadata["arguments"][-1] == 16
    assert task.metadata["src"] == BufferRegion("a", MemoryScope.GM, (16,), "float32")
    assert task.metadata["dst"] == BufferRegion("ub", MemoryScope.UB, (16,), "float32")


def test_real_tir_copy_executes_without_manual_task_metadata() -> None:
    program = build_kernel_program(copy_to_ub, platform="A2")
    simulator = FunctionalSimulator(program)
    source = BufferRegion("a", MemoryScope.GM, (16,), "float32")
    destination = BufferRegion("ub", MemoryScope.UB, (16,), "float32")
    values = np.arange(16, dtype=np.float32)

    simulator.write(source, values)
    simulator.run()

    np.testing.assert_array_equal(simulator.read(destination), values)


def test_real_tir_shmem_call_fails_closed() -> None:
    with pytest.raises(UnsupportedSimOpError, match="intentionally unsupported"):
        build_kernel_program(rejected_shmem, platform="A3")
