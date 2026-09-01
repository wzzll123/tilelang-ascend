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


def _strided_copy_primfunc(operation: str):
    source = tvm.tir.decl_buffer((4, 8), "float32", name="source", scope="global")
    destination = tvm.tir.decl_buffer(
        (2, 8), "float32", name="destination", scope="shared.ub"
    )
    if "ub_to_gm" in operation:
        source, destination = destination, source
    call = tvm.tir.call_extern(
        "handle",
        operation,
        source.access_ptr("r", offset=2 if source.scope() == "shared.ub" else 4, extent=16),
        destination.access_ptr(
            "w", offset=2 if destination.scope() == "shared.ub" else 4, extent=16
        ),
        8,
        2,
        5,
        0,
        2,
        8,
    )
    root = tvm.tir.Block(
        [],
        [],
        [],
        "root",
        tvm.tir.Evaluate(call),
        alloc_buffers=[buffer for buffer in (source, destination) if buffer.scope() != "global"],
    )
    parameters = [buffer for buffer in (source, destination) if buffer.scope() == "global"]
    return tvm.tir.PrimFunc(
        [buffer.data for buffer in parameters],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={buffer.data: buffer for buffer in parameters},
    )


def _vector_add_primfunc():
    x = tvm.tir.decl_buffer((8,), "float32", name="x", scope="global")
    y = tvm.tir.decl_buffer((8,), "float32", name="y", scope="global")
    output = tvm.tir.decl_buffer((8,), "float32", name="output", scope="global")
    ub_x = tvm.tir.decl_buffer((8,), "float32", name="ub_x", scope="shared.ub")
    ub_y = tvm.tir.decl_buffer((8,), "float32", name="ub_y", scope="shared.ub")
    ub_output = tvm.tir.decl_buffer(
        (8,), "float32", name="ub_output", scope="shared.ub"
    )

    def copy(operation, source, destination):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle",
            operation,
            source.access_ptr("r"),
            destination.access_ptr("w"),
            8,
            1,
            8,
            0,
            1,
            8,
        ))

    body = tvm.tir.SeqStmt([
        copy("tl::ascend::copy_gm_to_ub<float32, 8>", x, ub_x),
        copy("tl::ascend::copy_gm_to_ub<float32, 8>", y, ub_y),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle",
            "tl::ascend::add",
            ub_output.access_ptr("w"),
            ub_x.access_ptr("r"),
            ub_y.access_ptr("r"),
            8,
        )),
        copy("tl::ascend::copy_ub_to_gm<float32, 8>", ub_output, output),
    ])
    root = tvm.tir.Block(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=[ub_x, ub_y, ub_output],
    )
    return tvm.tir.PrimFunc(
        [x.data, y.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={x.data: x, y.data: y, output.data: output},
    )


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
    assert task.metadata["dst"] == BufferRegion(
        "ub", MemoryScope.UB, (16,), "float32", core_id=0
    )
    assert task.metadata["copy"] == {"valid_elements": 16}


def test_real_tir_copy_executes_without_manual_task_metadata() -> None:
    program = build_kernel_program(copy_to_ub, platform="A2")
    simulator = FunctionalSimulator(program)
    source = BufferRegion("a", MemoryScope.GM, (16,), "float32")
    destination = BufferRegion("ub", MemoryScope.UB, (16,), "float32")
    values = np.arange(16, dtype=np.float32)

    simulator.write(source, values)
    simulator.run()

    np.testing.assert_array_equal(simulator.read(destination), values)


@pytest.mark.parametrize(
    "operation",
    [
        "tl::ascend::copy_gm_to_ub<float32, 8, 2>",
        "tl::ascend::copy_ub_to_gm<float32, 8, 2>",
    ],
)
def test_access_ptr_offsets_and_valid_rectangle_execute(operation: str) -> None:
    program = build_kernel_program(_strided_copy_primfunc(operation), platform="A2")
    simulator = FunctionalSimulator(program)
    task = program.tasks[0]
    source = task.metadata["src"]
    destination = task.metadata["dst"]
    assert source.shape == destination.shape == (2, 5)
    assert source.strides_bytes == destination.strides_bytes == (32, 4)
    assert task.metadata["copy"] == {
        "valid_rows": 2,
        "valid_cols": 5,
        "stride_n": 8,
        "pad_value": 0,
        "physical_rows": 2,
        "physical_cols": 8,
    }

    whole_source = BufferRegion(
        source.buffer,
        source.scope,
        (2, 8) if source.scope is MemoryScope.UB else (4, 8),
        "float32",
    )
    values = np.arange(np.prod(whole_source.shape), dtype=np.float32).reshape(
        whole_source.shape
    )
    simulator.write(whole_source, values)
    simulator.run()

    expected = values.reshape(-1)[source.byte_offset // 4:]
    expected = np.stack((expected[:5], expected[8:13]))
    np.testing.assert_array_equal(simulator.read(destination), expected)


def test_real_tir_vector_add_builds_dependencies_and_executes_end_to_end() -> None:
    program = build_kernel_program(_vector_add_primfunc(), platform="A3")
    assert [task.operation for task in program.tasks] == [
        "copy_gm_to_ub",
        "copy_gm_to_ub",
        "add",
        "copy_ub_to_gm",
    ]
    load_x, load_y, add, store = program.tasks
    assert set(add.dependencies) == {load_x.task_id, load_y.task_id}
    assert store.dependencies == (add.task_id,)

    simulator = FunctionalSimulator(program)
    x_region = BufferRegion("x", MemoryScope.GM, (8,), "float32")
    y_region = BufferRegion("y", MemoryScope.GM, (8,), "float32")
    output_region = BufferRegion("output", MemoryScope.GM, (8,), "float32")
    x = np.arange(8, dtype=np.float32)
    y = np.linspace(0.5, 4, 8, dtype=np.float32)
    simulator.write(x_region, x)
    simulator.write(y_region, y)

    result = simulator.run()

    np.testing.assert_array_equal(simulator.read(output_region), x + y)
    assert result.schedule.stats.makespan_cycles == 4


def test_real_tir_shmem_call_fails_closed() -> None:
    with pytest.raises(UnsupportedSimOpError, match="intentionally unsupported"):
        build_kernel_program(rejected_shmem, platform="A3")
