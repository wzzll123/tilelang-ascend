# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Functional vertical-slice tests for the A2/A3 simulator."""

import numpy as np
import pytest

from tilelang.simulator import (
    AffineInt,
    BufferRegion,
    BufferSpec,
    CoreProgram,
    FunctionalSimulator,
    KernelProgram,
    Lane,
    MemoryScope,
    Pipe,
    ProgramValidationError,
    SymbolicInt,
    Task,
    UninitializedMemoryError,
)


def _region(
    name: str,
    scope: MemoryScope,
    length: int,
    *,
    offset: int = 0,
) -> BufferRegion:
    return BufferRegion(name, scope, (length,), "float32", byte_offset=offset * 4)


def _core_tasks(core_id: int, length: int, offset: int) -> CoreProgram:
    source_x = _region("x", MemoryScope.GM, length, offset=offset)
    source_y = _region("y", MemoryScope.GM, length, offset=offset)
    destination = _region("out", MemoryScope.GM, length, offset=offset)
    ub_x = _region("ub_x", MemoryScope.UB, length)
    ub_y = _region("ub_y", MemoryScope.UB, length)
    ub_out = _region("ub_out", MemoryScope.UB, length)
    prefix = f"c{core_id}"
    return CoreProgram(core_id, (
        Task(f"{prefix}-load-x", "copy_gm_to_ub", core_id, Lane.VECTOR_0, Pipe.MTE2, 4,
             metadata={"src": source_x, "dst": ub_x}),
        Task(f"{prefix}-load-y", "copy_gm_to_ub", core_id, Lane.VECTOR_0, Pipe.MTE2, 4,
             metadata={"src": source_y, "dst": ub_y}),
        Task(f"{prefix}-add", "add", core_id, Lane.VECTOR_0, Pipe.VECTOR, 3,
             dependencies=(f"{prefix}-load-x", f"{prefix}-load-y"),
             metadata={"lhs": ub_x, "rhs": ub_y, "dst": ub_out}),
        Task(f"{prefix}-store", "copy_ub_to_gm", core_id, Lane.VECTOR_0, Pipe.MTE3, 4,
             dependencies=(f"{prefix}-add",),
             metadata={"src": ub_out, "dst": destination}),
    ))


def test_gm_ub_vector_add_gm_with_two_aiv_cores_and_tail() -> None:
    program = KernelProgram(
        "vector-add-tail",
        "A2",
        (_core_tasks(0, 10, 0), _core_tasks(1, 9, 10)),
        buffers=(
            BufferSpec("x", MemoryScope.GM, (19,), "float32"),
            BufferSpec("y", MemoryScope.GM, (19,), "float32"),
            BufferSpec("out", MemoryScope.GM, (19,), "float32"),
            BufferSpec("ub_x", MemoryScope.UB, (10,), "float32"),
            BufferSpec("ub_y", MemoryScope.UB, (10,), "float32"),
            BufferSpec("ub_out", MemoryScope.UB, (10,), "float32"),
        ),
    )
    simulator = FunctionalSimulator(program)
    whole_x = BufferRegion("x", MemoryScope.GM, (19,), "float32")
    whole_y = BufferRegion("y", MemoryScope.GM, (19,), "float32")
    whole_out = BufferRegion("out", MemoryScope.GM, (19,), "float32")
    x = np.arange(19, dtype=np.float32)
    y = np.linspace(1, 2, 19, dtype=np.float32)
    simulator.write(whole_x, x)
    simulator.write(whole_y, y)

    result = simulator.run()

    np.testing.assert_array_equal(simulator.read(whole_out), x + y)
    assert result.schedule.stats.makespan_cycles == 15
    assert {record.core_id for record in result.schedule.records} == {0, 1}


def test_copy_reports_read_before_write() -> None:
    source = _region("x", MemoryScope.GM, 4)
    destination = _region("ub", MemoryScope.UB, 4)
    program = KernelProgram(
        "poison-copy",
        "A3",
        (CoreProgram(0, (
            Task("copy", "copy_gm_to_ub", 0, Lane.VECTOR_0, Pipe.MTE2, 1,
                 metadata={"src": source, "dst": destination}),
        )),),
        buffers=(
            BufferSpec("x", MemoryScope.GM, (4,), "float32"),
            BufferSpec("ub", MemoryScope.UB, (4,), "float32"),
        ),
    )

    with pytest.raises(UninitializedMemoryError, match="read-before-write"):
        FunctionalSimulator(program).run()


def test_binary_task_requires_explicit_operands() -> None:
    program = KernelProgram(
        "invalid-add",
        "A2",
        (CoreProgram(0, (
            Task("add", "add", 0, Lane.VECTOR_0, Pipe.VECTOR, 1),
        )),),
    )

    with pytest.raises(ProgramValidationError, match="requires BufferRegion"):
        FunctionalSimulator(program).run()


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("abs", lambda value: np.abs(value)),
        ("exp", lambda value: np.exp(value)),
        ("ln", lambda value: np.log(value)),
        ("reciprocal", lambda value: 1 / value),
        ("relu", lambda value: np.maximum(value, 0)),
        ("rsqrt", lambda value: 1 / np.sqrt(value)),
        ("sqrt", lambda value: np.sqrt(value)),
        ("sigmoid", lambda value: np.exp(-np.logaddexp(0, -value))),
        ("silu", lambda value: value * np.exp(-np.logaddexp(0, -value))),
        ("sin", lambda value: np.sin(value)),
        ("cos", lambda value: np.cos(value)),
    ],
)
def test_unary_vector_operations(operation, expected) -> None:
    source = _region("ub_source", MemoryScope.UB, 4)
    destination = _region("ub_destination", MemoryScope.UB, 4)
    program = KernelProgram(
        f"vector-{operation}",
        "A2",
        (CoreProgram(0, (
            Task(
                operation,
                operation,
                0,
                Lane.VECTOR_0,
                Pipe.VECTOR,
                1,
                metadata={"src": source, "dst": destination},
            ),
        )),),
        buffers=(
            BufferSpec("ub_source", MemoryScope.UB, (4,), "float32"),
            BufferSpec("ub_destination", MemoryScope.UB, (4,), "float32"),
        ),
    )
    simulator = FunctionalSimulator(program)
    values = np.array([0.25, 1, 4, 9], dtype=np.float32)
    simulator.write(source, values)

    simulator.run()

    np.testing.assert_allclose(simulator.read(destination), expected(values), rtol=1e-6)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("adds", lambda value: value + 2),
        ("subs", lambda value: value - 2),
        ("muls", lambda value: value * 2),
        ("divs", lambda value: value / 2),
        ("maxs", lambda value: np.maximum(value, 2)),
        ("mins", lambda value: np.minimum(value, 2)),
        ("leaky_relu", lambda value: np.where(value >= 0, value, value * 2)),
    ],
)
def test_scalar_vector_operations(operation, expected) -> None:
    source = _region("ub_source", MemoryScope.UB, 4)
    destination = _region("ub_destination", MemoryScope.UB, 4)
    program = KernelProgram(
        f"vector-{operation}",
        "A3",
        (CoreProgram(0, (
            Task(
                operation,
                operation,
                0,
                Lane.VECTOR_0,
                Pipe.VECTOR,
                1,
                metadata={"lhs": source, "scalar": 2, "dst": destination},
            ),
        )),),
        buffers=(
            BufferSpec("ub_source", MemoryScope.UB, (4,), "float32"),
            BufferSpec("ub_destination", MemoryScope.UB, (4,), "float32"),
        ),
    )
    simulator = FunctionalSimulator(program)
    values = np.array([-1, 1, 3, 5], dtype=np.float32)
    simulator.write(source, values)

    simulator.run()

    np.testing.assert_array_equal(simulator.read(destination), expected(values))


def test_fill_initializes_poisoned_ub_memory() -> None:
    destination = _region("ub_destination", MemoryScope.UB, 5)
    program = KernelProgram(
        "fill",
        "A3",
        (CoreProgram(0, (
            Task(
                "fill",
                "fill",
                0,
                Lane.VECTOR_0,
                Pipe.VECTOR,
                1,
                metadata={"dst": destination, "scalar": -2.5},
            ),
        )),),
        buffers=(
            BufferSpec("ub_destination", MemoryScope.UB, (5,), "float32"),
        ),
    )
    simulator = FunctionalSimulator(program)

    simulator.run()

    np.testing.assert_array_equal(
        simulator.read(destination), np.full((5,), -2.5, dtype=np.float32)
    )


def test_axpy_reads_and_updates_destination_in_place() -> None:
    source = _region("source", MemoryScope.UB, 4)
    destination = _region("destination", MemoryScope.UB, 4)
    task = Task(
        "axpy",
        "axpy",
        0,
        Lane.VECTOR_0,
        Pipe.VECTOR,
        1,
        metadata={
            "lhs": source,
            "dst": destination,
            "accumulator": destination,
            "scalar": 0.5,
        },
    )
    program = KernelProgram(
        "axpy",
        "A2",
        (CoreProgram(0, (task,)),),
        buffers=(
            BufferSpec("source", MemoryScope.UB, (4,), "float32"),
            BufferSpec("destination", MemoryScope.UB, (4,), "float32"),
        ),
    )
    simulator = FunctionalSimulator(program)
    simulator.write(source, np.array([2, 4, 6, 8], dtype=np.float32))
    simulator.write(destination, np.array([1, 1, 1, 1], dtype=np.float32))

    simulator.run()

    np.testing.assert_array_equal(
        simulator.read(destination), np.array([2, 3, 4, 5], dtype=np.float32)
    )


def test_dynamic_buffer_allocation_resolves_symbolic_shape() -> None:
    dynamic_extent = SymbolicInt(
        "max", (AffineInt.variable("element_count"), 4)
    )
    program = KernelProgram(
        "dynamic-allocation",
        "A2",
        (CoreProgram(0),),
        buffers=(
            BufferSpec("dynamic", MemoryScope.GM, (dynamic_extent,), "float32"),
        ),
    )
    simulator = FunctionalSimulator(program, bindings={"element_count": 3})
    region = BufferRegion("dynamic", MemoryScope.GM, (dynamic_extent,), "float32")
    values = np.arange(4, dtype=np.float32)

    simulator.write(region, values)

    np.testing.assert_array_equal(simulator.read(region), values)


def test_dynamic_buffer_allocation_requires_runtime_binding() -> None:
    program = KernelProgram(
        "missing-dynamic-allocation",
        "A3",
        (CoreProgram(0),),
        buffers=(
            BufferSpec(
                "dynamic",
                MemoryScope.GM,
                (AffineInt.variable("element_count"),),
                "float32",
            ),
        ),
    )

    with pytest.raises(ProgramValidationError, match="missing runtime binding"):
        FunctionalSimulator(program)


def test_symbolic_integer_division_matches_tir_floor_and_truncation() -> None:
    value = AffineInt.variable("value")

    assert SymbolicInt("floordiv", (value, 3)).evaluate({"value": -5}) == -2
    assert SymbolicInt("floormod", (value, 3)).evaluate({"value": -5}) == 1
    assert SymbolicInt("truncdiv", (value, 3)).evaluate({"value": -5}) == -1
    assert SymbolicInt("truncmod", (value, 3)).evaluate({"value": -5}) == -2
    with pytest.raises(ProgramValidationError, match="division by zero"):
        SymbolicInt("floordiv", (value, 0)).evaluate({"value": 1})
