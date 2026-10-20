# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Local scalar stores execute even when tensor arithmetic is sync-only."""

import numpy as np
import pytest
import tvm

from tilelang.simulator import (
    BufferRegion,
    FunctionalSimulator,
    MemoryScope,
    SimulatorConfig,
    UnsupportedSimOpError,
    UninitializedMemoryError,
    build_kernel_program,
)


def _program(*, poison=False, tensor=False):
    t = tvm.tir
    state = t.decl_buffer((1,), "int32", name="state", scope="local.var")
    target = t.decl_buffer((1,), "int32", name="target", scope="local.var")
    gm = t.decl_buffer((1,), "int32", name="gm")
    i = t.Var("i", "int32")
    value = gm[0] if tensor else state[0] * 2 + 1
    stores = [t.BufferStore(target, value, [0])]
    if not poison and not tensor:
        stores.insert(0, t.BufferStore(state, i % 3, [0]))
    body = t.SeqStmt(stores) if len(stores) > 1 else stores[0]
    body = t.For(i, 0, 5, t.ForKind.SERIAL, body)
    for b in (state, target):
        body = t.Allocate(b.data, b.dtype, b.shape, t.const(True, "bool"), body)
    body = t.AttrStmt(t.IntImm("int32", 1), "resource_scope", 1, body)
    func = t.PrimFunc([gm.data] if tensor else [], body, buffer_map={gm.data: gm} if tensor else {})
    return build_kernel_program(func, platform="A2")


@pytest.mark.parametrize("sync_only", [False, True])
def test_local_scalar_store_and_load_keep_real_values(sync_only):
    p = _program()
    sim = FunctionalSimulator(p, SimulatorConfig(sync_only=sync_only))
    sim.run()
    region = BufferRegion("target", MemoryScope.LOCAL, (1,), "int32")
    view = sim._resolve(region, 0)
    value = np.frombuffer(view.allocation.read(view), np.int32)[0]
    assert value == 3  # final i=4 -> state=1 -> target=3
    assert all(task.pipe.value == "s" for task in p.tasks)


@pytest.mark.parametrize("sync_only", [False, True])
def test_poison_scalar_load_is_not_silently_zero(sync_only):
    sim = FunctionalSimulator(_program(poison=True), SimulatorConfig(sync_only=sync_only))
    with pytest.raises(UninitializedMemoryError):
        sim.run()


def test_sync_only_tensor_value_cannot_drive_scalar_store():
    sim = FunctionalSimulator(_program(tensor=True), SimulatorConfig(sync_only=True))
    sim.write(BufferRegion("gm", MemoryScope.GM, (1,), "int32"), [123])
    with pytest.raises(UnsupportedSimOpError, match="sync-only.*tensor"):
        sim.run()


def test_full_tensor_scalar_load_remains_supported():
    sim = FunctionalSimulator(_program(tensor=True))
    sim.write(BufferRegion("gm", MemoryScope.GM, (1,), "int32"), [123])
    sim.run()
    np.testing.assert_array_equal(sim.read(BufferRegion("target", MemoryScope.LOCAL, (1,), "int32")), [123])


@pytest.mark.parametrize("sync_only", [False, True])
def test_scalar_store_honors_explicit_strides_and_element_offset(sync_only):
    t = tvm.tir
    backing = t.decl_buffer((12,), "int32", name="state", scope="local.var")
    view = t.decl_buffer((2, 2), "int32", name="view", data=backing.data, strides=(4, 1), elem_offset=2, scope="local.var")
    body = t.SeqStmt(
        [
            t.BufferStore(view, t.const(7, "int32"), [1, 1]),
            t.BufferStore(view, view[1, 1] * 3, [0, 0]),
        ]
    )
    body = t.Allocate(backing.data, backing.dtype, backing.shape, t.const(True, "bool"), body)
    body = t.AttrStmt(t.IntImm("int32", 0), "resource_scope", 0, body)
    program = build_kernel_program(t.PrimFunc([], body), platform="A2")
    assert all(task.lane.value == "cube" and task.pipe.value == "s" for task in program.tasks)
    assert [task.metadata["dst"].byte_offset for task in program.tasks] == [28, 8]
    sim = FunctionalSimulator(program, SimulatorConfig(sync_only=sync_only))
    sim.run()
    region = BufferRegion("state", MemoryScope.LOCAL, (1,), "int32", byte_offset=8)
    storage = sim._resolve(region, 0)
    assert np.frombuffer(storage.allocation.read(storage), dtype=np.int32)[0] == 21


def test_scalar_ub_store_requires_a_real_scalar_to_vector_fence():
    from tilelang.simulator import DiscreteEventScheduler, MemoryHazardError
    from tilelang.simulator.sync import validate_memory_synchronization

    t = tvm.tir
    source = t.decl_buffer((8,), "float32", name="source", scope="shared.ub")
    destination = t.decl_buffer((8,), "float32", name="destination", scope="shared.ub")

    def program(fence):
        statements = [t.BufferStore(source, t.const(1.0, "float32"), [0])]
        if fence:
            statements.extend(
                [
                    t.Evaluate(t.call_extern("handle", "set_flag", "s", "v", 0)),
                    t.Evaluate(t.call_extern("handle", "wait_flag", "s", "v", 0)),
                ]
            )
        statements.append(
            t.Evaluate(
                t.Call(
                    "handle",
                    tvm.ir.Op.get("tl.ascend_add"),
                    [
                        destination.access_ptr("w"),
                        source.access_ptr("r"),
                        source.access_ptr("r"),
                        t.const(8, "int32"),
                    ],
                )
            )
        )
        body = t.SeqStmt(statements)
        for buffer in (source, destination):
            body = t.Allocate(buffer.data, buffer.dtype, buffer.shape, t.const(True, "bool"), body)
        body = t.AttrStmt(t.IntImm("int32", 1), "resource_scope", 1, body)
        return build_kernel_program(t.PrimFunc([], body), platform="A2")

    broken = program(False)
    assert broken.tasks[0].pipe.value == "s"
    assert broken.tasks[-1].metadata["memory_dependencies"]
    assert not broken.tasks[-1].metadata.get("hardware_dependencies")
    assert DiscreteEventScheduler._execution_dependencies(broken.tasks[-1]) == ()
    with pytest.raises(MemoryHazardError, match="missing A2/A3 pipe synchronization"):
        validate_memory_synchronization(broken)
    assert not validate_memory_synchronization(program(True))


@pytest.mark.parametrize("sync_only", [False, True])
@pytest.mark.parametrize("dtype,value", [("float32", 2.5), ("int32", 16909060)])
def test_scalar_typed_alias_uses_view_dtype_not_backing_dtype(sync_only, dtype, value):
    t = tvm.tir
    storage = t.decl_buffer((16,), "uint8", name="bytes", scope="local.var")
    view = t.decl_buffer((4,), dtype, name="typed", data=storage.data, scope="local.var")
    target = t.decl_buffer((1,), dtype, name="target", scope="local.var")
    body = t.SeqStmt(
        [
            t.BufferStore(view, t.const(value, dtype), [1]),
            t.BufferStore(target, view[1], [0]),
        ]
    )
    for buffer in (storage, target):
        body = t.Allocate(buffer.data, buffer.dtype, buffer.shape, t.const(True, "bool"), body)
    body = t.AttrStmt(t.IntImm("int32", 0), "resource_scope", 0, body)
    program = build_kernel_program(t.PrimFunc([], body), platform="A2")
    assert program.tasks[0].metadata["dst"].byte_offset == 4
    assert program.tasks[0].metadata["dst"].dtype == dtype
    sim = FunctionalSimulator(program, SimulatorConfig(sync_only=sync_only))
    sim.run()
    region = BufferRegion("target", MemoryScope.LOCAL, (1,), dtype)
    allocation = sim._resolve(region, 0)
    actual = np.frombuffer(allocation.allocation.read(allocation), dtype=dtype)[0]
    assert actual == value
