# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Native VID must keep broadcast geometry and workspace capacity consistent."""

import pytest
import tilelang
from tilelang import language as T
import tvm
from tvm import tir


def fixture(axis=1, source_rank=1, threads=2, workspace=False, rows=32, offset=0):
    dst = tir.decl_buffer((rows, 64), "float32", name="dst", scope="shared.ub")
    shape = (rows,) if axis == 1 else (64,)
    if source_rank == 2:
        shape = (rows, 1) if axis == 1 else (1, 64)
    src = tir.decl_buffer(shape, "float32", name="src", scope="shared.ub")
    tmp = tir.decl_buffer((256,), "uint8", name="arena", scope="shared.ub")
    call = T.tile.broadcast(dst, src, axis=axis, tmp=tmp if workspace else None)
    if offset:
        pointer = call.args[1]
        modified = tir.Call(
            pointer.dtype, pointer.op,
            [*pointer.args[:2], tir.IntImm("int32", offset), *pointer.args[3:]])
        call = tir.Call(call.dtype, call.op, [call.args[0], modified, *call.args[2:]])
    root = tir.Block([], [], [],
                     "tilelang_root",
                     tir.Evaluate(call),
                     alloc_buffers=[dst, src, *([tmp] if workspace else [])])
    axis_var = tir.IterVar(
        tvm.ir.Range(0, threads), tir.Var("vid", "int32"), tir.IterVar.ThreadIndex, "threadIdx.x")
    body = tir.AttrStmt(axis_var, "thread_extent", threads, tir.BlockRealize([], True, root))
    return tir.PrimFunc([], body).with_attr("global_symbol", "probe")


def apply(function):
    return tilelang.transform.AscendVidReduction()(tvm.IRModule({"probe": function}))["probe"]


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("source_rank", [1, 2])
@pytest.mark.parametrize("workspace", [False, True])
def test_broadcast_geometry_tracks_lane_buffers_without_halving_arena(axis, source_rank, workspace):
    result = apply(fixture(axis, source_rank, workspace=workspace))
    block = result.body.body.block
    dst, src = block.alloc_buffers[:2]
    assert [int(x) for x in dst.shape] == [16, 64]
    expected = [16] if axis == 1 else [64]
    if source_rank == 2:
        expected = [16, 1] if axis == 1 else [1, 64]
    assert [int(x) for x in src.shape] == expected
    call = block.body.value
    index = 4 if workspace else 3
    assert [int(x) for x in call.args[index:]
           ] == [2, 16, 64, 16 if axis == 1 else 1, 1 if axis == 1 else 64]
    assert int(call.args[1].args[3]) == 16 * 64
    assert int(call.args[2].args[3]) == (16 if axis == 1 else 64)
    if workspace:
        assert [int(x) for x in block.alloc_buffers[2].shape] == [256]
        assert int(call.args[3].args[3]) == 256


def test_threads_one_broadcast_is_unchanged():
    function = fixture(threads=1, workspace=True)
    assert tvm.ir.structural_equal(function, apply(function))


@pytest.mark.parametrize("rows,offset", [(31, 0), (32, 8)])
def test_unproven_row_partition_and_slices_fail_closed(rows, offset):
    with pytest.raises(tvm.error.TVMError, match="VID broadcast"):
        apply(fixture(rows=rows, offset=offset))


def test_equal_element_count_does_not_prove_source_row_partition():
    function = fixture()
    root = function.body.body.block
    original = root.alloc_buffers[1]
    wrong = tir.decl_buffer((1, 32),
                            "float32",
                            name=original.name,
                            data=original.data,
                            scope="shared.ub")
    block = tir.Block([], [], [],
                      "tilelang_root",
                      root.body,
                      alloc_buffers=[root.alloc_buffers[0], wrong])
    function = function.with_body(
        tir.AttrStmt(function.body.node, "thread_extent", 2, tir.BlockRealize([], True, block)))
    with pytest.raises(tvm.error.TVMError, match="source shape must match"):
        apply(function)


def test_replicated_column_source_cannot_also_be_sharded_elementwise_data():
    function = fixture(axis=0)
    root = function.body.body.block
    src = root.alloc_buffers[1]
    out = tir.decl_buffer((64,), "float32", name="row_out", scope="shared.ub")
    body = tir.SeqStmt([root.body, tir.Evaluate(T.tile.add(out, src, src))])
    block = tir.Block([], [], [], "tilelang_root", body, alloc_buffers=[*root.alloc_buffers, out])
    function = function.with_body(
        tir.AttrStmt(function.body.node, "thread_extent", 2, tir.BlockRealize([], True, block)))
    with pytest.raises(tvm.error.TVMError, match="contradictory"):
        apply(function)


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("axis", [0, 1])
def test_native_generated_arrays_use_lane_shapes(platform, axis):
    function = apply(fixture(axis=axis, workspace=True)).with_attr("npu_platform", platform)
    target = tvm.target.Target({"kind": "llvm", "model": "ascendc"})
    mod = tir.transform.BindTarget(target)(tvm.IRModule({"probe": function}))
    mod = tilelang.transform.AscendLowerOpaqueBlock()(mod)
    mod = tilelang.transform.Flatten2DBuffer()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.AscendMemoryPlanning()(mod)
    source = tvm.get_global_func("target.build.tilelang_ascend")(mod, target, platform).get_source()
    assert "{16, 64}" in source
    assert ("{16, 1}" if axis == 1 else "{1, 64}") in source
    assert "GetWithOffset<uint8_t>(256," in source


@pytest.mark.parametrize("axis", [0, 1])
def test_distinct_values_on_both_vector_lanes(axis):
    import numpy as np
    from tilelang.simulator import (build_kernel_program, FunctionalSimulator, SimulatorConfig,
                                    BufferRegion, MemoryScope)

    function = apply(fixture(axis=axis)).with_attr("npu_cv_ratio", "cv_1_2")
    group = function.body
    function = function.with_body(
        tir.AttrStmt(group.node, "thread_extent", group.value,
                     tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, group.body)))
    program = build_kernel_program(function, platform="A2")
    sim = FunctionalSimulator(program, SimulatorConfig(platform="A2"))
    for owner in (0, sim.memory.vector1_owner(0)):
        count = 16 if axis == 1 else 64
        values = np.arange(count, dtype="float32") + (100 if owner != 0 else 0)
        sim.write(BufferRegion("src", MemoryScope.UB, (count,), "float32", core_id=owner), values)
    sim.run()
    for owner in (0, sim.memory.vector1_owner(0)):
        count = 16 if axis == 1 else 64
        values = np.arange(count, dtype="float32") + (100 if owner != 0 else 0)
        expected = np.broadcast_to(values[:, None] if axis == 1 else values[None, :], (16, 64))
        np.testing.assert_array_equal(
            sim.read(BufferRegion("dst", MemoryScope.UB, (16, 64), "float32", core_id=owner)),
            expected)
