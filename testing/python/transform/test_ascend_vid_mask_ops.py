# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Causal coordinates and Select counts must follow the logical VID partition."""

import pytest
import tilelang
from tilelang import language as T
import tvm
from tvm import tir


def function(body, buffers, threads=2):
    axis = tir.IterVar(None, tir.Var("threadIdx_x", "int32"), tir.IterVar.ThreadIndex,
                       "threadIdx.x")
    root = tir.Block([], [], [], "tilelang_root", body, alloc_buffers=buffers)
    return tir.PrimFunc([],
                        tir.AttrStmt(axis, "thread_extent", threads, tir.BlockRealize([], True,
                                                                                      root)))


def reduced(f):
    return tilelang.transform.AscendVidReduction()(tvm.IRModule({"main": f}))["main"]


def test_sharded_create_index_changes_count_and_global_lane_origin():
    vector = tir.decl_buffer((32,), "float32", name="indices", scope="shared.ub")
    f = reduced(
        function(tir.Evaluate(T.tile.createvecindex(vector, tir.FloatImm("float32", 5))), [vector]))
    call = f.body.body.block.body.value
    assert int(call.args[3]) == 16
    for vid in (0, 1):
        value = tir.stmt_functor.substitute(call.args[2],
                                            {f.body.node.var: tir.IntImm("int32", vid)})
        assert float(tvm.arith.Analyzer().simplify(value)) == 5 + vid * 16


def test_replicated_column_indices_keep_count_and_origin():
    vector = tir.decl_buffer((64,), "float32", name="columns", scope="shared.ub")
    dst = tir.decl_buffer((32, 64), "float32", name="column_grid", scope="shared.ub")
    body = tir.SeqStmt([
        tir.Evaluate(T.tile.createvecindex(vector, tir.FloatImm("float32", 5))),
        tir.Evaluate(T.tile.broadcast(dst, vector, axis=0))
    ])
    call = reduced(function(body, [vector, dst])).body.body.block.body.seq[0].value
    assert int(call.args[3]) == 64 and float(call.args[2]) == 5


@pytest.mark.parametrize("scalar", [True, False])
def test_select_halves_actual_count_not_dtype_or_mode(scalar):
    dst = tir.decl_buffer((32, 64), "float32", name="selected", scope="shared.ub")
    src = tir.decl_buffer((32, 64), "float32", name="src", scope="shared.ub")
    mask = tir.decl_buffer((32, 8), "uint8", name="mask", scope="shared.ub")
    second = -1.0 if scalar else src
    mode = "VSEL_TENSOR_SCALAR_MODE" if scalar else "VSEL_TENSOR_TENSOR_MODE"
    f = function(tir.Evaluate(T.tile.select(dst, mask, src, second, mode)), [dst, src, mask])
    original = f.body.body.block.body.value
    call = reduced(f).body.body.block.body.value
    assert int(call.args[6]) == 1024
    for index in range(7, len(call.args)):
        assert tvm.ir.structural_equal(original.args[index], call.args[index])
    assert call.args[5].value == mode


def test_odd_create_index_partition_fails_closed():
    vector = tir.decl_buffer((31,), "float32", name="indices", scope="shared.ub")
    with pytest.raises(tvm.error.TVMError, match="VID index"):
        reduced(
            function(
                tir.Evaluate(T.tile.createvecindex(vector, tir.FloatImm("float32", 0))), [vector]))


def test_lane_coordinates_and_packed_select_execute_numerically():
    import numpy as np
    from tilelang.simulator import (build_kernel_program, FunctionalSimulator, SimulatorConfig,
                                    BufferRegion, MemoryScope)

    rows = tir.decl_buffer((32,), "float32", name="rows", scope="shared.ub")
    grid = tir.decl_buffer((32, 64), "float32", name="grid", scope="shared.ub")
    mask = tir.decl_buffer((32, 8), "uint8", name="mask", scope="shared.ub")
    dst = tir.decl_buffer((32, 64), "float32", name="selected", scope="shared.ub")
    body = tir.SeqStmt([
        tir.Evaluate(T.tile.createvecindex(rows, tir.FloatImm("float32", 5))),
        tir.Evaluate(T.tile.broadcast(grid, rows, axis=1)),
        tir.Evaluate(T.tile.fill(mask, 0x55)),
        tir.Evaluate(T.tile.select(dst, mask, grid, -1.0, "VSEL_TENSOR_SCALAR_MODE")),
    ])
    f = reduced(function(body, [rows, grid, mask, dst]))
    group = f.body
    scoped = tir.AttrStmt(tir.IntImm("int32", 1), "resource_scope", 1, group.body)
    f = f.with_body(tir.AttrStmt(group.node, "thread_extent", group.value, scoped))
    sim = FunctionalSimulator(
        build_kernel_program(f, platform="A2"), SimulatorConfig(platform="A2"))
    sim.run()
    for vid, owner in enumerate((0, sim.memory.vector1_owner(0))):
        row_values = np.arange(16, dtype="float32") + 5 + vid * 16
        expected = np.where(np.arange(64)[None, :] % 2 == 0, row_values[:, None], -1)
        actual = sim.read(
            BufferRegion("selected", MemoryScope.UB, (16, 64), "float32", core_id=owner))
        np.testing.assert_array_equal(actual, expected)


def test_threads_one_index_is_unchanged():
    vector = tir.decl_buffer((32,), "float32", name="indices", scope="shared.ub")
    original = function(
        tir.Evaluate(T.tile.createvecindex(vector, tir.FloatImm("float32", 5))), [vector],
        threads=1)
    assert tvm.ir.structural_equal(original, reduced(original))


def test_tensor_scalar_select_is_not_treated_as_a_queued_vector_read():
    dst = tir.decl_buffer((32, 64), "float32", name="dst", scope="shared.ub")
    src = tir.decl_buffer((32, 64), "float32", name="src", scope="shared.ub")
    mask = tir.decl_buffer((32, 8), "uint8", name="mask", scope="shared.ub")
    call = T.tile.select(dst, mask, src, tir.BufferLoad(src, [0, 0]), "VSEL_TENSOR_TENSOR_MODE")
    with pytest.raises(tvm.error.TVMError, match="scalar decomposition"):
        reduced(function(tir.Evaluate(call), [dst, src, mask]))


def test_index_slice_is_rejected_before_origin_rebasing():
    vector = tir.decl_buffer((32,), "float32", name="indices", scope="shared.ub")
    call = T.tile.createvecindex(vector, tir.FloatImm("float32", 0))
    ptr = call.args[1]
    sliced = tir.Call(ptr.dtype, ptr.op, [*ptr.args[:2], tir.IntImm("int32", 8), *ptr.args[3:]])
    call = tir.Call(call.dtype, call.op, [call.args[0], sliced, *call.args[2:]])
    with pytest.raises(tvm.error.TVMError, match="VID index"):
        reduced(function(tir.Evaluate(call), [vector]))
