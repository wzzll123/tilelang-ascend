# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Tests that exercise the bridge against real TVM TIR nodes."""

import pytest
import numpy as np

tvm = pytest.importorskip("tvm")
from tvm.script import tir as T  # noqa: E402

from tilelang.simulator import (  # noqa: E402
    AffineInt,
    BufferRegion,
    FunctionalSimulator,
    Lane,
    MemoryScope,
    Pipe,
    ProgramValidationError,
    SymbolicInt,
    TimingProfile,
    UninitializedMemoryError,
    UnsupportedSimOpError,
    build_kernel_program,
)
from tilelang.simulator.layout import (  # noqa: E402
    pack_matrix,
    storage_elements,
    unpack_matrix,
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


def _gm_to_l1_linear_primfunc(valid_cols=8):
    source = tvm.tir.decl_buffer((2, 8), "float32", name="source", scope="global")
    l1 = tvm.tir.decl_buffer((3, 8), "float32", name="l1", scope="shared.l1")
    copy = tvm.tir.call_extern(
        "handle", "tl::ascend::copy_gm_to_l1_linear<float, 3, 8>",
        source.access_ptr("r"), l1.access_ptr("w"),
        8, 2, valid_cols, 3, 8,
    )
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.Evaluate(copy), alloc_buffers=[l1]
    )
    return tvm.tir.PrimFunc(
        [source.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source},
    )


def _gm_to_l1_zn_primfunc(physical_rows=16):
    source = tvm.tir.decl_buffer((16, 16), "float16", name="source", scope="global")
    l1 = tvm.tir.decl_buffer(
        (physical_rows * 16,), "float16", name="l1", scope="shared.l1"
    )
    copy = tvm.tir.call_extern(
        "handle",
        f"tl::ascend::copy_gm_to_l1<half, {physical_rows}, 16>",
        source.access_ptr("r"),
        l1.access_ptr("w"),
        16,
        13,
        15,
        physical_rows,
        16,
    )
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.Evaluate(copy), alloc_buffers=[l1]
    )
    return tvm.tir.PrimFunc(
        [source.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source},
    )


def _l1_to_l0_primfunc(
    destination_scope="wmma.matrix_a",
    transpose=False,
    source_rows=32,
    source_cols=32,
    destination_rows=None,
    destination_cols=None,
    source_offset=0,
):
    destination_rows = destination_rows or (source_cols if transpose else source_rows)
    destination_cols = destination_cols or (source_rows if transpose else source_cols)
    source_elements = ((source_rows + 15) // 16 * 16) * (
        (source_cols + 15) // 16 * 16
    )
    destination_elements = ((destination_rows + 15) // 16 * 16) * (
        (destination_cols + 15) // 16 * 16
    )
    l1 = tvm.tir.decl_buffer(
        (source_elements,), "float16", name="l1", scope="shared.l1"
    )
    l0 = tvm.tir.decl_buffer(
        (destination_elements,), "float16", name="l0", scope=destination_scope
    )
    suffix = "a" if destination_scope == "wmma.matrix_a" else "b"
    copy = tvm.tir.call_extern(
        "handle",
        f"tl::ascend::copy_l1_to_l0{suffix}<half, {source_rows}, "
        f"{source_cols}, {str(transpose).lower()}>",
        l1.access_ptr("r", offset=source_offset),
        l0.access_ptr("w"),
        destination_rows,
        destination_cols,
    )
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.Evaluate(copy), alloc_buffers=[l1, l0]
    )
    return tvm.tir.PrimFunc([], tvm.tir.BlockRealize([], True, root))


def _gm_l1_l0_primfunc():
    source = tvm.tir.decl_buffer((32, 32), "float16", name="source", scope="global")
    l1 = tvm.tir.decl_buffer((1024,), "float16", name="l1", scope="shared.l1")
    l0a = tvm.tir.decl_buffer(
        (1024,), "float16", name="l0a", scope="wmma.matrix_a"
    )
    gm_to_l1 = tvm.tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_l1<half, 32, 32>",
        source.access_ptr("r"),
        l1.access_ptr("w"),
        32,
        32,
        32,
        32,
        32,
    )
    l1_to_l0a = tvm.tir.call_extern(
        "handle",
        "tl::ascend::copy_l1_to_l0a<half, 32, 32, false>",
        l1.access_ptr("r"),
        l0a.access_ptr("w"),
        32,
        32,
    )
    root = tvm.tir.Block(
        [],
        [],
        [],
        "root",
        tvm.tir.SeqStmt([
            tvm.tir.Evaluate(gm_to_l1),
            tvm.tir.Evaluate(l1_to_l0a),
        ]),
        alloc_buffers=[l1, l0a],
    )
    return tvm.tir.PrimFunc(
        [source.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source},
    )


def _l0c_to_gm_primfunc(enable_relu=True, unit_flag=0):
    l0c = tvm.tir.decl_buffer(
        (16 * 32,), "float32", name="l0c", scope="wmma.accumulator"
    )
    output = tvm.tir.decl_buffer(
        (16, 32), "float16", name="output", scope="global"
    )
    copy = tvm.tir.call_extern(
        "handle",
        "tl::ascend::copy_l0c_to_gm<float, half, layout::RowMajor, "
        f"16, 32, {str(enable_relu).lower()}>",
        l0c.access_ptr("r"),
        output.access_ptr("w"),
        32,
        13,
        17,
        16,
        32,
        enable_relu,
        unit_flag,
    )
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.Evaluate(copy), alloc_buffers=[l0c]
    )
    return tvm.tir.PrimFunc(
        [output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={output.data: output},
    )


def _atomic_add_l0c_primfunc(
    *, source_dtype="float32", destination_dtype="float16",
    valid_rows=13, valid_cols=17,
):
    rows, cols = 16, 32
    l0c = tvm.tir.decl_buffer(
        (rows * cols,), source_dtype, name="l0c", scope="wmma.accumulator"
    )
    output = tvm.tir.decl_buffer(
        (rows, cols), destination_dtype, name="output", scope="global"
    )
    dtype_names = {
        "float16": "half", "float32": "float", "int32": "int",
    }
    atomic = tvm.tir.call_extern(
        "handle",
        "tl::ascend::atomic_add_l0c_to_gm<"
        f"{dtype_names[source_dtype]}, {dtype_names[destination_dtype]}, "
        f"layout::RowMajor, {rows}, {cols}>",
        l0c.access_ptr("r"), output.access_ptr("w"), cols,
        valid_rows, valid_cols,
    )
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.Evaluate(atomic), alloc_buffers=[l0c]
    )
    return tvm.tir.PrimFunc(
        [output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={output.data: output},
    )


def _mma_primfunc(
    init=True, inner=13, n_actual=None, unit_flag=0, cols=16,
    input_dtype="float16", accumulator_dtype="float32",
):
    rows = 16
    a_elements = storage_elements(
        "l0a", (rows, inner), np.dtype(input_dtype).itemsize
    )
    b_elements = storage_elements(
        "l0b", (inner, cols), np.dtype(input_dtype).itemsize
    )
    c_elements = storage_elements(
        "l0c", (rows, cols), np.dtype(accumulator_dtype).itemsize
    )
    l0a = tvm.tir.decl_buffer(
        (a_elements,), input_dtype, name="l0a", scope="wmma.matrix_a"
    )
    l0b = tvm.tir.decl_buffer(
        (b_elements,), input_dtype, name="l0b", scope="wmma.matrix_b"
    )
    l0c = tvm.tir.decl_buffer(
        (c_elements,), accumulator_dtype, name="l0c", scope="wmma.accumulator"
    )
    input_token = {"float16": "half", "int8": "int8_t"}[input_dtype]
    accumulator_token = {"float32": "float", "int32": "int"}[
        accumulator_dtype
    ]
    arguments = [
        f"mma<{input_token}, {accumulator_token}, 16, {cols}>",
        l0a.access_ptr("r"),
        l0b.access_ptr("r"),
        l0c.access_ptr("w" if init else "rw"),
        init,
        inner,
    ]
    if n_actual is not None:
        arguments.extend([n_actual, unit_flag])
    # The production lowering uses the registered ``tl.ascend_mma`` intrinsic.
    # Keep this bridge-only test independent of TileLang's registration side
    # effects by constructing the equivalent final-TIR call_extern form.
    mma = tvm.tir.call_extern("handle", "tl.ascend_mma", *arguments)
    root = tvm.tir.Block(
        [],
        [],
        [],
        "root",
        tvm.tir.Evaluate(mma),
        alloc_buffers=[l0a, l0b, l0c],
    )
    return tvm.tir.PrimFunc([], tvm.tir.BlockRealize([], True, root))


def _mma_bias_primfunc(*, bias_scope="shared.bt", bias_length=16, init=True):
    rows, cols, inner = 16, 16, 13
    l0a = tvm.tir.decl_buffer(
        (storage_elements("l0a", (rows, inner), 2),),
        "float16", name="l0a", scope="wmma.matrix_a",
    )
    l0b = tvm.tir.decl_buffer(
        (storage_elements("l0b", (inner, cols), 2),),
        "float16", name="l0b", scope="wmma.matrix_b",
    )
    l0c = tvm.tir.decl_buffer(
        (storage_elements("l0c", (rows, cols), 4),),
        "float32", name="l0c", scope="wmma.accumulator",
    )
    bias_l1 = tvm.tir.decl_buffer(
        (bias_length,), "float32", name="bias_l1", scope="shared.l1"
    )
    bias_bt = tvm.tir.decl_buffer(
        (bias_length,), "float32", name="bias_bt", scope=bias_scope
    )
    copy_bias = tvm.tir.call_extern(
        "handle", "tl::ascend::copy_l1_to_bt<float, float>",
        bias_l1.access_ptr("r"), bias_bt.access_ptr("w"), bias_length,
    )
    mma = tvm.tir.call_extern(
        "handle", "tl.ascend_mma", "mma_bias<half, float, 16, 16>",
        l0a.access_ptr("r"), l0b.access_ptr("r"),
        l0c.access_ptr("w" if init else "rw"),
        bias_bt.access_ptr("r"), init, inner,
    )
    root = tvm.tir.Block(
        [], [], [], "root",
        tvm.tir.SeqStmt([tvm.tir.Evaluate(copy_bias), tvm.tir.Evaluate(mma)]),
        alloc_buffers=[l0a, l0b, l0c, bias_l1, bias_bt],
    )
    return tvm.tir.PrimFunc([], tvm.tir.BlockRealize([], True, root))


def _mma_runtime_n_primfunc(nact_index=0):
    nact = tvm.tir.decl_buffer((1,), "int32", name="nact", scope="global")
    l0a = tvm.tir.decl_buffer(
        (16 * 16,), "float16", name="l0a", scope="wmma.matrix_a"
    )
    l0b = tvm.tir.decl_buffer(
        (16 * 32,), "float16", name="l0b", scope="wmma.matrix_b"
    )
    l0c = tvm.tir.decl_buffer(
        (16 * 32,), "float32", name="l0c", scope="wmma.accumulator"
    )
    mma = tvm.tir.call_extern(
        "handle", "tl.ascend_mma", "mma<half, float, 16, 32>",
        l0a.access_ptr("r"), l0b.access_ptr("r"), l0c.access_ptr("w"),
        True, 13, nact[nact_index], 0,
    )
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.Evaluate(mma),
        alloc_buffers=[l0a, l0b, l0c],
    )
    return tvm.tir.PrimFunc(
        [nact.data], tvm.tir.BlockRealize([], True, root),
        buffer_map={nact.data: nact},
    )


def _mma_chain_primfunc():
    inputs = [
        tvm.tir.decl_buffer(
            (256,), "float16", name=f"l0{role}{index}",
            scope=f"wmma.matrix_{role}",
        )
        for index in range(2)
        for role in ("a", "b")
    ]
    l0c = tvm.tir.decl_buffer(
        (256,), "float32", name="l0c", scope="wmma.accumulator"
    )
    calls = []
    for index in range(2):
        l0a, l0b = inputs[index * 2:index * 2 + 2]
        calls.append(tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_mma", "mma<half, float, 16, 16>",
            l0a.access_ptr("r"), l0b.access_ptr("r"),
            l0c.access_ptr("w" if index == 0 else "rw"), index == 0, 13,
        )))
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(calls),
        alloc_buffers=[*inputs, l0c],
    )
    return tvm.tir.PrimFunc([], tvm.tir.BlockRealize([], True, root))


def _mma_fixpipe_primfunc(
    *, n_actual=16, mma_unit_flag=3, fix_unit_flag=3, fix_valid_cols=None,
    accumulate=False,
):
    rows, cols, inner = 16, 32, 13
    valid_cols = n_actual if fix_valid_cols is None else fix_valid_cols
    operand_count = 2 if accumulate else 1
    operands = [
        (
            tvm.tir.decl_buffer(
                (16 * 16,), "float16", name=f"l0a{index}",
                scope="wmma.matrix_a",
            ),
            tvm.tir.decl_buffer(
                (16 * 32,), "float16", name=f"l0b{index}",
                scope="wmma.matrix_b",
            ),
        )
        for index in range(operand_count)
    ]
    l0c = tvm.tir.decl_buffer(
        (16 * 32,), "float32", name="l0c", scope="wmma.accumulator"
    )
    output = tvm.tir.decl_buffer(
        (rows, cols), "float16", name="output", scope="global"
    )
    mma_calls = [
        tvm.tir.call_extern(
            "handle", "tl.ascend_mma", "mma<half, float, 16, 32>",
            l0a.access_ptr("r"), l0b.access_ptr("r"),
            l0c.access_ptr("w" if index == 0 else "rw"), index == 0, inner,
            n_actual, 2 if index + 1 < operand_count else mma_unit_flag,
        )
        for index, (l0a, l0b) in enumerate(operands)
    ]
    fixpipe = tvm.tir.call_extern(
        "handle",
        "tl::ascend::copy_l0c_to_gm<float, half, layout::RowMajor, "
        "16, 32, false>",
        l0c.access_ptr("r"), output.access_ptr("w"), cols, rows, valid_cols,
        rows, cols, False, fix_unit_flag,
    )
    root = tvm.tir.Block(
        [], [], [], "root",
        tvm.tir.SeqStmt([
            *(tvm.tir.Evaluate(mma) for mma in mma_calls),
            tvm.tir.Evaluate(fixpipe),
        ]),
        alloc_buffers=[
            *(buffer for pair in operands for buffer in pair), l0c,
        ],
    )
    return tvm.tir.PrimFunc(
        [output.data], tvm.tir.BlockRealize([], True, root),
        buffer_map={output.data: output},
    )


def _gemm_v0_primfunc(
    transpose_a=False, transpose_b=False, init=True, n_actual=None,
    k_l0_size=16, inner=13, rows=16, cols=16,
    input_dtype="float16", accumulator_dtype="float32",
):
    shape_a = (inner, rows) if transpose_a else (rows, inner)
    shape_b = (cols, inner) if transpose_b else (inner, cols)
    a_elements = storage_elements(
        "zn", shape_a, np.dtype(input_dtype).itemsize
    )
    b_elements = storage_elements(
        "zn", shape_b, np.dtype(input_dtype).itemsize
    )
    c_elements = storage_elements(
        "l0c", (rows, cols), np.dtype(accumulator_dtype).itemsize
    )
    l1a = tvm.tir.decl_buffer(
        (a_elements,), input_dtype, name="l1a", scope="shared.l1"
    )
    l1b = tvm.tir.decl_buffer(
        (b_elements,), input_dtype, name="l1b", scope="shared.l1"
    )
    l0c = tvm.tir.decl_buffer(
        (c_elements,), accumulator_dtype, name="l0c", scope="wmma.accumulator"
    )
    input_token = {"float16": "half", "int8": "int8_t"}[input_dtype]
    accumulator_token = {"float32": "float", "int32": "int"}[
        accumulator_dtype
    ]
    call = tvm.tir.call_extern(
        "handle",
        "tl.ascend_gemm_v0",
        f"gemm_v0<{input_token}, {accumulator_token}, {rows}, {cols}, {inner}, "
        f"{str(transpose_a).lower()}, {str(transpose_b).lower()}, {k_l0_size}>",
        l1a.access_ptr("r"),
        l1b.access_ptr("r"),
        l0c.access_ptr("w" if init else "rw"),
        init,
        cols if n_actual is None else n_actual,
    )
    root = tvm.tir.Block(
        [],
        [],
        [],
        "root",
        tvm.tir.Evaluate(call),
        alloc_buffers=[l1a, l1b, l0c],
    )
    return (
        tvm.tir.PrimFunc([], tvm.tir.BlockRealize([], True, root)),
        shape_a,
        shape_b,
    )


def _gemm_v0_runtime_n_primfunc():
    rows, cols, inner, k_l0_size = 16, 32, 48, 16
    nact = tvm.tir.decl_buffer((1,), "int32", name="nact", scope="global")
    l1a = tvm.tir.decl_buffer(
        (rows * inner,), "float16", name="l1a", scope="shared.l1"
    )
    l1b = tvm.tir.decl_buffer(
        (cols * inner,), "float16", name="l1b", scope="shared.l1"
    )
    l0c = tvm.tir.decl_buffer(
        (rows * cols,), "float32", name="l0c", scope="wmma.accumulator"
    )
    call = tvm.tir.call_extern(
        "handle", "tl.ascend_gemm_v0",
        "gemm_v0<half, float, 16, 32, 48, false, true, 16>",
        l1a.access_ptr("r"), l1b.access_ptr("r"), l0c.access_ptr("w"),
        True, nact[0],
    )
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.Evaluate(call),
        alloc_buffers=[l1a, l1b, l0c],
    )
    return tvm.tir.PrimFunc(
        [nact.data], tvm.tir.BlockRealize([], True, root),
        buffer_map={nact.data: nact},
    )


def _gemm_pipeline_primfunc(use_gemm_v0=False):
    shape = (16, 16)
    left = tvm.tir.decl_buffer(shape, "float16", name="left", scope="global")
    right = tvm.tir.decl_buffer(shape, "float16", name="right", scope="global")
    output = tvm.tir.decl_buffer(shape, "float32", name="output", scope="global")
    l1a = tvm.tir.decl_buffer((256,), "float16", name="l1a", scope="shared.l1")
    l1b = tvm.tir.decl_buffer((256,), "float16", name="l1b", scope="shared.l1")
    l0a = tvm.tir.decl_buffer(
        (256,), "float16", name="l0a", scope="wmma.matrix_a"
    )
    l0b = tvm.tir.decl_buffer(
        (256,), "float16", name="l0b", scope="wmma.matrix_b"
    )
    l0c = tvm.tir.decl_buffer(
        (256,), "float32", name="l0c", scope="wmma.accumulator"
    )

    def gm_to_l1(source, destination):
        return tvm.tir.Evaluate(
            tvm.tir.call_extern(
                "handle",
                "tl::ascend::copy_gm_to_l1<half, 16, 16>",
                source.access_ptr("r"),
                destination.access_ptr("w"),
                16,
                16,
                16,
                16,
                16,
            )
        )

    operations = [
        gm_to_l1(left, l1a),
        gm_to_l1(right, l1b),
    ]
    allocations = [l1a, l1b, l0c]
    if use_gemm_v0:
        operations.append(
            tvm.tir.Evaluate(
                tvm.tir.call_extern(
                    "handle",
                    "tl.ascend_gemm_v0",
                    "gemm_v0<half, float, 16, 16, 16, false, false, 16>",
                    l1a.access_ptr("r"),
                    l1b.access_ptr("r"),
                    l0c.access_ptr("w"),
                    True,
                    16,
                )
            )
        )
    else:
        operations.extend(
            [
                tvm.tir.Evaluate(
                    tvm.tir.call_extern(
                        "handle",
                        "tl::ascend::copy_l1_to_l0a<half, 16, 16, false>",
                        l1a.access_ptr("r"),
                        l0a.access_ptr("w"),
                        16,
                        16,
                    )
                ),
                tvm.tir.Evaluate(
                    tvm.tir.call_extern(
                        "handle",
                        "tl::ascend::copy_l1_to_l0b<half, 16, 16, false>",
                        l1b.access_ptr("r"),
                        l0b.access_ptr("w"),
                        16,
                        16,
                    )
                ),
                tvm.tir.Evaluate(
                    tvm.tir.call_extern(
                        "handle",
                        "tl.ascend_mma",
                        "mma<half, float, 16, 16>",
                        l0a.access_ptr("r"),
                        l0b.access_ptr("r"),
                        l0c.access_ptr("w"),
                        True,
                        16,
                    )
                ),
            ]
        )
        allocations.extend([l0a, l0b])
    operations.append(
        tvm.tir.Evaluate(
            tvm.tir.call_extern(
                "handle",
                "tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, "
                "16, 16, false>",
                l0c.access_ptr("r"),
                output.access_ptr("w"),
                16,
                16,
                16,
                16,
                16,
                False,
                0,
            )
        )
    )
    body = tvm.tir.SeqStmt(operations)
    root = tvm.tir.Block([], [], [], "root", body, alloc_buffers=allocations)
    return tvm.tir.PrimFunc(
        [left.data, right.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={left.data: left, right.data: right, output.data: output},
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


def _dynamic_copy_primfunc():
    source = tvm.tir.decl_buffer((32,), "float32", name="source", scope="global")
    destination = tvm.tir.decl_buffer(
        (32,), "float32", name="destination", scope="shared.ub"
    )
    valid_cols = tvm.tir.Var("valid_cols", "int32")
    call = tvm.tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float32, 32>",
        source.access_ptr("r", offset=valid_cols + 1, extent=valid_cols),
        destination.access_ptr("w", offset=valid_cols * 2, extent=valid_cols),
        32,
        1,
        valid_cols,
        0,
        1,
        32,
    )
    root = tvm.tir.Block(
        [],
        [],
        [],
        "root",
        tvm.tir.Evaluate(call),
        alloc_buffers=[destination],
    )
    return tvm.tir.PrimFunc(
        [source.data, valid_cols],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source},
    )


def _tail_vector_primfunc(unary_tag="Relu"):
    x = tvm.tir.decl_buffer((1, 8), "float32", name="x", scope="global")
    y = tvm.tir.decl_buffer((1, 8), "float32", name="y", scope="global")
    output = tvm.tir.decl_buffer((1, 8), "float32", name="output", scope="global")
    ub_x = tvm.tir.decl_buffer((1, 8), "float32", name="ub_x", scope="shared.ub")
    ub_y = tvm.tir.decl_buffer((1, 8), "float32", name="ub_y", scope="shared.ub")
    ub_relu = tvm.tir.decl_buffer(
        (1, 8), "float32", name="ub_relu", scope="shared.ub"
    )
    ub_scaled = tvm.tir.decl_buffer(
        (1, 8), "float32", name="ub_scaled", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (1, 8), "float32", name="ub_output", scope="shared.ub"
    )

    def copy(operation, source, destination):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle",
            operation,
            source.access_ptr("r"),
            destination.access_ptr("w"),
            8,
            1,
            5,
            0,
            1,
            8,
        ))

    def tail(operation, tag, *arguments):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", operation, tag, *arguments, 1, 5, 8
        ))

    body = tvm.tir.SeqStmt([
        copy("tl::ascend::copy_gm_to_ub<float32, 8>", x, ub_x),
        copy("tl::ascend::copy_gm_to_ub<float32, 8>", y, ub_y),
        tail(
            "tl.ascend_tail_unary",
            unary_tag,
            ub_relu.access_ptr("w"),
            ub_x.access_ptr("r"),
        ),
        tail(
            "tl.ascend_tail_scalar",
            "Adds",
            ub_scaled.access_ptr("w"),
            ub_relu.access_ptr("r"),
            tvm.tir.FloatImm("float32", 1.5),
        ),
        tail(
            "tl.ascend_tail_binary",
            "Mul",
            ub_output.access_ptr("w"),
            ub_scaled.access_ptr("r"),
            ub_y.access_ptr("r"),
        ),
        copy("tl::ascend::copy_ub_to_gm<float32, 8>", ub_output, output),
    ])
    root = tvm.tir.Block(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=[ub_x, ub_y, ub_relu, ub_scaled, ub_output],
    )
    return tvm.tir.PrimFunc(
        [x.data, y.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={x.data: x, y.data: y, output.data: output},
    )


def _cast_primfunc(round_mode="CAST_RINT", destination_dtype="int32"):
    source = tvm.tir.decl_buffer((8,), "float32", name="source", scope="global")
    output = tvm.tir.decl_buffer(
        (8,), destination_dtype, name="output", scope="global"
    )
    ub_source = tvm.tir.decl_buffer(
        (8,), "float32", name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (8,), destination_dtype, name="ub_output", scope="shared.ub"
    )

    def copy(operation, source_buffer, destination_buffer):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle",
            operation,
            source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"),
            5,
        ))

    body = tvm.tir.SeqStmt([
        copy("copy_gm_to_ub", source, ub_source),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle",
            "tl.ascend_cast",
            ub_output.access_ptr("w"),
            ub_source.access_ptr("r"),
            round_mode,
            5,
        )),
        copy("copy_ub_to_gm", ub_output, output),
    ])
    root = tvm.tir.Block(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=[ub_source, ub_output],
    )
    return tvm.tir.PrimFunc(
        [source.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source, output.data: output},
    )


def _fill_primfunc(count=5, scalar=-3.5):
    output = tvm.tir.decl_buffer((8,), "float32", name="output", scope="global")
    ub_output = tvm.tir.decl_buffer(
        (8,), "float32", name="ub_output", scope="shared.ub"
    )
    body = tvm.tir.SeqStmt([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle",
            "tl.ascend_fill",
            ub_output.access_ptr("w", offset=1),
            tvm.tir.FloatImm("float32", scalar),
            count,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle",
            "copy_ub_to_gm",
            ub_output.access_ptr("r", offset=1),
            output.access_ptr("w", offset=2),
            count,
        )),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=[ub_output]
    )
    parameters = [output.data]
    if isinstance(count, tvm.tir.Var):
        parameters.append(count)
    return tvm.tir.PrimFunc(
        parameters,
        tvm.tir.BlockRealize([], True, root),
        buffer_map={output.data: output},
    )


def _sequence_primfunc(
    operation, *, dtype="int32", count=7, first=5, difference=-2,
    template_dtype=None,
):
    output = tvm.tir.decl_buffer((12,), dtype, name="output", scope="global")
    ub_output = tvm.tir.decl_buffer(
        (12,), dtype, name="ub_output", scope="shared.ub"
    )
    dtype_names = {
        "float16": "half", "float32": "float", "int16": "int16_t",
        "int32": "int", "uint16": "uint16_t", "uint32": "uint32_t",
    }
    template_name = "CreateVecIndex" if operation == "createvecindex" else "ArithProgression"
    arguments = [
        f"{template_name}<{template_dtype or dtype_names[dtype]}>",
        ub_output.access_ptr("w", offset=2), first,
    ]
    if operation == "arith_progression":
        arguments.append(difference)
    arguments.append(count)
    body = tvm.tir.SeqStmt([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", f"tl.ascend_{operation}", *arguments,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "copy_ub_to_gm", ub_output.access_ptr("r", offset=2),
            output.access_ptr("w", offset=1), count,
        )),
    ])
    root = tvm.tir.Block([], [], [], "root", body, alloc_buffers=[ub_output])
    parameters = [output.data]
    for value in (count, first, difference):
        if isinstance(value, tvm.tir.Var) and not any(
            value.same_as(parameter) for parameter in parameters
        ):
            parameters.append(value)
    return tvm.tir.PrimFunc(
        parameters,
        tvm.tir.BlockRealize([], True, root),
        buffer_map={output.data: output},
    )


def _scalar_vector_primfunc(operation):
    source = tvm.tir.decl_buffer((5,), "float32", name="source", scope="global")
    initial = tvm.tir.decl_buffer((5,), "float32", name="initial", scope="global")
    output = tvm.tir.decl_buffer((5,), "float32", name="output", scope="global")
    ub_source = tvm.tir.decl_buffer(
        (5,), "float32", name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (5,), "float32", name="ub_output", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle",
            name,
            source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"),
            5,
        ))

    statements = [copy("copy_gm_to_ub", source, ub_source)]
    if operation == "axpy":
        statements.append(copy("copy_gm_to_ub", initial, ub_output))
    statements.extend([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle",
            f"tl.ascend_{operation}",
            ub_output.access_ptr("w"),
            ub_source.access_ptr("r"),
            tvm.tir.FloatImm("float32", 0.25),
            5,
        )),
        copy("copy_ub_to_gm", ub_output, output),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=[ub_source, ub_output],
    )
    return tvm.tir.PrimFunc(
        [source.data, initial.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={
            source.data: source,
            initial.data: initial,
            output.data: output,
        },
    )


def _bitwise_primfunc(operation, *, dtype="int16", shift=1, with_scratch=False):
    left = tvm.tir.decl_buffer((8,), dtype, name="left", scope="global")
    right = tvm.tir.decl_buffer((8,), dtype, name="right", scope="global")
    output = tvm.tir.decl_buffer((8,), dtype, name="output", scope="global")
    ub_left = tvm.tir.decl_buffer((8,), dtype, name="ub_left", scope="shared.ub")
    ub_right = tvm.tir.decl_buffer((8,), dtype, name="ub_right", scope="shared.ub")
    ub_output = tvm.tir.decl_buffer((8,), dtype, name="ub_output", scope="shared.ub")
    scratch = tvm.tir.decl_buffer((6,), dtype, name="scratch", scope="shared.ub")

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    destination = ub_output.access_ptr("w", offset=1, extent=6)
    source = ub_left.access_ptr("r", offset=1, extent=6)
    arguments = [destination, source]
    if operation in {"bitwise_and", "bitwise_or"}:
        arguments.extend([ub_right.access_ptr("r", offset=1, extent=6), 6])
    elif operation == "bitwise_xor":
        arguments.append(ub_right.access_ptr("r", offset=1, extent=6))
        if with_scratch:
            arguments.append(scratch.access_ptr("w"))
    elif operation == "bitwise_not":
        arguments.append(6)
    else:
        arguments.extend([shift, 6])
    statements = [copy("copy_gm_to_ub", left, ub_left, 8)]
    if operation in {"bitwise_and", "bitwise_or", "bitwise_xor"}:
        statements.append(copy("copy_gm_to_ub", right, ub_right, 8))
    statements.extend([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", f"tl.ascend_{operation}", *arguments,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "copy_ub_to_gm", ub_output.access_ptr("r", offset=1),
            output.access_ptr("w", offset=1), 6,
        )),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=[ub_left, ub_right, ub_output, scratch],
    )
    parameters = [left.data, right.data, output.data]
    if isinstance(shift, tvm.tir.Var):
        parameters.append(shift)
    return tvm.tir.PrimFunc(
        parameters, tvm.tir.BlockRealize([], True, root),
        buffer_map={left.data: left, right.data: right, output.data: output},
    )


def _mul_add_dst_primfunc(*, dtype="float32", right_dtype=None):
    right_dtype = right_dtype or dtype
    left = tvm.tir.decl_buffer((8,), dtype, name="left", scope="global")
    right = tvm.tir.decl_buffer((8,), right_dtype, name="right", scope="global")
    initial = tvm.tir.decl_buffer((8,), dtype, name="initial", scope="global")
    output = tvm.tir.decl_buffer((8,), dtype, name="output", scope="global")
    ub_left = tvm.tir.decl_buffer((8,), dtype, name="ub_left", scope="shared.ub")
    ub_right = tvm.tir.decl_buffer(
        (8,), right_dtype, name="ub_right", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (8,), dtype, name="ub_output", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    statements = [
        copy("copy_gm_to_ub", left, ub_left, 8),
        copy("copy_gm_to_ub", right, ub_right, 8),
        copy("copy_gm_to_ub", initial, ub_output, 8),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_mul_add_dst",
            ub_output.access_ptr("rw", offset=1, extent=6),
            ub_left.access_ptr("r", offset=1, extent=6),
            ub_right.access_ptr("r", offset=1, extent=6), 6,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "copy_ub_to_gm", ub_output.access_ptr("r", offset=1),
            output.access_ptr("w", offset=1), 6,
        )),
    ]
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=[ub_left, ub_right, ub_output],
    )
    return tvm.tir.PrimFunc(
        [left.data, right.data, initial.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={
            left.data: left,
            right.data: right,
            initial.data: initial,
            output.data: output,
        },
    )


def _gather_primfunc(
    *, dtype="float32", offset_dtype="uint32", base=0, with_scratch=False
):
    itemsize = np.dtype(dtype).itemsize
    aligned_elements = 32 // itemsize
    source_size = aligned_elements + 16
    destination_size = aligned_elements + 5
    offset_start = 8
    source = tvm.tir.decl_buffer(
        (source_size,), dtype, name="source", scope="global"
    )
    offsets = tvm.tir.decl_buffer(
        (offset_start + 5,), offset_dtype, name="offsets", scope="global"
    )
    output = tvm.tir.decl_buffer((5,), dtype, name="output", scope="global")
    ub_source = tvm.tir.decl_buffer(
        (source_size,), dtype, name="ub_source", scope="shared.ub"
    )
    ub_offsets = tvm.tir.decl_buffer(
        (offset_start + 5,), offset_dtype, name="ub_offsets", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (destination_size,), dtype, name="ub_output", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer((64,), "uint8", name="scratch", scope="shared.ub")

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    gather_arguments = [
        ub_output.access_ptr("w", offset=aligned_elements, extent=5),
        ub_source.access_ptr("r", offset=aligned_elements, extent=16),
        ub_offsets.access_ptr("r", offset=offset_start, extent=5),
        base,
        5,
    ]
    if with_scratch:
        gather_arguments.append(scratch.access_ptr("w", offset=32, extent=32))
    statements = [
        copy("copy_gm_to_ub", source, ub_source, source_size),
        copy("copy_gm_to_ub", offsets, ub_offsets, offset_start + 5),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_gather", *gather_arguments,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "copy_ub_to_gm",
            ub_output.access_ptr("r", offset=aligned_elements),
            output.access_ptr("w"), 5,
        )),
    ]
    alloc_buffers = [ub_source, ub_offsets, ub_output]
    if with_scratch:
        alloc_buffers.append(scratch)
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=alloc_buffers,
    )
    parameters = [source.data, offsets.data, output.data]
    if isinstance(base, tvm.tir.Var):
        parameters.append(base)
    return tvm.tir.PrimFunc(
        parameters,
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source, offsets.data: offsets, output.data: output},
    )


def _gatherb_primfunc(
    *, dtype="uint16", offset_dtype="uint32", repeat=2,
    dst_block_stride=1, dst_repeat_stride=8, template_dtype=None,
):
    itemsize = np.dtype(dtype).itemsize
    elements_per_block = 32 // itemsize
    source_count = 24 * elements_per_block
    storage_repeat = repeat if isinstance(repeat, int) else 2
    offset_count = max(0, storage_repeat) * 8
    destination_blocks = max(0, storage_repeat - 1) * dst_repeat_stride + 8
    destination_count = max(1, destination_blocks * elements_per_block)
    source = tvm.tir.decl_buffer(
        (source_count,), dtype, name="source", scope="global"
    )
    offsets = tvm.tir.decl_buffer(
        (max(1, offset_count),), offset_dtype, name="offsets", scope="global"
    )
    output = tvm.tir.decl_buffer(
        (destination_count,), dtype, name="output", scope="global"
    )
    ub_source = tvm.tir.decl_buffer(
        (source_count,), dtype, name="ub_source", scope="shared.ub"
    )
    ub_offsets = tvm.tir.decl_buffer(
        (max(1, offset_count),), offset_dtype,
        name="ub_offsets", scope="shared.ub",
    )
    ub_output = tvm.tir.decl_buffer(
        (destination_count,), dtype, name="ub_output", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    template_dtype = template_dtype or {
        "float16": "half", "float32": "float", "int8": "int8_t",
        "uint8": "uint8_t", "int16": "int16_t", "uint16": "uint16_t",
        "int32": "int", "uint32": "uint32_t",
    }[dtype]
    body = tvm.tir.SeqStmt([
        copy("copy_gm_to_ub", source, ub_source, source_count),
        copy("copy_gm_to_ub", offsets, ub_offsets, max(1, offset_count)),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_gatherb", f"Gatherb<{template_dtype}>",
            ub_output.access_ptr("w"), ub_source.access_ptr("r"),
            ub_offsets.access_ptr("r"), repeat,
            dst_block_stride, dst_repeat_stride,
        )),
        copy("copy_ub_to_gm", ub_output, output, destination_count),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body,
        alloc_buffers=[ub_source, ub_offsets, ub_output],
    )
    parameters = [source.data, offsets.data, output.data]
    if isinstance(repeat, tvm.tir.Var):
        parameters.append(repeat)
    return tvm.tir.PrimFunc(
        parameters,
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source, offsets.data: offsets, output.data: output},
    )


def _gather_mask_primfunc(
    *, dtype="float32", pattern="P0101", custom=False,
    index_dtype="uint32", with_scratch=False,
):
    source_count = 32
    destination_count = 8 if custom else source_count
    source = tvm.tir.decl_buffer(
        (source_count,), dtype, name="source", scope="global"
    )
    output = tvm.tir.decl_buffer(
        (destination_count,), dtype, name="output", scope="global"
    )
    indices = tvm.tir.decl_buffer(
        (8,), index_dtype, name="indices", scope="global"
    )
    ub_source = tvm.tir.decl_buffer(
        (source_count,), dtype, name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (destination_count,), dtype, name="ub_output", scope="shared.ub"
    )
    ub_indices = tvm.tir.decl_buffer(
        (8,), index_dtype, name="ub_indices", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer((64,), "uint8", name="scratch", scope="shared.ub")

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    template_dtype = {
        "float16": "half", "float32": "float", "int8": "int8_t",
        "uint8": "uint8_t", "int16": "int16_t", "uint16": "uint16_t",
        "int32": "int", "uint32": "uint32_t",
    }[dtype]
    selector = ub_indices.access_ptr("r") if custom else pattern
    operation_arguments = [
        f"GatherMask<{template_dtype}>",
        ub_output.access_ptr("w"), ub_source.access_ptr("r"), selector,
    ]
    if with_scratch:
        operation_arguments.append(scratch.access_ptr("w"))
    statements = [copy("copy_gm_to_ub", source, ub_source, source_count)]
    if custom:
        statements.append(copy("copy_gm_to_ub", indices, ub_indices, 8))
    statements.extend([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_gather_mask", *operation_arguments,
        )),
        copy("copy_ub_to_gm", ub_output, output, destination_count),
    ])
    alloc_buffers = [ub_source, ub_output, ub_indices]
    if with_scratch:
        alloc_buffers.append(scratch)
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=alloc_buffers,
    )
    parameters = [source.data, output.data]
    buffer_map = {source.data: source, output.data: output}
    if custom:
        parameters.insert(1, indices.data)
        buffer_map[indices.data] = indices
    return tvm.tir.PrimFunc(
        parameters, tvm.tir.BlockRealize([], True, root), buffer_map=buffer_map
    )


def _transpose_primfunc(*, shape=(16, 32), dtype="float32", dst_shape=None):
    dst_shape = dst_shape or tuple(reversed(shape))
    source = tvm.tir.decl_buffer(shape, dtype, name="source", scope="global")
    output = tvm.tir.decl_buffer(dst_shape, dtype, name="output", scope="global")
    ub_source = tvm.tir.decl_buffer(
        shape, dtype, name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        dst_shape, dtype, name="ub_output", scope="shared.ub"
    )
    source_count = int(np.prod(shape))
    destination_count = int(np.prod(dst_shape))

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    body = tvm.tir.SeqStmt([
        copy("copy_gm_to_ub", source, ub_source, source_count),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_transpose",
            ub_output.access_ptr("w"), ub_source.access_ptr("r"),
        )),
        copy("copy_ub_to_gm", ub_output, output, destination_count),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=[ub_source, ub_output]
    )
    return tvm.tir.PrimFunc(
        [source.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source, output.data: output},
    )


def _reinterpretcast_primfunc(*, destination_extent=8, cast_type="uint16_t"):
    source = tvm.tir.decl_buffer((4,), "uint32", name="source", scope="global")
    view_output = tvm.tir.decl_buffer(
        (destination_extent,), "uint16", name="view_output", scope="global"
    )
    after_output = tvm.tir.decl_buffer(
        (4,), "uint32", name="after_output", scope="global"
    )
    ub_source = tvm.tir.decl_buffer(
        (4,), "uint32", name="ub_source", scope="shared.ub"
    )
    ub_view = tvm.tir.decl_buffer(
        (destination_extent,), "uint16", name="ub_view", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    body = tvm.tir.SeqStmt([
        copy("copy_gm_to_ub", source, ub_source, 4),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_reinterpretcast",
            ub_view.access_ptr("w"), ub_source.access_ptr("r"), cast_type,
        )),
        copy("copy_ub_to_gm", ub_view, view_output, destination_extent),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_fill", ub_view.access_ptr("w"),
            0x1234, destination_extent,
        )),
        copy("copy_ub_to_gm", ub_source, after_output, 4),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=[ub_source, ub_view]
    )
    return tvm.tir.PrimFunc(
        [source.data, view_output.data, after_output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={
            source.data: source,
            view_output.data: view_output,
            after_output.data: after_output,
        },
    )


def _topk_primfunc(
    *, dtype="float32", k=4, actual_num=7, max_actual_num=9,
    repeat_times=None, destination_extent=None, template_dtype=None,
):
    aligned_count = ((max_actual_num + 31) // 32) * 32
    repeat_times = repeat_times or aligned_count // 32
    destination_extent = destination_extent or 2 * k
    source = tvm.tir.decl_buffer(
        (aligned_count,), dtype, name="source", scope="global"
    )
    output = tvm.tir.decl_buffer(
        (destination_extent,), dtype, name="output", scope="global"
    )
    ub_source = tvm.tir.decl_buffer(
        (aligned_count,), dtype, name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (destination_extent,), dtype, name="ub_output", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer(
        (aligned_count * 6,), dtype, name="scratch", scope="shared.ub"
    )
    dtype_name = template_dtype or {"float16": "half", "float32": "float"}[dtype]

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    body = tvm.tir.SeqStmt([
        copy("copy_gm_to_ub", source, ub_source, aligned_count),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_topk", f"TopK<{dtype_name}>",
            ub_output.access_ptr("w"), ub_source.access_ptr("r"),
            scratch.access_ptr("w"), k, repeat_times, actual_num,
            max_actual_num,
        )),
        copy("copy_ub_to_gm", ub_output, output, 2 * k),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body,
        alloc_buffers=[ub_source, ub_output, scratch],
    )
    parameters = [source.data, output.data]
    if isinstance(actual_num, tvm.tir.Var):
        parameters.append(actual_num)
    return tvm.tir.PrimFunc(
        parameters,
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source, output.data: output},
    )


def _sort32_primfunc(*, dtype="float32", repeat_times=2, output_elements=None):
    count = 64
    multiplier = 4 if dtype == "float16" else 2
    output_elements = output_elements or multiplier * count
    source = tvm.tir.decl_buffer((count,), dtype, name="source", scope="global")
    indices = tvm.tir.decl_buffer(
        (count,), "uint32", name="indices", scope="global"
    )
    output = tvm.tir.decl_buffer(
        (output_elements,), dtype, name="output", scope="global"
    )
    ub_source = tvm.tir.decl_buffer(
        (count,), dtype, name="ub_source", scope="shared.ub"
    )
    ub_indices = tvm.tir.decl_buffer(
        (count,), "uint32", name="ub_indices", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (output_elements,), dtype, name="ub_output", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer, extent):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), extent,
        ))

    body = tvm.tir.SeqStmt([
        copy("copy_gm_to_ub", source, ub_source, count),
        copy("copy_gm_to_ub", indices, ub_indices, count),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_sort32", ub_output.access_ptr("w"),
            ub_source.access_ptr("r"), ub_indices.access_ptr("r"),
            repeat_times,
        )),
        copy(
            "copy_ub_to_gm", ub_output, output,
            multiplier * 32 * repeat_times,
        ),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body,
        alloc_buffers=[ub_source, ub_indices, ub_output],
    )
    return tvm.tir.PrimFunc(
        [source.data, indices.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={
            source.data: source,
            indices.data: indices,
            output.data: output,
        },
    )


def _sort_primfunc(
    *, dtype="float32", actual_num=37, capacity=64, repeat_times=None,
    destination_extent=None, template_dtype=None,
):
    repeat_times = (
        (actual_num + 31) // 32 if repeat_times is None else repeat_times
    )
    destination_extent = destination_extent or 2 * capacity
    source = tvm.tir.decl_buffer(
        (capacity,), dtype, name="source", scope="global"
    )
    output = tvm.tir.decl_buffer(
        (2 * capacity,), dtype, name="output", scope="global"
    )
    ub_source = tvm.tir.decl_buffer(
        (capacity,), dtype, name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (destination_extent,), dtype, name="ub_output", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer(
        (capacity * 8,), dtype, name="scratch", scope="shared.ub"
    )
    dtype_name = template_dtype or {"float16": "half", "float32": "float"}[dtype]

    def copy(name, source_buffer, destination_buffer, extent):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), extent,
        ))

    body = tvm.tir.SeqStmt([
        copy("copy_gm_to_ub", source, ub_source, capacity),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_sort", f"Sort<{dtype_name}>",
            ub_output.access_ptr("w"), ub_source.access_ptr("r"),
            scratch.access_ptr("w"), repeat_times, actual_num,
        )),
        copy(
            "copy_ub_to_gm", ub_output, output,
            actual_num * 2,
        ),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body,
        alloc_buffers=[ub_source, ub_output, scratch],
    )
    parameters = [source.data, output.data]
    if isinstance(actual_num, tvm.tir.Var):
        parameters.append(actual_num)
    return tvm.tir.PrimFunc(
        parameters,
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source, output.data: output},
    )


def _merge_sort_primfunc(
    num_ways=2, *, with_scratch=False, destination_extent=None, block_length=4,
    dtype="float32", reported_block_length=None,
):
    record_width = 4 if dtype == "float16" else 2
    dtype_name = "half" if dtype == "float16" else "float"
    source_extent = record_width * block_length
    destination_extent = destination_extent or num_ways * source_extent
    sources = [
        tvm.tir.decl_buffer(
            (source_extent,), dtype, name=f"source{index}", scope="global"
        )
        for index in range(num_ways)
    ]
    output = tvm.tir.decl_buffer(
        (num_ways * source_extent,), dtype, name="output", scope="global"
    )
    ub_sources = [
        tvm.tir.decl_buffer(
            (source_extent,), dtype, name=f"ub_source{index}",
            scope="shared.ub",
        )
        for index in range(num_ways)
    ]
    ub_output = tvm.tir.decl_buffer(
        (destination_extent,), dtype, name="ub_output", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer(
        (num_ways * source_extent * 4,), "uint8", name="scratch",
        scope="shared.ub",
    )

    def copy(name, source_buffer, destination_buffer, extent):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), extent,
        ))

    arguments = [
        f"MergeSort<{dtype_name}>", num_ways, ub_output.access_ptr("w"),
    ]
    if with_scratch:
        arguments.append(scratch.access_ptr("w"))
    arguments.extend(source.access_ptr("r") for source in ub_sources)
    arguments.extend([
        block_length if reported_block_length is None else reported_block_length
    ] * num_ways)
    statements = [
        copy("copy_gm_to_ub", source, ub_source, source_extent)
        for source, ub_source in zip(sources, ub_sources)
    ]
    statements.extend([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_merge_sort", *arguments,
        )),
        copy(
            "copy_ub_to_gm", ub_output, output,
            num_ways * source_extent,
        ),
    ])
    alloc_buffers = [*ub_sources, ub_output]
    if with_scratch:
        alloc_buffers.append(scratch)
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=alloc_buffers,
    )
    return tvm.tir.PrimFunc(
        [*(source.data for source in sources), output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={
            **{source.data: source for source in sources},
            output.data: output,
        },
    )


def _atomic_add_ub_primfunc(*, dtype="float32", rows=2, cols=None, stride=None):
    cols = cols or (16 if dtype == "float16" else 8)
    stride = stride or cols + 4
    output = tvm.tir.decl_buffer(
        (rows, stride), dtype, name="output", scope="global"
    )
    first = tvm.tir.decl_buffer(
        (rows, cols), dtype, name="first", scope="shared.ub"
    )
    second = tvm.tir.decl_buffer(
        (rows, cols), dtype, name="second", scope="shared.ub"
    )
    template_dtype = {"float16": "half", "float32": "float"}[dtype]

    def fill(buffer, value):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_fill", buffer.access_ptr("w"), value,
            rows * cols,
        ))

    def atomic(source):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle",
            f"tl::ascend::atomic_add_ub_to_gm<{template_dtype}, {cols}, {rows}>",
            source.access_ptr("r"), output.access_ptr("w"), stride, rows, cols,
        ))

    root = tvm.tir.Block(
        [], [], [], "root",
        tvm.tir.SeqStmt([
            fill(first, 1), fill(second, 2), atomic(first), atomic(second),
        ]),
        alloc_buffers=[first, second],
    )
    return tvm.tir.PrimFunc(
        [output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={output.data: output},
    )


def _atomic_add_ub_multicore_primfunc(num_cores=2):
    output = tvm.tir.decl_buffer((8,), "float32", name="output", scope="global")
    source = tvm.tir.decl_buffer(
        (1, 8), "float32", name="source", scope="shared.ub"
    )
    body = tvm.tir.SeqStmt([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_fill", source.access_ptr("w"), 1.0, 8,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl::ascend::atomic_add_ub_to_gm<float, 8, 1>",
            source.access_ptr("r"), output.access_ptr("w"), 8, 1, 8,
        )),
    ])
    root = tvm.tir.BlockRealize(
        [], True,
        tvm.tir.Block([], [], [], "root", body, alloc_buffers=[source]),
    )
    block_var = tvm.tir.Var("blockIdx.x", "int32")
    thread_axis = tvm.tir.IterVar(
        tvm.ir.Range(0, num_cores), block_var,
        tvm.tir.IterVar.ThreadIndex, "blockIdx.x",
    )
    threaded = tvm.tir.AttrStmt(
        thread_axis, "thread_extent", num_cores, root
    )
    return tvm.tir.PrimFunc(
        [output.data], threaded, buffer_map={output.data: output}
    )


def _scratch_unary_primfunc(
    operation, *, with_scratch=False, in_place=False, dtype="float32"
):
    source = tvm.tir.decl_buffer((5,), dtype, name="source", scope="global")
    output = tvm.tir.decl_buffer((5,), dtype, name="output", scope="global")
    ub_source = tvm.tir.decl_buffer(
        (5,), dtype, name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (5,), dtype, name="ub_output", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer(
        (5,), "float32", name="scratch", scope="shared.ub"
    )
    destination = ub_source if in_place else ub_output

    def copy(name, source_buffer, destination_buffer):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle",
            name,
            source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"),
            5,
        ))

    operation_arguments = [
        destination.access_ptr("w"),
        ub_source.access_ptr("r"),
    ]
    if with_scratch:
        operation_arguments.append(scratch.access_ptr("w"))
    operation_arguments.append(5)
    body = tvm.tir.SeqStmt([
        copy("copy_gm_to_ub", source, ub_source),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", f"tl.ascend_{operation}", *operation_arguments
        )),
        copy("copy_ub_to_gm", destination, output),
    ])
    alloc_buffers = [ub_source, ub_output]
    if with_scratch:
        alloc_buffers.append(scratch)
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=alloc_buffers
    )
    return tvm.tir.PrimFunc(
        [source.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source, output.data: output},
    )


def _elementwise_experiment_primfunc(operation, *, dtype="float32", count=8):
    source = tvm.tir.decl_buffer((8,), dtype, name="source", scope="global")
    right = tvm.tir.decl_buffer((8,), dtype, name="right", scope="global")
    output = tvm.tir.decl_buffer((8,), dtype, name="output", scope="global")
    ub_source = tvm.tir.decl_buffer(
        (8,), dtype, name="ub_source", scope="shared.ub"
    )
    ub_right = tvm.tir.decl_buffer(
        (8,), dtype, name="ub_right", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (8,), dtype, name="ub_output", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), 8,
        ))

    arguments = [ub_output.access_ptr("w"), ub_source.access_ptr("r")]
    if operation == "sub_experiment":
        arguments.append(ub_right.access_ptr("r"))
    elif operation == "mins_experiment":
        arguments.append(1.5)
    arguments.append(count)
    statements = [copy("copy_gm_to_ub", source, ub_source)]
    parameters = [source.data]
    buffer_map = {source.data: source, output.data: output}
    if operation == "sub_experiment":
        statements.append(copy("copy_gm_to_ub", right, ub_right))
        parameters.append(right.data)
        buffer_map[right.data] = right
    statements.extend([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", f"tl.ascend_{operation}", *arguments,
        )),
        copy("copy_ub_to_gm", ub_output, output),
    ])
    parameters.append(output.data)
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=[ub_source, ub_right, ub_output],
    )
    return tvm.tir.PrimFunc(
        parameters, tvm.tir.BlockRealize([], True, root), buffer_map=buffer_map
    )


def _reduce_sum_experiment_primfunc(
    *, dtype="float32", count=8, with_scratch=True, scratch_dtype=None
):
    source = tvm.tir.decl_buffer((8,), dtype, name="source", scope="global")
    output = tvm.tir.decl_buffer((1,), dtype, name="output", scope="global")
    ub_source = tvm.tir.decl_buffer(
        (8,), dtype, name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (8,), dtype, name="ub_output", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer(
        (8,), scratch_dtype or dtype, name="scratch", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer, elements):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), elements,
        ))

    arguments = [ub_output.access_ptr("w"), ub_source.access_ptr("r")]
    if with_scratch:
        arguments.append(scratch.access_ptr("w"))
    arguments.append(count)
    statements = [
        copy("copy_gm_to_ub", source, ub_source, 8),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_reducesum_experiment", *arguments,
        )),
        copy("copy_ub_to_gm", ub_output, output, 1),
    ]
    alloc_buffers = [ub_source, ub_output]
    if with_scratch:
        alloc_buffers.append(scratch)
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=alloc_buffers,
    )
    return tvm.tir.PrimFunc(
        [source.data, output.data], tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source, output.data: output},
    )


def _sum_experiment_primfunc(
    *, dtype="float32", outer=3, inner=8, valid=5, template_dtype=None
):
    source = tvm.tir.decl_buffer(
        (outer * inner,), dtype, name="source", scope="global"
    )
    output = tvm.tir.decl_buffer((outer,), dtype, name="output", scope="global")
    ub_source = tvm.tir.decl_buffer(
        (outer * inner,), dtype, name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (outer,), dtype, name="ub_output", scope="shared.ub"
    )
    dtype_name = template_dtype or ("half" if dtype == "float16" else "float")

    def copy(name, source_buffer, destination_buffer, elements):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), elements,
        ))

    statements = [
        copy("copy_gm_to_ub", source, ub_source, outer * inner),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_sum_experiment",
            f"Sum_experiment<{dtype_name}>", ub_output.access_ptr("w"),
            ub_source.access_ptr("r"), outer, inner, valid,
        )),
        copy("copy_ub_to_gm", ub_output, output, outer),
    ]
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=[ub_source, ub_output],
    )
    return tvm.tir.PrimFunc(
        [source.data, output.data], tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source, output.data: output},
    )


def _pow_clamp_primfunc(operation, *, with_scratch=False):
    source = tvm.tir.decl_buffer((5,), "float32", name="source", scope="global")
    exponent = tvm.tir.decl_buffer((5,), "float32", name="exponent", scope="global")
    output = tvm.tir.decl_buffer((5,), "float32", name="output", scope="global")
    ub_source = tvm.tir.decl_buffer(
        (5,), "float32", name="ub_source", scope="shared.ub"
    )
    ub_exponent = tvm.tir.decl_buffer(
        (5,), "float32", name="ub_exponent", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (5,), "float32", name="ub_output", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer(
        (5,), "float32", name="scratch", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), 5,
        ))

    statements = [copy("copy_gm_to_ub", source, ub_source)]
    if operation == "pow":
        statements.append(copy("copy_gm_to_ub", exponent, ub_exponent))
        arguments = [
            ub_output.access_ptr("w"), ub_source.access_ptr("r"),
            ub_exponent.access_ptr("r"),
        ]
        if with_scratch:
            arguments.append(scratch.access_ptr("w"))
    elif operation == "clamp":
        arguments = [ub_output.access_ptr("w"), ub_source.access_ptr("r")]
        if with_scratch:
            arguments.append(scratch.access_ptr("w"))
        arguments.extend([
            tvm.tir.FloatImm("float32", -1.0),
            tvm.tir.FloatImm("float32", 2.0),
            5,
        ])
    else:
        arguments = [ub_output.access_ptr("w"), ub_source.access_ptr("r")]
        if with_scratch:
            arguments.append(scratch.access_ptr("w"))
        arguments.extend([tvm.tir.FloatImm("float32", 1.5), 5])
    statements.extend([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", f"tl.ascend_{operation}", *arguments
        )),
        copy("copy_ub_to_gm", ub_output, output),
    ])
    alloc_buffers = [ub_source, ub_exponent, ub_output]
    if with_scratch:
        alloc_buffers.append(scratch)
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=alloc_buffers,
    )
    return tvm.tir.PrimFunc(
        [source.data, exponent.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={
            source.data: source,
            exponent.data: exponent,
            output.data: output,
        },
    )


def _broadcast_primfunc(source_shape, *, with_scratch=False):
    destination_shape = (2, 3)
    source = tvm.tir.decl_buffer(
        source_shape, "float32", name="source", scope="global"
    )
    output = tvm.tir.decl_buffer(
        destination_shape, "float32", name="output", scope="global"
    )
    ub_source = tvm.tir.decl_buffer(
        source_shape, "float32", name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        destination_shape, "float32", name="ub_output", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer(
        (6,), "float32", name="scratch", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    arguments = [ub_output.access_ptr("w"), ub_source.access_ptr("r")]
    if with_scratch:
        arguments.append(scratch.access_ptr("w"))
    arguments.extend([2, *destination_shape, *source_shape])
    body = tvm.tir.SeqStmt([
        copy("copy_gm_to_ub", source, ub_source, int(np.prod(source_shape))),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_broadcast", *arguments
        )),
        copy("copy_ub_to_gm", ub_output, output, 6),
    ])
    alloc_buffers = [ub_source, ub_output]
    if with_scratch:
        alloc_buffers.append(scratch)
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=alloc_buffers
    )
    return tvm.tir.PrimFunc(
        [source.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source, output.data: output},
    )


def _compare_primfunc(mode, *, scalar=False):
    left = tvm.tir.decl_buffer((8,), "float32", name="left", scope="global")
    right = tvm.tir.decl_buffer((8,), "float32", name="right", scope="global")
    output = tvm.tir.decl_buffer((1,), "uint8", name="output", scope="global")
    ub_left = tvm.tir.decl_buffer(
        (8,), "float32", name="ub_left", scope="shared.ub"
    )
    ub_right = tvm.tir.decl_buffer(
        (8,), "float32", name="ub_right", scope="shared.ub"
    )
    ub_mask = tvm.tir.decl_buffer(
        (1,), "uint8", name="ub_mask", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    statements = [copy("copy_gm_to_ub", left, ub_left, 8)]
    operation = "compare_scalar" if scalar else "compare"
    if scalar:
        right_argument = tvm.tir.FloatImm("float32", 0.0)
    else:
        statements.append(copy("copy_gm_to_ub", right, ub_right, 8))
        right_argument = ub_right.access_ptr("r")
    statements.extend([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", f"tl.ascend_{operation}", ub_mask.access_ptr("w"),
            ub_left.access_ptr("r"), right_argument, mode, 8,
        )),
        copy("copy_ub_to_gm", ub_mask, output, 1),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=[ub_left, ub_right, ub_mask],
    )
    return tvm.tir.PrimFunc(
        [left.data, right.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={left.data: left, right.data: right, output.data: output},
    )


def _compare_scalar_buffer_primfunc(mode, scalar_index=1):
    left = tvm.tir.decl_buffer((8,), "float32", name="left", scope="global")
    scalars = tvm.tir.decl_buffer(
        (2,), "float32", name="scalars", scope="global"
    )
    output = tvm.tir.decl_buffer((1,), "uint8", name="output", scope="global")
    ub_left = tvm.tir.decl_buffer(
        (8,), "float32", name="ub_left", scope="shared.ub"
    )
    ub_scalars = tvm.tir.decl_buffer(
        (2,), "float32", name="ub_scalars", scope="shared.ub"
    )
    ub_mask = tvm.tir.decl_buffer(
        (1,), "uint8", name="ub_mask", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    statements = [
        copy("copy_gm_to_ub", left, ub_left, 8),
        copy("copy_gm_to_ub", scalars, ub_scalars, 2),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_compare_scalar", ub_mask.access_ptr("w"),
            ub_left.access_ptr("r"), ub_scalars.access_ptr("r"), scalar_index,
            mode, 8,
        )),
        copy("copy_ub_to_gm", ub_mask, output, 1),
    ]
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=[ub_left, ub_scalars, ub_mask],
    )
    return tvm.tir.PrimFunc(
        [left.data, scalars.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={
            left.data: left, scalars.data: scalars, output.data: output,
        },
    )


def _compare_select_primfunc(
    *, scalar_select=False, buffer_select=False, with_scratch=False,
    select_mode=None,
):
    left = tvm.tir.decl_buffer((8,), "float32", name="left", scope="global")
    right = tvm.tir.decl_buffer((8,), "float32", name="right", scope="global")
    output = tvm.tir.decl_buffer((8,), "float32", name="output", scope="global")
    ub_left = tvm.tir.decl_buffer(
        (8,), "float32", name="ub_left", scope="shared.ub"
    )
    ub_right = tvm.tir.decl_buffer(
        (8,), "float32", name="ub_right", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (8,), "float32", name="ub_output", scope="shared.ub"
    )
    ub_mask = tvm.tir.decl_buffer(
        (1,), "uint8", name="ub_mask", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer(
        (8,), "float32", name="scratch", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    select_arguments = [
        ub_output.access_ptr("w"), ub_mask.access_ptr("r"),
        ub_left.access_ptr("r"),
    ]
    if with_scratch:
        select_arguments.append(scratch.access_ptr("w"))
    if buffer_select:
        select_arguments.extend([
            0, ub_right.access_ptr("r"), 1,
            select_mode or "VSEL_CMPMASK_SPR", 8,
        ])
    elif scalar_select:
        select_arguments.extend([
            1, tvm.tir.FloatImm("float32", 10.0),
            "VSEL_TENSOR_SCALAR_MODE", 8, "float32", "uint8",
        ])
    else:
        select_arguments.extend([
            2, ub_right.access_ptr("r"),
            select_mode or "VSEL_TENSOR_TENSOR_MODE", 8,
        ])
    statements = [
        copy("copy_gm_to_ub", left, ub_left, 8),
        copy("copy_gm_to_ub", right, ub_right, 8),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_compare", ub_mask.access_ptr("w"),
            ub_left.access_ptr("r"), ub_right.access_ptr("r"), "LT", 8,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_select", *select_arguments
        )),
        copy("copy_ub_to_gm", ub_output, output, 8),
    ]
    alloc_buffers = [ub_left, ub_right, ub_output, ub_mask]
    if with_scratch:
        alloc_buffers.append(scratch)
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=alloc_buffers,
    )
    return tvm.tir.PrimFunc(
        [left.data, right.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={left.data: left, right.data: right, output.data: output},
    )


def _tail_compare_primfunc(*, scalar=False, storage_cols=2):
    left = tvm.tir.decl_buffer((2, 10), "float32", name="left", scope="global")
    right = tvm.tir.decl_buffer((2, 10), "float32", name="right", scope="global")
    ub_left = tvm.tir.decl_buffer(
        (2, 10), "float32", name="ub_left", scope="shared.ub"
    )
    ub_right = tvm.tir.decl_buffer(
        (2, 10), "float32", name="ub_right", scope="shared.ub"
    )
    ub_mask = tvm.tir.decl_buffer(
        (2, 32), "uint8", name="ub_mask", scope="shared.ub"
    )

    def load(source_buffer, destination_buffer):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "copy_gm_to_ub", source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), 10, 2, 9, 0, 2, 10,
        ))

    statements = [load(left, ub_left)]
    if scalar:
        right_argument = tvm.tir.FloatImm("float32", 0.0)
        operation = "tl.ascend_tail_compare_scalar"
    else:
        statements.append(load(right, ub_right))
        right_argument = ub_right.access_ptr("r")
        operation = "tl.ascend_tail_compare"
    statements.append(tvm.tir.Evaluate(tvm.tir.call_extern(
        "handle", operation, ub_mask.access_ptr("w"), ub_left.access_ptr("r"),
        right_argument, "GE", 2, 9, 2, 10, storage_cols,
    )))
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=[ub_left, ub_right, ub_mask],
    )
    return tvm.tir.PrimFunc(
        [left.data, right.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={left.data: left, right.data: right},
    )


def _tail_compare_select_primfunc(
    *, scalar_select=False, with_scratch=False, select_kind=None
):
    left = tvm.tir.decl_buffer((2, 10), "float32", name="left", scope="global")
    right = tvm.tir.decl_buffer((2, 10), "float32", name="right", scope="global")
    output = tvm.tir.decl_buffer((2, 10), "float32", name="output", scope="global")
    ub_left = tvm.tir.decl_buffer(
        (2, 10), "float32", name="ub_left", scope="shared.ub"
    )
    ub_right = tvm.tir.decl_buffer(
        (2, 10), "float32", name="ub_right", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (2, 10), "float32", name="ub_output", scope="shared.ub"
    )
    ub_mask = tvm.tir.decl_buffer(
        (2, 32), "uint8", name="ub_mask", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer(
        (2, 10), "float32", name="scratch", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), 10, 2, 9, 0, 2, 10,
        ))

    kind = select_kind or ("Scalar" if scalar_select else "Tensor")
    source_type = 1 if scalar_select else 2
    fallback = (
        tvm.tir.FloatImm("float32", 10.0)
        if scalar_select else ub_right.access_ptr("r")
    )
    mode = (
        "VSEL_TENSOR_SCALAR_MODE"
        if scalar_select else "VSEL_TENSOR_TENSOR_MODE"
    )
    tmp = scratch.access_ptr("w") if with_scratch else ub_mask.access_ptr("r")
    statements = [
        copy("copy_gm_to_ub", left, ub_left),
        copy("copy_gm_to_ub", right, ub_right),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_tail_compare", ub_mask.access_ptr("w"),
            ub_left.access_ptr("r"), ub_right.access_ptr("r"),
            "LT", 2, 9, 2, 10, 2,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_tail_select", kind,
            ub_output.access_ptr("w"), ub_mask.access_ptr("r"),
            ub_left.access_ptr("r"), tmp, source_type, fallback, mode,
            2, 9, 2, 10, 2,
        )),
        copy("copy_ub_to_gm", ub_output, output),
    ]
    alloc_buffers = [ub_left, ub_right, ub_output, ub_mask]
    if with_scratch:
        alloc_buffers.append(scratch)
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.SeqStmt(statements),
        alloc_buffers=alloc_buffers,
    )
    return tvm.tir.PrimFunc(
        [left.data, right.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={left.data: left, right.data: right, output.data: output},
    )


def _tail_broadcast_primfunc(axis, *, with_scratch=False):
    output = tvm.tir.decl_buffer((3, 5), "float32", name="output", scope="global")
    ub_output = tvm.tir.decl_buffer(
        (3, 5), "float32", name="ub_output", scope="shared.ub"
    )
    source_shape = (3, 8) if axis == 1 else (1, 5)
    logical_source_shape = (3, 1) if axis == 1 else (1, 5)
    ub_source = tvm.tir.decl_buffer(
        source_shape, "float32", name="ub_source", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer(
        (15,), "float32", name="scratch", scope="shared.ub"
    )
    valid_shape = (2, 5, 2, 1) if axis == 1 else (3, 4, 1, 4)
    arguments = [
        f"Broadcast<float32, 2, {axis}, false>",
        ub_output.access_ptr("w"),
        ub_source.access_ptr("r", extent=int(np.prod(logical_source_shape))),
    ]
    if with_scratch:
        arguments.append(scratch.access_ptr("w"))
    arguments.extend([2, 3, 5, *logical_source_shape, *valid_shape])
    broadcast = tvm.tir.call_extern(
        "handle", "tl.ascend_tail_broadcast", *arguments
    )
    store = tvm.tir.call_extern(
        "handle", "copy_ub_to_gm", ub_output.access_ptr("r"),
        output.access_ptr("w"), 5, valid_shape[0], valid_shape[1],
        0, 3, 5,
    )
    alloc_buffers = [ub_source, ub_output]
    if with_scratch:
        alloc_buffers.append(scratch)
    root = tvm.tir.Block(
        [], [], [], "root",
        tvm.tir.SeqStmt([tvm.tir.Evaluate(broadcast), tvm.tir.Evaluate(store)]),
        alloc_buffers=alloc_buffers,
    )
    return tvm.tir.PrimFunc(
        [output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={output.data: output},
    )


def _padded_copy_primfunc(pad_value=-3.5):
    source = tvm.tir.decl_buffer((2, 5), "float32", name="source", scope="global")
    output = tvm.tir.decl_buffer((3, 8), "float32", name="output", scope="global")
    ub = tvm.tir.decl_buffer((3, 8), "float32", name="ub", scope="shared.ub")
    load = tvm.tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float32, 8, 3>",
        source.access_ptr("r"),
        ub.access_ptr("w"),
        5,
        2,
        5,
        tvm.tir.FloatImm("float32", pad_value),
        3,
        8,
    )
    store = tvm.tir.call_extern(
        "handle",
        "copy_ub_to_gm",
        ub.access_ptr("r"),
        output.access_ptr("w"),
        24,
    )
    root = tvm.tir.Block(
        [],
        [],
        [],
        "root",
        tvm.tir.SeqStmt([tvm.tir.Evaluate(load), tvm.tir.Evaluate(store)]),
        alloc_buffers=[ub],
    )
    return tvm.tir.PrimFunc(
        [source.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source, output.data: output},
    )


def _non_affine_copy_primfunc():
    source = tvm.tir.decl_buffer((32,), "float32", name="source", scope="global")
    destination = tvm.tir.decl_buffer(
        (32,), "float32", name="destination", scope="shared.ub"
    )
    extent = tvm.tir.Var("extent", "int32")
    valid_cols = tvm.tir.min(extent, 8)
    element_offset = tvm.tir.Select(
        extent < 10,
        tvm.tir.floormod(extent, 3),
        tvm.tir.floordiv(extent, 4),
    ) * 2
    load = tvm.tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float32, 32>",
        source.access_ptr("r", offset=element_offset, extent=valid_cols),
        destination.access_ptr("w"),
        32,
        1,
        valid_cols,
        0,
        1,
        32,
    )
    root = tvm.tir.Block(
        [],
        [],
        [],
        "root",
        tvm.tir.Evaluate(load),
        alloc_buffers=[destination],
    )
    return tvm.tir.PrimFunc(
        [source.data, extent],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source},
    )


def _dynamic_allocation_primfunc():
    extent = tvm.tir.Var("extent", "int32")
    source = tvm.tir.decl_buffer(
        (extent,), "float32", name="source", scope="global"
    )
    destination = tvm.tir.decl_buffer(
        (8,), "float32", name="destination", scope="shared.ub"
    )
    valid_cols = tvm.tir.min(extent, 8)
    load = tvm.tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float32, 8>",
        source.access_ptr("r", extent=valid_cols),
        destination.access_ptr("w"),
        extent,
        1,
        valid_cols,
        0,
        1,
        8,
    )
    root = tvm.tir.Block(
        [],
        [],
        [],
        "root",
        tvm.tir.Evaluate(load),
        alloc_buffers=[destination],
    )
    return tvm.tir.PrimFunc(
        [source.data, extent],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source},
    )


def _planned_alias_primfunc():
    x = tvm.tir.decl_buffer((4,), "float32", name="x", scope="global")
    y = tvm.tir.decl_buffer((4,), "float32", name="y", scope="global")
    output = tvm.tir.decl_buffer((4,), "float32", name="output", scope="global")
    ub_x = tvm.tir.decl_buffer((4,), "float32", name="ub_x", scope="shared.ub")
    ub_y = tvm.tir.decl_buffer((4,), "float32", name="ub_y", scope="shared.ub")

    def copy(operation, source, destination):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle",
            operation,
            source.access_ptr("r"),
            destination.access_ptr("w"),
            4,
        ))

    root = tvm.tir.Block(
        [],
        [],
        [],
        "root",
        tvm.tir.SeqStmt([
            copy("copy_gm_to_ub", x, ub_x),
            copy("copy_gm_to_ub", y, ub_y),
            copy("copy_ub_to_gm", ub_x, output),
        ]),
        alloc_buffers=[ub_x, ub_y],
    )
    function = tvm.tir.PrimFunc(
        [x.data, y.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={x.data: x, y.data: y, output.data: output},
    )
    function = function.with_attr("address_map", {
        ub_x.data: tvm.tir.IntImm("int64", 0),
        ub_y.data: tvm.tir.IntImm("int64", 8),
    })
    return function.with_attr("size_map", {
        ub_x.data: tvm.tir.IntImm("int64", 16),
        ub_y.data: tvm.tir.IntImm("int64", 16),
    })


def _tail_reduce_primfunc(kind, dimension=0, clear=1, dtype="float32"):
    source = tvm.tir.decl_buffer((3, 5), dtype, name="source", scope="global")
    output = tvm.tir.decl_buffer((5,), dtype, name="output", scope="global")
    ub_source = tvm.tir.decl_buffer(
        (4, 8), dtype, name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (8,), dtype, name="ub_output", scope="shared.ub"
    )
    load = tvm.tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_ub<float32, 8, 4>",
        source.access_ptr("r"),
        ub_source.access_ptr("w"),
        5,
        3,
        5,
        0,
        4,
        8,
    )
    reduce = tvm.tir.call_extern(
        "handle",
        "tl.ascend_tail_reduce",
        kind,
        ub_output.access_ptr("w"),
        ub_source.access_ptr("r"),
        dimension,
        3,
        5,
        8,
        clear,
    )
    store = tvm.tir.call_extern(
        "handle",
        "copy_ub_to_gm",
        ub_output.access_ptr("r"),
        output.access_ptr("w"),
        5,
    )
    root = tvm.tir.Block(
        [],
        [],
        [],
        "root",
        tvm.tir.SeqStmt([
            tvm.tir.Evaluate(load),
            tvm.tir.Evaluate(reduce),
            tvm.tir.Evaluate(store),
        ]),
        alloc_buffers=[ub_source, ub_output],
    )
    return tvm.tir.PrimFunc(
        [source.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source, output.data: output},
    )


def _reduce_primfunc(kind, axis, *, clear=True):
    source = tvm.tir.decl_buffer((3, 5), "float32", name="source", scope="global")
    output_count = 5 if axis == 0 else 3
    initial = tvm.tir.decl_buffer(
        (output_count,), "float32", name="initial", scope="global"
    )
    output = tvm.tir.decl_buffer(
        (output_count,), "float32", name="output", scope="global"
    )
    ub_source = tvm.tir.decl_buffer(
        (3, 5), "float32", name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (output_count,), "float32", name="ub_output", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer(
        (15,), "float32", name="scratch", scope="shared.ub"
    )
    output_scratch = tvm.tir.decl_buffer(
        (output_count,), "float32", name="output_scratch", scope="shared.ub"
    )
    load = tvm.tir.call_extern(
        "handle", "copy_gm_to_ub", source.access_ptr("r"),
        ub_source.access_ptr("w"), 15,
    )
    arguments = [
        f"{kind}<float, 3, 5, {axis}>", ub_output.access_ptr("w"),
        ub_source.access_ptr("r"),
    ]
    if axis == -1:
        arguments.append(scratch.access_ptr("w"))
    if not clear:
        arguments.append(output_scratch.access_ptr("w"))
    arguments.append(tvm.tir.const(clear, "bool"))
    reduce_call = tvm.tir.call_extern(
        "handle", "tl.ascend_reduce", *arguments
    )
    store = tvm.tir.call_extern(
        "handle", "copy_ub_to_gm", ub_output.access_ptr("r"),
        output.access_ptr("w"), output_count,
    )
    statements = [tvm.tir.Evaluate(load)]
    if not clear:
        statements.append(tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "copy_gm_to_ub", initial.access_ptr("r"),
            ub_output.access_ptr("w"), output_count,
        )))
    statements.extend([
        tvm.tir.Evaluate(reduce_call), tvm.tir.Evaluate(store),
    ])
    root = tvm.tir.Block(
        [], [], [], "root",
        tvm.tir.SeqStmt(statements),
        alloc_buffers=[ub_source, ub_output, scratch, output_scratch],
    )
    return tvm.tir.PrimFunc(
        [source.data, initial.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={
            source.data: source, initial.data: initial, output.data: output,
        },
    )


def _narrow_reduce_primfunc(
    kind, *, rows=3, logical_cols=5, physical_cols=8, offset=2, clear=True,
    axis=-1,
):
    source = tvm.tir.decl_buffer(
        (rows, physical_cols), "float32", name="source", scope="global"
    )
    output = tvm.tir.decl_buffer((rows,), "float32", name="output", scope="global")
    ub_source = tvm.tir.decl_buffer(
        (rows, physical_cols), "float32", name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (rows,), "float32", name="ub_output", scope="shared.ub"
    )
    scratch = tvm.tir.decl_buffer(
        (rows * logical_cols,), "float32", name="scratch", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    body = tvm.tir.SeqStmt([
        copy("copy_gm_to_ub", source, ub_source, rows * physical_cols),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_reduce",
            f"reduce_{kind}<float, {rows}, {logical_cols}, {axis}>",
            ub_output.access_ptr("w"),
            ub_source.access_ptr("r", offset=offset),
            scratch.access_ptr("w"), tvm.tir.const(clear, "bool"), physical_cols,
        )),
        copy("copy_ub_to_gm", ub_output, output, rows),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body,
        alloc_buffers=[ub_source, ub_output, scratch],
    )
    return tvm.tir.PrimFunc(
        [source.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={source.data: source, output.data: output},
    )


def _block_reduce_primfunc(kind, *, dtype="float16", mask=37):
    source = tvm.tir.decl_buffer((256,), dtype, name="source", scope="global")
    initial = tvm.tir.decl_buffer((16,), dtype, name="initial", scope="global")
    output = tvm.tir.decl_buffer((16,), dtype, name="output", scope="global")
    ub_source = tvm.tir.decl_buffer(
        (256,), dtype, name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (16,), dtype, name="ub_output", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    body = tvm.tir.SeqStmt([
        copy("copy_gm_to_ub", source, ub_source, 256),
        copy("copy_gm_to_ub", initial, ub_output, 16),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", f"tl.ascend_block_reduce_{kind}",
            ub_output.access_ptr("w"), ub_source.access_ptr("r"),
            2, mask, 1, 1, 8,
        )),
        copy("copy_ub_to_gm", ub_output, output, 16),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=[ub_source, ub_output]
    )
    return tvm.tir.PrimFunc(
        [source.data, initial.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={
            source.data: source, initial.data: initial, output.data: output,
        },
    )


def _whole_reduce_primfunc(
    kind, *, dtype="float16", mask=37, order="ORDER_VALUE_INDEX"
):
    source = tvm.tir.decl_buffer((256,), dtype, name="source", scope="global")
    initial = tvm.tir.decl_buffer((16,), dtype, name="initial", scope="global")
    output = tvm.tir.decl_buffer((16,), dtype, name="output", scope="global")
    ub_source = tvm.tir.decl_buffer(
        (256,), dtype, name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (16,), dtype, name="ub_output", scope="shared.ub"
    )

    def copy(name, source_buffer, destination_buffer, count):
        return tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", name, source_buffer.access_ptr("r"),
            destination_buffer.access_ptr("w"), count,
        ))

    arguments = [
        ub_output.access_ptr("w"), ub_source.access_ptr("r"),
        mask, 2, 2, 1, 8,
    ]
    if kind != "sum":
        arguments.append(order)
    body = tvm.tir.SeqStmt([
        copy("copy_gm_to_ub", source, ub_source, 256),
        copy("copy_gm_to_ub", initial, ub_output, 16),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", f"tl.ascend_wholereduce{kind}", *arguments,
        )),
        copy("copy_ub_to_gm", ub_output, output, 16),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=[ub_source, ub_output]
    )
    return tvm.tir.PrimFunc(
        [source.data, initial.data, output.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={
            source.data: source, initial.data: initial, output.data: output,
        },
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


def test_real_tir_row_major_gm_to_l1_copy_executes() -> None:
    program = build_kernel_program(_gm_to_l1_linear_primfunc(), platform="A3")
    task = program.tasks[0]
    assert (task.operation, task.lane, task.pipe) == (
        "copy_gm_to_l1_linear", Lane.CUBE, Pipe.MTE2,
    )
    assert task.metadata["src"].scope is MemoryScope.GM
    assert task.metadata["dst"].scope is MemoryScope.L1
    assert task.metadata["copy"] == {
        "layout": "row_major",
        "valid_rows": 2,
        "valid_cols": 8,
        "source_cols": 8,
        "physical_rows": 3,
        "physical_cols": 8,
    }

    simulator = FunctionalSimulator(program)
    values = np.arange(16, dtype=np.float32).reshape(2, 8)
    simulator.write(task.metadata["src"], values)
    simulator.run()

    np.testing.assert_array_equal(simulator.read(task.metadata["dst"]), values)


def test_row_major_gm_to_l1_rejects_unaligned_rows() -> None:
    with pytest.raises(ProgramValidationError, match="32-byte aligned"):
        build_kernel_program(_gm_to_l1_linear_primfunc(valid_cols=7), platform="A2")


def test_real_tir_zn_gm_to_l1_copy_packs_and_clears_tail() -> None:
    program = build_kernel_program(_gm_to_l1_zn_primfunc(), platform="A3")
    task = program.tasks[0]
    assert (task.operation, task.lane, task.pipe) == (
        "copy_gm_to_l1", Lane.CUBE, Pipe.MTE2,
    )
    assert task.metadata["copy"] == {
        "layout": "zN",
        "valid_rows": 13,
        "valid_cols": 15,
        "source_cols": 16,
        "physical_rows": 16,
        "physical_cols": 16,
        "need_clear": True,
    }

    simulator = FunctionalSimulator(program)
    values = np.arange(13 * 15, dtype=np.float16).reshape(13, 15)
    simulator.write(task.metadata["src"], values)
    simulator.run()

    physical = simulator.read(task.metadata["dst"])
    logical = unpack_matrix(physical, "zN", (16, 16))
    expected = np.zeros((16, 16), dtype=np.float16)
    expected[:13, :15] = values
    np.testing.assert_array_equal(logical, expected)


def test_zn_gm_to_l1_rejects_non_fractal_physical_tile() -> None:
    with pytest.raises(UnsupportedSimOpError, match="fractal/C0-aligned"):
        build_kernel_program(_gm_to_l1_zn_primfunc(physical_rows=15), platform="A2")


def _gm_to_l1_zn_splice_primfunc(
    *,
    physical_rows=48,
    cols=16,
    first_valid_rows=32,
    second_valid_rows=16,
    anchor_row=32,
    dst_offset_elements=None,
    capacity_rows=None,
    second_dma=True,
):
    elements_per_fractal = 512 // 2
    if dst_offset_elements is None:
        dst_offset_elements = anchor_row // 16 * elements_per_fractal
    first_source = tvm.tir.decl_buffer(
        (first_valid_rows, cols), "float16", name="source0", scope="global"
    )
    second_source = tvm.tir.decl_buffer(
        (second_valid_rows, cols), "float16", name="source1", scope="global"
    )
    capacity_rows = physical_rows if capacity_rows is None else capacity_rows
    l1 = tvm.tir.decl_buffer(
        (capacity_rows * cols,), "float16", name="l1", scope="shared.l1"
    )

    def splice_copy(source_buffer, dst_offset):
        return tvm.tir.call_extern(
            "handle",
            f"tl::ascend::copy_gm_to_l1<half, {physical_rows}, {cols}>",
            source_buffer.access_ptr("r"),
            l1.access_ptr("w", offset=dst_offset),
            cols,
            (
                first_valid_rows
                if source_buffer is first_source
                else second_valid_rows
            ),
            cols,
            physical_rows,
            cols,
        )

    if second_dma:
        body = tvm.tir.SeqStmt([
            tvm.tir.Evaluate(splice_copy(first_source, 0)),
            tvm.tir.Evaluate(splice_copy(second_source, dst_offset_elements)),
        ])
    else:
        body = tvm.tir.Evaluate(splice_copy(second_source, dst_offset_elements))
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=[l1],
    )
    return tvm.tir.PrimFunc(
        [first_source.data, second_source.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={
            first_source.data: first_source,
            second_source.data: second_source,
        },
    )


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_zn_gm_to_l1_splice_merges_without_clobbering(platform) -> None:
    program = build_kernel_program(_gm_to_l1_zn_splice_primfunc(), platform=platform)
    primary, splice = program.tasks
    assert "dst" in primary.metadata and "dst_regions" not in primary.metadata
    assert primary.metadata["copy"]["need_clear"] is True
    assert "dst" not in splice.metadata
    written = splice.metadata["dst_regions"]
    assert len(written) == 1
    assert written[0].shape == (16 * 16,)
    assert written[0].byte_offset == 512 * 2
    assert splice.metadata["copy"]["written_rows"] == 16
    assert splice.metadata["copy"]["need_clear"] is False
    assert splice.dependencies == (primary.task_id,)

    first = np.arange(32 * 16, dtype=np.float16).reshape(32, 16) + 1
    second = -(np.arange(16 * 16, dtype=np.float16).reshape(16, 16) + 1)
    simulator = FunctionalSimulator(program)
    simulator.write(primary.metadata["src"], first)
    simulator.write(splice.metadata["src"], second)
    simulator.run()
    logical = unpack_matrix(simulator.read(primary.metadata["dst"]), "zN", (48, 16))
    expected = np.zeros((48, 16), dtype=np.float16)
    expected[:32] = first
    expected[32:] = second
    np.testing.assert_array_equal(logical, expected)


def test_zn_gm_to_l1_splice_writes_every_fractal_column_band() -> None:
    program = build_kernel_program(
        _gm_to_l1_zn_splice_primfunc(cols=32), platform="A3"
    )
    primary, splice = program.tasks
    written = splice.metadata["dst_regions"]
    assert len(written) == 2
    assert written[0].byte_offset == 512 * 2
    assert written[1].byte_offset == (512 + 48 * 16) * 2
    assert all(region.shape == (16 * 16,) for region in written)
    assert splice.dependencies == (primary.task_id,)

    first = np.arange(32 * 32, dtype=np.float16).reshape(32, 32) + 3
    second = -(np.arange(16 * 32, dtype=np.float16).reshape(16, 32) + 3)
    simulator = FunctionalSimulator(program)
    simulator.write(primary.metadata["src"], first)
    simulator.write(splice.metadata["src"], second)
    simulator.run()
    logical = unpack_matrix(simulator.read(primary.metadata["dst"]), "zN", (48, 32))
    expected = np.zeros((48, 32), dtype=np.float16)
    expected[:32] = first
    expected[32:] = second
    np.testing.assert_array_equal(logical, expected)


def test_zn_gm_to_l1_splice_alone_keeps_other_rows_poisoned() -> None:
    program = build_kernel_program(
        _gm_to_l1_zn_splice_primfunc(second_dma=False), platform="A2"
    )
    (splice,) = program.tasks
    assert splice.metadata["dst_regions"][0].byte_offset == 512 * 2

    second = np.full((16, 16), -7, dtype=np.float16)
    simulator = FunctionalSimulator(program)
    simulator.write(splice.metadata["src"], second)
    simulator.run()
    written = simulator.read(
        BufferRegion(
            splice.metadata["dst_regions"][0].buffer,
            MemoryScope.L1,
            (16 * 16,),
            "float16",
            byte_offset=512 * 2,
        )
    )
    np.testing.assert_array_equal(
        unpack_matrix(written, "zN", (16, 16)), second
    )
    with pytest.raises(UninitializedMemoryError, match="read-before-write"):
        simulator.read(
            BufferRegion(
                splice.metadata["dst_regions"][0].buffer,
                MemoryScope.L1,
                (32 * 16,),
                "float16",
            )
        )


def test_zn_gm_to_l1_splice_rejects_unaligned_offset_and_overflow() -> None:
    with pytest.raises(UnsupportedSimOpError, match="fractal-row-aligned"):
        build_kernel_program(
            _gm_to_l1_zn_splice_primfunc(second_dma=False, dst_offset_elements=100),
            platform="A2",
        )
    with pytest.raises(ProgramValidationError, match="bands exceed"):
        build_kernel_program(
            _gm_to_l1_zn_splice_primfunc(
                second_dma=False, capacity_rows=32,
            ),
            platform="A2",
        )


def test_zn_gm_to_l1_ring_slot_offset_stays_primary() -> None:
    program = build_kernel_program(
        _gm_to_l1_zn_primfunc(), platform="A2"
    )
    primary = program.tasks[0]
    tile_elements = primary.metadata["copy"]["physical_rows"] * (
        primary.metadata["copy"]["physical_cols"]
    )
    ring_slot = build_kernel_program(
        _gm_to_l1_zn_splice_primfunc(
            physical_rows=16,
            cols=16,
            first_valid_rows=13,
            second_valid_rows=13,
            anchor_row=0,
            second_dma=False,
            dst_offset_elements=tile_elements,
            capacity_rows=32,
        ),
        platform="A2",
    )
    (task,) = ring_slot.tasks
    assert "dst" in task.metadata and "dst_regions" not in task.metadata
    assert task.metadata["dst"].byte_offset == tile_elements * 2
    assert task.metadata["copy"]["need_clear"] is True
    assert primary.metadata["dst"].byte_offset == 0


_TEMPLATE_DTYPE_NAMES = {"float16": "half", "float32": "float"}


def _ub_to_ub_primfunc(
    *,
    src_dtype="float16",
    dst_dtype="float32",
    rows=4,
    cols=6,
    src_stride=None,
    dst_stride=None,
    template_dst_dtype=None,
    template_len=None,
):
    src_stride = src_stride if src_stride is not None else cols
    dst_stride = dst_stride if dst_stride is not None else cols
    template_len = template_len if template_len is not None else rows * cols
    ub_src = tvm.tir.decl_buffer(
        (rows, src_stride), src_dtype, name="ub_src", scope="shared.ub"
    )
    ub_dst = tvm.tir.decl_buffer(
        (rows, dst_stride), dst_dtype, name="ub_dst", scope="shared.ub"
    )
    dst_name = template_dst_dtype or dst_dtype
    template = (
        f"tl::ascend::copy_ub_to_ub<"
        f"{_TEMPLATE_DTYPE_NAMES.get(dst_name, dst_name)}, "
        f"{_TEMPLATE_DTYPE_NAMES.get(src_dtype, src_dtype)}, {template_len}>"
    )
    copy = tvm.tir.call_extern(
        "handle", template,
        ub_src.access_ptr("r"), ub_dst.access_ptr("w"),
        rows, cols, src_stride, rows, cols, dst_stride,
    )
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.Evaluate(copy), alloc_buffers=[ub_src, ub_dst],
    )
    return tvm.tir.PrimFunc([], tvm.tir.BlockRealize([], True, root))


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_ub_to_ub_copies_contiguous_tiles(platform) -> None:
    program = build_kernel_program(_ub_to_ub_primfunc(), platform=platform)
    (copy,) = program.tasks
    assert (copy.operation, copy.lane, copy.pipe) == (
        "copy_ub_to_ub", Lane.VECTOR_0, Pipe.VECTOR,
    )
    assert copy.metadata["copy"]["cast_mode"] == "CAST_NONE"

    values = np.array(
        [[1, -2, 0.25, 65504, -0.5, 3], [7, -8, 0.125, 9, -10, 11.5]] * 2,
        dtype=np.float16,
    )
    simulator = FunctionalSimulator(program)
    simulator.write(copy.metadata["src"], values)
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(copy.metadata["dst"]), values.astype(np.float32)
    )


def test_ub_to_ub_copies_strided_tiles_without_touching_gaps() -> None:
    program = build_kernel_program(
        _ub_to_ub_primfunc(
            src_dtype="float32", dst_dtype="float32", src_stride=8, dst_stride=8
        ),
        platform="A2",
    )
    (copy,) = program.tasks
    assert copy.metadata["copy"]["cast_mode"] is None

    values = (np.arange(4 * 6, dtype=np.float32).reshape(4, 6) - 11) / 4
    simulator = FunctionalSimulator(program)
    simulator.write(copy.metadata["src"], values)
    simulator.run()
    result = simulator.read(copy.metadata["dst"])
    np.testing.assert_array_equal(result, values)
    with pytest.raises(UninitializedMemoryError, match="read-before-write"):
        simulator.read(
            BufferRegion(
                copy.metadata["dst"].buffer,
                MemoryScope.UB,
                (4, 2),
                "float32",
                byte_offset=6 * 4,
                strides_bytes=(8 * 4, 4),
            )
        )


def test_ub_to_ub_narrows_float_with_nearest_even_conversion() -> None:
    program = build_kernel_program(
        _ub_to_ub_primfunc(src_dtype="float32", dst_dtype="float16"),
        platform="A2",
    )
    (copy,) = program.tasks
    assert copy.metadata["copy"]["cast_mode"] == "CAST_RINT"

    values = np.array(
        [[2.51, 3.49, -2.75, -0.333, 65504, 1.6],
         [4.5, 5.5, -4.5, -5.5, 2.49, 2.51]] * 2,
        dtype=np.float32,
    )
    simulator = FunctionalSimulator(program)
    simulator.write(copy.metadata["src"], values)
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(copy.metadata["dst"]),
        values.astype(np.float16),
    )


def test_ub_to_ub_orders_raw_against_producer() -> None:
    gm_source = tvm.tir.decl_buffer(
        (4, 6), "float32", name="gm_source", scope="global"
    )
    ub_src = tvm.tir.decl_buffer(
        (4, 6), "float32", name="ub_src", scope="shared.ub"
    )
    ub_dst = tvm.tir.decl_buffer(
        (4, 6), "float32", name="ub_dst", scope="shared.ub"
    )
    body = tvm.tir.SeqStmt([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "copy_gm_to_ub",
            gm_source.access_ptr("r"), ub_src.access_ptr("w"), 24,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle",
            "tl::ascend::copy_ub_to_ub<float, float, 24>",
            ub_src.access_ptr("r"), ub_dst.access_ptr("w"),
            4, 6, 6, 4, 6, 6,
        )),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=[ub_src, ub_dst],
    )
    primfunc = tvm.tir.PrimFunc(
        [],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={gm_source.data: gm_source},
    )
    program = build_kernel_program(primfunc, platform="A2")
    load, copy = program.tasks
    assert copy.dependencies == (load.task_id,)

    values = (np.arange(24, dtype=np.float32).reshape(4, 6) - 7) / 3
    simulator = FunctionalSimulator(program)
    simulator.write(load.metadata["src"], values.reshape(-1))
    simulator.run()
    np.testing.assert_array_equal(simulator.read(copy.metadata["dst"]), values)


def test_ub_to_ub_validates_template_and_extents() -> None:
    with pytest.raises(ProgramValidationError, match="template dtypes"):
        build_kernel_program(
            _ub_to_ub_primfunc(template_dst_dtype="float16"), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="template length"):
        build_kernel_program(
            _ub_to_ub_primfunc(template_len=999), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="tile columns must not exceed"):
        build_kernel_program(
            _ub_to_ub_primfunc(src_stride=4, cols=6), platform="A2"
        )


def _ub_to_l1_primfunc(*, rows=32, cols=16, dtype="float16", template_dtype=None):
    ub = tvm.tir.decl_buffer(
        (rows, cols), dtype, name="ub_tile", scope="shared.ub"
    )
    l1 = tvm.tir.decl_buffer(
        (rows * cols,), dtype, name="l1_tile", scope="shared.l1"
    )
    dtype_name = template_dtype or _TEMPLATE_DTYPE_NAMES.get(dtype, dtype)
    copy = tvm.tir.call_extern(
        "handle", f"tl::ascend::copy_ub_to_l1<{dtype_name}, {cols}, {rows}>",
        ub.access_ptr("r"), l1.access_ptr("w"),
        cols, rows, rows, cols,
    )
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.Evaluate(copy), alloc_buffers=[ub, l1],
    )
    return tvm.tir.PrimFunc([], tvm.tir.BlockRealize([], True, root))


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_ub_to_l1_packs_row_major_tile_into_zn(platform) -> None:
    program = build_kernel_program(_ub_to_l1_primfunc(), platform=platform)
    (copy,) = program.tasks
    assert (copy.operation, copy.lane, copy.pipe) == (
        "copy_ub_to_l1", Lane.VECTOR_0, Pipe.MTE3,
    )
    assert copy.metadata["copy"]["layout"] == "zN"

    values = np.arange(32 * 16, dtype=np.float16).reshape(32, 16) + 1
    simulator = FunctionalSimulator(program)
    simulator.write(copy.metadata["src"], values)
    simulator.run()
    np.testing.assert_array_equal(
        unpack_matrix(simulator.read(copy.metadata["dst"]), "zN", (32, 16)),
        values,
    )


def test_ub_to_l1_validates_dtype_alignment_and_template() -> None:
    with pytest.raises(UnsupportedSimOpError, match="half tiles only"):
        build_kernel_program(
            _ub_to_l1_primfunc(dtype="float32"), platform="A2"
        )
    with pytest.raises(UnsupportedSimOpError, match="fractal/C0-aligned"):
        build_kernel_program(
            _ub_to_l1_primfunc(rows=31), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="template dtype"):
        build_kernel_program(
            _ub_to_l1_primfunc(template_dtype="float"), platform="A3"
        )

def _brcb_primfunc(
    *,
    dtype="float16",
    repeat=2,
    blk_stride=1,
    rep_stride=8,
    src_offset=0,
    dst_offset=0,
    template_dtype=None,
    src_dtype=None,
    dst_scope="shared.ub",
):
    src_dtype = src_dtype or dtype
    itemsize = {"float16": 2, "float32": 4, "int32": 4}[dtype]
    elements_per_block = 32 // itemsize
    footprint = (
        ((repeat - 1) * rep_stride + 7 * blk_stride + 1) * elements_per_block
        if repeat > 0
        else 0
    )
    src = tvm.tir.decl_buffer(
        (src_offset + repeat * 8,), src_dtype, name="brcb_src", scope="shared.ub"
    )
    dst = tvm.tir.decl_buffer(
        (dst_offset + footprint,), dtype, name="brcb_dst", scope=dst_scope
    )
    dtype_name = template_dtype or _TEMPLATE_DTYPE_NAMES.get(dtype, dtype)
    body = tvm.tir.Evaluate(tvm.tir.call_extern(
        "handle", "tl.ascend_brcb_experiment", f"brcb<{dtype_name}>",
        dst.access_ptr("w", offset=dst_offset),
        src.access_ptr("r", offset=src_offset),
        repeat, blk_stride, rep_stride,
    ))
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=[src, dst],
    )
    return tvm.tir.PrimFunc([], tvm.tir.BlockRealize([], True, root))


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_brcb_broadcasts_elements_into_consecutive_blocks(platform) -> None:
    program = build_kernel_program(_brcb_primfunc(), platform=platform)
    (brcb,) = program.tasks
    assert (brcb.operation, brcb.lane, brcb.pipe) == (
        "brcb_experiment", Lane.VECTOR_0, Pipe.VECTOR,
    )
    assert brcb.metadata["brcb"] == {
        "repeat": 2, "blk_stride": 1, "rep_stride": 8,
    }

    scalars = (np.arange(16, dtype=np.float16) + 1) / 4
    simulator = FunctionalSimulator(program)
    simulator.write(brcb.metadata["src"], scalars)
    simulator.run()
    expected = np.repeat(scalars, 16)
    np.testing.assert_array_equal(simulator.read(brcb.metadata["dst"]), expected)


def test_brcb_strided_blocks_leave_gaps_poisoned() -> None:
    program = build_kernel_program(
        _brcb_primfunc(dtype="float32", repeat=2, blk_stride=2, rep_stride=16),
        platform="A2",
    )
    (brcb,) = program.tasks
    scalars = np.arange(16, dtype=np.float32) - 8
    simulator = FunctionalSimulator(program)
    simulator.write(brcb.metadata["src"], scalars)
    simulator.run()
    for repeat in range(2):
        for block in range(8):
            offset = (repeat * 16 + block * 2) * 32
            np.testing.assert_array_equal(
                simulator.read(
                    BufferRegion(
                        brcb.metadata["dst"].buffer,
                        MemoryScope.UB,
                        (8,),
                        "float32",
                        byte_offset=offset,
                    )
                ),
                np.full(8, scalars[repeat * 8 + block], dtype=np.float32),
            )
    with pytest.raises(UninitializedMemoryError, match="read-before-write"):
        simulator.read(
            BufferRegion(
                brcb.metadata["dst"].buffer,
                MemoryScope.UB,
                (8,),
                "float32",
                byte_offset=32,
            )
        )


def test_brcb_orders_waw_and_raw_in_pipeline() -> None:
    gm_output = tvm.tir.decl_buffer(
        (16 * 8,), "int32", name="gm_output", scope="global"
    )
    ub_src = tvm.tir.decl_buffer(
        (16,), "int32", name="brcb_src", scope="shared.ub"
    )
    ub_dst = tvm.tir.decl_buffer(
        (16 * 8,), "int32", name="brcb_dst", scope="shared.ub"
    )
    body = tvm.tir.SeqStmt([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_fill",
            ub_src.access_ptr("w"), tvm.tir.FloatImm("float32", 5.0), 16,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_brcb_experiment", "brcb<int32>",
            ub_dst.access_ptr("w"), ub_src.access_ptr("r"), 2, 1, 8,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "copy_ub_to_gm",
            ub_dst.access_ptr("r"), gm_output.access_ptr("w"), 16 * 8,
        )),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=[ub_src, ub_dst],
    )
    primfunc = tvm.tir.PrimFunc(
        [],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={gm_output.data: gm_output},
    )
    program = build_kernel_program(primfunc, platform="A3")
    fill, brcb, store = program.tasks
    assert brcb.dependencies == (fill.task_id,)
    assert store.dependencies == (brcb.task_id,)

    simulator = FunctionalSimulator(program)
    simulator.run()
    values = simulator.read(store.metadata["dst"]).reshape(-1)
    np.testing.assert_array_equal(values, np.full(values.size, 5, np.int32))


def test_brcb_validates_template_scope_alignment_and_dtypes() -> None:
    with pytest.raises(ProgramValidationError, match="template dtype"):
        build_kernel_program(
            _brcb_primfunc(template_dtype="float"), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="source/destination dtypes"):
        build_kernel_program(
            _brcb_primfunc(
                dtype="float32", src_dtype="float16", template_dtype="half",
            ),
            platform="A2",
        )
    with pytest.raises(ProgramValidationError, match="32-byte-aligned source"):
        build_kernel_program(
            _brcb_primfunc(src_offset=1), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="UB destination"):
        build_kernel_program(
            _brcb_primfunc(dst_scope="global"), platform="A2"
        )

_ROW_EXPAND_TAG_NAMES = {
    "mul": "RowExpandMulExperiment",
    "sub": "RowExpandSubExperiment",
    "div": "RowExpandDivExperiment",
}


def _row_expand_primfunc(
    *,
    op="mul",
    dtype="float32",
    rows=8,
    with_tmp=False,
    dst_offset=0,
    cols_override=None,
    tag_dtype_override=None,
    scalar_count_override=None,
):
    elements_per_block = 32 // (2 if dtype == "float16" else 4)
    cols = cols_override or 8 * elements_per_block
    extent = rows * cols
    scalar_count = scalar_count_override or (
        rows if with_tmp else rows * elements_per_block
    )
    ub_dst = tvm.tir.decl_buffer(
        (rows, cols), dtype, name="re_dst", scope="shared.ub"
    )
    ub_src0 = tvm.tir.decl_buffer(
        (rows, cols), dtype, name="re_src0", scope="shared.ub"
    )
    ub_src1 = tvm.tir.decl_buffer(
        (scalar_count,), dtype, name="re_src1", scope="shared.ub"
    )
    buffers = [ub_dst, ub_src0, ub_src1]
    tag_dtype = tag_dtype_override or _TEMPLATE_DTYPE_NAMES.get(dtype, dtype)
    args = [
        f"{_ROW_EXPAND_TAG_NAMES[op]}<{tag_dtype}>",
        ub_dst.access_ptr("w", offset=dst_offset, extent=extent),
        ub_src0.access_ptr("r", extent=extent),
        ub_src1.access_ptr("r"),
    ]
    if with_tmp:
        ub_tmp = tvm.tir.decl_buffer(
            (rows, elements_per_block), dtype, name="re_tmp", scope="shared.ub"
        )
        buffers.append(ub_tmp)
        args.append(ub_tmp.access_ptr("rw"))
    body = tvm.tir.Evaluate(tvm.tir.call_extern(
        "handle", f"tl.ascend_row_expand_{op}_experiment", *args,
    ))
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=buffers,
    )
    return tvm.tir.PrimFunc([], tvm.tir.BlockRealize([], True, root))


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("op", ["mul", "sub", "div"])
def test_row_expand_broadcasts_row_scalars(platform, op) -> None:
    program = build_kernel_program(
        _row_expand_primfunc(op=op), platform=platform
    )
    (task,) = program.tasks
    assert (task.operation, task.lane, task.pipe) == (
        f"row_expand_{op}_experiment", Lane.VECTOR_0, Pipe.VECTOR,
    )
    assert task.metadata["row_expand"] == {"rows": 8, "row_elements": 64}
    assert "scratch" not in task.metadata

    values = (np.arange(8 * 64, dtype=np.float32).reshape(8, 64) - 100) / 16
    scalars = np.array([4, -2, 0.5, 8, -1, 2, 0.25, -4], dtype=np.float32)
    broadcast = np.repeat(scalars, 8)
    simulator = FunctionalSimulator(program)
    simulator.write(task.metadata["src"], values)
    simulator.write(task.metadata["scalar_src"], broadcast)
    simulator.run()
    expected = {
        "mul": values * scalars[:, None],
        "sub": values - scalars[:, None],
        "div": values / scalars[:, None],
    }[op]
    np.testing.assert_allclose(
        simulator.read(task.metadata["dst"]), expected, rtol=1e-6, atol=1e-6
    )


def test_row_expand_with_tmp_writes_broadcast_scratch() -> None:
    program = build_kernel_program(
        _row_expand_primfunc(op="mul", dtype="float16", with_tmp=True),
        platform="A2",
    )
    (task,) = program.tasks
    assert isinstance(task.metadata["scratch"], BufferRegion)

    cols = 8 * (32 // 2)  # float16 rows are 128 elements wide
    values = (np.arange(8 * cols, dtype=np.float16).reshape(8, cols) - 40) / 8
    scalars = np.arange(8, dtype=np.float16) + 1
    simulator = FunctionalSimulator(program)
    simulator.write(task.metadata["src"], values)
    simulator.write(task.metadata["scalar_src"], scalars)
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(task.metadata["scratch"]).reshape(-1),
        np.repeat(scalars, cols // 8),
    )
    np.testing.assert_array_equal(
        simulator.read(task.metadata["dst"]),
        values * scalars[:, None],
    )


def test_row_expand_orders_raw_against_producer() -> None:
    extent = 8 * 64
    ub_src0 = tvm.tir.decl_buffer(
        (8, 64), "float32", name="re_src0", scope="shared.ub"
    )
    ub_src1 = tvm.tir.decl_buffer(
        (64,), "float32", name="re_src1", scope="shared.ub"
    )
    ub_dst = tvm.tir.decl_buffer(
        (8, 64), "float32", name="re_dst", scope="shared.ub"
    )
    body = tvm.tir.SeqStmt([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_fill",
            ub_src0.access_ptr("w"), tvm.tir.FloatImm("float32", 2.0), extent,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_row_expand_mul_experiment",
            "RowExpandMulExperiment<float>",
            ub_dst.access_ptr("w", extent=extent),
            ub_src0.access_ptr("r", extent=extent),
            ub_src1.access_ptr("r"),
        )),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=[ub_src0, ub_src1, ub_dst],
    )
    primfunc = tvm.tir.PrimFunc([], tvm.tir.BlockRealize([], True, root))
    program = build_kernel_program(primfunc, platform="A3")
    fill, task = program.tasks
    assert task.dependencies == (fill.task_id,)

    scalars = np.arange(8, dtype=np.float32) + 1
    broadcast = np.repeat(scalars, 8)
    simulator = FunctionalSimulator(program)
    simulator.write(task.metadata["scalar_src"], broadcast)
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(task.metadata["dst"]),
        np.full((8, 64), 2.0, dtype=np.float32) * scalars[:, None],
    )


def test_row_expand_validates_rows_dtype_alignment_and_tag() -> None:
    with pytest.raises(ProgramValidationError, match="256-byte"):
        build_kernel_program(
            _row_expand_primfunc(dtype="float16", cols_override=32),
            platform="A2",
        )
    with pytest.raises(ProgramValidationError, match="32-byte-aligned"):
        build_kernel_program(
            _row_expand_primfunc(dst_offset=1), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="scalar source must hold"):
        build_kernel_program(
            _row_expand_primfunc(scalar_count_override=7), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="tag dtype"):
        build_kernel_program(
            _row_expand_primfunc(tag_dtype_override="half"), platform="A2"
        )


@pytest.mark.parametrize(
    ("destination_scope", "destination_layout"),
    [("wmma.matrix_a", "l0a"), ("wmma.matrix_b", "l0b")],
)
def test_real_tir_l1_to_l0_copy_converts_physical_layout(
    destination_scope, destination_layout
) -> None:
    program = build_kernel_program(
        _l1_to_l0_primfunc(destination_scope=destination_scope), platform="A3"
    )
    task = program.tasks[0]
    expected_operation = (
        "copy_l1_to_l0a" if destination_scope == "wmma.matrix_a"
        else "copy_l1_to_l0b"
    )
    assert (task.operation, task.lane, task.pipe) == (
        expected_operation, Lane.CUBE, Pipe.MTE1,
    )
    assert task.metadata["copy"]["source_layout"] == "zN"
    assert task.metadata["copy"]["destination_layout"] == destination_layout

    simulator = FunctionalSimulator(program)
    logical = np.arange(32 * 32, dtype=np.float16).reshape(32, 32)
    simulator.write(task.metadata["src"], pack_matrix(logical, "zN"))
    simulator.run()

    physical = simulator.read(task.metadata["dst"])
    np.testing.assert_array_equal(
        unpack_matrix(physical, destination_layout, (32, 32)), logical
    )


def test_real_tir_transposed_l1_to_l0b_reinterprets_zn_as_nz() -> None:
    program = build_kernel_program(
        _l1_to_l0_primfunc(
            destination_scope="wmma.matrix_b",
            transpose=True,
            source_rows=16,
            source_cols=32,
        ),
        platform="A2",
    )
    task = program.tasks[0]
    assert task.metadata["copy"] == {
        "layout_transform": True,
        "source_layout": "nZ",
        "destination_layout": "l0b",
        "source_shape": (32, 16),
        "destination_shape": (32, 16),
        "source_origin": (0, 0),
        "source_window_direct": False,
        "source_region_axis": 0,
        "transpose": True,
    }

    simulator = FunctionalSimulator(program)
    original = np.arange(16 * 32, dtype=np.float16).reshape(16, 32)
    source_storage = pack_matrix(original, "zN")
    np.testing.assert_array_equal(source_storage, pack_matrix(original.T, "nZ"))
    simulator.write(task.metadata["src"], source_storage)
    simulator.run()

    physical = simulator.read(task.metadata["dst"])
    np.testing.assert_array_equal(
        unpack_matrix(physical, "l0b", (32, 16)), original.T
    )


def test_l1_to_l0_rejects_destination_larger_than_logical_source() -> None:
    with pytest.raises(ProgramValidationError, match="must fit"):
        build_kernel_program(
            _l1_to_l0_primfunc(destination_rows=48), platform="A3"
        )


@pytest.mark.parametrize(
    "transpose,destination_scope,destination_layout",
    [
        (False, "wmma.matrix_a", "l0a"),
        (True, "wmma.matrix_b", "l0b"),
    ],
)
def test_real_tir_l1_to_l0_decodes_aligned_source_window(
    transpose, destination_scope, destination_layout
) -> None:
    program = build_kernel_program(
        _l1_to_l0_primfunc(
            destination_scope=destination_scope,
            transpose=transpose,
            source_rows=32,
            source_cols=32,
            destination_rows=16,
            destination_cols=16,
            source_offset=768,
        ),
        platform="A2",
    )
    task = program.tasks[0]
    assert task.metadata["copy"]["source_origin"] == (16, 16)
    assert task.metadata["copy"]["source_window_direct"] is True
    assert task.metadata["src"].shape == (16, 16)
    assert task.metadata["src"].byte_offset == 768 * 2
    expected_strides = (2, 16 * 2) if transpose else (16 * 2, 2)
    assert task.metadata["src"].strides_bytes == expected_strides
    simulator = FunctionalSimulator(program)
    logical = np.arange(32 * 32, dtype=np.float16).reshape(32, 32)
    whole_l1 = BufferRegion(
        task.metadata["src"].buffer,
        MemoryScope.L1,
        (32 * 32,),
        "float16",
        core_id=task.core_id,
    )
    simulator.write(whole_l1, pack_matrix(logical, "zN"))
    simulator.run()
    source_logical = logical.T if transpose else logical
    np.testing.assert_array_equal(
        unpack_matrix(
            simulator.read(task.metadata["dst"]), destination_layout, (16, 16)
        ),
        source_logical[16:32, 16:32],
    )


def test_real_tir_l1_to_l0_uses_multiple_regions_across_c0_blocks() -> None:
    program = build_kernel_program(
        _l1_to_l0_primfunc(
            source_rows=32,
            source_cols=32,
            destination_rows=16,
            destination_cols=32,
            source_offset=256,
        ),
        platform="A3",
    )
    task = program.tasks[0]
    regions = task.metadata["src_regions"]
    assert task.metadata["copy"]["source_origin"] == (16, 0)
    assert task.metadata["copy"]["source_region_axis"] == 1
    assert [region.shape for region in regions] == [(16, 16), (16, 16)]
    assert [region.byte_offset for region in regions] == [256 * 2, 768 * 2]

    simulator = FunctionalSimulator(program)
    logical = np.arange(32 * 32, dtype=np.float16).reshape(32, 32)
    simulator.write(task.metadata["src"], pack_matrix(logical, "zN"))
    simulator.run()
    np.testing.assert_array_equal(
        unpack_matrix(simulator.read(task.metadata["dst"]), "l0a", (16, 32)),
        logical[16:32, :],
    )


def test_real_tir_gm_l1_l0a_pipeline_executes_with_raw_dependency() -> None:
    program = build_kernel_program(_gm_l1_l0_primfunc(), platform="A3")
    assert [task.operation for task in program.tasks] == [
        "copy_gm_to_l1", "copy_l1_to_l0a",
    ]
    assert program.tasks[1].dependencies == (program.tasks[0].task_id,)

    simulator = FunctionalSimulator(program)
    logical = np.arange(32 * 32, dtype=np.float16).reshape(32, 32)
    simulator.write(program.tasks[0].metadata["src"], logical)
    simulator.run()

    physical = simulator.read(program.tasks[1].metadata["dst"])
    np.testing.assert_array_equal(unpack_matrix(physical, "l0a", (32, 32)), logical)


def test_real_tir_l0c_to_gm_executes_relu_tail_and_dtype_conversion() -> None:
    program = build_kernel_program(_l0c_to_gm_primfunc(), platform="A3")
    task = program.tasks[0]
    assert (task.operation, task.lane, task.pipe) == (
        "copy_l0c_to_gm", Lane.CUBE, Pipe.FIX,
    )
    assert task.metadata["copy"] == {
        "layout_transform": True,
        "source_layout": "l0c",
        "destination_layout": "row_major",
        "source_shape": (16, 17),
        "destination_shape": (13, 17),
        "relu": True,
        "unit_flag": 0,
        "destination_cols": 32,
    }

    simulator = FunctionalSimulator(program)
    logical = np.linspace(-4, 4, 16 * 32, dtype=np.float32).reshape(16, 32)
    simulator.write(task.metadata["src"], pack_matrix(logical, "l0c"))
    simulator.run()

    expected = np.maximum(logical[:13, :17], 0).astype(np.float16)
    np.testing.assert_array_equal(simulator.read(task.metadata["dst"]), expected)


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize(
    ("source_dtype", "destination_dtype"),
    [("float32", "float32"), ("float32", "float16"), ("int32", "int32")],
)
def test_l0c_atomic_add_decodes_tail_converts_then_accumulates(
    platform, source_dtype, destination_dtype
) -> None:
    program = build_kernel_program(
        _atomic_add_l0c_primfunc(
            source_dtype=source_dtype,
            destination_dtype=destination_dtype,
        ),
        platform=platform,
    )
    task = program.tasks[0]
    assert (task.operation, task.lane, task.pipe) == (
        "atomic_add_l0c_to_gm", Lane.CUBE, Pipe.FIX,
    )
    assert task.metadata["accumulator"] == task.metadata["dst"]
    logical = np.arange(16 * 32, dtype=np.dtype(source_dtype)).reshape(16, 32)
    output = BufferRegion(
        "output", MemoryScope.GM, (16, 32), destination_dtype
    )
    initial = np.full((16, 32), 2, dtype=np.dtype(destination_dtype))
    simulator = FunctionalSimulator(program)
    simulator.write(task.metadata["src"], pack_matrix(logical, "l0c"))
    simulator.write(output, initial)
    simulator.run()

    expected = initial.copy()
    expected[:13, :17] += logical[:13, :17].astype(destination_dtype)
    np.testing.assert_array_equal(simulator.read(output), expected)


def test_l0c_atomic_add_rejects_unsupported_dtype_pair() -> None:
    with pytest.raises(UnsupportedSimOpError, match="does not support"):
        build_kernel_program(
            _atomic_add_l0c_primfunc(destination_dtype="int32"),
            platform="A3",
        )


def test_l0c_to_gm_rejects_unpaired_unit_flag() -> None:
    with pytest.raises(ProgramValidationError, match="preceding paired MMA"):
        build_kernel_program(_l0c_to_gm_primfunc(unit_flag=3), platform="A2")


def test_mma_fixpipe_unit_flag_pair_executes_partial_columns() -> None:
    program = build_kernel_program(_mma_fixpipe_primfunc(), platform="A3")
    mma, fixpipe = program.tasks
    assert mma.metadata["unit_flag_role"] == "release"
    assert mma.metadata["unit_flag_pair"] == fixpipe.task_id
    assert fixpipe.metadata["unit_flag_role"] == "consume"
    assert fixpipe.metadata["unit_flag_pair"] == mma.task_id
    assert mma.task_id in fixpipe.dependencies
    assert fixpipe.metadata["src"].shape == (16 * 16,)

    simulator = FunctionalSimulator(program)
    left = (np.arange(16 * 13, dtype=np.float16).reshape(16, 13) - 50) / 32
    right = (np.arange(13 * 16, dtype=np.float16).reshape(13, 16) - 70) / 64
    simulator.write(mma.metadata["lhs"], pack_matrix(left, "l0a"))
    simulator.write(mma.metadata["rhs"], pack_matrix(right, "l0b"))
    simulator.run()
    np.testing.assert_allclose(
        simulator.read(fixpipe.metadata["dst"]),
        (left.astype(np.float32) @ right.astype(np.float32)).astype(np.float16),
        rtol=1e-3,
        atol=1e-3,
    )


def test_mma_fixpipe_unit_flag_hold_accumulates_then_releases() -> None:
    program = build_kernel_program(
        _mma_fixpipe_primfunc(accumulate=True), platform="A2"
    )
    first, second, fixpipe = program.tasks
    assert first.metadata["unit_flag_role"] == "hold"
    assert second.metadata["unit_flag_role"] == "release"
    assert fixpipe.metadata["unit_flag_role"] == "consume"
    assert first.task_id in second.dependencies
    assert second.task_id in fixpipe.dependencies

    simulator = FunctionalSimulator(program)
    expected = np.zeros((16, 16), dtype=np.float32)
    for index, mma in enumerate((first, second)):
        left = (
            np.arange(16 * 13, dtype=np.float16).reshape(16, 13) - 50 + index
        ) / 32
        right = (
            np.arange(13 * 16, dtype=np.float16).reshape(13, 16) - 70 - index
        ) / 64
        simulator.write(mma.metadata["lhs"], pack_matrix(left, "l0a"))
        simulator.write(mma.metadata["rhs"], pack_matrix(right, "l0b"))
        expected += left.astype(np.float32) @ right.astype(np.float32)
    simulator.run()
    np.testing.assert_allclose(
        simulator.read(fixpipe.metadata["dst"]),
        expected.astype(np.float16),
        rtol=1e-3,
        atol=1e-3,
    )


def test_mma_fixpipe_unit_flag_pair_rejects_column_mismatch() -> None:
    with pytest.raises(ProgramValidationError, match="column counts disagree"):
        build_kernel_program(
            _mma_fixpipe_primfunc(fix_valid_cols=32), platform="A2"
        )


def test_mma_rejects_unmatched_release_unit_flag() -> None:
    with pytest.raises(ProgramValidationError, match="following paired fixpipe"):
        build_kernel_program(
            _mma_primfunc(cols=32, n_actual=16, unit_flag=3), platform="A3"
        )


def test_mma_rejects_unknown_unit_flag() -> None:
    with pytest.raises(UnsupportedSimOpError, match="unitFlag"):
        build_kernel_program(
            _mma_primfunc(cols=32, n_actual=16, unit_flag=1), platform="A2"
        )


def test_fixpipe_rejects_unknown_unit_flag() -> None:
    with pytest.raises(UnsupportedSimOpError, match="unitFlag"):
        build_kernel_program(_l0c_to_gm_primfunc(unit_flag=2), platform="A3")


@pytest.mark.parametrize("initialize", [True, False])
def test_real_tir_mma_executes_k_tail_and_accumulation(initialize) -> None:
    program = build_kernel_program(_mma_primfunc(init=initialize), platform="A2")
    task = program.tasks[0]
    assert (task.operation, task.lane, task.pipe) == (
        "mma", Lane.CUBE, Pipe.MATRIX,
    )
    assert task.metadata["mma"] == {
        "rows": 16,
        "cols": 16,
        "inner": 13,
        "init": initialize,
        "n_actual": 16,
        "unit_flag": 0,
    }
    assert ("accumulator" in task.metadata) is (not initialize)

    simulator = FunctionalSimulator(program)
    left = (np.arange(16 * 13, dtype=np.float16).reshape(16, 13) - 50) / 32
    right = (np.arange(13 * 16, dtype=np.float16).reshape(13, 16) - 70) / 64
    simulator.write(task.metadata["lhs"], pack_matrix(left, "l0a"))
    simulator.write(task.metadata["rhs"], pack_matrix(right, "l0b"))
    expected = left.astype(np.float32) @ right.astype(np.float32)
    if not initialize:
        previous = np.arange(16 * 16, dtype=np.float32).reshape(16, 16) / 17
        simulator.write(
            task.metadata["accumulator"], pack_matrix(previous, "l0c")
        )
        expected += previous
    simulator.run()

    physical = simulator.read(task.metadata["dst"])
    np.testing.assert_allclose(
        unpack_matrix(physical, "l0c", (16, 16)), expected, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("initialize", [True, False])
def test_real_tir_mma_executes_int8_to_int32(platform, initialize) -> None:
    program = build_kernel_program(
        _mma_primfunc(
            init=initialize,
            inner=35,
            input_dtype="int8",
            accumulator_dtype="int32",
        ),
        platform=platform,
    )
    task = program.tasks[0]
    simulator = FunctionalSimulator(program)
    left = (np.arange(16 * 35).reshape(16, 35) % 11 - 5).astype(np.int8)
    right = (np.arange(35 * 16).reshape(35, 16) % 13 - 6).astype(np.int8)
    simulator.write(task.metadata["lhs"], pack_matrix(left, "l0a"))
    simulator.write(task.metadata["rhs"], pack_matrix(right, "l0b"))
    expected = left.astype(np.int32) @ right.astype(np.int32)
    if not initialize:
        previous = np.arange(16 * 16, dtype=np.int32).reshape(16, 16)
        simulator.write(
            task.metadata["accumulator"], pack_matrix(previous, "l0c")
        )
        expected += previous
    simulator.run()

    np.testing.assert_array_equal(
        unpack_matrix(simulator.read(task.metadata["dst"]), "l0c", (16, 16)),
        expected,
    )


def test_mma_rejects_unsupported_int8_accumulator_dtype() -> None:
    with pytest.raises(UnsupportedSimOpError, match="int8-to-int32"):
        build_kernel_program(
            _mma_primfunc(input_dtype="int8", accumulator_dtype="float32"),
            platform="A2",
        )


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("initialize", [True, False])
def test_mma_bias_copies_bt_and_broadcasts_across_rows(
    platform, initialize
) -> None:
    program = build_kernel_program(
        _mma_bias_primfunc(init=initialize), platform=platform
    )
    copy_bias, mma = program.tasks
    assert (copy_bias.operation, copy_bias.lane, copy_bias.pipe) == (
        "copy_l1_to_bt", Lane.CUBE, Pipe.MTE1,
    )
    assert (mma.operation, mma.lane, mma.pipe) == (
        "mma_bias", Lane.CUBE, Pipe.MATRIX,
    )
    assert mma.dependencies == (copy_bias.task_id,)
    assert mma.metadata["mma"]["bias"] is True
    assert "accumulator" not in mma.metadata

    left = (np.arange(16 * 13, dtype=np.float16).reshape(16, 13) - 50) / 32
    right = (np.arange(13 * 16, dtype=np.float16).reshape(13, 16) - 70) / 64
    bias = np.linspace(-2, 2, 16, dtype=np.float32)
    simulator = FunctionalSimulator(program)
    simulator.write(mma.metadata["lhs"], pack_matrix(left, "l0a"))
    simulator.write(mma.metadata["rhs"], pack_matrix(right, "l0b"))
    simulator.write(copy_bias.metadata["src"], bias)
    simulator.run()

    expected = left.astype(np.float32) @ right.astype(np.float32) + bias
    np.testing.assert_allclose(
        unpack_matrix(simulator.read(mma.metadata["dst"]), "l0c", (16, 16)),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_copy_l1_to_bt_rejects_non_64_byte_transfer() -> None:
    with pytest.raises(UnsupportedSimOpError, match="64-byte"):
        build_kernel_program(_mma_bias_primfunc(bias_length=15), platform="A2")


def test_bias_pipeline_rejects_non_bt_destination() -> None:
    with pytest.raises(ProgramValidationError, match="BT destination"):
        build_kernel_program(
            _mma_bias_primfunc(bias_scope="shared.ub"), platform="A3"
        )


@pytest.mark.parametrize("initialize", [True, False])
def test_real_tir_mma_executes_partial_n_actual(initialize) -> None:
    program = build_kernel_program(
        _mma_primfunc(init=initialize, cols=32, n_actual=16), platform="A3"
    )
    task = program.tasks[0]
    assert task.metadata["mma"]["cols"] == 32
    assert task.metadata["mma"]["n_actual"] == 16
    assert task.metadata["rhs"].shape == (16 * 16,)
    assert task.metadata["dst"].shape == (16 * 16,)

    simulator = FunctionalSimulator(program)
    left = (np.arange(16 * 13, dtype=np.float16).reshape(16, 13) - 50) / 32
    right = (np.arange(13 * 16, dtype=np.float16).reshape(13, 16) - 70) / 64
    simulator.write(task.metadata["lhs"], pack_matrix(left, "l0a"))
    simulator.write(task.metadata["rhs"], pack_matrix(right, "l0b"))
    expected = left.astype(np.float32) @ right.astype(np.float32)
    if not initialize:
        previous = np.arange(16 * 16, dtype=np.float32).reshape(16, 16) / 17
        simulator.write(
            task.metadata["accumulator"], pack_matrix(previous, "l0c")
        )
        expected += previous
    simulator.run()

    np.testing.assert_allclose(
        unpack_matrix(simulator.read(task.metadata["dst"]), "l0c", (16, 16)),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("n_actual", [0, 8, 48])
def test_mma_rejects_illegal_partial_n_actual(n_actual) -> None:
    error = ProgramValidationError if n_actual in {0, 48} else UnsupportedSimOpError
    with pytest.raises(error, match="n_actual"):
        build_kernel_program(
            _mma_primfunc(cols=32, n_actual=n_actual), platform="A3"
        )


@pytest.mark.parametrize("n_actual", [16, 32])
def test_real_tir_mma_resolves_buffer_load_n_actual(n_actual) -> None:
    program = build_kernel_program(_mma_runtime_n_primfunc(), platform="A2")
    task = program.tasks[0]
    assert task.metadata["mma"]["n_actual"] == AffineInt.variable("nact[0]")
    simulator = FunctionalSimulator(program, bindings={"nact[0]": n_actual})
    left = (np.arange(16 * 13, dtype=np.float16).reshape(16, 13) - 50) / 32
    right = (
        np.arange(13 * n_actual, dtype=np.float16).reshape(13, n_actual) - 70
    ) / 64
    simulator.write(task.metadata["lhs"], pack_matrix(left, "l0a"))
    simulator.write(task.metadata["rhs"], pack_matrix(right, "l0b"))
    simulator.run()
    np.testing.assert_allclose(
        unpack_matrix(
            simulator.read(task.metadata["dst"]), "l0c", (16, n_actual)
        ),
        left.astype(np.float32) @ right.astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("bindings", [{}, {"nact[0]": 0}, {"nact[0]": 8}, {"nact[0]": 48}])
def test_mma_rejects_invalid_runtime_n_actual_binding(bindings) -> None:
    program = build_kernel_program(_mma_runtime_n_primfunc(), platform="A3")
    with pytest.raises(ProgramValidationError, match="nact|n_actual"):
        FunctionalSimulator(program, bindings=bindings)


def test_mma_rejects_out_of_bounds_runtime_scalar_load() -> None:
    with pytest.raises(ProgramValidationError, match="outside its buffer"):
        build_kernel_program(_mma_runtime_n_primfunc(nact_index=1), platform="A2")


def test_real_tir_mma_chain_accumulates_k_tiles_with_raw_dependency() -> None:
    program = build_kernel_program(_mma_chain_primfunc(), platform="A3")
    first, second = program.tasks
    assert second.dependencies == (first.task_id,)
    simulator = FunctionalSimulator(program)
    expected = np.zeros((16, 16), dtype=np.float32)
    for index, task in enumerate(program.tasks):
        left = (
            np.arange(16 * 13, dtype=np.float16).reshape(16, 13) - 40 + index
        ) / 32
        right = (
            np.arange(13 * 16, dtype=np.float16).reshape(13, 16) - 60 - index
        ) / 64
        simulator.write(task.metadata["lhs"], pack_matrix(left, "l0a"))
        simulator.write(task.metadata["rhs"], pack_matrix(right, "l0b"))
        expected += left.astype(np.float32) @ right.astype(np.float32)
    simulator.run()
    np.testing.assert_allclose(
        unpack_matrix(simulator.read(second.metadata["dst"]), "l0c", (16, 16)),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "transpose_a,transpose_b",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("initialize", [True, False])
def test_real_tir_gemm_v0_executes_transpose_and_accumulation(
    transpose_a, transpose_b, initialize
) -> None:
    prim_func, shape_a, shape_b = _gemm_v0_primfunc(
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        init=initialize,
    )
    program = build_kernel_program(prim_func, platform="A2")
    task = program.tasks[0]
    assert (task.operation, task.lane, task.pipe) == (
        "gemm_v0",
        Lane.CUBE,
        Pipe.MATRIX,
    )
    assert task.metadata["gemm"]["step_count"] == 1
    simulator = FunctionalSimulator(program)
    left = (np.arange(np.prod(shape_a), dtype=np.float16).reshape(shape_a) - 50) / 32
    right = (
        np.arange(np.prod(shape_b), dtype=np.float16).reshape(shape_b) - 70
    ) / 64
    simulator.write(task.metadata["lhs"], pack_matrix(left, "zn"))
    simulator.write(task.metadata["rhs"], pack_matrix(right, "zn"))
    expected = (left.T if transpose_a else left).astype(np.float32) @ (
        right.T if transpose_b else right
    ).astype(np.float32)
    if not initialize:
        previous = np.arange(256, dtype=np.float32).reshape(16, 16) / 17
        simulator.write(
            task.metadata["accumulator"], pack_matrix(previous, "l0c")
        )
        expected += previous
    simulator.run()
    np.testing.assert_allclose(
        unpack_matrix(simulator.read(task.metadata["dst"]), "l0c", (16, 16)),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (True, True)])
@pytest.mark.parametrize("initialize", [True, False])
def test_gemm_v0_int8_to_int32_expands_k_tail(
    platform, transpose_a, transpose_b, initialize
) -> None:
    prim_func, shape_a, shape_b = _gemm_v0_primfunc(
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        init=initialize,
        inner=61,
        k_l0_size=32,
        input_dtype="int8",
        accumulator_dtype="int32",
    )
    program = build_kernel_program(prim_func, platform=platform)
    assert [task.operation for task in program.tasks] == [
        "copy_l1_to_l0a", "copy_l1_to_l0b", "mma",
        "copy_l1_to_l0a", "copy_l1_to_l0b", "mma",
    ]
    left = (np.arange(np.prod(shape_a)).reshape(shape_a) % 11 - 5).astype(np.int8)
    right = (np.arange(np.prod(shape_b)).reshape(shape_b) % 13 - 6).astype(np.int8)
    packed_left = pack_matrix(left, "zn")
    packed_right = pack_matrix(right, "zn")
    output = BufferRegion(
        "l0c", MemoryScope.L0C,
        (storage_elements("l0c", (16, 16), 4),), "int32",
    )
    simulator = FunctionalSimulator(program)
    simulator.write(
        BufferRegion("l1a", MemoryScope.L1, packed_left.shape, "int8"),
        packed_left,
    )
    simulator.write(
        BufferRegion("l1b", MemoryScope.L1, packed_right.shape, "int8"),
        packed_right,
    )
    expected = (left.T if transpose_a else left).astype(np.int32) @ (
        right.T if transpose_b else right
    ).astype(np.int32)
    if not initialize:
        previous = np.arange(16 * 16, dtype=np.int32).reshape(16, 16)
        simulator.write(output, pack_matrix(previous, "l0c"))
        expected += previous
    simulator.run()

    np.testing.assert_array_equal(
        unpack_matrix(simulator.read(output), "l0c", (16, 16)), expected
    )


@pytest.mark.parametrize("initialize", [True, False])
def test_gemm_v0_executes_partial_n_actual_prefix(initialize) -> None:
    prim_func, shape_a, shape_b = _gemm_v0_primfunc(
        transpose_b=True, init=initialize, cols=32, inner=48,
        k_l0_size=16, n_actual=16,
    )
    program = build_kernel_program(prim_func, platform="A2")
    assert [task.operation for task in program.tasks] == [
        "copy_l1_to_l0a", "copy_l1_to_l0b", "mma",
        "copy_l1_to_l0a", "copy_l1_to_l0b", "mma",
        "copy_l1_to_l0a", "copy_l1_to_l0b", "mma",
    ]
    assert program.tasks[1].metadata["copy"]["source_window_shape"] == (16, 16)
    assert program.tasks[2].metadata["mma"]["cols"] == 16
    assert program.tasks[2].metadata["mma"]["n_actual"] == 16
    assert program.tasks[2].metadata["dst"].shape == (16 * 16,)
    assert [
        program.tasks[index].metadata["src"].byte_offset
        for index in (1, 4, 7)
    ] == [0, 512 * 2, 1024 * 2]

    simulator = FunctionalSimulator(program)
    left = (np.arange(np.prod(shape_a), dtype=np.float16).reshape(shape_a) - 50) / 32
    right = (
        np.arange(np.prod(shape_b), dtype=np.float16).reshape(shape_b) - 70
    ) / 64
    for task, values in zip(program.tasks[:2], (left, right)):
        payload = pack_matrix(values, "zn")
        source = task.metadata["src"]
        simulator.write(
            BufferRegion(
                source.buffer, MemoryScope.L1, (payload.size,), source.dtype,
                core_id=task.core_id,
            ),
            payload,
        )
    expected = left.astype(np.float32) @ right[:16].T.astype(np.float32)
    final = program.tasks[-1]
    if not initialize:
        previous = np.arange(16 * 16, dtype=np.float32).reshape(16, 16) / 17
        simulator.write(
            program.tasks[2].metadata["accumulator"], pack_matrix(previous, "l0c")
        )
        expected += previous
    simulator.run()
    np.testing.assert_allclose(
        unpack_matrix(
            simulator.read(final.metadata["dst"]), "l0c", (16, 16)
        ),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_gemm_v0_rejects_partial_n_actual_without_transpose_b() -> None:
    prim_func, _, _ = _gemm_v0_primfunc(cols=32, n_actual=16)
    with pytest.raises(UnsupportedSimOpError, match="transpose_B"):
        build_kernel_program(prim_func, platform="A2")


@pytest.mark.parametrize("n_actual", [0, 8, 48])
def test_gemm_v0_rejects_illegal_partial_n_actual(n_actual) -> None:
    prim_func, _, _ = _gemm_v0_primfunc(
        transpose_b=True, cols=32, n_actual=n_actual,
    )
    error = ProgramValidationError if n_actual in {0, 48} else UnsupportedSimOpError
    with pytest.raises(error, match="n_actual"):
        build_kernel_program(prim_func, platform="A3")


@pytest.mark.parametrize("n_actual", [16, 32])
def test_gemm_v0_resolves_buffer_load_n_actual(n_actual) -> None:
    program = build_kernel_program(_gemm_v0_runtime_n_primfunc(), platform="A3")
    assert program.tasks[2].metadata["mma"]["n_actual"] == AffineInt.variable(
        "nact[0]"
    )
    simulator = FunctionalSimulator(program, bindings={"nact[0]": n_actual})
    left = (np.arange(16 * 48, dtype=np.float16).reshape(16, 48) - 50) / 32
    right = (np.arange(32 * 48, dtype=np.float16).reshape(32, 48) - 70) / 64
    for task, values in zip(program.tasks[:2], (left, right)):
        payload = pack_matrix(values, "zn")
        source = task.metadata["src"]
        simulator.write(
            BufferRegion(
                source.buffer, MemoryScope.L1, (payload.size,), source.dtype,
                core_id=task.core_id,
            ),
            payload,
        )
    simulator.run()
    final = program.tasks[-1]
    np.testing.assert_allclose(
        unpack_matrix(
            simulator.read(final.metadata["dst"]), "l0c", (16, n_actual)
        ),
        left.astype(np.float32) @ right[:n_actual].T.astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_gemm_v0_rejects_internal_l0_tile_overflow() -> None:
    prim_func, _, _ = _gemm_v0_primfunc(k_l0_size=4080)
    with pytest.raises(ProgramValidationError, match="L0A slot"):
        build_kernel_program(prim_func, platform="A2")


@pytest.mark.parametrize(
    "transpose_a,transpose_b",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_gemm_v0_expands_multi_k_steps_with_real_l0_payloads(
    transpose_a, transpose_b
) -> None:
    prim_func, shape_a, shape_b = _gemm_v0_primfunc(
        inner=48,
        k_l0_size=16,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )
    timing = TimingProfile(
        platform="A3",
        operation_cycles={
            "gemm_v0.load_a": 3,
            "gemm_v0.load_b": 5,
            "gemm_v0.mma": 11,
        },
    )
    program = build_kernel_program(
        prim_func, platform="A3", timing_profile=timing
    )
    assert [task.operation for task in program.tasks] == [
        "copy_l1_to_l0a", "copy_l1_to_l0b", "mma",
        "copy_l1_to_l0a", "copy_l1_to_l0b", "mma",
        "copy_l1_to_l0a", "copy_l1_to_l0b", "mma",
    ]
    assert [task.duration_cycles for task in program.tasks[:3]] == [3, 5, 11]
    assert [task.metadata["timing_key"] for task in program.tasks[:3]] == [
        "gemm_v0.load_a", "gemm_v0.load_b", "gemm_v0.mma",
    ]
    assert all(
        task.metadata["timing_calibration"] == "uncalibrated-unit-cost"
        for task in program.tasks
    )
    assert "wait_event" not in program.tasks[0].metadata
    assert program.tasks[0].metadata["transfer_bytes"] == 16 * 16 * 2
    assert program.tasks[2].metadata["math_ops"] == 2 * 16 * 16 * 16
    assert program.tasks[2].metadata["wait_event"] == "MTE1_M"
    assert program.tasks[2].metadata["set_event"] == "M_MTE1"
    assert program.tasks[6].metadata["wait_event"] == "M_MTE1"
    final = program.tasks[-1]
    assert program.tasks[2].task_id in program.tasks[6].dependencies
    assert [program.tasks[index].metadata["src"].byte_offset for index in (0, 3, 6)] == [
        0, 256 * 2, 512 * 2,
    ]
    assert [program.tasks[index].metadata["src"].byte_offset for index in (1, 4, 7)] == [
        0, 256 * 2, 512 * 2,
    ]
    simulator = FunctionalSimulator(program)
    left = np.arange(np.prod(shape_a), dtype=np.float16).reshape(shape_a) / 64
    right = (
        np.arange(np.prod(shape_b), dtype=np.float16).reshape(shape_b) - 100
    ) / 128
    for task, values in zip(program.tasks[:2], (left, right)):
        payload = pack_matrix(values, "zn")
        source = task.metadata["src"]
        simulator.write(
            BufferRegion(
                source.buffer, MemoryScope.L1, (payload.size,), source.dtype,
                core_id=task.core_id,
            ),
            payload,
        )
    result = simulator.run()
    records = {record.task_id: record for record in result.schedule.records}
    assert records[program.tasks[2].task_id].start_cycle == records[
        program.tasks[3].task_id
    ].start_cycle
    np.testing.assert_allclose(
        unpack_matrix(simulator.read(final.metadata["dst"]), "l0c", (16, 16)),
        (left.T if transpose_a else left).astype(np.float32)
        @ (right.T if transpose_b else right).astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_gemm_v0_expands_n_tiles_into_l0c_column_bands() -> None:
    prim_func, shape_a, shape_b = _gemm_v0_primfunc(
        rows=16, cols=256, inner=16, k_l0_size=128
    )
    program = build_kernel_program(prim_func, platform="A2")
    assert [task.metadata["gemm_stage"]["n_index"] for task in program.tasks] == [
        0, 0, 0, 1, 1, 1,
    ]
    assert len(program.tasks[1].metadata["src_regions"]) == 8
    assert len(program.tasks[4].metadata["src_regions"]) == 8
    assert program.tasks[4].metadata["src_regions"][0].byte_offset == 2048 * 2
    simulator = FunctionalSimulator(program)
    left = np.arange(np.prod(shape_a), dtype=np.float16).reshape(shape_a) / 64
    right = (
        np.arange(np.prod(shape_b), dtype=np.float16).reshape(shape_b) - 100
    ) / 128
    for task, values in zip(program.tasks[:2], (left, right)):
        payload = pack_matrix(values, "zn")
        source = task.metadata["src"]
        simulator.write(
            BufferRegion(
                source.buffer, MemoryScope.L1, (payload.size,), source.dtype,
                core_id=task.core_id,
            ),
            payload,
        )
    simulator.run()
    output = BufferRegion(
        program.tasks[2].metadata["dst"].buffer,
        MemoryScope.L0C,
        (16 * 256,),
        "float32",
        core_id=program.tasks[2].core_id,
    )
    np.testing.assert_allclose(
        unpack_matrix(simulator.read(output), "l0c", (16, 256)),
        left.astype(np.float32) @ right.astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_real_tir_gemm_pipeline_executes_without_bisheng() -> None:
    program = build_kernel_program(_gemm_pipeline_primfunc(), platform="A3")
    assert [task.operation for task in program.tasks] == [
        "copy_gm_to_l1",
        "copy_gm_to_l1",
        "copy_l1_to_l0a",
        "copy_l1_to_l0b",
        "mma",
        "copy_l0c_to_gm",
    ]
    simulator = FunctionalSimulator(program)
    left = (np.arange(256, dtype=np.float16).reshape(16, 16) - 80) / 32
    right = (np.arange(256, dtype=np.float16).reshape(16, 16) - 120) / 64
    simulator.write(program.tasks[0].metadata["src"], left)
    simulator.write(program.tasks[1].metadata["src"], right)
    simulator.run()

    expected = left.astype(np.float32) @ right.astype(np.float32)
    np.testing.assert_allclose(
        simulator.read(program.tasks[-1].metadata["dst"]),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_real_tir_gemm_v0_pipeline_executes_without_bisheng() -> None:
    program = build_kernel_program(
        _gemm_pipeline_primfunc(use_gemm_v0=True), platform="A3"
    )
    assert [task.operation for task in program.tasks] == [
        "copy_gm_to_l1",
        "copy_gm_to_l1",
        "gemm_v0",
        "copy_l0c_to_gm",
    ]
    simulator = FunctionalSimulator(program)
    left = (np.arange(256, dtype=np.float16).reshape(16, 16) - 80) / 32
    right = (np.arange(256, dtype=np.float16).reshape(16, 16) - 120) / 64
    simulator.write(program.tasks[0].metadata["src"], left)
    simulator.write(program.tasks[1].metadata["src"], right)
    simulator.run()

    expected = left.astype(np.float32) @ right.astype(np.float32)
    np.testing.assert_allclose(
        simulator.read(program.tasks[-1].metadata["dst"]),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


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


def test_symbolic_extent_and_affine_offsets_bind_at_runtime() -> None:
    program = build_kernel_program(_dynamic_copy_primfunc(), platform="A2")
    task = program.tasks[0]
    source = task.metadata["src"]
    destination = task.metadata["dst"]
    assert source.shape == destination.shape == (1, AffineInt.variable("valid_cols"))
    assert source.byte_offset == AffineInt((("valid_cols", 4),), constant=4)
    assert destination.byte_offset == AffineInt((("valid_cols", 8),))

    simulator = FunctionalSimulator(program, bindings={"valid_cols": 5})
    whole_source = BufferRegion("source", MemoryScope.GM, (32,), "float32")
    values = np.arange(32, dtype=np.float32)
    simulator.write(whole_source, values)
    simulator.run()

    np.testing.assert_array_equal(
        simulator.read(destination),
        values[6:11].reshape(1, 5),
    )


def test_missing_symbolic_binding_fails_before_memory_access() -> None:
    program = build_kernel_program(_dynamic_copy_primfunc(), platform="A2")
    simulator = FunctionalSimulator(program)
    simulator.write(
        BufferRegion("source", MemoryScope.GM, (32,), "float32"),
        np.arange(32, dtype=np.float32),
    )

    with pytest.raises(ProgramValidationError, match="missing runtime binding"):
        simulator.run()


def test_real_tir_tail_unary_scalar_and_binary_execute_valid_rectangle() -> None:
    program = build_kernel_program(_tail_vector_primfunc(), platform="A3")
    assert [task.operation for task in program.tasks] == [
        "copy_gm_to_ub",
        "copy_gm_to_ub",
        "relu",
        "adds",
        "mul",
        "copy_ub_to_gm",
    ]
    load_x, load_y, relu, adds, mul, store = program.tasks
    assert relu.metadata["tail_kind"] == "tail_unary"
    assert adds.metadata["tail_kind"] == "tail_scalar"
    assert mul.metadata["tail_kind"] == "tail_binary"
    assert relu.metadata["tail"] == {
        "valid_rows": 1,
        "valid_cols": 5,
        "physical_cols": 8,
    }
    assert relu.dependencies == (load_x.task_id,)
    assert adds.dependencies == (relu.task_id,)
    assert set(mul.dependencies) == {load_y.task_id, adds.task_id}
    assert store.dependencies == (mul.task_id,)

    simulator = FunctionalSimulator(program)
    x = np.array([[-3, -1, 0, 2, 4, 99, 99, 99]], dtype=np.float32)
    y = np.array([[2, 3, 4, 5, 6, 99, 99, 99]], dtype=np.float32)
    simulator.write(BufferRegion("x", MemoryScope.GM, (1, 8), "float32"), x)
    simulator.write(BufferRegion("y", MemoryScope.GM, (1, 8), "float32"), y)

    simulator.run()

    output = store.metadata["dst"]
    np.testing.assert_array_equal(
        simulator.read(output),
        (np.maximum(x[:, :5], 0) + 1.5) * y[:, :5],
    )


def test_tail_operation_tag_must_match_intrinsic_family() -> None:
    with pytest.raises(UnsupportedSimOpError, match="not valid for tail_unary"):
        build_kernel_program(_tail_vector_primfunc(unary_tag="Add"), platform="A2")


def test_real_tir_cast_rint_executes_and_converts_destination_dtype() -> None:
    program = build_kernel_program(_cast_primfunc(), platform="A2")
    assert [task.operation for task in program.tasks] == [
        "copy_gm_to_ub",
        "cast",
        "copy_ub_to_gm",
    ]
    load, cast, store = program.tasks
    assert cast.metadata["round_mode"] == "CAST_RINT"
    assert cast.metadata["src"].dtype == "float32"
    assert cast.metadata["dst"].dtype == "int32"
    assert cast.dependencies == (load.task_id,)
    assert store.dependencies == (cast.task_id,)

    simulator = FunctionalSimulator(program)
    values = np.array([-1.5, -0.5, 0.5, 1.5, 2.6, 99, 99, 99], dtype=np.float32)
    simulator.write(
        BufferRegion("source", MemoryScope.GM, (8,), "float32"), values
    )
    simulator.run()

    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]),
        np.array([-2, 0, 0, 2, 3], dtype=np.int32),
    )


@pytest.mark.parametrize(
    ("round_mode", "expected"),
    [
        ("CAST_FLOOR", [-3, -3, -2, -1, 0]),
        ("CAST_CEIL", [-2, -2, -1, 0, 1]),
        ("CAST_ROUND", [-3, -2, -2, -1, 1]),
        ("CAST_TRUNC", [-2, -2, -1, 0, 0]),
    ],
)
def test_real_tir_cast_round_modes_execute(round_mode, expected) -> None:
    program = build_kernel_program(_cast_primfunc(round_mode), platform="A3")
    load, _, store = program.tasks
    simulator = FunctionalSimulator(program)
    values = np.array(
        [-2.5, -2.1, -1.5, -0.5, 0.5, 1.5, 2.1, 2.5], dtype=np.float32
    )
    simulator.write(
        BufferRegion("source", MemoryScope.GM, (8,), "float32"), values
    )

    simulator.run()

    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]), np.asarray(expected, dtype=np.int32)
    )


def test_real_tir_fill_initializes_offset_region_and_builds_dependency() -> None:
    count = tvm.tir.Var("count", "int32")
    program = build_kernel_program(_fill_primfunc(count=count), platform="A3")
    assert [task.operation for task in program.tasks] == [
        "fill",
        "copy_ub_to_gm",
    ]
    fill, store = program.tasks
    assert fill.metadata["scalar"] == -3.5
    assert fill.metadata["dst"].byte_offset == 4
    assert store.metadata["dst"].byte_offset == 8
    assert store.dependencies == (fill.task_id,)

    simulator = FunctionalSimulator(program, bindings={"count": 4})
    simulator.run()

    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]),
        np.full((4,), -3.5, dtype=np.float32),
    )


def test_real_tir_fill_rejects_negative_count() -> None:
    with pytest.raises(ProgramValidationError, match="fill count must not be negative"):
        build_kernel_program(_fill_primfunc(count=-1), platform="A2")


@pytest.mark.parametrize(
    ("operation", "dtype", "first", "difference", "expected"),
    [
        ("arith_progression", "int32", 5, -2, [5, 3, 1, -1, -3, -5, -7]),
        ("arith_progression", "float16", 0.5, 0.25,
         [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]),
        ("createvecindex", "uint16", 65534, 99, [65534, 65535, 0, 1, 2, 3, 4]),
    ],
)
def test_sequence_intrinsics_execute_offset_region(
    operation, dtype, first, difference, expected
) -> None:
    program = build_kernel_program(
        _sequence_primfunc(
            operation, dtype=dtype, first=first, difference=difference,
        ),
        platform="A2",
    )
    sequence, store = program.tasks
    assert sequence.operation == operation
    assert sequence.metadata["dst"].byte_offset == 2 * np.dtype(dtype).itemsize
    assert sequence.metadata["count"] == 7
    assert sequence.metadata["difference"] == (
        1 if operation == "createvecindex" else difference
    )
    assert store.dependencies == (sequence.task_id,)

    simulator = FunctionalSimulator(program)
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]), np.asarray(expected, dtype=dtype)
    )


def test_arith_progression_resolves_runtime_integer_scalars_and_count() -> None:
    count = tvm.tir.Var("count", "int32")
    first = tvm.tir.Var("first", "int32")
    difference = tvm.tir.Var("difference", "int32")
    program = build_kernel_program(
        _sequence_primfunc(
            "arith_progression", count=count, first=first, difference=difference,
        ),
        platform="A3",
    )
    sequence, store = program.tasks
    assert sequence.metadata["count"] == AffineInt.variable("count")
    assert sequence.metadata["first_value"] == AffineInt.variable("first")
    assert sequence.metadata["difference"] == AffineInt.variable("difference")

    simulator = FunctionalSimulator(
        program, bindings={"count": 5, "first": -4, "difference": 3}
    )
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]),
        np.array([-4, -1, 2, 5, 8], dtype=np.int32),
    )


def test_sequence_intrinsics_reject_negative_count_and_dtype_mismatch() -> None:
    with pytest.raises(ProgramValidationError, match="count must not be negative"):
        build_kernel_program(
            _sequence_primfunc("createvecindex", count=-1), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="template dtype must match"):
        build_kernel_program(
            _sequence_primfunc(
                "arith_progression", dtype="float32", template_dtype="half"
            ),
            platform="A3",
        )


@pytest.mark.parametrize("dtype", ["int16", "uint16"])
@pytest.mark.parametrize(
    ("operation", "implementation", "with_scratch"),
    [
        ("bitwise_and", np.bitwise_and, False),
        ("bitwise_or", np.bitwise_or, False),
        ("bitwise_xor", np.bitwise_xor, True),
    ],
)
def test_bitwise_binary_executes_offset_region(
    dtype, operation, implementation, with_scratch
) -> None:
    program = build_kernel_program(
        _bitwise_primfunc(operation, dtype=dtype, with_scratch=with_scratch),
        platform="A2",
    )
    left_load, right_load, bitwise, store = program.tasks
    assert bitwise.metadata["dst"].byte_offset == np.dtype(dtype).itemsize
    assert set(bitwise.dependencies) == {left_load.task_id, right_load.task_id}
    assert ("scratch" in bitwise.metadata) is with_scratch
    assert store.dependencies == (bitwise.task_id,)

    left = np.array([0, 1, 3, 7, 15, 31, 0x5555, 0x7FFF], dtype=dtype)
    right = np.array([0, 2, 5, 8, 17, 0x1234, 0x3333, 0x7F00], dtype=dtype)
    simulator = FunctionalSimulator(program)
    simulator.write(left_load.metadata["src"], left)
    simulator.write(right_load.metadata["src"], right)
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]),
        implementation(left[1:7], right[1:7]),
    )


@pytest.mark.parametrize("dtype", ["int16", "uint16", "int32", "uint32"])
def test_bitwise_not_preserves_fixed_width_dtype(dtype) -> None:
    program = build_kernel_program(
        _bitwise_primfunc("bitwise_not", dtype=dtype), platform="A3"
    )
    left_load, bitwise, store = program.tasks
    assert bitwise.dependencies == (left_load.task_id,)
    values = np.arange(8, dtype=dtype)
    simulator = FunctionalSimulator(program)
    simulator.write(left_load.metadata["src"], values)
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]), np.bitwise_not(values[1:7])
    )


@pytest.mark.parametrize("operation", ["bitwise_lshift", "bitwise_rshift"])
@pytest.mark.parametrize("dtype", ["int16", "uint16", "int32", "uint32"])
def test_bitwise_shift_matches_signed_or_unsigned_numpy(operation, dtype) -> None:
    program = build_kernel_program(
        _bitwise_primfunc(operation, dtype=dtype, shift=3), platform="A2"
    )
    left_load, shift_task, store = program.tasks
    signed = dtype.startswith("int")
    values = np.array(
        [0, -9 if signed else 9, -1 if signed else 0xFFFF, 1, 7, 0x1234, 31, 2],
        dtype=dtype,
    )
    simulator = FunctionalSimulator(program)
    simulator.write(left_load.metadata["src"], values)
    simulator.run()
    implementation = np.left_shift if operation == "bitwise_lshift" else np.right_shift
    expected = implementation(values[1:7], 3).astype(dtype)
    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)
    assert shift_task.dependencies == (left_load.task_id,)


def test_runtime_bitwise_shift_validates_width() -> None:
    shift = tvm.tir.Var("shift", "int32")
    program = build_kernel_program(
        _bitwise_primfunc("bitwise_lshift", dtype="int16", shift=shift),
        platform="A3",
    )
    simulator = FunctionalSimulator(program, bindings={"shift": 17})
    simulator.write(program.tasks[0].metadata["src"], np.arange(8, dtype=np.int16))
    with pytest.raises(ProgramValidationError, match=r"\[0, 16\]"):
        simulator.run()
    with pytest.raises(UnsupportedSimOpError, match="does not support dtype"):
        build_kernel_program(
            _bitwise_primfunc("bitwise_not", dtype="float32"), platform="A2"
        )


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_mul_add_dst_executes_accumulator_region(dtype, platform) -> None:
    program = build_kernel_program(_mul_add_dst_primfunc(dtype=dtype), platform=platform)
    left_load, right_load, initial_load, multiply_add, store = program.tasks
    assert multiply_add.operation == "mul_add_dst"
    assert set(multiply_add.dependencies) == {
        left_load.task_id,
        right_load.task_id,
        initial_load.task_id,
    }
    assert multiply_add.metadata["accumulator"] == multiply_add.metadata["dst"]
    assert store.dependencies == (multiply_add.task_id,)

    left = np.linspace(-2, 2, 8, dtype=dtype)
    right = np.linspace(3, -1, 8, dtype=dtype)
    initial = np.linspace(1, 4, 8, dtype=dtype)
    simulator = FunctionalSimulator(program)
    simulator.write(left_load.metadata["src"], left)
    simulator.write(right_load.metadata["src"], right)
    simulator.write(initial_load.metadata["src"], initial)
    simulator.run()
    expected = left[1:7] * right[1:7] + initial[1:7]
    np.testing.assert_allclose(
        simulator.read(store.metadata["dst"]), expected, rtol=1e-3, atol=1e-3
    )


def test_mul_add_dst_validates_operand_dtypes() -> None:
    with pytest.raises(ProgramValidationError, match="operand dtypes must match"):
        build_kernel_program(
            _mul_add_dst_primfunc(dtype="float32", right_dtype="float16"),
            platform="A2",
        )
    with pytest.raises(UnsupportedSimOpError, match="does not support dtype"):
        build_kernel_program(_mul_add_dst_primfunc(dtype="int16"), platform="A3")


@pytest.mark.parametrize(
    ("platform", "dtype", "with_scratch"),
    [
        ("A2", "float16", False),
        ("A2", "int32", True),
        ("A3", "float32", True),
        ("A3", "uint16", False),
    ],
)
def test_gather_executes_flat_byte_offsets(platform, dtype, with_scratch) -> None:
    itemsize = np.dtype(dtype).itemsize
    program = build_kernel_program(
        _gather_primfunc(
            dtype=dtype, base=itemsize, with_scratch=with_scratch
        ),
        platform=platform,
    )
    source_load, offset_load, gather, store = program.tasks
    assert gather.operation == "gather"
    assert set(gather.dependencies) == {
        source_load.task_id, offset_load.task_id,
    }
    assert ("scratch" in gather.metadata) is with_scratch
    assert store.dependencies == (gather.task_id,)

    source_values = np.arange(source_load.metadata["src"].shape[0], dtype=dtype)
    offset_values = np.zeros(offset_load.metadata["src"].shape, dtype=np.uint32)
    relative_indices = np.array([0, 2, 5, 7, 9], dtype=np.uint32)
    offset_values[8:13] = relative_indices * itemsize
    simulator = FunctionalSimulator(program)
    simulator.write(source_load.metadata["src"], source_values)
    simulator.write(offset_load.metadata["src"], offset_values)
    simulator.run()

    source_start = 32 // itemsize
    expected_indices = source_start + relative_indices.astype(np.int64) + 1
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]), source_values[expected_indices]
    )


@pytest.mark.parametrize(
    ("base", "offset", "message"),
    [(1, 0, "align with element size"), (0, 4096, "index out of range")],
)
def test_gather_rejects_invalid_byte_address(base, offset, message) -> None:
    program = build_kernel_program(
        _gather_primfunc(dtype="float32", base=base), platform="A2"
    )
    source_load, offset_load, _, _ = program.tasks
    simulator = FunctionalSimulator(program)
    simulator.write(
        source_load.metadata["src"],
        np.arange(source_load.metadata["src"].shape[0], dtype=np.float32),
    )
    offset_values = np.zeros(offset_load.metadata["src"].shape, dtype=np.uint32)
    offset_values[8] = offset
    simulator.write(offset_load.metadata["src"], offset_values)
    with pytest.raises(ProgramValidationError, match=message):
        simulator.run()


def test_gather_rejects_non_32_bit_offsets() -> None:
    with pytest.raises(ProgramValidationError, match="int32 or uint32"):
        build_kernel_program(
            _gather_primfunc(offset_dtype="int16"), platform="A3"
        )


@pytest.mark.parametrize(
    ("platform", "dtype"),
    [
        ("A2", "uint16"),
        ("A2", "float16"),
        ("A3", "uint32"),
        ("A3", "int32"),
    ],
)
def test_gatherb_executes_eight_blocks_per_repeat(platform, dtype) -> None:
    program = build_kernel_program(
        _gatherb_primfunc(dtype=dtype, repeat=2), platform=platform
    )
    source_load, offset_load, gatherb, store = program.tasks
    assert gatherb.operation == "gatherb"
    assert gatherb.metadata["dst"].shape == (
        2, 8, 32 // np.dtype(dtype).itemsize
    )
    assert set(gatherb.dependencies) == {
        source_load.task_id, offset_load.task_id,
    }
    assert store.dependencies == (gatherb.task_id,)

    source_values = np.arange(source_load.metadata["src"].shape[0], dtype=dtype)
    offset_values = (
        np.arange(offset_load.metadata["src"].shape[0], dtype=np.uint32)[::-1]
        * 32
    )
    simulator = FunctionalSimulator(program)
    simulator.write(source_load.metadata["src"], source_values)
    simulator.write(offset_load.metadata["src"], offset_values)
    simulator.run()
    block_elements = 32 // np.dtype(dtype).itemsize
    expected = np.concatenate([
        source_values[int(offset) // np.dtype(dtype).itemsize:
                      int(offset) // np.dtype(dtype).itemsize + block_elements]
        for offset in offset_values
    ])
    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)


@pytest.mark.parametrize(
    ("offset", "message"),
    [(2, "32-byte aligned"), (24 * 32, "block out of range")],
)
def test_gatherb_rejects_invalid_block_offset(offset, message) -> None:
    program = build_kernel_program(_gatherb_primfunc(repeat=1), platform="A2")
    source_load, offset_load, _, _ = program.tasks
    simulator = FunctionalSimulator(program)
    simulator.write(
        source_load.metadata["src"],
        np.arange(source_load.metadata["src"].shape[0], dtype=np.uint16),
    )
    offsets = np.zeros(offset_load.metadata["src"].shape, dtype=np.uint32)
    offsets[0] = offset
    simulator.write(offset_load.metadata["src"], offsets)
    with pytest.raises(ProgramValidationError, match=message):
        simulator.run()


def test_gatherb_validates_control_and_offset_dtype() -> None:
    with pytest.raises(ProgramValidationError, match="must not exceed 255"):
        build_kernel_program(_gatherb_primfunc(repeat=256), platform="A3")
    with pytest.raises(ProgramValidationError, match="must not be negative"):
        build_kernel_program(
            _gatherb_primfunc(dst_block_stride=-1), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="int32 or uint32"):
        build_kernel_program(
            _gatherb_primfunc(offset_dtype="int16"), platform="A3"
        )
    with pytest.raises(ProgramValidationError, match="template dtype must match"):
        build_kernel_program(
            _gatherb_primfunc(dtype="uint16", template_dtype="uint32_t"),
            platform="A2",
        )

    repeat = tvm.tir.Var("repeat", "int32")
    program = build_kernel_program(
        _gatherb_primfunc(repeat=repeat), platform="A3"
    )
    source_load, offset_load, _, _ = program.tasks
    simulator = FunctionalSimulator(program, bindings={"repeat": 256})
    simulator.write(
        source_load.metadata["src"],
        np.arange(source_load.metadata["src"].shape[0], dtype=np.uint16),
    )
    simulator.write(
        offset_load.metadata["src"],
        np.zeros(offset_load.metadata["src"].shape, dtype=np.uint32),
    )
    with pytest.raises(ProgramValidationError, match="must not exceed 255"):
        simulator.run()


@pytest.mark.parametrize(
    ("pattern", "residues"),
    [
        ("P0101", (0, 2)),
        ("P1010", (1, 3)),
        ("P0001", (0,)),
        ("P0010", (1,)),
        ("P0100", (2,)),
        ("P1000", (3,)),
        ("P1111", (0, 1, 2, 3)),
    ],
)
def test_gather_mask_fixed_patterns_compact_and_zero_tail(pattern, residues) -> None:
    program = build_kernel_program(
        _gather_mask_primfunc(pattern=pattern), platform="A2"
    )
    load, gather_mask, store = program.tasks
    assert gather_mask.operation == "gather_mask"
    assert gather_mask.dependencies == (load.task_id,)
    assert store.dependencies == (gather_mask.task_id,)
    values = np.arange(1, 33, dtype=np.float32)
    simulator = FunctionalSimulator(program)
    simulator.write(load.metadata["src"], values)
    simulator.run()
    selected = values[np.isin(np.arange(values.size) % 4, residues)]
    expected = np.zeros_like(values)
    expected[:selected.size] = selected
    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)


def test_gather_mask_custom_indices_execute_with_scratch() -> None:
    program = build_kernel_program(
        _gather_mask_primfunc(dtype="int32", custom=True, with_scratch=True),
        platform="A3",
    )
    source_load, index_load, gather_mask, store = program.tasks
    assert set(gather_mask.dependencies) == {
        source_load.task_id, index_load.task_id,
    }
    assert "scratch" in gather_mask.metadata
    values = np.arange(32, dtype=np.int32) * 3
    indices = np.array([1, 2, 2, 5, 4, 6, 7, 8], dtype=np.uint32)
    simulator = FunctionalSimulator(program)
    simulator.write(source_load.metadata["src"], values)
    simulator.write(index_load.metadata["src"], indices)
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]), values[indices]
    )


def test_gather_mask_validates_pattern_indices_and_bounds() -> None:
    with pytest.raises(ProgramValidationError, match="fixed pattern"):
        build_kernel_program(
            _gather_mask_primfunc(pattern="P0110"), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="must use uint32"):
        build_kernel_program(
            _gather_mask_primfunc(custom=True, index_dtype="int32"),
            platform="A3",
        )

    program = build_kernel_program(
        _gather_mask_primfunc(custom=True), platform="A2"
    )
    source_load, index_load, _, _ = program.tasks
    simulator = FunctionalSimulator(program)
    simulator.write(source_load.metadata["src"], np.arange(32, dtype=np.float32))
    simulator.write(
        index_load.metadata["src"],
        np.array([0, 1, 2, 3, 4, 5, 6, 32], dtype=np.uint32),
    )
    with pytest.raises(ProgramValidationError, match="index out of range"):
        simulator.run()


@pytest.mark.parametrize(
    ("platform", "dtype", "shape"),
    [
        ("A2", "float16", (16, 32)),
        ("A2", "int8", (32, 64)),
        ("A3", "float32", (16, 32)),
        ("A3", "uint16", (32, 16)),
    ],
)
def test_transpose_executes_rank_two_tile(platform, dtype, shape) -> None:
    program = build_kernel_program(
        _transpose_primfunc(shape=shape, dtype=dtype), platform=platform
    )
    load, transpose, store = program.tasks
    assert transpose.operation == "transpose"
    assert transpose.metadata["src"].shape == shape
    assert transpose.metadata["dst"].shape == tuple(reversed(shape))
    assert transpose.dependencies == (load.task_id,)
    assert store.dependencies == (transpose.task_id,)

    values = np.arange(int(np.prod(shape)), dtype=dtype)
    simulator = FunctionalSimulator(program)
    simulator.write(load.metadata["src"], values)
    simulator.run()
    expected = values.reshape(shape).T.reshape(-1)
    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)


def test_transpose_validates_shape_alignment_and_dtype() -> None:
    with pytest.raises(ProgramValidationError, match="reverse the source shape"):
        build_kernel_program(
            _transpose_primfunc(shape=(16, 32), dst_shape=(16, 32)),
            platform="A2",
        )
    with pytest.raises(ProgramValidationError, match="32-byte aligned"):
        build_kernel_program(
            _transpose_primfunc(shape=(16, 20), dtype="float32"), platform="A3"
        )
    with pytest.raises(UnsupportedSimOpError, match="does not support dtype"):
        build_kernel_program(
            _transpose_primfunc(shape=(16, 16), dtype="bfloat16"), platform="A2"
        )


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_reinterpretcast_establishes_bidirectional_byte_alias(platform) -> None:
    program = build_kernel_program(_reinterpretcast_primfunc(), platform=platform)
    load, reinterpret, view_store, fill, after_store = program.tasks
    assert reinterpret.operation == "reinterpretcast"
    assert reinterpret.pipe is Pipe.SCALAR
    assert reinterpret.dependencies == (load.task_id,)
    assert reinterpret.task_id in view_store.dependencies
    assert view_store.task_id in fill.dependencies
    assert fill.task_id in after_store.dependencies

    values = np.array(
        [0x11223344, 0x55667788, 0x99AABBCC, 0xDDEEFF00], dtype=np.uint32
    )
    simulator = FunctionalSimulator(program)
    simulator.write(load.metadata["src"], values)
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(view_store.metadata["dst"]), values.view(np.uint16)
    )
    np.testing.assert_array_equal(
        simulator.read(after_store.metadata["dst"]),
        np.full((4,), 0x12341234, dtype=np.uint32),
    )


def test_reinterpretcast_validates_byte_size_and_cast_type() -> None:
    with pytest.raises(ProgramValidationError, match="byte sizes must match"):
        build_kernel_program(
            _reinterpretcast_primfunc(destination_extent=7), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="casttype must match"):
        build_kernel_program(
            _reinterpretcast_primfunc(cast_type="int16_t"), platform="A3"
        )


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_topk_executes_stable_interleaved_value_index_output(platform, dtype) -> None:
    program = build_kernel_program(_topk_primfunc(dtype=dtype), platform=platform)
    load, topk, store = program.tasks
    assert topk.operation == "topk"
    assert topk.pipe is Pipe.VECTOR
    assert topk.dependencies == (load.task_id,)
    assert store.dependencies == (topk.task_id,)
    assert topk.metadata["scratch"].scope is MemoryScope.UB

    values = np.array(
        [3, 8, 8, -2, 11, 1, 7] + [1000] * 25,
        dtype=np.dtype(dtype),
    )
    simulator = FunctionalSimulator(program)
    simulator.write(load.metadata["src"], values)
    simulator.run()
    expected = np.array([11, 4, 8, 1, 8, 2, 7, 6], dtype=np.dtype(dtype))
    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)


def test_topk_resolves_runtime_actual_num() -> None:
    actual_num = tvm.tir.Var("actual_num", "int32")
    program = build_kernel_program(
        _topk_primfunc(k=3, actual_num=actual_num, max_actual_num=12),
        platform="A3",
    )
    load, topk, store = program.tasks
    assert isinstance(topk.metadata["topk"]["actual_num"], AffineInt)
    values = np.array([2, 9, 4, 8, 5] + [100] * 27, dtype=np.float32)
    simulator = FunctionalSimulator(program, bindings={"actual_num": 5})
    simulator.write(load.metadata["src"], values)
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]),
        np.array([9, 1, 8, 3, 5, 4], dtype=np.float32),
    )
    with pytest.raises(ProgramValidationError, match=r"must be in \[3, 12\]"):
        FunctionalSimulator(program, bindings={"actual_num": 2})


def test_topk_validates_static_contract() -> None:
    with pytest.raises(ProgramValidationError, match="repeatTimes must be"):
        build_kernel_program(_topk_primfunc(repeat_times=2), platform="A2")
    with pytest.raises(ProgramValidationError, match=r"at least 2\*K"):
        build_kernel_program(
            _topk_primfunc(destination_extent=7), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="template dtype"):
        build_kernel_program(
            _topk_primfunc(template_dtype="half"), platform="A3"
        )


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_sort32_sorts_each_block_and_bitcasts_indices(platform, dtype) -> None:
    program = build_kernel_program(_sort32_primfunc(dtype=dtype), platform=platform)
    load_values, load_indices, sort32, store = program.tasks
    assert sort32.operation == "sort32"
    assert sort32.dependencies == tuple(
        sorted((load_values.task_id, load_indices.task_id))
    )
    assert store.dependencies == (sort32.task_id,)

    values = np.array(
        list(range(32)) + [7, 9, 9, 1] + list(range(28)),
        dtype=np.dtype(dtype),
    )
    indices = np.arange(100, 164, dtype=np.uint32)
    simulator = FunctionalSimulator(program)
    simulator.write(load_values.metadata["src"], values)
    simulator.write(load_indices.metadata["src"], indices)
    simulator.run()
    encoded = simulator.read(store.metadata["dst"])

    expected_orders = [
        np.argsort(-values[offset:offset + 32].astype(np.float64), kind="stable")
        for offset in (0, 32)
    ]
    expected_values = np.concatenate([
        values[offset:offset + 32][order]
        for offset, order in zip((0, 32), expected_orders)
    ])
    expected_indices = np.concatenate([
        indices[offset:offset + 32][order]
        for offset, order in zip((0, 32), expected_orders)
    ])
    multiplier = 4 if dtype == "float16" else 2
    records = encoded.reshape(64, multiplier)
    np.testing.assert_array_equal(records[:, 0], expected_values)
    if dtype == "float32":
        np.testing.assert_array_equal(
            np.ascontiguousarray(records[:, 1]).view(np.uint32),
            expected_indices,
        )
    else:
        np.testing.assert_array_equal(records[:, 1].view(np.uint16), 0)
        np.testing.assert_array_equal(
            np.ascontiguousarray(records[:, 2:4]).view(np.uint32).reshape(-1),
            expected_indices,
        )


def test_sort32_validates_repeat_and_destination_extent() -> None:
    with pytest.raises(ProgramValidationError, match="repeatTimes must be"):
        build_kernel_program(_sort32_primfunc(repeat_times=0), platform="A2")
    with pytest.raises(ProgramValidationError, match="destination extent"):
        build_kernel_program(
            _sort32_primfunc(output_elements=127), platform="A3"
        )


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_sort_executes_valid_prefix_with_numeric_indices(platform, dtype) -> None:
    program = build_kernel_program(_sort_primfunc(dtype=dtype), platform=platform)
    load, sort, store = program.tasks
    assert sort.operation == "sort"
    assert sort.dependencies == (load.task_id,)
    assert store.dependencies == (sort.task_id,)
    assert sort.metadata["scratch"].scope is MemoryScope.UB

    valid = np.array(
        [3, 8, 8, -2, 11, 1, 7] + list(range(30)), dtype=np.dtype(dtype)
    )
    values = np.concatenate(
        (valid, np.full(64 - valid.size, 1000, dtype=np.dtype(dtype)))
    )
    order = np.argsort(-valid.astype(np.float64), kind="stable")
    expected = np.empty(valid.size * 2, dtype=np.dtype(dtype))
    expected[0::2] = valid[order]
    expected[1::2] = order

    simulator = FunctionalSimulator(program)
    simulator.write(load.metadata["src"], values)
    simulator.run()
    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)


def test_sort_resolves_runtime_actual_num_and_repeat() -> None:
    actual_num = tvm.tir.Var("actual_num", "int32")
    program = build_kernel_program(
        _sort_primfunc(actual_num=actual_num), platform="A3"
    )
    load, sort, store = program.tasks
    assert isinstance(sort.metadata["sort"]["actual_num"], AffineInt)
    assert isinstance(sort.metadata["sort"]["repeat_times"], SymbolicInt)
    values = np.arange(64, dtype=np.float32)
    simulator = FunctionalSimulator(program, bindings={"actual_num": 33})
    simulator.write(load.metadata["src"], values)
    simulator.run()
    result = simulator.read(store.metadata["dst"])
    np.testing.assert_array_equal(result[0::2], np.arange(32, -1, -1))
    np.testing.assert_array_equal(result[1::2], np.arange(32, -1, -1))


def test_sort_validates_repeat_extent_and_template() -> None:
    with pytest.raises(ProgramValidationError, match="does not match actual_num"):
        build_kernel_program(_sort_primfunc(repeat_times=1), platform="A2")
    with pytest.raises(ProgramValidationError, match="twice the source extent"):
        build_kernel_program(
            _sort_primfunc(destination_extent=127), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="template dtype"):
        build_kernel_program(
            _sort_primfunc(template_dtype="half"), platform="A3"
        )


def _init_sort_buf_primfunc(
    *,
    num=128,
    capacity=None,
    dtype="float32",
    scope="shared.ub",
    template_dtype=None,
    drop_rsv=False,
):
    capacity = num if capacity is None else capacity
    workspace = tvm.tir.decl_buffer(
        (capacity,), dtype, name="workspace", scope=scope
    )
    dtype_name = template_dtype or {"float16": "half", "float32": "float"}[dtype]
    call_args = [
        f"InitSortBuf<{dtype_name}>",
        workspace.access_ptr("w"),
        num,
    ]
    if not drop_rsv:
        call_args.append(0)
    body = tvm.tir.Evaluate(tvm.tir.call_extern(
        "handle", "tl.ascend_init_sort_buf", *call_args,
    ))
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=[workspace],
    )
    return tvm.tir.PrimFunc([], tvm.tir.BlockRealize([], True, root))


def _init_sort_buf_pipeline_primfunc(*, num=128):
    output = tvm.tir.decl_buffer((num,), "float32", name="output", scope="global")
    workspace = tvm.tir.decl_buffer(
        (num,), "float32", name="workspace", scope="shared.ub"
    )
    body = tvm.tir.SeqStmt([
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_fill",
            workspace.access_ptr("w"), tvm.tir.FloatImm("float32", 1.5), num,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "tl.ascend_init_sort_buf", "InitSortBuf<float>",
            workspace.access_ptr("w"), num, 0,
        )),
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "handle", "copy_ub_to_gm",
            workspace.access_ptr("r"), output.access_ptr("w"), num,
        )),
    ])
    root = tvm.tir.Block(
        [], [], [], "root", body, alloc_buffers=[workspace],
    )
    return tvm.tir.PrimFunc(
        [],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={output.data: output},
    )


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_init_sort_buf_fills_alternating_value_index_lanes(platform) -> None:
    program = build_kernel_program(_init_sort_buf_primfunc(), platform=platform)
    (init,) = program.tasks
    assert (init.operation, init.lane, init.pipe) == (
        "init_sort_buf", Lane.VECTOR_0, Pipe.VECTOR,
    )
    assert init.metadata["init_sort_buf"]["element_count"] == 128
    assert init.metadata["init_sort_buf"]["covered_elements"] == 128

    simulator = FunctionalSimulator(program)
    simulator.run()
    values = simulator.read(init.metadata["dst"]).reshape(-1)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(
        values.view(np.int32)[0::2], np.int32(-8388608)
    )
    np.testing.assert_array_equal(values.view(np.int32)[1::2], np.int32(-1))
    np.testing.assert_array_equal(values[0::2], np.float32("-inf"))


@pytest.mark.parametrize("num", [130, 191])
def test_init_sort_buf_leaves_non_block_tail_unwritten(num) -> None:
    program = build_kernel_program(
        _init_sort_buf_primfunc(num=num), platform="A2"
    )
    (init,) = program.tasks
    covered = num - num % 64
    assert init.metadata["init_sort_buf"]["covered_elements"] == covered

    simulator = FunctionalSimulator(program)
    simulator.run()
    prefix = simulator.read(
        BufferRegion(
            init.metadata["dst"].buffer,
            MemoryScope.UB,
            (covered,),
            "float32",
        )
    ).reshape(-1).view(np.int32)
    assert prefix.size == covered
    np.testing.assert_array_equal(prefix[0::2], np.int32(-8388608))
    np.testing.assert_array_equal(prefix[1::2], np.int32(-1))
    with pytest.raises(UninitializedMemoryError, match="read-before-write"):
        simulator.read(
            BufferRegion(
                init.metadata["dst"].buffer,
                MemoryScope.UB,
                (num - covered,),
                "float32",
                byte_offset=covered * 4,
            )
        )


def test_init_sort_buf_skips_entirely_below_one_block() -> None:
    program = build_kernel_program(
        _init_sort_buf_primfunc(num=32), platform="A3"
    )
    (init,) = program.tasks
    assert init.metadata["init_sort_buf"]["covered_elements"] == 0
    simulator = FunctionalSimulator(program)
    simulator.run()
    with pytest.raises(UninitializedMemoryError, match="read-before-write"):
        simulator.read(init.metadata["dst"])


def test_init_sort_buf_orders_waw_and_raw_in_pipeline() -> None:
    program = build_kernel_program(
        _init_sort_buf_pipeline_primfunc(), platform="A2"
    )
    fill, init, store = program.tasks
    assert init.dependencies == (fill.task_id,)
    assert store.dependencies == (init.task_id,)

    simulator = FunctionalSimulator(program)
    simulator.run()
    values = simulator.read(store.metadata["dst"]).reshape(-1)
    np.testing.assert_array_equal(
        values.view(np.int32)[0::2], np.int32(-8388608)
    )
    np.testing.assert_array_equal(values.view(np.int32)[1::2], np.int32(-1))


def test_init_sort_buf_rejects_non_static_count_and_extent() -> None:
    num = tvm.tir.Var("num", "int32")
    with pytest.raises(UnsupportedSimOpError, match="static element count"):
        build_kernel_program(_init_sort_buf_primfunc(num=num), platform="A2")
    with pytest.raises(UnsupportedSimOpError, match="static destination extent"):
        build_kernel_program(
            _init_sort_buf_primfunc(capacity=tvm.tir.Var("cap", "int32")),
            platform="A2",
        )


def test_init_sort_buf_rejects_legacy_three_argument_abi() -> None:
    with pytest.raises(UnsupportedSimOpError, match="four-argument"):
        build_kernel_program(_init_sort_buf_primfunc(drop_rsv=True), platform="A2")


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_init_sort_buf_rejects_gm_scope_and_dtype_mismatch(platform) -> None:
    with pytest.raises(ProgramValidationError, match="UB destination"):
        build_kernel_program(
            _init_sort_buf_primfunc(scope="global"), platform=platform
        )
    with pytest.raises(ProgramValidationError, match="template dtype"):
        build_kernel_program(
            _init_sort_buf_primfunc(template_dtype="half"), platform=platform
        )
    with pytest.raises(UnsupportedSimOpError, match="float32 workspaces"):
        build_kernel_program(
            _init_sort_buf_primfunc(dtype="float16"), platform=platform
        )
    with pytest.raises(ProgramValidationError, match="exceeds destination extent"):
        build_kernel_program(
            _init_sort_buf_primfunc(num=96, capacity=64), platform=platform
        )


@pytest.mark.parametrize("num_ways", [2, 3, 4])
@pytest.mark.parametrize(
    ("platform", "with_scratch"), [("A2", False), ("A3", True)]
)
def test_merge_sort_executes_two_to_four_way_abi(
    num_ways, platform, with_scratch
) -> None:
    program = build_kernel_program(
        _merge_sort_primfunc(num_ways, with_scratch=with_scratch),
        platform=platform,
    )
    *loads, merge, store = program.tasks
    assert merge.operation == "merge_sort"
    assert merge.dependencies == tuple(sorted(task.task_id for task in loads))
    assert store.dependencies == (merge.task_id,)
    assert ("scratch" in merge.metadata) is with_scratch

    source_pairs = []
    simulator = FunctionalSimulator(program)
    for source_number, load in enumerate(loads):
        values = np.array(
            [12 - source_number, 8, 4 + source_number, -source_number],
            dtype=np.float32,
        )
        indices = np.arange(
            source_number * 10, source_number * 10 + 4, dtype=np.float32
        )
        pairs = np.empty(8, dtype=np.float32)
        pairs[0::2] = values
        pairs[1::2] = indices
        source_pairs.append(pairs.reshape(4, 2))
        simulator.write(load.metadata["src"], pairs)
    simulator.run()

    concatenated = np.concatenate(source_pairs)
    order = np.argsort(-concatenated[:, 0].astype(np.float64), kind="stable")
    expected = concatenated[order].reshape(-1)
    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("num_ways", [2, 4])
def test_merge_sort_executes_float16_eight_byte_records(
    platform, num_ways
) -> None:
    program = build_kernel_program(
        _merge_sort_primfunc(num_ways, dtype="float16"), platform=platform
    )
    *loads, merge, store = program.tasks
    assert merge.metadata["merge_sort"]["record_width"] == 4
    assert all(region.shape == (16,) for region in merge.metadata["src_regions"])

    source_records = []
    simulator = FunctionalSimulator(program)
    for source_number, load in enumerate(loads):
        records = np.empty((4, 4), dtype=np.float16)
        records[:, 0] = np.array(
            [12 - source_number, 8, 4 + source_number, -source_number],
            dtype=np.float16,
        )
        records[:, 1] = np.float16(100 + source_number)
        encoded_indices = np.arange(
            source_number * 10, source_number * 10 + 4, dtype=np.int32
        ).view(np.float16).reshape(4, 2)
        records[:, 2:4] = encoded_indices
        source_records.append(records)
        simulator.write(load.metadata["src"], records.reshape(-1))
    simulator.run()

    concatenated = np.concatenate(source_records)
    order = np.argsort(-concatenated[:, 0].astype(np.float64), kind="stable")
    expected = concatenated[order]
    actual = simulator.read(store.metadata["dst"]).reshape(-1, 4)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        actual[:, 2:4].reshape(-1).view(np.int32),
        expected[:, 2:4].reshape(-1).view(np.int32),
    )


def test_merge_sort_rejects_legacy_float16_pair_block_length() -> None:
    with pytest.raises(ProgramValidationError, match="source extent"):
        build_kernel_program(
            _merge_sort_primfunc(
                dtype="float16", block_length=4, reported_block_length=8,
                destination_extent=64,
            ),
            platform="A2",
        )


def test_merge_sort_rejects_bad_extent_and_unsorted_input() -> None:
    with pytest.raises(ProgramValidationError, match="destination extent"):
        build_kernel_program(
            _merge_sort_primfunc(destination_extent=15), platform="A2"
        )

    program = build_kernel_program(_merge_sort_primfunc(), platform="A3")
    first, second, _, _ = program.tasks
    simulator = FunctionalSimulator(program)
    simulator.write(
        first.metadata["src"],
        np.array([1, 0, 3, 1, 2, 2, 0, 3], dtype=np.float32),
    )
    simulator.write(
        second.metadata["src"],
        np.array([4, 4, 3, 5, 2, 6, 1, 7], dtype=np.float32),
    )
    with pytest.raises(ProgramValidationError, match="source0 is not descending"):
        simulator.run()


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_atomic_add_ub_accumulates_and_preserves_gm_stride(platform, dtype) -> None:
    program = build_kernel_program(
        _atomic_add_ub_primfunc(dtype=dtype), platform=platform
    )
    first_fill, second_fill, first_atomic, second_atomic = program.tasks
    assert first_atomic.operation == "atomic_add_ub_to_gm"
    assert first_atomic.pipe is Pipe.MTE3
    assert first_atomic.dependencies == (first_fill.task_id,)
    assert second_atomic.dependencies == tuple(
        sorted((second_fill.task_id, first_atomic.task_id))
    )
    assert first_atomic.metadata["accumulator"] == first_atomic.metadata["dst"]

    destination = first_atomic.metadata["dst"]
    rows, cols = destination.shape
    stride = first_atomic.metadata["atomic"]["stride"]
    whole_output = BufferRegion(
        destination.buffer, MemoryScope.GM, (rows, stride), destination.dtype
    )
    simulator = FunctionalSimulator(program)
    simulator.write(
        whole_output, np.zeros((rows, stride), dtype=np.dtype(dtype))
    )
    simulator.run()
    result = simulator.read(whole_output)
    np.testing.assert_array_equal(result[:, :cols], 3)
    np.testing.assert_array_equal(result[:, cols:], 0)


def test_atomic_add_ub_requires_initialized_destination_and_aligned_rows() -> None:
    program = build_kernel_program(_atomic_add_ub_primfunc(), platform="A2")
    with pytest.raises(UninitializedMemoryError, match="read-before-write"):
        FunctionalSimulator(program).run()
    with pytest.raises(ProgramValidationError, match="32-byte aligned"):
        build_kernel_program(
            _atomic_add_ub_primfunc(cols=7), platform="A3"
        )


def test_atomic_add_ub_serializes_shared_gm_updates_across_cores() -> None:
    program = build_kernel_program(
        _atomic_add_ub_multicore_primfunc(), platform="A3"
    )
    assert tuple(core.core_id for core in program.cores) == (0, 1)
    first_atomic = program.cores[0].tasks[1]
    second_atomic = program.cores[1].tasks[1]
    assert first_atomic.task_id in second_atomic.dependencies

    output = BufferRegion("output", MemoryScope.GM, (8,), "float32")
    simulator = FunctionalSimulator(program)
    simulator.write(output, np.zeros(8, dtype=np.float32))
    result = simulator.run()
    np.testing.assert_array_equal(simulator.read(output), 2)
    atomic_records = [
        record for record in result.schedule.records
        if record.operation == "atomic_add_ub_to_gm"
    ]
    assert atomic_records[1].start_cycle >= atomic_records[0].end_cycle


@pytest.mark.parametrize("operation", ["leaky_relu", "axpy"])
def test_real_tir_scalar_activation_builds_operands_and_executes(operation) -> None:
    program = build_kernel_program(_scalar_vector_primfunc(operation), platform="A3")
    scalar_task = next(task for task in program.tasks if task.operation == operation)
    store = program.tasks[-1]
    assert scalar_task.metadata["scalar"] == 0.25
    assert store.dependencies == (scalar_task.task_id,)

    simulator = FunctionalSimulator(program)
    source_values = np.array([-4, -1, 0, 2, 8], dtype=np.float32)
    initial_values = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    simulator.write(
        BufferRegion("source", MemoryScope.GM, (5,), "float32"), source_values
    )
    if operation == "axpy":
        simulator.write(
            BufferRegion("initial", MemoryScope.GM, (5,), "float32"),
            initial_values,
        )
        assert len(scalar_task.dependencies) == 2
        expected = initial_values + 0.25 * source_values
    else:
        expected = np.where(source_values >= 0, source_values, 0.25 * source_values)

    simulator.run()

    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("sub_experiment", lambda source, right: source - right),
        ("abs_experiment", lambda source, right: np.abs(source)),
        ("mins_experiment", lambda source, right: np.minimum(source, 1.5)),
    ],
)
def test_elementwise_experiment_intrinsics_execute(
    platform, dtype, operation, expected
) -> None:
    program = build_kernel_program(
        _elementwise_experiment_primfunc(operation, dtype=dtype),
        platform=platform,
    )
    vector = next(task for task in program.tasks if task.operation == operation)
    store = program.tasks[-1]
    expected_dependencies = 2 if operation == "sub_experiment" else 1
    assert len(vector.dependencies) == expected_dependencies
    assert store.dependencies == (vector.task_id,)

    source = np.array([-4, -2, -0.5, 0, 1, 2, 4, 8], dtype=dtype)
    right = np.array([1, -1, 2, 3, -2, 0.5, 4, 9], dtype=dtype)
    simulator = FunctionalSimulator(program)
    simulator.write(program.tasks[0].metadata["src"], source)
    if operation == "sub_experiment":
        simulator.write(program.tasks[1].metadata["src"], right)
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]), expected(source, right)
    )


def test_elementwise_experiment_rejects_negative_count() -> None:
    with pytest.raises(ProgramValidationError, match="must not be negative"):
        build_kernel_program(
            _elementwise_experiment_primfunc("abs_experiment", count=-1),
            platform="A2",
        )


@pytest.mark.parametrize(
    ("platform", "dtype", "with_scratch"),
    [
        ("A2", "float16", False),
        ("A2", "float32", True),
        ("A3", "float16", True),
        ("A3", "float32", False),
    ],
)
def test_reduce_sum_experiment_executes_final_abi(
    platform, dtype, with_scratch
) -> None:
    program = build_kernel_program(
        _reduce_sum_experiment_primfunc(
            dtype=dtype, with_scratch=with_scratch
        ),
        platform=platform,
    )
    reduction = next(
        task for task in program.tasks
        if task.operation == "reducesum_experiment"
    )
    load, store = program.tasks[0], program.tasks[-1]
    assert reduction.metadata["dst"].shape == (1,)
    assert reduction.metadata["src"].shape == (8,)
    assert reduction.dependencies == (load.task_id,)
    assert store.dependencies == (reduction.task_id,)
    assert ("scratch" in reduction.metadata) is with_scratch

    values = np.array([-4, -2, -0.5, 0, 1, 2, 4, 8], dtype=dtype)
    simulator = FunctionalSimulator(program)
    simulator.write(load.metadata["src"], values)
    simulator.run()
    expected = np.asarray([np.sum(values, dtype=values.dtype)], dtype=dtype)
    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)


def test_reduce_sum_experiment_validates_count_and_scratch_dtype() -> None:
    with pytest.raises(ProgramValidationError, match="count must not be negative"):
        build_kernel_program(
            _reduce_sum_experiment_primfunc(count=-1), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="scratch must match"):
        build_kernel_program(
            _reduce_sum_experiment_primfunc(scratch_dtype="float16"),
            platform="A3",
        )


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_sum_experiment_ignores_padded_row_tail(platform, dtype) -> None:
    inner = 16 if dtype == "float16" else 8
    program = build_kernel_program(
        _sum_experiment_primfunc(dtype=dtype, inner=inner), platform=platform
    )
    load, row_sum, store = program.tasks
    assert row_sum.operation == "sum_experiment"
    assert row_sum.metadata["src"].shape == (3, inner)
    assert row_sum.metadata["dst"].shape == (3,)
    assert row_sum.dependencies == (load.task_id,)
    assert store.dependencies == (row_sum.task_id,)

    values = np.arange(3 * inner, dtype=np.dtype(dtype)).reshape(3, inner)
    values[:, 5:] = 1000
    simulator = FunctionalSimulator(program)
    simulator.write(load.metadata["src"], values.reshape(-1))
    simulator.run()
    expected = np.sum(values[:, :5], axis=1, dtype=values.dtype)
    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)


def test_sum_experiment_validates_width_and_template() -> None:
    with pytest.raises(ProgramValidationError, match="must not exceed"):
        build_kernel_program(
            _sum_experiment_primfunc(valid=9), platform="A2"
        )
    with pytest.raises(ProgramValidationError, match="template dtype"):
        build_kernel_program(
            _sum_experiment_primfunc(template_dtype="half"), platform="A3"
        )
    with pytest.raises(ProgramValidationError, match="32-byte aligned"):
        build_kernel_program(
            _sum_experiment_primfunc(dtype="float16", inner=8, valid=5),
            platform="A2",
        )


@pytest.mark.parametrize(
    ("operation", "with_scratch", "in_place", "expected"),
    [
        ("sigmoid", False, False, lambda x: np.exp(-np.logaddexp(0, -x))),
        ("sin", True, False, np.sin),
        ("cos", False, False, np.cos),
        (
            "silu",
            True,
            True,
            lambda x: x * np.exp(-np.logaddexp(0, -x)),
        ),
    ],
)
def test_real_tir_scratch_unary_forms_execute(
    operation, with_scratch, in_place, expected
) -> None:
    program = build_kernel_program(
        _scratch_unary_primfunc(
            operation, with_scratch=with_scratch, in_place=in_place
        ),
        platform="A2",
    )
    unary = next(task for task in program.tasks if task.operation == operation)
    load, store = program.tasks[0], program.tasks[-1]
    assert unary.dependencies == (load.task_id,)
    assert store.dependencies == (unary.task_id,)
    assert ("scratch" in unary.metadata) is with_scratch

    simulator = FunctionalSimulator(program)
    values = np.array([-4, -1, 0, 2, 8], dtype=np.float32)
    simulator.write(
        BufferRegion("source", MemoryScope.GM, (5,), "float32"), values
    )
    simulator.run()

    np.testing.assert_allclose(
        simulator.read(store.metadata["dst"]), expected(values), rtol=1e-6
    )


@pytest.mark.parametrize(
    ("platform", "dtype", "with_scratch"),
    [
        ("A2", "float16", True),
        ("A2", "float32", False),
        ("A3", "float16", False),
        ("A3", "float32", True),
    ],
)
def test_round_executes_ties_away_from_zero(platform, dtype, with_scratch) -> None:
    program = build_kernel_program(
        _scratch_unary_primfunc(
            "round", dtype=dtype, with_scratch=with_scratch
        ),
        platform=platform,
    )
    load, operation, store = program.tasks
    assert ("scratch" in operation.metadata) is with_scratch
    values = np.array([-2.5, -1.5, -0.5, 0.5, 2.5], dtype=dtype)
    simulator = FunctionalSimulator(program)
    simulator.write(load.metadata["src"], values)
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]),
        np.array([-3, -2, -1, 1, 3], dtype=dtype),
    )


def test_round_rejects_non_floating_dtype() -> None:
    with pytest.raises(UnsupportedSimOpError, match="does not support dtype"):
        build_kernel_program(
            _scratch_unary_primfunc("round", dtype="int16"), platform="A2"
        )


@pytest.mark.parametrize(
    ("operation", "with_scratch", "expected"),
    [
        ("pow", True, lambda x, y: np.power(x, y)),
        ("clamp_max", False, lambda x, _: np.minimum(x, 1.5)),
        ("clamp_min", True, lambda x, _: np.maximum(x, 1.5)),
        ("clamp", True, lambda x, _: np.clip(x, -1.0, 2.0)),
    ],
)
def test_real_tir_pow_and_clamp_forms_execute(operation, with_scratch, expected) -> None:
    program = build_kernel_program(
        _pow_clamp_primfunc(operation, with_scratch=with_scratch), platform="A3"
    )
    vector = next(task for task in program.tasks if task.operation == operation)
    store = program.tasks[-1]
    assert ("scratch" in vector.metadata) is with_scratch
    assert store.dependencies == (vector.task_id,)

    simulator = FunctionalSimulator(program)
    source_values = np.array([0.25, 1, 2, 4, 8], dtype=np.float32)
    exponent_values = np.array([2, 3, 0.5, 1, -1], dtype=np.float32)
    simulator.write(
        BufferRegion("source", MemoryScope.GM, (5,), "float32"), source_values
    )
    if operation == "pow":
        simulator.write(
            BufferRegion("exponent", MemoryScope.GM, (5,), "float32"),
            exponent_values,
        )
        assert len(vector.dependencies) == 2
    simulator.run()

    np.testing.assert_allclose(
        simulator.read(store.metadata["dst"]),
        expected(source_values, exponent_values),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("source_shape", "with_scratch"),
    [((1, 3), False), ((2, 1), True)],
)
def test_real_tir_broadcast_executes_both_axes(source_shape, with_scratch) -> None:
    program = build_kernel_program(
        _broadcast_primfunc(source_shape, with_scratch=with_scratch), platform="A2"
    )
    load, broadcast, store = program.tasks
    assert broadcast.operation == "broadcast"
    assert broadcast.metadata["src"].shape == source_shape
    assert broadcast.metadata["dst"].shape == (2, 3)
    assert broadcast.dependencies == (load.task_id,)
    assert store.dependencies == (broadcast.task_id,)
    assert ("scratch" in broadcast.metadata) is with_scratch

    simulator = FunctionalSimulator(program)
    values = np.arange(1, int(np.prod(source_shape)) + 1, dtype=np.float32)
    simulator.write(load.metadata["src"], values)
    simulator.run()

    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]),
        np.broadcast_to(values.reshape(source_shape), (2, 3)).reshape(-1),
    )


@pytest.mark.parametrize("mode", ["EQ", "NE", "GT", "GE", "LT", "LE"])
@pytest.mark.parametrize("scalar", [False, True])
def test_real_tir_compare_packs_little_endian_predicate_bits(mode, scalar) -> None:
    program = build_kernel_program(
        _compare_primfunc(mode, scalar=scalar), platform="A3"
    )
    compare = next(
        task for task in program.tasks
        if task.operation in {"compare", "compare_scalar"}
    )
    store = program.tasks[-1]
    assert compare.metadata["compare_mode"] == mode
    assert store.dependencies == (compare.task_id,)

    simulator = FunctionalSimulator(program)
    left = np.array([-2, -1, 0, 1, 2, 3, 4, 5], dtype=np.float32)
    right = np.array([-2, 0, -1, 1, 3, 2, 5, 4], dtype=np.float32)
    simulator.write(
        BufferRegion("left", MemoryScope.GM, (8,), "float32"), left
    )
    if not scalar:
        simulator.write(
            BufferRegion("right", MemoryScope.GM, (8,), "float32"), right
        )
    simulator.run()

    rhs = 0.0 if scalar else right
    predicates = {
        "EQ": np.equal,
        "NE": np.not_equal,
        "GT": np.greater,
        "GE": np.greater_equal,
        "LT": np.less,
        "LE": np.less_equal,
    }[mode](left, rhs)
    expected = np.packbits(predicates, bitorder="little")
    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)


@pytest.mark.parametrize("mode", ["EQ", "NE", "GT", "GE", "LT", "LE"])
def test_compare_scalar_reads_indexed_buffer_value(mode) -> None:
    program = build_kernel_program(
        _compare_scalar_buffer_primfunc(mode), platform="A2"
    )
    left_load, scalar_load, compare, store = program.tasks
    assert compare.operation == "compare_scalar"
    assert compare.metadata["scalar_src"].shape == (1,)
    assert compare.metadata["scalar_src"].byte_offset == 4
    assert compare.dependencies == (left_load.task_id, scalar_load.task_id)
    assert store.dependencies == (compare.task_id,)

    simulator = FunctionalSimulator(program)
    left = np.array([-2, -1, 0, 1, 2, 3, 4, 5], dtype=np.float32)
    scalars = np.array([-100, 1], dtype=np.float32)
    simulator.write(left_load.metadata["src"], left)
    simulator.write(scalar_load.metadata["src"], scalars)
    simulator.run()
    predicates = {
        "EQ": np.equal,
        "NE": np.not_equal,
        "GT": np.greater,
        "GE": np.greater_equal,
        "LT": np.less,
        "LE": np.less_equal,
    }[mode](left, scalars[1])
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]),
        np.packbits(predicates, bitorder="little"),
    )


@pytest.mark.parametrize(
    ("scalar_select", "with_scratch", "select_mode"),
    [
        (False, False, None),
        (False, True, "VSEL_CMPMASK_SPR"),
        (True, True, None),
    ],
)
def test_real_tir_select_reads_packed_compare_mask(
    scalar_select, with_scratch, select_mode
) -> None:
    program = build_kernel_program(
        _compare_select_primfunc(
            scalar_select=scalar_select, with_scratch=with_scratch,
            select_mode=select_mode,
        ),
        platform="A3",
    )
    left_load, right_load, compare, select, store = program.tasks
    assert select.operation == "select"
    assert select.metadata["source_type"] == (1 if scalar_select else 2)
    if select_mode is not None:
        assert select.metadata["select_mode"] == select_mode
    assert compare.task_id in select.dependencies
    assert left_load.task_id in select.dependencies
    if not scalar_select:
        assert right_load.task_id in select.dependencies
    assert store.dependencies == (select.task_id,)
    assert ("scratch" in select.metadata) is with_scratch

    simulator = FunctionalSimulator(program)
    left = np.array([-2, 4, 0, 7, 2, 3, 9, 5], dtype=np.float32)
    right = np.array([-1, 3, 1, 6, 3, 2, 8, 6], dtype=np.float32)
    simulator.write(left_load.metadata["src"], left)
    simulator.write(right_load.metadata["src"], right)
    simulator.run()

    fallback = 10.0 if scalar_select else right
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]),
        np.where(left < right, left, fallback),
    )


@pytest.mark.parametrize(
    "mode", ["VSEL_CMPMASK_SPR", "VSEL_TENSOR_TENSOR_MODE"]
)
@pytest.mark.parametrize("with_scratch", [False, True])
def test_select_reads_indexed_buffer_scalar(mode, with_scratch) -> None:
    program = build_kernel_program(
        _compare_select_primfunc(
            buffer_select=True, with_scratch=with_scratch, select_mode=mode,
        ),
        platform="A2",
    )
    left_load, scalar_load, compare, select, store = program.tasks
    assert select.metadata["source_type"] == 0
    assert select.metadata["select_mode"] == mode
    assert select.metadata["scalar_src"].shape == (1,)
    assert select.metadata["scalar_src"].byte_offset == 4
    assert set(select.dependencies) == {
        left_load.task_id, scalar_load.task_id, compare.task_id,
    }
    assert ("scratch" in select.metadata) is with_scratch
    assert store.dependencies == (select.task_id,)

    simulator = FunctionalSimulator(program)
    left = np.array([-2, 4, 0, 7, 2, 3, 9, 5], dtype=np.float32)
    right = np.array([-1, 3, 1, 6, 3, 2, 8, 6], dtype=np.float32)
    simulator.write(left_load.metadata["src"], left)
    simulator.write(scalar_load.metadata["src"], right)
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]),
        np.where(left < right, left, right[1]),
    )


@pytest.mark.parametrize("scalar", [False, True])
def test_real_tail_compare_packs_each_valid_row_independently(scalar) -> None:
    program = build_kernel_program(_tail_compare_primfunc(scalar=scalar), platform="A2")
    compare = program.tasks[-1]
    assert compare.operation == (
        "tail_compare_scalar" if scalar else "tail_compare"
    )
    assert compare.metadata["valid_rows"] == 2
    assert compare.metadata["valid_cols"] == 9
    assert compare.metadata["physical_cols"] == 10
    assert compare.metadata["storage_cols"] == 2
    assert set(compare.dependencies) == {
        task.task_id for task in program.tasks[:-1]
    }

    simulator = FunctionalSimulator(program)
    left = np.array([
        [-2, -1, 0, 1, 2, 3, 4, 5, 6, 99],
        [6, 5, 4, 3, 2, 1, 0, -1, -2, 99],
    ], dtype=np.float32)
    right = np.array([
        [-1, -1, 1, 0, 3, 2, 5, 4, 7, 99],
        [7, 4, 5, 2, 3, 0, 1, -2, -1, 99],
    ], dtype=np.float32)
    simulator.write(program.tasks[0].metadata["src"], left[:, :9])
    if not scalar:
        simulator.write(program.tasks[1].metadata["src"], right[:, :9])
    simulator.run()

    rhs = 0.0 if scalar else right[:, :9]
    expected = np.packbits(left[:, :9] >= rhs, axis=1, bitorder="little")
    np.testing.assert_array_equal(simulator.read(compare.metadata["dst"]), expected)


def test_tail_compare_rejects_narrow_packed_storage() -> None:
    with pytest.raises(
        ProgramValidationError, match="packed storage width is too small"
    ):
        build_kernel_program(
            _tail_compare_primfunc(storage_cols=1), platform="A3"
        )


@pytest.mark.parametrize(
    ("scalar_select", "with_scratch"), [(False, False), (True, True)]
)
def test_real_tail_select_unpacks_each_mask_row(
    scalar_select, with_scratch
) -> None:
    program = build_kernel_program(
        _tail_compare_select_primfunc(
            scalar_select=scalar_select, with_scratch=with_scratch
        ),
        platform="A3",
    )
    left_load, right_load, compare, select, store = program.tasks
    assert select.operation == "tail_select"
    assert select.metadata["select_kind"] == (
        "Scalar" if scalar_select else "Tensor"
    )
    assert compare.task_id in select.dependencies
    assert left_load.task_id in select.dependencies
    if not scalar_select:
        assert right_load.task_id in select.dependencies
    assert ("scratch" in select.metadata) is with_scratch
    assert store.dependencies == (select.task_id,)

    simulator = FunctionalSimulator(program)
    left = np.array([
        [-2, 4, 0, 7, 2, 3, 9, 5, -4],
        [6, 5, 4, 3, 2, 1, 0, -1, -2],
    ], dtype=np.float32)
    right = np.array([
        [-1, 3, 1, 6, 3, 2, 8, 6, -3],
        [7, 4, 5, 2, 3, 0, 1, -2, -1],
    ], dtype=np.float32)
    simulator.write(left_load.metadata["src"], left)
    simulator.write(right_load.metadata["src"], right)
    simulator.run()

    fallback = 10.0 if scalar_select else right
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]),
        np.where(left < right, left, fallback),
    )


def test_tail_select_rejects_kind_and_source_type_mismatch() -> None:
    with pytest.raises(UnsupportedSimOpError, match="kind/type/mode"):
        build_kernel_program(
            _tail_compare_select_primfunc(
                scalar_select=False, select_kind="Scalar"
            ),
            platform="A2",
        )


@pytest.mark.parametrize(
    ("axis", "with_scratch"), [(0, False), (1, True)]
)
def test_real_tail_broadcast_executes_valid_rectangle(axis, with_scratch) -> None:
    program = build_kernel_program(
        _tail_broadcast_primfunc(axis, with_scratch=with_scratch), platform="A2"
    )
    broadcast, store = program.tasks
    assert broadcast.operation == "tail_broadcast"
    assert broadcast.metadata["broadcast"]["axis"] == axis
    assert ("scratch" in broadcast.metadata) is with_scratch
    assert store.dependencies == (broadcast.task_id,)

    simulator = FunctionalSimulator(program)
    source = (
        np.array([[1, 2, 3, 4]], dtype=np.float32)
        if axis == 0 else np.array([[2], [5]], dtype=np.float32)
    )
    simulator.write(broadcast.metadata["src"], source)
    simulator.run()

    expected = np.broadcast_to(source, broadcast.metadata["dst"].shape)
    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)


def test_unconfirmed_cast_round_mode_fails_closed_during_execution() -> None:
    program = build_kernel_program(_cast_primfunc("CAST_ODD"), platform="A3")
    simulator = FunctionalSimulator(program)
    simulator.write(
        BufferRegion("source", MemoryScope.GM, (8,), "float32"),
        np.arange(8, dtype=np.float32),
    )

    with pytest.raises(UnsupportedSimOpError, match="CAST_ODD"):
        simulator.run()


def test_real_tir_cast_none_converts_float32_to_float16() -> None:
    program = build_kernel_program(
        _cast_primfunc("CAST_NONE", destination_dtype="float16"), platform="A3"
    )
    simulator = FunctionalSimulator(program)
    values = np.array([0.1, -1.25, 3.14, 1024.5, 0, 99, 99, 99], dtype=np.float32)
    simulator.write(
        BufferRegion("source", MemoryScope.GM, (8,), "float32"), values
    )

    simulator.run()

    np.testing.assert_array_equal(
        simulator.read(program.tasks[-1].metadata["dst"]),
        values[:5].astype(np.float16),
    )


def test_gm_to_ub_copy_fills_physical_tail_with_literal_pad_value() -> None:
    program = build_kernel_program(_padded_copy_primfunc(), platform="A2")
    load, store = program.tasks
    assert load.metadata["pad_dst"].shape == (3, 8)
    assert load.metadata["copy"]["pad_value"] == -3.5
    assert store.dependencies == (load.task_id,)

    simulator = FunctionalSimulator(program)
    values = np.arange(10, dtype=np.float32).reshape(2, 5)
    simulator.write(
        BufferRegion("source", MemoryScope.GM, (2, 5), "float32"), values
    )
    simulator.run()

    expected = np.full((3, 8), -3.5, dtype=np.float32)
    expected[:2, :5] = values
    np.testing.assert_array_equal(
        simulator.read(BufferRegion("output", MemoryScope.GM, (3, 8), "float32")),
        expected,
    )


def test_non_affine_runtime_region_executes_min_mod_div_and_select() -> None:
    program = build_kernel_program(_non_affine_copy_primfunc(), platform="A3")
    task = program.tasks[0]
    assert isinstance(task.metadata["src"].byte_offset, SymbolicInt)
    assert isinstance(task.metadata["src"].shape[1], SymbolicInt)

    simulator = FunctionalSimulator(program, bindings={"extent": 11})
    values = np.arange(32, dtype=np.float32)
    simulator.write(
        BufferRegion("source", MemoryScope.GM, (32,), "float32"), values
    )
    simulator.run()

    np.testing.assert_array_equal(
        simulator.read(task.metadata["dst"]),
        values[4:12].reshape(1, 8),
    )


def test_real_tir_dynamic_parameter_buffer_allocates_from_binding() -> None:
    program = build_kernel_program(_dynamic_allocation_primfunc(), platform="A2")
    source_spec = next(buffer for buffer in program.buffers if buffer.name == "source")
    assert source_spec.shape == (AffineInt.variable("extent"),)

    simulator = FunctionalSimulator(program, bindings={"extent": 6})
    values = np.arange(6, dtype=np.float32)
    simulator.write(
        BufferRegion(
            "source",
            MemoryScope.GM,
            (AffineInt.variable("extent"),),
            "float32",
        ),
        values,
    )
    simulator.run()

    np.testing.assert_array_equal(
        simulator.read(
            BufferRegion("destination", MemoryScope.UB, (8,), "float32")
        ),
        np.pad(values, (0, 2)),
    )


def test_planned_physical_alias_drives_dependencies_and_shared_bytes() -> None:
    program = build_kernel_program(_planned_alias_primfunc(), platform="A3")
    specs = {buffer.name: buffer for buffer in program.buffers}
    assert specs["ub_x"].address == 0
    assert specs["ub_y"].address == 8
    assert specs["ub_x"].size_bytes == specs["ub_y"].size_bytes == 16
    load_x, load_y, store = program.tasks
    assert load_y.dependencies == (load_x.task_id,)
    assert store.dependencies == (load_y.task_id,)

    simulator = FunctionalSimulator(program)
    x = np.arange(4, dtype=np.float32)
    y = np.arange(10, 14, dtype=np.float32)
    simulator.write(BufferRegion("x", MemoryScope.GM, (4,), "float32"), x)
    simulator.write(BufferRegion("y", MemoryScope.GM, (4,), "float32"), y)
    simulator.run()

    np.testing.assert_array_equal(
        simulator.read(BufferRegion("output", MemoryScope.GM, (4,), "float32")),
        np.concatenate((x[:2], y[:2])),
    )


@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize(
    ("kind", "reducer"),
    [
        ("reduce_sum", np.sum),
        ("reduce_max", np.max),
        ("reduce_min", np.min),
    ],
)
def test_real_tir_reduce_executes_both_axes(kind, reducer, axis) -> None:
    program = build_kernel_program(_reduce_primfunc(kind, axis), platform="A3")
    load, reduce, store = program.tasks
    assert reduce.operation == "reduce"
    assert reduce.metadata["reduce_kind"] == kind
    assert reduce.metadata["reduce_axis"] == axis
    assert reduce.dependencies == (load.task_id,)
    assert store.dependencies == (reduce.task_id,)
    assert ("scratch" in reduce.metadata) is (axis == -1)

    simulator = FunctionalSimulator(program)
    values = np.arange(15, dtype=np.float32).reshape(3, 5) - 4
    simulator.write(load.metadata["src"], values.reshape(-1))
    simulator.run()

    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]),
        reducer(values, axis=0 if axis == 0 else 1),
    )


@pytest.mark.parametrize(
    ("kind", "reducer"),
    [("sum", np.sum), ("max", np.max), ("min", np.min)],
)
def test_narrow_row_reduce_uses_physical_stride(kind, reducer) -> None:
    program = build_kernel_program(_narrow_reduce_primfunc(kind), platform="A2")
    load, reduction, store = program.tasks
    assert reduction.operation == "reduce"
    assert reduction.metadata["physical_cols"] == 8
    assert reduction.metadata["src"].shape == (3, 5)
    assert reduction.metadata["src"].byte_offset == 8
    assert reduction.metadata["src"].strides_bytes == (32, 4)
    assert reduction.dependencies == (load.task_id,)
    assert store.dependencies == (reduction.task_id,)

    values = np.arange(24, dtype=np.float32).reshape(3, 8)
    poison = {"sum": 100, "max": 1000, "min": -1000}[kind]
    values[:, :2] = poison
    values[:, 7:] = poison
    simulator = FunctionalSimulator(program)
    simulator.write(load.metadata["src"], values.reshape(-1))
    simulator.run()
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]), reducer(values[:, 2:7], axis=1)
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"logical_cols": 9}, "logical width must fit"),
        ({"physical_cols": 7}, "32-byte aligned"),
        ({"logical_cols": 65, "physical_cols": 72}, "256-byte vector repeat"),
        ({"axis": 0}, "only row reduction"),
        ({"clear": False}, "clear=false"),
    ],
)
def test_narrow_row_reduce_rejects_unsupported_contract(kwargs, message) -> None:
    with pytest.raises((ProgramValidationError, UnsupportedSimOpError), match=message):
        build_kernel_program(_narrow_reduce_primfunc("sum", **kwargs), platform="A3")


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(
    ("kind", "reducer"),
    [("sum", np.sum), ("max", np.max), ("min", np.min)],
)
def test_block_reduce_executes_masked_strided_repeats(dtype, kind, reducer) -> None:
    mask = 37
    program = build_kernel_program(
        _block_reduce_primfunc(kind, dtype=dtype, mask=mask), platform="A2"
    )
    source_load, initial_load, reduction, store = program.tasks
    itemsize = np.dtype(dtype).itemsize
    elements_per_block = 32 // itemsize
    active_blocks = (mask + elements_per_block - 1) // elements_per_block
    assert reduction.operation == f"block_reduce_{kind}"
    assert reduction.metadata["src"].shape == (2, active_blocks, elements_per_block)
    assert reduction.metadata["src"].strides_bytes == (256, 32, itemsize)
    assert reduction.metadata["dst"].shape == (2, active_blocks)
    assert reduction.metadata["dst"].strides_bytes == (8 * itemsize, itemsize)
    assert set(reduction.dependencies) == {
        source_load.task_id, initial_load.task_id,
    }
    assert store.dependencies == (reduction.task_id,)

    values = np.linspace(-8, 8, 256, dtype=np.float32).astype(dtype)
    initial = np.full(16, -99, dtype=dtype)
    simulator = FunctionalSimulator(program)
    simulator.write(source_load.metadata["src"], values)
    simulator.write(initial_load.metadata["src"], initial)
    simulator.run()

    expected = initial.copy()
    for repeat in range(2):
        for block_index in range(active_blocks):
            active = min(
                elements_per_block, mask - block_index * elements_per_block
            )
            start = repeat * 8 * elements_per_block + block_index * elements_per_block
            block = values[start:start + active].astype(np.float32)
            expected[repeat * 8 + block_index] = reducer(block)
    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)


def test_block_reduce_rejects_invalid_mask_and_dtype() -> None:
    with pytest.raises(ProgramValidationError, match="mask must be in"):
        build_kernel_program(
            _block_reduce_primfunc("sum", mask=129), platform="A3"
        )
    with pytest.raises(UnsupportedSimOpError, match="only float16/float32"):
        build_kernel_program(
            _block_reduce_primfunc("max", dtype="int32", mask=32), platform="A2"
        )


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(
    ("kind", "order"),
    [
        ("sum", "ORDER_ONLY_VALUE"),
        ("max", "ORDER_ONLY_VALUE"),
        ("min", "ORDER_ONLY_VALUE"),
        ("max", "ORDER_VALUE_INDEX"),
        ("min", "ORDER_VALUE_INDEX"),
    ],
)
def test_whole_reduce_executes_masked_strided_repeats(dtype, kind, order) -> None:
    mask = 37
    program = build_kernel_program(
        _whole_reduce_primfunc(kind, dtype=dtype, mask=mask, order=order),
        platform="A3",
    )
    source_load, initial_load, reduction, store = program.tasks
    itemsize = np.dtype(dtype).itemsize
    elements_per_block = 32 // itemsize
    active_blocks = (mask + elements_per_block - 1) // elements_per_block
    output_width = 2 if order == "ORDER_VALUE_INDEX" else 1
    assert reduction.operation == f"wholereduce{kind}"
    assert reduction.metadata["src"].shape == (2, active_blocks, elements_per_block)
    assert reduction.metadata["src"].strides_bytes == (256, 32, itemsize)
    assert reduction.metadata["dst"].shape == (2, output_width)
    assert reduction.metadata["dst"].strides_bytes == (
        2 * output_width * itemsize, itemsize,
    )
    assert set(reduction.dependencies) == {
        source_load.task_id, initial_load.task_id,
    }
    assert store.dependencies == (reduction.task_id,)

    values = np.linspace(-8, 8, 256, dtype=np.float32).astype(dtype)
    initial = np.full(16, -99, dtype=dtype)
    simulator = FunctionalSimulator(program)
    simulator.write(source_load.metadata["src"], values)
    simulator.write(initial_load.metadata["src"], initial)
    simulator.run()

    expected = initial.copy()
    for repeat in range(2):
        start = repeat * 8 * elements_per_block
        lanes = values[start:start + mask].astype(np.float32)
        destination_start = repeat * 2 * output_width
        if kind == "sum":
            expected[destination_start] = np.sum(lanes)
        else:
            index = int(np.argmax(lanes) if kind == "max" else np.argmin(lanes))
            expected[destination_start] = lanes[index]
            if order == "ORDER_VALUE_INDEX":
                index_dtype = np.uint16 if itemsize == 2 else np.uint32
                expected[destination_start + 1] = np.asarray(
                    [index], dtype=index_dtype
                ).view(np.dtype(dtype))[0]
    np.testing.assert_array_equal(simulator.read(store.metadata["dst"]), expected)


def test_whole_reduce_rejects_invalid_order() -> None:
    with pytest.raises(UnsupportedSimOpError, match="does not support order"):
        build_kernel_program(
            _whole_reduce_primfunc("max", order="ORDER_INDEX_VALUE"),
            platform="A2",
        )


@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize(
    ("kind", "reducer", "merger"),
    [
        ("reduce_sum", np.sum, np.add),
        ("reduce_max", np.max, np.maximum),
        ("reduce_min", np.min, np.minimum),
    ],
)
def test_real_tir_reduce_clear_false_merges_destination(
    kind, reducer, merger, axis
) -> None:
    program = build_kernel_program(
        _reduce_primfunc(kind, axis, clear=False), platform="A2"
    )
    source_load, initial_load, reduce, store = program.tasks
    assert reduce.metadata["clear"] is False
    assert reduce.metadata["accumulator"] == reduce.metadata["dst"]
    assert set(reduce.dependencies) == {
        source_load.task_id, initial_load.task_id,
    }
    assert "scratch" in reduce.metadata
    assert ("output_scratch" in reduce.metadata) is (axis == -1)
    assert store.dependencies == (reduce.task_id,)

    simulator = FunctionalSimulator(program)
    values = np.arange(15, dtype=np.float32).reshape(3, 5) - 4
    initial = np.linspace(-2, 2, 5 if axis == 0 else 3, dtype=np.float32)
    simulator.write(source_load.metadata["src"], values.reshape(-1))
    simulator.write(initial_load.metadata["src"], initial)
    simulator.run()

    reduced = reducer(values, axis=0 if axis == 0 else 1)
    np.testing.assert_array_equal(
        simulator.read(store.metadata["dst"]), merger(initial, reduced)
    )


@pytest.mark.parametrize(
    ("kind", "reducer"),
    [
        ("reduce_sum", lambda value: np.sum(value, axis=0)),
        ("reduce_max", lambda value: np.max(value, axis=0)),
        ("reduce_min", lambda value: np.min(value, axis=0)),
    ],
)
def test_tail_reduce_axis_zero_executes_valid_rectangle(kind, reducer) -> None:
    program = build_kernel_program(_tail_reduce_primfunc(kind), platform="A2")
    load, reduce, store = program.tasks
    assert reduce.operation == kind
    assert reduce.metadata["tail_kind"] == "tail_reduce"
    assert reduce.metadata["reduce"] == {
        "dimension": 0,
        "clear": True,
        "valid_rows": 3,
        "valid_cols": 5,
        "physical_cols": 8,
    }
    assert reduce.dependencies == (load.task_id,)
    assert store.dependencies == (reduce.task_id,)

    simulator = FunctionalSimulator(program)
    values = np.arange(15, dtype=np.float32).reshape(3, 5) - 4
    simulator.write(
        BufferRegion("source", MemoryScope.GM, (3, 5), "float32"), values
    )
    simulator.run()

    np.testing.assert_array_equal(
        simulator.read(BufferRegion("output", MemoryScope.GM, (5,), "float32")),
        reducer(values),
    )


@pytest.mark.parametrize(
    ("dimension", "clear"),
    [(1, 1), (0, 0)],
)
def test_tail_reduce_unvalidated_contract_fails_closed(dimension, clear) -> None:
    with pytest.raises(
        UnsupportedSimOpError, match="supports only dim=0 and clear=true"
    ):
        build_kernel_program(
            _tail_reduce_primfunc("reduce_sum", dimension, clear), platform="A3"
        )


def test_tail_reduce_unvalidated_dtype_fails_closed() -> None:
    with pytest.raises(UnsupportedSimOpError, match="only float32"):
        build_kernel_program(
            _tail_reduce_primfunc("reduce_sum", dtype="float16"), platform="A2"
        )


def test_real_tir_shmem_call_fails_closed() -> None:
    with pytest.raises(UnsupportedSimOpError, match="intentionally unsupported"):
        build_kernel_program(rejected_shmem, platform="A3")
