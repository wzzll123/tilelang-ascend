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
    UnsupportedSimOpError,
    build_kernel_program,
)
from tilelang.simulator.layout import pack_matrix, unpack_matrix  # noqa: E402


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


def _mma_primfunc(init=True, inner=13, n_actual=None, unit_flag=0, cols=16):
    rows = 16
    a_elements = rows * ((inner + 15) // 16 * 16)
    b_elements = ((inner + 15) // 16 * 16) * cols
    c_elements = rows * cols
    l0a = tvm.tir.decl_buffer(
        (a_elements,), "float16", name="l0a", scope="wmma.matrix_a"
    )
    l0b = tvm.tir.decl_buffer(
        (b_elements,), "float16", name="l0b", scope="wmma.matrix_b"
    )
    l0c = tvm.tir.decl_buffer(
        (c_elements,), "float32", name="l0c", scope="wmma.accumulator"
    )
    arguments = [
        f"mma<half, float, 16, {cols}>",
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
):
    shape_a = (inner, rows) if transpose_a else (rows, inner)
    shape_b = (cols, inner) if transpose_b else (inner, cols)
    a_elements = ((shape_a[0] + 15) // 16 * 16) * (
        (shape_a[1] + 15) // 16 * 16
    )
    b_elements = ((shape_b[0] + 15) // 16 * 16) * (
        (shape_b[1] + 15) // 16 * 16
    )
    c_elements = ((rows + 15) // 16 * 16) * ((cols + 15) // 16 * 16)
    l1a = tvm.tir.decl_buffer(
        (a_elements,), "float16", name="l1a", scope="shared.l1"
    )
    l1b = tvm.tir.decl_buffer(
        (b_elements,), "float16", name="l1b", scope="shared.l1"
    )
    l0c = tvm.tir.decl_buffer(
        (c_elements,), "float32", name="l0c", scope="wmma.accumulator"
    )
    call = tvm.tir.call_extern(
        "handle",
        "tl.ascend_gemm_v0",
        f"gemm_v0<half, float, {rows}, {cols}, {inner}, "
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


def _scratch_unary_primfunc(operation, *, with_scratch=False, in_place=False):
    source = tvm.tir.decl_buffer((5,), "float32", name="source", scope="global")
    output = tvm.tir.decl_buffer((5,), "float32", name="output", scope="global")
    ub_source = tvm.tir.decl_buffer(
        (5,), "float32", name="ub_source", scope="shared.ub"
    )
    ub_output = tvm.tir.decl_buffer(
        (5,), "float32", name="ub_output", scope="shared.ub"
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
