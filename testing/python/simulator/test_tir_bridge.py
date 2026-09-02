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


def _compare_select_primfunc(*, scalar_select=False, with_scratch=False):
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
    if scalar_select:
        select_arguments.extend([
            1, tvm.tir.FloatImm("float32", 10.0),
            "VSEL_TENSOR_SCALAR_MODE", 8, "float32", "uint8",
        ])
    else:
        select_arguments.extend([
            2, ub_right.access_ptr("r"), "VSEL_TENSOR_TENSOR_MODE", 8,
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


@pytest.mark.parametrize(
    ("scalar_select", "with_scratch"), [(False, False), (True, True)]
)
def test_real_tir_select_reads_packed_compare_mask(
    scalar_select, with_scratch
) -> None:
    program = build_kernel_program(
        _compare_select_primfunc(
            scalar_select=scalar_select, with_scratch=with_scratch
        ),
        platform="A3",
    )
    left_load, right_load, compare, select, store = program.tasks
    assert select.operation == "select"
    assert select.metadata["source_type"] == (1 if scalar_select else 2)
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
