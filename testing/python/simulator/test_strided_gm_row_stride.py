# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Minimal reproduction for the GQA simulator NaN (read-before-write on output).

Root cause under test: ``_TirBridge._access_buffer_region`` derives a 2-D GM
region's row stride from ``spec.shape[-1]`` (the buffer's last logical dim).
For a sliced view of a higher-rank GM tensor — e.g. GQA's BSND ``output``
``[B, S, Hq, D]`` written as a ``[rows, D]`` tile — the true GM row stride is
the product of the trailing dims (``Hq * D``), carried by the copy's
``stride_n`` argument, not ``D``.  The bridge ignored ``stride_n`` and used
``spec.shape[-1]``, so the strided ``copy_ub_to_gm`` marked only a tightly
packed byte range as initialized.  Rows beyond the first few were left poisoned,
and the host read of ``output`` reported read-before-write (NaN) even though the
real kernel writes every row (verified on hardware: 20/20 PASS).

These tests are RED until the GM-side region of a strided copy uses the copy's
row-stride argument instead of ``spec.shape[-1]``.
"""

import numpy as np
import pytest

tvm = pytest.importorskip("tvm")

from tilelang.simulator import (  # noqa: E402
    FunctionalSimulator,
    MemoryScope,
    build_kernel_program,
)


def _ub_to_gm_wide_stride_primfunc(stride_n: int):
    """UB->GM copy whose GM row stride exceeds the buffer's last dim.

    GM ``out`` is declared (4, 8) fp32, but the copy writes a (2, 5) tile whose
    GM row stride is ``stride_n`` (= 16) elements — modelling a [rows, D] slice
    of a wider row-major GM tensor (rows separated by ``stride_n`` elements).
    """
    out = tvm.tir.decl_buffer((4, 8), "float32", name="out", scope="global")
    ub = tvm.tir.decl_buffer((2, 8), "float32", name="ub", scope="shared.ub")
    call = tvm.tir.call_extern(
        "handle",
        "tl::ascend::copy_ub_to_gm<float32, 8, 2>",
        ub.access_ptr("r", offset=0, extent=16),
        out.access_ptr("w", offset=0, extent=16),
        stride_n,  # arguments[2] -> stride_n (GM row stride, elements)
        2,         # arguments[3] -> valid_rows
        5,         # arguments[4] -> valid_cols
        2,         # physical_rows
        8,         # physical_cols
    )
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.Evaluate(call), alloc_buffers=[ub]
    )
    return tvm.tir.PrimFunc(
        [out.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={out.data: out},
    )


def test_ub_to_gm_region_uses_copy_row_stride_not_last_dim() -> None:
    """GM destination row stride must come from stride_n, not spec.shape[-1]."""
    program = build_kernel_program(
        _ub_to_gm_wide_stride_primfunc(stride_n=16), platform="A2"
    )
    destination = program.tasks[0].metadata["dst"]
    assert destination.scope is MemoryScope.GM
    # True GM row stride is 16 elements = 64 bytes, not shape[-1]=8 (32 bytes).
    assert destination.strides_bytes == (16 * 4, 4)


def test_ub_to_gm_wide_stride_marks_all_rows_initialized() -> None:
    """Functional write must initialize every strided row (no poison gap)."""
    program = build_kernel_program(
        _ub_to_gm_wide_stride_primfunc(stride_n=16), platform="A2"
    )
    simulator = FunctionalSimulator(program)
    task = program.tasks[0]
    # Pre-initialize the UB source tile so the copy has real data to move.
    simulator.write(task.metadata["src"], np.ones((2, 5), dtype=np.float32))
    simulator.run()
    # Every byte the strided write touched must be initialized: rows 0..1,
    # each covering [r*64, r*64 + 20) bytes within the wide-stride GM view.
    allocation = simulator.memory.get("out", scope=MemoryScope.GM)
    for row in range(2):
        base = row * 16 * 4
        for byte in range(base, base + 5 * 4):
            assert allocation._backing.initialized[byte], (
                f"row {row} byte {byte} left poisoned by strided ub_to_gm"
            )


def _gm_to_l1_wide_stride_primfunc(source_cols: int):
    """GM->L1 copy whose GM source row stride exceeds the buffer's last dim.

    GM ``src`` is declared (4, 8) fp16, but the copy reads a (2, 8) tile whose
    GM row stride is ``source_cols`` (= 16) elements — modelling a [rows, D]
    Q/K/V tile of a wider BSND tensor.
    """
    src = tvm.tir.decl_buffer((4, 8), "float16", name="src", scope="global")
    # zN (2, 16) fp16 fractal stores ceil(2/16)*16 * 16 = 256 elements.
    l1 = tvm.tir.decl_buffer((256,), "float16", name="l1", scope="shared.l1")
    call = tvm.tir.call_extern(
        "handle",
        "tl::ascend::copy_gm_to_l1<half, 2, 16>",
        src.access_ptr("r"),
        l1.access_ptr("w"),
        source_cols,  # realSrcN -> GM source row stride (elements)
        2,            # validM
        8,            # validN
        2,            # dstM
        16,           # dstN
    )
    root = tvm.tir.Block(
        [], [], [], "root", tvm.tir.Evaluate(call), alloc_buffers=[l1]
    )
    return tvm.tir.PrimFunc(
        [src.data],
        tvm.tir.BlockRealize([], True, root),
        buffer_map={src.data: src},
    )


def test_gm_to_l1_source_region_uses_source_cols_row_stride() -> None:
    """GM source row stride must come from source_cols, not spec.shape[-1]."""
    program = build_kernel_program(
        _gm_to_l1_wide_stride_primfunc(source_cols=16), platform="A2"
    )
    source = program.tasks[0].metadata["src"]
    assert source.scope is MemoryScope.GM
    # True GM row stride is 16 elements = 32 bytes, not shape[-1]=8 (16 bytes).
    assert source.strides_bytes == (16 * 2, 2)
