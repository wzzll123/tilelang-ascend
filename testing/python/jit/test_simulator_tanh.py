# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Native lowering through selected Tanh terminals into the CPU simulator."""

import numpy as np
import pytest

import tilelang
from tilelang import language as T
from tilelang.simulator import MemoryHazardError, ProgramValidationError, build_kernel_program


def _kernel(dtype, platform, *, sync_only=False, auto_sync=True):
    @T.prim_func
    def main(
        source: T.Tensor((128,), dtype),
        output: T.Tensor((128,), dtype),
        source_copy: T.Tensor((128,), dtype),
    ):
        with T.Kernel(1, threads=1, is_npu=True):
            src_ub = T.alloc_ub((128,), dtype)
            dst_ub = T.alloc_ub((128,), dtype)
            with T.Scope("V"):
                T.copy(source, src_ub)
                T.tile.tanh(dst_ub, src_ub)
                T.copy(dst_ub, output)
                T.copy(src_ub, source_copy)

    return tilelang.compile(
        main,
        out_idx=[1, 2],
        target="ascendc",
        platform=platform,
        simulator=True,
        sim_config={"sync_only": sync_only},
        pass_configs={"tl.ascend_auto_sync": auto_sync, "tl.ascend_memory_planning": True},
    )


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("sync_only", [False, True])
def test_tanh_round_trip_matches_numpy_and_preserves_source(platform, dtype, sync_only):
    kernel = _kernel(dtype, platform, sync_only=sync_only)
    source = np.linspace(-4, 4, 128).astype(dtype)
    result = kernel(source)
    if sync_only:
        assert result is None
    else:
        output, source_copy = result
        np.testing.assert_array_equal(output, np.tanh(source))
        np.testing.assert_array_equal(source_copy, source)
    assert kernel.get_simulator_report().sync_only is sync_only


def test_tanh_internal_vector_barriers_do_not_replace_dma_fences():
    with pytest.raises(MemoryHazardError, match="missing A2/A3 pipe synchronization"):
        _kernel("float32", "A2", auto_sync=False)


@pytest.mark.parametrize("dtype,minimum", [("float16", 1024), ("float32", 256)])
@pytest.mark.parametrize("undersized", [False, True])
def test_raw_tanh_bridge_checks_official_scratch_minimum(dtype, minimum, undersized):
    from tilelang import tvm

    t = tvm.tir
    source = t.decl_buffer((128,), dtype, name="source", scope="shared.ub")
    destination = t.decl_buffer((128,), dtype, name="destination", scope="shared.ub")
    scratch = t.decl_buffer((minimum - 32 if undersized else minimum,), "uint8", name="scratch", scope="shared.ub")
    body = t.Evaluate(
        t.Call(
            "handle",
            tvm.ir.Op.get("tl.ascend_tanh"),
            [
                destination.access_ptr("w"),
                source.access_ptr("r"),
                scratch.access_ptr("w"),
                t.const(128, "int32"),
            ],
        )
    )
    for buffer in (source, destination, scratch):
        body = t.Allocate(buffer.data, buffer.dtype, buffer.shape, t.const(True, "bool"), body)
    body = t.AttrStmt(t.IntImm("int32", 1), "resource_scope", 1, body)
    if undersized:
        with pytest.raises(ProgramValidationError, match=f"at least {minimum} bytes"):
            build_kernel_program(t.PrimFunc([], body), platform="A2")
    else:
        program = build_kernel_program(t.PrimFunc([], body), platform="A2")
        assert program.tasks[0].metadata["scratch"].shape == (minimum,)
