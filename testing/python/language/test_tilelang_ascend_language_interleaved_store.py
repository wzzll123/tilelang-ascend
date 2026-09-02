# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# Interleaved store (strided store) vectorization regression test.
#
# Background: interleaved RoPE even/odd recombination `rf[2j]=oa[j], rf[2j+1]=ob[j]`
# has no vector primitive in tilelang codegen, so it lowers to scalar
# SetValue/GetValue loops + full-pipe barriers (27x slowdown on memory-bound ops).
# GM-side strided copy (`q[...,0::2]`) fails with "Cannot convert type int32x32".
#
# This test captures the expected vectorized lowering: the compiler should
# recognize the interleaved store pattern and emit Gather-based vectorized
# recombination (index-Gather with dynamically generated offsets), not scalar
# SetValue/GetValue loops.

import pytest
import torch

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _run_interleaved_store(target, dtype):
    """Test interleaved store pattern: rf[2j]=oa[j], rf[2j+1]=ob[j].

    This is the core pattern from apply_rotary_pos_emb interleaved mode.
    The compiler should vectorize this, not lower to scalar SetValue/GetValue.
    """
    torch_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
    TILE_ROWS = 8
    half = 32  # half-width of each row
    D = 2 * half  # full width
    d_align = D  # aligned width (D is already 32B-aligned for fp32/fp16)

    @tilelang.jit(out_idx=[2], pass_configs=pass_configs, target=target)
    def kernel():
        @T.prim_func
        def main(
            oa: T.Tensor([TILE_ROWS, half], dtype),
            ob: T.Tensor([TILE_ROWS, half], dtype),
            rf: T.Tensor([TILE_ROWS, D], dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                oa_ub = T.alloc_ub([TILE_ROWS, half], dtype)
                ob_ub = T.alloc_ub([TILE_ROWS, half], dtype)
                rf_ub = T.alloc_ub([TILE_ROWS, d_align], dtype)

                T.copy(oa, oa_ub)
                T.copy(ob, ob_ub)

                # Interleaved store: rf[2j]=oa[j], rf[2j+1]=ob[j]
                # This should be vectorized, not scalar SetValue/GetValue
                for i, j in T.Parallel(TILE_ROWS, half):
                    rf_ub[i, 2 * j] = oa_ub[i, j]
                    rf_ub[i, 2 * j + 1] = ob_ub[i, j]

                T.copy(rf_ub, rf)

        return main

    torch.manual_seed(0)
    oa = torch.randn(TILE_ROWS, half, dtype=torch_dtype)
    ob = torch.randn(TILE_ROWS, half, dtype=torch_dtype)

    got = kernel()(oa.npu(), ob.npu()).cpu().float()

    # Reference: manual interleave
    ref = torch.zeros(TILE_ROWS, D, dtype=torch_dtype)
    ref[:, 0::2] = oa
    ref[:, 1::2] = ob
    ref = ref.float()

    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-3)


def _run_interleaved_store_fp32(target):
    """FP32 variant of interleaved store test."""
    _run_interleaved_store(target, "float32")


def _run_interleaved_store_fp16(target):
    """FP16 variant of interleaved store test."""
    _run_interleaved_store(target, "float16")


@pytest.mark.parametrize("target", ["ascendc"])
def test_interleaved_store_fp32(target):
    """Test that interleaved store is vectorized, not scalar lowered."""
    _run_interleaved_store_fp32(target)


@pytest.mark.parametrize("target", ["ascendc"])
def test_interleaved_store_fp16(target):
    """Test that interleaved store is vectorized, not scalar lowered."""
    _run_interleaved_store_fp16(target)


def _check_generated_code_vectorized(target, dtype):
    """Check that generated code does NOT contain scalar SetValue/GetValue
    for the interleaved store pattern.

    This is a stronger test: it verifies the compiler actually vectorized
    the pattern, not just that the result is correct (scalar code can also
    produce correct results, just slowly).
    """
    torch_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
    TILE_ROWS = 8
    half = 32
    D = 2 * half
    d_align = D

    @tilelang.jit(out_idx=[2], pass_configs=pass_configs, target=target)
    def kernel():
        @T.prim_func
        def main(
            oa: T.Tensor([TILE_ROWS, half], dtype),
            ob: T.Tensor([TILE_ROWS, half], dtype),
            rf: T.Tensor([TILE_ROWS, D], dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                oa_ub = T.alloc_ub([TILE_ROWS, half], dtype)
                ob_ub = T.alloc_ub([TILE_ROWS, half], dtype)
                rf_ub = T.alloc_ub([TILE_ROWS, d_align], dtype)

                T.copy(oa, oa_ub)
                T.copy(ob, ob_ub)

                for i, j in T.Parallel(TILE_ROWS, half):
                    rf_ub[i, 2 * j] = oa_ub[i, j]
                    rf_ub[i, 2 * j + 1] = ob_ub[i, j]

                T.copy(rf_ub, rf)

        return main

    torch.manual_seed(0)
    oa = torch.randn(TILE_ROWS, half, dtype=torch_dtype)
    ob = torch.randn(TILE_ROWS, half, dtype=torch_dtype)

    # Get the compiled kernel and inspect generated code
    compiled = kernel()
    ker = compiled(oa.npu(), ob.npu())

    # Get generated source code
    source = compiled.get_kernel_source()

    # Check for scalar SetValue/GetValue pattern (bad)
    # The interleaved store should NOT produce per-element SetValue/GetValue
    # with PipeBarrier<PIPE_ALL> (full pipe drain)
    scalar_setvalue_count = source.count(".SetValue(")
    scalar_getvalue_count = source.count(".GetValue(")

    # Check for full-pipe-drain barriers (a hallmark of scalar lowering)
    pipe_all_count = source.count("PipeBarrier<PIPE_ALL>")

    # Check for vectorized Gather-based recombination (good)
    gather_count = source.count("AscendC::Gather(")

    # For a vectorized implementation, we expect:
    # - DataCopy for staging [oa|ob] into contiguous buffer
    # - Gather for index-based recombination
    # - NO per-element SetValue/GetValue in the hot loop
    # - NO PipeBarrier<PIPE_ALL> (full pipe drain)
    #
    # Current buggy behavior: SetValue/GetValue inside a for loop that iterates
    #   TILE_ROWS * half = 256 times, with PipeBarrier<PIPE_ALL> per element.
    #   The generated code shows 2 SetValue + 2 GetValue calls in the loop body,
    #   but the loop executes 256 times -> 512 scalar ops + 512 full-pipe drains.
    # Expected fixed behavior: Gather-based vectorized recombination with
    #   one-time offset buffer initialization (D SetValue calls, but no
    #   per-element GetValue/SetValue in the hot loop).

    # The test fails if we see per-element scalar lowering:
    # - GetValue calls (indicates scalar load from UB)
    # - PipeBarrier<PIPE_ALL> (indicates full-pipe-drain synchronization)
    # - SetValue calls in a loop pattern (indicates per-element scalar store)
    #
    # Note: One-time offset buffer initialization with SetValue is acceptable
    # (it's O(D) setup, not O(rows*D) per-element). We detect this by checking
    # that Gather is present (indicating vectorized recombination).
    assert scalar_getvalue_count == 0, (
        f"Interleaved store was scalar-lowered: found {scalar_getvalue_count} "
        f"GetValue calls in generated code. "
        f"Generated code should use vectorized Gather-based recombination, "
        f"not scalar SetValue/GetValue loops."
    )
    assert pipe_all_count == 0, (
        f"Interleaved store caused full-pipe drains: found {pipe_all_count} "
        f"PipeBarrier<PIPE_ALL> in generated code. "
        f"Vectorized code should use pipe-specific barriers, not full-pipe drains."
    )
    assert gather_count > 0, (
        f"Interleaved store was not vectorized: found {gather_count} "
        f"AscendC::Gather calls in generated code. "
        f"Generated code should use Gather-based vectorized recombination."
    )


@pytest.mark.parametrize("target", ["ascendc"])
def test_interleaved_store_vectorized_fp32(target):
    """Test that interleaved store generates vectorized code, not scalar."""
    _check_generated_code_vectorized(target, "float32")


@pytest.mark.parametrize("target", ["ascendc"])
def test_interleaved_store_vectorized_fp16(target):
    """Test that interleaved store generates vectorized code, not scalar."""
    _check_generated_code_vectorized(target, "float16")


def _run_dual_interleaved_store(target, dtype):
    """Two independent interleaved-store pairs in ONE T.Parallel body.

    This mirrors the apply_rotary_pos_emb interleaved path, which recombines BOTH
    q and k in the same parallel loop (4 stores total: 2 for q_out, 2 for k_out).
    The compiler must group stores by output buffer and vectorize each pair.
    """
    torch_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
    TILE_ROWS = 8
    half = 32
    D = 2 * half

    @tilelang.jit(out_idx=[4, 5], pass_configs=pass_configs, target=target)
    def kernel():
        @T.prim_func
        def main(
            qa: T.Tensor([TILE_ROWS, half], dtype),
            qb: T.Tensor([TILE_ROWS, half], dtype),
            ka: T.Tensor([TILE_ROWS, half], dtype),
            kb: T.Tensor([TILE_ROWS, half], dtype),
            qf: T.Tensor([TILE_ROWS, D], dtype),
            kf: T.Tensor([TILE_ROWS, D], dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                qa_ub = T.alloc_ub([TILE_ROWS, half], dtype)
                qb_ub = T.alloc_ub([TILE_ROWS, half], dtype)
                ka_ub = T.alloc_ub([TILE_ROWS, half], dtype)
                kb_ub = T.alloc_ub([TILE_ROWS, half], dtype)
                qf_ub = T.alloc_ub([TILE_ROWS, D], dtype)
                kf_ub = T.alloc_ub([TILE_ROWS, D], dtype)

                T.copy(qa, qa_ub)
                T.copy(qb, qb_ub)
                T.copy(ka, ka_ub)
                T.copy(kb, kb_ub)

                # Two interleaved pairs in one body: qf (from qa/qb) + kf (from ka/kb)
                for i, j in T.Parallel(TILE_ROWS, half):
                    qf_ub[i, 2 * j] = qa_ub[i, j]
                    qf_ub[i, 2 * j + 1] = qb_ub[i, j]
                    kf_ub[i, 2 * j] = ka_ub[i, j]
                    kf_ub[i, 2 * j + 1] = kb_ub[i, j]

                T.copy(qf_ub, qf)
                T.copy(kf_ub, kf)

        return main

    torch.manual_seed(0)
    qa = torch.randn(TILE_ROWS, half, dtype=torch_dtype)
    qb = torch.randn(TILE_ROWS, half, dtype=torch_dtype)
    ka = torch.randn(TILE_ROWS, half, dtype=torch_dtype)
    kb = torch.randn(TILE_ROWS, half, dtype=torch_dtype)

    compiled = kernel()
    qf_got, kf_got = compiled(qa.npu(), qb.npu(), ka.npu(), kb.npu())
    qf_got = qf_got.cpu().float()
    kf_got = kf_got.cpu().float()

    qf_ref = torch.zeros(TILE_ROWS, D, dtype=torch_dtype)
    qf_ref[:, 0::2] = qa
    qf_ref[:, 1::2] = qb
    kf_ref = torch.zeros(TILE_ROWS, D, dtype=torch_dtype)
    kf_ref[:, 0::2] = ka
    kf_ref[:, 1::2] = kb

    torch.testing.assert_close(qf_got, qf_ref.float(), rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(kf_got, kf_ref.float(), rtol=2e-2, atol=2e-3)

    # Both pairs must be vectorized: 2 Gather calls, no scalar GetValue, no
    # full-pipe drains in the recombination.
    source = compiled.get_kernel_source()
    assert source.count("AscendC::Gather(") == 2, (
        f"Expected 2 Gather calls (one per pair), got "
        f"{source.count('AscendC::Gather(')}"
    )
    assert source.count(".GetValue(") == 0, (
        f"Dual interleaved store was scalar-lowered: "
        f"{source.count('.GetValue(')} GetValue calls"
    )
    assert source.count("PipeBarrier<PIPE_ALL>") == 0, (
        "Dual interleaved store caused full-pipe drains"
    )


@pytest.mark.parametrize("target", ["ascendc"])
def test_dual_interleaved_store_fp32(target):
    """Two interleaved pairs (q+k) in one loop body must both be vectorized."""
    _run_dual_interleaved_store(target, "float32")


@pytest.mark.parametrize("target", ["ascendc"])
def test_dual_interleaved_store_fp16(target):
    """Two interleaved pairs (q+k) in one loop body must both be vectorized."""
    _run_dual_interleaved_store(target, "float16")


if __name__ == "__main__":
    test_interleaved_store_fp32("ascendc")
    print("interleaved store fp32 PASS")
    test_interleaved_store_fp16("ascendc")
    print("interleaved store fp16 PASS")
    test_interleaved_store_vectorized_fp32("ascendc")
    print("interleaved store vectorized fp32 PASS")
    test_interleaved_store_vectorized_fp16("ascendc")
    print("interleaved store vectorized fp16 PASS")
    test_dual_interleaved_store_fp32("ascendc")
    print("dual interleaved store fp32 PASS")
    test_dual_interleaved_store_fp16("ascendc")
    print("dual interleaved store fp16 PASS")
