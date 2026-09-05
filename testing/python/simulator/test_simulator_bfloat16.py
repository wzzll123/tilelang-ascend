# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""bfloat16 functional support for the CPU-only A2/A3 simulator.

The wqbm bf16 epilogue narrows fp32 accumulators to bf16 in UB via
``T.copy(acc_ub, y_t_ub)`` (lowered to ``copy_ub_to_ub<bfloat16, float32,
len>``) and ``T.tile.cast(..., "CAST_RINT", ...)``.  NumPy has no native
bfloat16, so the simulator maps it to ``ml_dtypes.bfloat16`` (RNE casts,
verified bit-exact against torch.bfloat16).
"""

import numpy as np
import pytest

import ml_dtypes
import torch

import tilelang
import tilelang.language as T


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def _torch_ref_f32_to_bf16(values):
    return torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)


def _assert_bf16_bits_equal(actual, expected_torch_bf16):
    actual_bits = np.asarray(actual, dtype=ml_dtypes.bfloat16).view(np.uint16)
    expected_bits = expected_torch_bf16.view(torch.uint16).numpy()
    np.testing.assert_array_equal(actual_bits, expected_bits)


def _ub_to_ub_narrow_kernel(platform):
    @tilelang.jit(
        out_idx=[1],
        simulator=True,
        platform=platform,
        pass_configs=PASS_CONFIGS,
    )
    def kernel():
        @T.prim_func
        def main(
            source: T.Tensor([4, 32], "float32"),
            output: T.Tensor([4, 32], "bfloat16"),
        ):
            with T.Kernel(1, is_npu=True):
                source_ub = T.alloc_ub([4, 32], "float32")
                output_ub = T.alloc_ub([4, 32], "bfloat16")
                with T.Scope("V"):
                    T.copy(source, source_ub)
                    T.copy(source_ub, output_ub)
                    T.copy(output_ub, output)

        return main

    return kernel()


def _ub_to_ub_widen_kernel(platform):
    @tilelang.jit(
        out_idx=[1],
        simulator=True,
        platform=platform,
        pass_configs=PASS_CONFIGS,
    )
    def kernel():
        @T.prim_func
        def main(
            source: T.Tensor([4, 32], "bfloat16"),
            output: T.Tensor([4, 32], "float32"),
        ):
            with T.Kernel(1, is_npu=True):
                source_ub = T.alloc_ub([4, 32], "bfloat16")
                output_ub = T.alloc_ub([4, 32], "float32")
                with T.Scope("V"):
                    T.copy(source, source_ub)
                    T.copy(source_ub, output_ub)
                    T.copy(output_ub, output)

        return main

    return kernel()


def _tile_cast_rint_kernel(platform):
    @tilelang.jit(
        out_idx=[1],
        simulator=True,
        platform=platform,
        pass_configs=PASS_CONFIGS,
    )
    def kernel():
        @T.prim_func
        def main(
            source: T.Tensor([4, 32], "float32"),
            output: T.Tensor([4, 32], "bfloat16"),
        ):
            with T.Kernel(1, is_npu=True):
                source_ub = T.alloc_ub([4, 32], "float32")
                output_ub = T.alloc_ub([4, 32], "bfloat16")
                with T.Scope("V"):
                    T.copy(source, source_ub)
                    T.tile.cast(output_ub, source_ub, "CAST_RINT", 4 * 32)
                    T.copy(output_ub, output)

        return main

    return kernel()


def _bf16_arithmetic_kernel(platform):
    @tilelang.jit(
        out_idx=[1],
        simulator=True,
        platform=platform,
        pass_configs=PASS_CONFIGS,
    )
    def kernel():
        @T.prim_func
        def main(
            source: T.Tensor([4, 32], "bfloat16"),
            output: T.Tensor([4, 32], "bfloat16"),
        ):
            with T.Kernel(1, is_npu=True):
                source_ub = T.alloc_ub([4, 32], "bfloat16")
                scaled_ub = T.alloc_ub([4, 32], "bfloat16")
                bias_ub = T.alloc_ub([4, 32], "bfloat16")
                out_ub = T.alloc_ub([4, 32], "bfloat16")
                with T.Scope("V"):
                    T.copy(source, source_ub)
                    T.copy(source, bias_ub)
                    T.tile.mul(scaled_ub, source_ub, source_ub)
                    T.tile.add(out_ub, scaled_ub, bias_ub)
                    T.copy(out_ub, output)

        return main

    return kernel()


def test_copy_ub_to_ub_float32_to_bfloat16_narrowing() -> None:
    rng = np.random.default_rng(0)
    source = (rng.standard_normal((4, 32)).astype(np.float32)
              * np.exp(rng.standard_normal((4, 32)).astype(np.float32) * 5))
    expected = _torch_ref_f32_to_bf16(source)

    for platform in ("A2", "A3"):
        output = _ub_to_ub_narrow_kernel(platform)(source)
        _assert_bf16_bits_equal(output, expected)


def test_copy_ub_to_ub_bfloat16_to_float32_widening_is_exact() -> None:
    rng = np.random.default_rng(1)
    source_f32 = rng.standard_normal((4, 32)).astype(np.float32)
    source = source_f32.astype(ml_dtypes.bfloat16)
    expected = torch.tensor(source_f32, dtype=torch.float32).to(torch.bfloat16)

    for platform in ("A2", "A3"):
        output = _ub_to_ub_widen_kernel(platform)(source)
        np.testing.assert_array_equal(
            output, expected.to(torch.float32).numpy()
        )


def test_tile_cast_float32_to_bfloat16_cast_rint() -> None:
    rng = np.random.default_rng(2)
    source = (rng.standard_normal((4, 32)).astype(np.float32)
              * np.exp(rng.standard_normal((4, 32)).astype(np.float32) * 5))
    expected = _torch_ref_f32_to_bf16(source)

    for platform in ("A2", "A3"):
        output = _tile_cast_rint_kernel(platform)(source)
        _assert_bf16_bits_equal(output, expected)


def test_bfloat16_gm_ub_roundtrip_and_arithmetic() -> None:
    rng = np.random.default_rng(3)
    source_f32 = rng.standard_normal((4, 32)).astype(np.float32) * 0.5
    source = source_f32.astype(ml_dtypes.bfloat16)
    source_t = torch.tensor(source_f32, dtype=torch.float32).to(torch.bfloat16)
    expected = source_t * source_t + source_t

    for platform in ("A2", "A3"):
        output = _bf16_arithmetic_kernel(platform)(source)
        _assert_bf16_bits_equal(output, expected)


def _bf16_mma_kernel(platform):
    from tilelang.intrinsics import make_zn_layout

    @tilelang.jit(
        out_idx=[2],
        simulator=True,
        platform=platform,
        pass_configs=PASS_CONFIGS,
    )
    def kernel():
        @T.prim_func
        def main(
            left: T.Tensor([16, 32], "bfloat16"),
            right: T.Tensor([32, 16], "bfloat16"),
            output: T.Tensor([16, 16], "float32"),
        ):
            with T.Kernel(1, is_npu=True):
                left_l1 = T.alloc_L1([16, 32], "bfloat16")
                right_l1 = T.alloc_L1([32, 16], "bfloat16")
                T.annotate_layout({
                    left_l1: make_zn_layout(left_l1),
                    right_l1: make_zn_layout(right_l1),
                })
                left_l0 = T.alloc_L0A([16, 32], "bfloat16")
                right_l0 = T.alloc_L0B([32, 16], "bfloat16")
                accumulator = T.alloc_L0C([16, 16], "float32")
                with T.Scope("C"):
                    T.copy(left, left_l1)
                    T.copy(right, right_l1)
                    T.copy(left_l1, left_l0)
                    T.copy(right_l1, right_l0)
                    T.mma(left_l0, right_l0, accumulator, init=True)
                    T.copy(accumulator, output)

        return main

    return kernel()


def test_bfloat16_mma_accumulates_in_float32() -> None:
    rng = np.random.default_rng(4)
    left_f32 = rng.standard_normal((16, 32)).astype(np.float32)
    right_f32 = rng.standard_normal((32, 16)).astype(np.float32)
    left = left_f32.astype(ml_dtypes.bfloat16)
    right = right_f32.astype(ml_dtypes.bfloat16)
    # bf16 mma widens inputs exactly to fp32 and accumulates in fp32.
    expected = (
        torch.tensor(left_f32).to(torch.bfloat16).to(torch.float32)
        @ torch.tensor(right_f32).to(torch.bfloat16).to(torch.float32)
    ).numpy()

    for platform in ("A2", "A3"):
        output = _bf16_mma_kernel(platform)(left, right)
        np.testing.assert_array_equal(output, expected)


def test_torch_bfloat16_tensors_round_trip_through_adapter() -> None:
    # The wqbm pilot passes CPU torch.bfloat16 tensors straight into the
    # simulator kernel; the adapter must convert them in both directions
    # without a native NumPy bfloat16.
    rng = np.random.default_rng(5)
    source_f32 = rng.standard_normal((4, 32)).astype(np.float32) * 0.5
    source_t = torch.tensor(source_f32, dtype=torch.float32).to(torch.bfloat16)
    expected = source_t * source_t + source_t

    for platform in ("A2", "A3"):
        output = _bf16_arithmetic_kernel(platform)(source_t)
        assert isinstance(output, torch.Tensor)
        assert output.dtype == torch.bfloat16
        assert torch.equal(output, expected)
