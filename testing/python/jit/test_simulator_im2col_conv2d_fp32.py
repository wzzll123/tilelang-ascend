"""Execute the repository's fp32 im2col/MMA convolution path on CPU."""

import numpy as np
import ml_dtypes

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def _ceil16(value):
    return (value + 15) // 16 * 16


def _conv2d_kernel(
    platform, *, height=8, width=8, dtype="float32", channel_groups=2,
    output_channels=32, kernel=(3, 3), stride=(1, 1), dilation=(1, 1),
    padding=(1, 1, 1, 1),
):
    channels_per_c0 = 8 if dtype == "float32" else 16
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    pad_left, pad_right, pad_top, pad_bottom = padding
    output_h = (
        height + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    output_w = (
        width + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    output_positions = output_h * output_w
    output_positions_round = _ceil16(output_positions)
    k_per_group = kernel_h * kernel_w * channels_per_c0

    @tilelang.jit(
        out_idx=[2],
        simulator=True,
        platform=platform,
        pass_configs=PASS_CONFIGS,
    )
    def kernel():
        @T.prim_func
        def main(
            feature: T.Tensor(
                [channel_groups * height * width, channels_per_c0], dtype
            ),
            weight: T.Tensor([channel_groups * k_per_group, output_channels], dtype),
            output: T.Tensor([output_positions_round, output_channels], "float32"),
        ):
            with T.Kernel(1, is_npu=True):
                feature_l1 = T.alloc_L1([height * width, channels_per_c0], dtype)
                weight_l1 = T.alloc_L1([k_per_group, output_channels], dtype)
                T.annotate_layout({
                    feature_l1: make_zn_layout(feature_l1),
                    weight_l1: make_zn_layout(weight_l1),
                })
                feature_l0a = T.alloc_L0A([output_positions_round, k_per_group], dtype)
                weight_l0b = T.alloc_L0B([k_per_group, output_channels], dtype)
                accumulator = T.alloc_L0C([output_positions_round, output_channels], "float32")
                with T.Scope("C"):
                    for group in T.serial(channel_groups):
                        T.copy(
                            feature[group * height * width:(group + 1) * height * width, :],
                            feature_l1,
                        )
                        T.copy(
                            weight[group * k_per_group:(group + 1) * k_per_group, :],
                            weight_l1,
                        )
                        T.tile.im2col(
                            feature_l0a, feature_l1, (height, width),
                            (kernel_h, kernel_w), (stride_h, stride_w),
                            (dilation_h, dilation_w),
                            (pad_left, pad_right, pad_top, pad_bottom),
                            0, 0, output_positions, k_per_group,
                        )
                        T.copy(weight_l1, weight_l0b)
                        T.mma(feature_l0a, weight_l0b, accumulator, init=(group == 0))
                    T.copy(accumulator, output)

        return main

    return kernel(), {
        "height": height, "width": width, "channels_per_c0": channels_per_c0,
        "groups": channel_groups, "k_per_group": k_per_group,
        "kernel_shape": (kernel_h, kernel_w), "stride": stride, "dilation": dilation,
        "padding": padding, "output_h": output_h, "output_w": output_w,
        "output_positions": output_positions, "dtype": dtype,
        "output_channels": output_channels,
    }


def _reference(feature, weight, **config):
    height, width = config["height"], config["width"]
    channels_per_c0, groups = config["channels_per_c0"], config["groups"]
    k_per_group = config["k_per_group"]
    kernel_h, kernel_w = config["kernel_shape"]
    stride_h, stride_w = config["stride"]
    dilation_h, dilation_w = config["dilation"]
    pad_left, _pad_right, pad_top, _pad_bottom = config["padding"]
    output = np.zeros((config["output_positions"], weight.shape[1]), dtype=np.float32)
    for output_row in range(config["output_h"]):
        for output_col in range(config["output_w"]):
            m = output_row * config["output_w"] + output_col
            for group in range(groups):
                for kernel_row in range(kernel_h):
                    for kernel_col in range(kernel_w):
                        image_row = output_row * stride_h - pad_top + kernel_row * dilation_h
                        image_col = output_col * stride_w - pad_left + kernel_col * dilation_w
                        if 0 <= image_row < height and 0 <= image_col < width:
                            image = feature[
                                group * height * width + image_row * width + image_col
                            ]
                            k_start = group * k_per_group + (
                                kernel_row * kernel_w + kernel_col
                            ) * channels_per_c0
                            output[m] += image.astype(np.float32) @ weight[
                                k_start:k_start + channels_per_c0
                            ].astype(np.float32)
    return output


def test_fp32_im2col_mma_convolution_matches_reference() -> None:
    rng = np.random.default_rng(0)
    for platform in ("A2", "A3"):
        kernel, config = _conv2d_kernel(platform)
        feature = rng.normal(
            size=(config["groups"] * config["height"] * config["width"],
                  config["channels_per_c0"])
        ).astype("float32")
        weight = rng.normal(
            size=(config["groups"] * config["k_per_group"], 32)
        ).astype("float32")
        expected = _reference(feature, weight, **config)
        actual = kernel(feature, weight)[:config["output_positions"]]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_fp16_convolution_stride_dilation_asymmetric_padding_and_tail() -> None:
    cases = (
        {"height": 7, "width": 8, "stride": (2, 2)},
        {
            "height": 7, "width": 9, "stride": (1, 2),
            "dilation": (2, 1), "padding": (2, 0, 1, 2),
        },
    )
    rng = np.random.default_rng(7)
    for platform in ("A2", "A3"):
        for case in cases:
            kernel, config = _conv2d_kernel(
                platform, dtype="float16", output_channels=16, **case
            )
            feature = rng.normal(size=(
                config["groups"] * config["height"] * config["width"],
                config["channels_per_c0"],
            )).astype("float16")
            weight = rng.normal(size=(
                config["groups"] * config["k_per_group"],
                config["output_channels"],
            )).astype("float16")
            expected = _reference(feature, weight, **config)
            actual = kernel(feature, weight)[:config["output_positions"]]
            np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)


def test_bfloat16_im2col_mma_convolution_matches_reference() -> None:
    rng = np.random.default_rng(11)
    for platform in ("A2", "A3"):
        kernel, config = _conv2d_kernel(
            platform, dtype="bfloat16", height=5, width=7,
            output_channels=16,
        )
        feature = rng.normal(size=(
            config["groups"] * config["height"] * config["width"],
            config["channels_per_c0"],
        )).astype(ml_dtypes.bfloat16)
        weight = rng.normal(size=(
            config["groups"] * config["k_per_group"],
            config["output_channels"],
        )).astype(ml_dtypes.bfloat16)
        expected = _reference(feature, weight, **config)
        actual = kernel(feature, weight)[:config["output_positions"]]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
