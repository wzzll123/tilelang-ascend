"""Execute the repository's fp32 im2col/MMA convolution path on CPU."""

import numpy as np

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def _conv2d_kernel(platform):
    height = width = 8
    channels_per_c0 = 8
    channel_groups = 2
    output_channels = 32
    kernel_extent = 3
    output_positions = height * width
    k_per_group = kernel_extent * kernel_extent * channels_per_c0

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
                [channel_groups * height * width, channels_per_c0], "float32"
            ),
            weight: T.Tensor([channel_groups * k_per_group, output_channels], "float32"),
            output: T.Tensor([output_positions, output_channels], "float32"),
        ):
            with T.Kernel(1, is_npu=True):
                feature_l1 = T.alloc_L1([height * width, channels_per_c0], "float32")
                weight_l1 = T.alloc_L1([k_per_group, output_channels], "float32")
                T.annotate_layout({
                    feature_l1: make_zn_layout(feature_l1),
                    weight_l1: make_zn_layout(weight_l1),
                })
                feature_l0a = T.alloc_L0A([output_positions, k_per_group], "float32")
                weight_l0b = T.alloc_L0B([k_per_group, output_channels], "float32")
                accumulator = T.alloc_L0C([output_positions, output_channels], "float32")
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
                            (kernel_extent, kernel_extent), (1, 1), (1, 1),
                            (1, 1, 1, 1), 0, 0, output_positions, k_per_group,
                        )
                        T.copy(weight_l1, weight_l0b)
                        T.mma(feature_l0a, weight_l0b, accumulator, init=(group == 0))
                    T.copy(accumulator, output)

        return main

    return kernel(), (height, width, channels_per_c0, channel_groups, k_per_group)


def _reference(feature, weight, *, height, width, channels_per_c0, groups, k_per_group):
    output = np.zeros((height * width, weight.shape[1]), dtype=np.float32)
    for output_row in range(height):
        for output_col in range(width):
            m = output_row * width + output_col
            for group in range(groups):
                for kernel_row in range(3):
                    for kernel_col in range(3):
                        image_row = output_row + kernel_row - 1
                        image_col = output_col + kernel_col - 1
                        if 0 <= image_row < height and 0 <= image_col < width:
                            image = feature[
                                group * height * width + image_row * width + image_col
                            ]
                            k_start = group * k_per_group + (
                                kernel_row * 3 + kernel_col
                            ) * channels_per_c0
                            output[m] += image @ weight[
                                k_start:k_start + channels_per_c0
                            ]
    return output


def test_fp32_im2col_mma_convolution_matches_reference() -> None:
    rng = np.random.default_rng(0)
    for platform in ("A2", "A3"):
        kernel, shape = _conv2d_kernel(platform)
        height, width, channels_per_c0, groups, k_per_group = shape
        feature = rng.normal(
            size=(groups * height * width, channels_per_c0)
        ).astype("float32")
        weight = rng.normal(
            size=(groups * k_per_group, 32)
        ).astype("float32")
        expected = _reference(
            feature, weight, height=height, width=width,
            channels_per_c0=channels_per_c0, groups=groups, k_per_group=k_per_group,
        )
        np.testing.assert_allclose(kernel(feature, weight), expected, rtol=1e-5, atol=1e-5)
