"""Small im2col + MMA + fixpipe convolution for NPU or CPU simulation."""

import argparse

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout
import torch


def ceil16(value):
    return (value + 15) // 16 * 16


def make_kernel(*, simulator=False, platform="A2", trace_path=None):
    height, width = 8, 8
    channels, output_channels = 32, 16
    c0 = 16
    c1 = channels // c0
    kernel_h = kernel_w = 3
    padding = 1
    output_h, output_w = height, width
    m = output_h * output_w
    m_round = ceil16(m)
    k_per_c1 = kernel_h * kernel_w * c0
    k = k_per_c1 * c1

    jit_options = {}
    if simulator:
        jit_options = {
            "simulator": True,
            "platform": platform,
            "sim_config": {"trace_path": trace_path} if trace_path else None,
        }

    @tilelang.jit(out_idx=[2], **jit_options)
    def conv2d():
        @T.prim_func
        def main(
            feature: T.Tensor([c1 * height * width, c0], "float16"),
            weight: T.Tensor([k, output_channels], "float16"),
            output: T.Tensor([m_round, output_channels], "float32"),
        ):
            with T.Kernel(1, is_npu=True) as (_cid, _vid):
                feature_l1 = T.alloc_L1([height * width, c0], "float16")
                weight_l1 = T.alloc_L1([k_per_c1, output_channels], "float16")
                T.annotate_layout({
                    feature_l1: make_zn_layout(feature_l1),
                    weight_l1: make_zn_layout(weight_l1),
                })
                feature_l0a = T.alloc_L0A([m_round, k_per_c1], "float16")
                weight_l0b = T.alloc_L0B([k_per_c1, output_channels], "float16")
                accumulator = T.alloc_L0C([m_round, output_channels], "float32")
                with T.Scope("C"):
                    for block in T.serial(c1):
                        T.copy(
                            feature[block * height * width:(block + 1) * height * width, :],
                            feature_l1,
                        )
                        T.copy(
                            weight[block * k_per_c1:(block + 1) * k_per_c1, :],
                            weight_l1,
                        )
                        T.tile.im2col(
                            feature_l0a,
                            feature_l1,
                            (height, width),
                            (kernel_h, kernel_w),
                            (1, 1),
                            (1, 1),
                            (padding, padding, padding, padding),
                            0,
                            0,
                            m,
                            k_per_c1,
                        )
                        T.copy(weight_l1, weight_l0b)
                        T.mma(
                            feature_l0a,
                            weight_l0b,
                            accumulator,
                            init=(block == 0),
                        )
                    T.copy(accumulator, output)

        return main

    return conv2d(), (height, width, channels, output_channels, m, m_round, k)


def main():
    parser = argparse.ArgumentParser(description="TileLang Ascend im2col convolution")
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--platform", choices=["A2", "A3"], default="A2")
    parser.add_argument("--trace", default=None)
    args = parser.parse_args()

    tilelang.disable_cache()
    kernel, shape = make_kernel(
        simulator=args.simulator, platform=args.platform, trace_path=args.trace
    )
    height, width, channels, output_channels, m, _m_round, k = shape
    torch.manual_seed(0)
    feature_nchw = torch.randn(1, channels, height, width, dtype=torch.float16)
    weight_oihw = torch.randn(
        output_channels, channels, 3, 3, dtype=torch.float16
    )
    c1 = channels // 16
    feature = feature_nchw.view(1, c1, 16, height, width).permute(
        0, 1, 3, 4, 2
    ).contiguous().view(c1 * height * width, 16)
    weight = weight_oihw.view(output_channels, c1, 16, 3, 3).permute(
        1, 3, 4, 2, 0
    ).contiguous().view(k, output_channels)
    if not args.simulator:
        feature = feature.npu()
        weight = weight.npu()

    output = kernel(feature, weight)
    reference = torch.nn.functional.conv2d(
        feature_nchw.float(), weight_oihw.float(), padding=1
    )[0].permute(1, 2, 0).reshape(m, output_channels)
    torch.testing.assert_close(
        output.cpu()[:m], reference, rtol=1e-2, atol=1e-2
    )
    print(
        f"Test Passed! mode={'simulator' if args.simulator else 'npu'} "
        f"platform={args.platform}"
    )


if __name__ == "__main__":
    main()
