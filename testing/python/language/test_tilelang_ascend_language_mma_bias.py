#!/usr/bin/env python3
"""BT bias-init 最小回归:C = A@B + bias(四操作数 Mmad,L0C 从 BT 初始化)。

数据流(对齐 catlass block_conv3d_pingpong_bias / ops-nn conv2d_v2):
bias GM -> L1 -> BT(DataCopy) -> 首个 mma 带 bias 操作数(cmatrixInitVal=false)
-> L0C 预置 bias(沿 M broadcast)-> 正常累加。

修复前预期失败: T.alloc_BT 不存在 / scope shared.bt 未注册。
"""
import torch

import tilelang
import tilelang.language as T
from tilelang.intrinsics.ascend_layout import AscendLayout

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

M, N, K = 64, 64, 64


def build():
    @tilelang.jit(out_idx=[3], pass_configs=pass_configs, target="ascendc")
    def kernel():
        @T.prim_func
        def main(A: T.Tensor([M, K], "float16"),
                 B: T.Tensor([K, N], "float16"),
                 Bias: T.Tensor([1, N], "float"),
                 Out: T.Tensor([M, N], "float")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a_l1 = T.alloc_L1([M, K], "float16")
                b_l1 = T.alloc_L1([K, N], "float16")
                bias_l1 = T.alloc_L1([1, N], "float")
                bias_bt = T.alloc_BT([1, N], "float")
                a_l0 = T.alloc_L0A([M, K], "float16")
                b_l0 = T.alloc_L0B([K, N], "float16")
                c_l0 = T.alloc_L0C([M, N], "float")
                # bias_l1 要线性布局: 默认 zN 分形会把 [1,N] 按 16 列块打散,
                # copy_l1_to_bt 线性读会拿到错位的 bias
                T.annotate_layout({
                    bias_l1: T.Layout([1, N], lambda *args: args,
                                      layout_tag=AscendLayout.kRowMajor.value)
                })
                with T.Scope("C"):
                    T.copy(A, a_l1)
                    T.copy(B, b_l1)
                    T.copy(Bias, bias_l1)
                    T.copy(bias_l1, bias_bt)
                    T.copy(a_l1, a_l0)
                    T.copy(b_l1, b_l0)
                    T.mma(a_l0, b_l0, c_l0, bias=bias_bt)
                    T.copy(c_l0, Out)
        return main

    return kernel()


def main():
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16).npu()
    b = (torch.randn(K, N, dtype=torch.float16) * 0.05).npu()
    bias = torch.randn(N, dtype=torch.float32).npu()  # 官方文档: bias 全程 fp32
    got = build()(a, b, bias.view(1, N)).cpu().double()
    ref = a.cpu().double() @ b.cpu().double() + bias.cpu().double()
    d = (got - ref).abs()
    bad = int((d > 1e-2 + 1e-2 * ref.abs()).sum())
    print(f"max={d.max():.4f} bad={bad}")
    assert bad == 0, "BT bias mma 数值错误"
    print("PASS: BT bias-init mma")


if __name__ == "__main__":
    main()
