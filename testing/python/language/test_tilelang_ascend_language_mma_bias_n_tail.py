#!/usr/bin/env python3
"""BT bias-init N 非对齐回归: C = A@B + bias，N=255（非 32B 对齐）。

复现 grouped_matmul ISSUE-3 的 bias 路径根因：copy_gm_to_l1_linear 用
DataCopy（32B 块粒度），bias 行宽 tailN*sizeof(float)=255*4=1020B 非 32B
倍数，整数除法 blockLen=31 只拷 992B=248 个 float，末尾 7 个 float
（列 248-254）未拷贝 → BT 尾段脏数据 → 有效输出列 248-254 的 bias 错。

修复前预期失败：列 248-254 与 golden 不符（max 误差 ~bias 量级）。
修复（copy_gm_to_l1_linear 改 DataCopyPad + 尾列零填充）后转绿。
"""
import torch

import tilelang
import tilelang.language as T
from tilelang.intrinsics.ascend_layout import AscendLayout

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

M, K = 64, 64
BLOCK_N = 256  # L0B/BT 的 N 维（编译期 tile）
N = 255        # GM 实际 N（非对齐）


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
                b_l1 = T.alloc_L1([K, BLOCK_N], "float16")
                bias_l1 = T.alloc_L1([1, BLOCK_N], "float")
                bias_bt = T.alloc_BT([1, BLOCK_N], "float")
                a_l0 = T.alloc_L0A([M, K], "float16")
                b_l0 = T.alloc_L0B([K, BLOCK_N], "float16")
                c_l0 = T.alloc_L0C([M, BLOCK_N], "float")
                T.annotate_layout({
                    bias_l1: T.Layout([1, BLOCK_N], lambda *args: args,
                                      layout_tag=AscendLayout.kRowMajor.value)
                })
                with T.Scope("C"):
                    T.copy(A, a_l1)
                    # B 的 N 维切片 [0:N]（N=255 < BLOCK_N=256），尾列零填充
                    T.copy(B[:, 0:N], b_l1)
                    T.copy(Bias[:, 0:N], bias_l1)
                    T.copy(bias_l1, bias_bt)
                    T.copy(a_l1, a_l0)
                    T.copy(b_l1, b_l0)
                    T.mma(a_l0, b_l0, c_l0, bias=bias_bt)
                    # fixpipe 写出钳位回 N=255
                    T.copy(c_l0, Out[:, 0:N])
        return main

    return kernel()


def main():
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16).npu()
    b = (torch.randn(K, N, dtype=torch.float16) * 0.05).npu()
    # bias 用列索引标记，便于定位哪些列错
    bias = (torch.arange(N, dtype=torch.float32) * 0.01).npu()
    got = build()(a, b, bias.view(1, N)).cpu().double()
    ref = a.cpu().double() @ b.cpu().double() + bias.cpu().double()
    d = (got - ref).abs()
    thr = 1e-2 + 1e-2 * ref.abs()
    bad_mask = d > thr
    bad = int(bad_mask.sum())
    print(f"max={d.max():.4f} bad={bad}")
    if bad:
        bad_cols = bad_mask.any(dim=0).nonzero().flatten().tolist()
        print(f"bad cols ({len(bad_cols)}): {bad_cols[:20]}{'...' if len(bad_cols)>20 else ''}")
        # 期望：修复前坏列集中在尾部（248-254，DataCopy 32B 截尾区）
    assert bad == 0, f"BT bias N 非对齐数值错误，坏列={bad_cols if bad else []}"
    print("PASS: BT bias-init mma N 非对齐")


if __name__ == "__main__":
    main()
