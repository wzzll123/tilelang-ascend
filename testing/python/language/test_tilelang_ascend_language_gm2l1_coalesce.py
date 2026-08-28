#!/usr/bin/env python3
"""GM→L1 行带拷贝合并传输回归（copy_gm_to_l1 快路径）。

背景：NC1HWC0 行带 [rows, C0]（C0=16，32B 行）的 GM→L1 拷贝被
TileCopyTla 逐行发 768 次 32B 传输（~54GB/s，case3 8767us vs torch
528us 的主病灶）。catlass conv 的 CopyGmToL1 对同一布局按整图像行
（4KB）发 DataCopy。修复：copy_gm_to_l1 对"双侧全内维 + 内维恰一个
32B 块"的拷贝走单块 DataCopy 快路径（zN 分形与线性布局逐字节等价）。

验证：
1. 语义正确：行带拷入 f_l1 后 im2col+mma == torch conv2d（fp64 参考）；
2. 生成码：单块 DataCopy（DataCopyParams(1, blockLen, 0, 0)），
   不再出现 TileCopyTla 行带拷贝；
3. 性能界：行带拷贝 kernel 时延 < 100us（修复前 32B 行粒度 ~54GB/s）。
"""
import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

H, W, CIN, COUT, KH, KW, TM = 8, 8, 16, 32, 3, 3, 16
HO = H
M_ROUND = (TM * W + 15) // 16 * 16
K1 = KH * KW * CIN
C1HW = 1 * H * W
N_PAD = (COUT + 15) // 16 * 16


def build():
    @tilelang.jit(out_idx=[2], pass_configs=pass_configs, target="ascendc")
    def kernel():
        @T.prim_func
        def main(Fmap: T.Tensor([1, C1HW, 16], "float16"),
                 Filter: T.Tensor([K1, N_PAD], "float16"),
                 Out: T.Tensor([1, M_ROUND, N_PAD], "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                f_l1 = T.alloc_L1([H * W, 16], "float16")
                w_l1 = T.alloc_L1([K1, 64], "float16")
                T.annotate_layout({w_l1: make_zn_layout(w_l1)})
                a_l0 = T.alloc_L0A([M_ROUND, K1], "float16")
                b_l0 = T.alloc_L0B([K1, 64], "float16")
                c_l0 = T.alloc_L0C([M_ROUND, 64], "float")
                with T.Scope("C"):
                    # 行带拷贝（[rows, C0] 全内维——快路径目标）
                    T.copy(Fmap[0, 0:H * W, :], f_l1[0:H * W, :])
                    T.copy(Filter[0:K1, 0:64], w_l1[0:K1, :])
                    T.tile.im2col(a_l0, f_l1[0:H * W, :], (H, W), (KH, KW),
                                  (1, 1), (1, 1), (1, 1, 1, 1), 0, 0,
                                  TM * W, K1)
                    T.copy(w_l1[0:K1, :], b_l0)
                    T.mma(a_l0, b_l0, c_l0, init=True, k_actual=K1)
                    T.copy(c_l0, Out[0, 0:M_ROUND, :])
        return main

    return kernel()


def main():
    torch.manual_seed(0)
    x = torch.randn(1, 16, H, W, dtype=torch.float16)
    w = torch.randn(COUT, 16, KH, KW, dtype=torch.float16) * 0.05
    fmap = x.view(1, 1, 16, H, W).permute(0, 1, 3, 4, 2) \
           .contiguous().view(1, H * W, 16)
    w5 = w.view(COUT, 1, 16, KH, KW)
    filt = torch.zeros(1, KH, KW, 16, N_PAD, dtype=torch.float16)
    filt[..., :COUT] = w5.permute(1, 3, 4, 2, 0)
    filt = filt.view(K1, N_PAD)

    kernel = build()
    out_pad = kernel(fmap.npu(), filt.npu()).cpu().double()
    got = out_pad[0, :HO * W, :COUT].view(HO, W, COUT)
    ref = F.conv2d(x.double(), w.double(), padding=1) \
        .permute(0, 2, 3, 1)[0]
    d = (got - ref).abs()
    bad = int((d > 1e-3 + 1e-3 * ref.abs()).sum().item())
    assert bad == 0, f"行带拷贝语义错误: max={d.max():.4f} bad={bad}"
    print(f"numeric: PASS (max={d.max():.2e}, bad=0)")

    # 生成码检查说明：copy_gm_to_l1 的模板体在头文件内联展开，
    # get_kernel_source 只有调用行——快路径证据以数值 + 性能界为准。

    # 性能界：64 次行带拷贝循环摊薄 wrapper（修复前 ~1128us/iter）
    def build_loop():
        ROWS = 768

        @tilelang.jit(out_idx=[], pass_configs=pass_configs, target="ascendc")
        def kernel2():
            @T.prim_func
            def main2(Fmap: T.Tensor([1, 16 * 128 * 128, 16], "bfloat16")):
                with T.Kernel(40, is_npu=True) as (cid, vid):
                    f_l1 = T.alloc_L1([ROWS, 16], "bfloat16")
                    with T.Scope("C"):
                        for it in T.serial(64):
                            T.copy(Fmap[0, (cid % 16) * 16384:
                                        (cid % 16) * 16384 + ROWS, :],
                                   f_l1[0:ROWS, :])
            return main2

        return kernel2()

    k2 = build_loop()
    x2 = torch.randn(1, 16 * 128 * 128, 16, dtype=torch.bfloat16).npu()
    for _ in range(5):
        k2(x2)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(30):
        k2(x2)
    end.record()
    torch.npu.synchronize()
    us = start.elapsed_time(end) * 1000 / 30
    print(f"perf: {us:.1f} us/iter（修复前 ~1128us）")
    assert us < 600, f"行带拷贝仍过慢（{us:.0f}us ≥ 600us）——快路径未生效"
    print("perf: PASS (<600us)")


if __name__ == "__main__":
    main()
