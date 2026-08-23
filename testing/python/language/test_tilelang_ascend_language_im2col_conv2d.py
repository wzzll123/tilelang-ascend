# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# T.tile.im2col 实战验收：真实随机 conv2d（非恒等 mma 抽取）。
#
# 与 test_tilelang_ascend_language_im2col.py（恒等抽取）的区别：
# 用真实随机 filter 替换单位矩阵，完整跑 "im2col + mma = conv2d"，
# 与 torch F.conv2d（float64 参考，输入取 fp16 舍入后的值）比对。
#
# 覆盖矩阵（PLAN_conv2d_sample.md §4.1）：
#   base      : H=W=8  cin=16 cout=32 k=3 pad=1 s=1 d=1
#   stride2   : 同上 s=(2,2)
#   dilation2 : 同上 pad=2 d=(2,2)
#   cin32     : cin=32（两个 cin1 块，im2col 分块 pos_k 累积）
#   asym_pad  : pad=(0,1,2,0) 非对称 + s=2（HO/WO 公式含 pt+pb/pl+pr）
#   tail_m    : H=5 W=7（HO*WO=35 非 16 对齐，valid_m 放宽路径）
#   cin8_pad  : cin=8 host 零填到 16（小通道策略：零填而非 channelSize 尾块）
#   cout_tail : cout=24（N 尾块，filter 零填到 32，输出切片）
#
# 布局约定（与恒等测试相同）：C0=16 且 L1 tile N 维==C0 时 zN 分形退化为
# 行主序，GM [C1*H*W, 16] 的 2D copy 即合法 NC1HWC0。filter 排布为
# B[k=(kh,kw,ci), n=co]（c 最快，与 im2col 的 K 序一致）。

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

C0 = 16


def _ceil16(x):
    return (x + 15) // 16 * 16


def _run_conv2d(target, H, W, CIN, COUT, KH, KW,
                stride=(1, 1), dilation=(1, 1), pad4=(1, 1, 1, 1),
                cin_pad=False, seed=0):
    """单 tile 真实卷积：CIN 须为 16 倍数（cin_pad=True 时 host 侧零填）。"""
    pl, pr, pt, pb = pad4
    sh, sw = stride
    dh, dw = dilation
    HO = (H + pt + pb - dh * (KH - 1) - 1) // sh + 1
    WO = (W + pl + pr - dw * (KW - 1) - 1) // sw + 1
    assert HO > 0 and WO > 0

    CIN_K = _ceil16(CIN) if cin_pad else CIN
    assert CIN_K % C0 == 0
    C1 = CIN_K // C0
    M = HO * WO
    M_ROUND = _ceil16(M)
    K1 = KH * KW * C0                    # 单 cin1 块的 im2col K 维
    K = K1 * C1
    N_ROUND = _ceil16(COUT)

    # 结构对齐 catlass BlockConv2d：逐 cin1 块 GM->L1->L0A(im2col)/L0B->mma 累加。
    # 注意：kStartPt/mStartPt 恒为 0——硬件（A2 实测）拒绝非零 K_M_START_POS，
    # 多 cin1 块走 dst/src 指针偏移 + 多次 mma 累加（init 仅首块置位）。
    @tilelang.jit(out_idx=[2], pass_configs=pass_configs, target=target)
    def kernel():
        @T.prim_func
        def main(Fmap: T.Tensor([C1 * H * W, C0], "float16"),
                 Filter: T.Tensor([K, N_ROUND], "float16"),
                 Out: T.Tensor([M_ROUND, N_ROUND], "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                f_l1 = T.alloc_L1([H * W, C0], "float16")
                w_l1 = T.alloc_L1([K1, N_ROUND], "float16")
                T.annotate_layout({
                    f_l1: make_zn_layout(f_l1),
                    w_l1: make_zn_layout(w_l1),
                })
                a_l0 = T.alloc_L0A([M_ROUND, K1], "float16")
                b_l0 = T.alloc_L0B([K1, N_ROUND], "float16")
                c_l0c = T.alloc_L0C([M_ROUND, N_ROUND], "float")
                with T.Scope("C"):
                    for c1 in range(C1):
                        T.copy(Fmap[c1 * H * W:(c1 + 1) * H * W, :], f_l1)
                        T.copy(Filter[c1 * K1:(c1 + 1) * K1, :], w_l1)
                        T.tile.im2col(a_l0, f_l1, (H, W), (KH, KW),
                                      (sh, sw), (dh, dw), (pl, pr, pt, pb),
                                      0, 0, M, K1)
                        T.copy(w_l1, b_l0)
                        T.mma(a_l0, b_l0, c_l0c, init=(c1 == 0))
                    T.copy(c_l0c, Out)

        return main

    torch.manual_seed(seed)
    x = torch.randn(1, CIN, H, W, dtype=torch.float16)          # NCHW
    w = torch.randn(COUT, CIN, KH, KW, dtype=torch.float16)     # OHWI

    # host 布局转换（样例 host 侧要做的两件事的迷你版）：
    # fmap: NCHW -> NC1HWC0（cin 零填到 16 倍数）-> [C1*H*W, 16]。
    # 注意 C1>1 时必须先 view 出 C1 维再 permute：直接 NHWC.view(C1*H*W, 16)
    # 得到的是 [H,W,C1,C0]（c 跨块交错），不是 [C1,H,W,C0]——C1=1 时两者
    # 退化相同，只有多 cin1 块才暴露。
    xk = torch.zeros(1, CIN_K, H, W, dtype=torch.float16)
    xk[:, :CIN] = x
    fmap_nc1hwc0 = xk.view(1, C1, C0, H, W).permute(0, 1, 3, 4, 2) \
        .contiguous().view(C1 * H * W, C0).npu()
    # filter: OHWI -> 按 cin1 块分块堆叠 [c1, kh, kw, c0, co] -> [K, N] 零填 N。
    # 注意：K 序必须与 im2col 分块堆叠一致——块内 (kh,kw,c0) c0 最快、块间 c1 慢序；
    # 若直接 permute(2,3,1,0) 会把 ci 跨块交错（c fastest over full CIN），必然错。
    # filter cin 同步零填到 CIN_K（cin_pad 时 fmap/filter 都要补，贡献为 0）
    wk = torch.zeros(COUT, CIN_K, KH, KW, dtype=torch.float16)
    wk[:, :CIN] = w
    w5 = wk.view(COUT, C1, C0, KH, KW)
    filt = torch.zeros(K, N_ROUND, dtype=torch.float16)
    filt[:, :COUT] = w5.permute(1, 3, 4, 2, 0).contiguous().view(K, COUT)
    filt = filt.npu()

    got = kernel()(fmap_nc1hwc0, filt).cpu()

    # float64 参考（输入取 fp16 舍入值，消除输入舍入差）
    ref = torch.nn.functional.conv2d(
        x.double(), w.double(), stride=stride, dilation=dilation,
        padding=(pt, pl))  # torch padding=(pad_h, pad_w) 对称形式
    # torch 对称 padding 只接受 (ph, pw)；非对称用 F.pad 后 conv
    if not (pl == pr and pt == pb):
        xp = torch.nn.functional.pad(x.double(), (pl, pr, pt, pb))
        ref = torch.nn.functional.conv2d(xp, w.double(), stride=stride,
                                         dilation=dilation)
    ref = ref[0].permute(1, 2, 0).contiguous().view(M, COUT)  # [HO*WO, COUT]

    torch.testing.assert_close(got[:M, :COUT].double(), ref, rtol=1e-3, atol=1e-3)


CASES = [
    dict(name="base",      H=8, W=8, CIN=16, COUT=32, KH=3, KW=3),
    dict(name="stride2",   H=8, W=8, CIN=16, COUT=32, KH=3, KW=3, stride=(2, 2)),
    dict(name="dilation2", H=8, W=8, CIN=16, COUT=32, KH=3, KW=3,
         dilation=(2, 2), pad4=(2, 2, 2, 2)),
    dict(name="cin32",     H=8, W=8, CIN=32, COUT=32, KH=3, KW=3),
    dict(name="asym_pad",  H=8, W=8, CIN=16, COUT=32, KH=3, KW=3,
         stride=(2, 2), pad4=(0, 1, 2, 0)),
    dict(name="tail_m",    H=5, W=7, CIN=16, COUT=16, KH=3, KW=3),
    dict(name="cin8_pad",  H=8, W=8, CIN=8,  COUT=32, KH=3, KW=3, cin_pad=True),
    dict(name="cout_tail", H=8, W=8, CIN=16, COUT=24, KH=3, KW=3),
    dict(name="k5_s1",     H=7, W=7, CIN=16, COUT=16, KH=5, KW=5, pad4=(2, 2, 2, 2)),
    # 注：k5 的 K1=5*5*16=400，M_ROUND*K1*2B 须 < L0A 64KB——H=W=9 会溢出(76800B)。
    # 大 kernel 大 feature map 的 M 切分留给泛化 kernel（tile 层职责）。
    dict(name="k1",        H=8, W=8, CIN=16, COUT=32, KH=1, KW=1, pad4=(0, 0, 0, 0)),
]


@pytest.mark.parametrize("target", ["ascendc"])
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_im2col_conv2d(target, case):
    kw = dict(case)
    kw.pop("name")
    _run_conv2d(target, **kw)


if __name__ == "__main__":
    for case in CASES:
        kw = dict(case)
        name = kw.pop("name")
        _run_conv2d("ascendc", **kw)
        print(f"{name} PASS")
    print("all ascendc PASS")
