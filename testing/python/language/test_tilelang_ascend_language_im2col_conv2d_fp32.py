# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# fp32 conv2d（im2col+mma 路径 A）回归测试——56_PSA 前置探针转正。
# 验证：fp32 im2col（C0=8, channelSize=8）+ fp32 mma 在 A2 直接可用，
# 精度满足官方 fp32 容差（max_rel 2.35e-5, MERE 7.8e-7 << 2e-5）。
import torch
import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

C0 = 8  # fp32
H = W = 8
CIN = 16       # 2 个 cin1 块（C0=8）
COUT = 32
KH = KW = 3
PAD = 1
C1 = CIN // C0
M = H * W
M_ROUND = (M + 15) // 16 * 16
K1 = KH * KW * C0
K = K1 * C1
N_ROUND = (COUT + 15) // 16 * 16


@tilelang.jit(out_idx=[2], pass_configs=pass_configs, target="ascendc")
def kernel():
    @T.prim_func
    def main(Fmap: T.Tensor([C1 * H * W, C0], "float32"),
             Filter: T.Tensor([K, N_ROUND], "float32"),
             Out: T.Tensor([M_ROUND, N_ROUND], "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            f_l1 = T.alloc_L1([H * W, C0], "float32")
            w_l1 = T.alloc_L1([K1, N_ROUND], "float32")
            T.annotate_layout({f_l1: make_zn_layout(f_l1),
                               w_l1: make_zn_layout(w_l1)})
            a_l0 = T.alloc_L0A([M_ROUND, K1], "float32")
            b_l0 = T.alloc_L0B([K1, N_ROUND], "float32")
            c_l0c = T.alloc_L0C([M_ROUND, N_ROUND], "float")
            with T.Scope("C"):
                for c1 in range(C1):
                    T.copy(Fmap[c1 * H * W:(c1 + 1) * H * W, :], f_l1)
                    T.copy(Filter[c1 * K1:(c1 + 1) * K1, :], w_l1)
                    T.tile.im2col(a_l0, f_l1, (H, W), (KH, KW), (1, 1), (1, 1),
                                  (PAD, PAD, PAD, PAD), 0, 0, M, K1)
                    T.copy(w_l1, b_l0)
                    T.mma(a_l0, b_l0, c_l0c, init=(c1 == 0))
                T.copy(c_l0c, Out)
    return main


torch.manual_seed(0)
x = torch.randn(1, CIN, H, W, dtype=torch.float32)
w = torch.randn(COUT, CIN, KH, KW, dtype=torch.float32)
fmap = x.view(1, C1, C0, H, W).permute(0, 1, 3, 4, 2).contiguous().view(C1 * H * W, C0).npu()
w5 = w.view(COUT, C1, C0, KH, KW)
filt = w5.permute(1, 3, 4, 2, 0).contiguous().view(K, COUT)
filt = torch.nn.functional.pad(filt, (0, N_ROUND - COUT)).npu()

got = kernel()(fmap, filt).cpu().double()
ref = torch.nn.functional.conv2d(x.double(), w.double(), padding=PAD)[0].permute(1, 2, 0).reshape(M, COUT)

d = (got[:M, :COUT] - ref).abs()
rel = d / ref.abs().clamp(min=1e-6)
print(f"max_abs={d.max().item():.3e}  max_rel={rel.max().item():.3e}  mean_rel={rel.mean().item():.3e}")
# 官方 fp32 口径
thr = 2e-5 + 2 ** -13 * ref.abs()
bad = (d > thr).sum().item()
print(f"official fp32 tol: bad={bad}/{d.numel()} matched_ratio={1 - bad / d.numel():.4f} (req>=0.9)")
print(f"MERE={d.mean().item():.3e} (rel_thr=2e-5)")
