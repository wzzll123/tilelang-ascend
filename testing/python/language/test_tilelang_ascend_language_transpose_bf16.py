#!/usr/bin/env python3
"""bf16 transpose 回归：tl::ascend::transpose 的 bf16 硬件路径修复。

背景：tl_templates/ascend/common.h 的 transpose/transpose_block 对
bfloat16_t 显式排除，bf16 落标量 GetValue/SetValue 双重循环——
[32,1024,1024] 转置 msprof 实测 24.2ms（torch 官方 0.102ms，0.004x）。
TransDataTo5HD 是 2B 位模式搬运（官方 transpose_v2 让 bf16 走 half 实例
化），修复后按 uint16_t 位宽实例化。

验证两点：
1. 数值正确（对 torch.permute 按位比对）；
2. 生成码走 TransDataTo5HDImpl（不再出现标量 SetValue 转置循环）。
"""
import os

import torch

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def build_bf16_transpose2d(batch, H, W):
    TH, TW = 64, 64  # UB 砖（16 倍数），逐砖转置
    tasks = batch * (H // TH) * (W // TW)

    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target="ascendc")
    def kernel():
        @T.prim_func
        def main(x: T.Tensor((batch, H, W), "bfloat16"),
                 out: T.Tensor((batch, W, H), "bfloat16")):
            with T.Kernel(tasks, is_npu=True) as (cid, vid):
                ub = T.alloc_ub((TH, TW), "bfloat16")
                ubt = T.alloc_ub((TW, TH), "bfloat16")
                with T.Scope("V"):
                    b = cid // ((H // TH) * (W // TW))
                    rem = cid % ((H // TH) * (W // TW))
                    h0 = (rem // (W // TW)) * TH
                    w0 = (rem % (W // TW)) * TW
                    T.copy(x[b, h0:h0 + TH, w0:w0 + TW], ub)
                    T.tile.transpose(ubt, ub)
                    T.copy(ubt, out[b, w0:w0 + TW, h0:h0 + TH])
        return main

    return kernel()


def main():
    torch.manual_seed(0)
    batch, H, W = 8, 1024, 1024
    x = torch.randn(batch, H, W, dtype=torch.bfloat16)
    kernel = build_bf16_transpose2d(batch, H, W)
    got = kernel(x.npu()).cpu()
    ref = x.permute(0, 2, 1).contiguous()
    bad = int((got != ref).sum().item())
    assert bad == 0, f"bf16 transpose 数值错误: bad={bad}/{ref.numel()}"
    print(f"numeric: PASS (bad=0/{ref.numel()})")

    # 硬件路径行为验证：标量回退在该 shape 为毫秒级（实测修复前
    # [32,1024,1024] 24.2ms，等比 ~6ms），硬件路径 ~0.2ms——以 1ms 为界
    xin = x.npu()
    for _ in range(3):
        kernel(xin)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        kernel(xin)
    end.record()
    torch.npu.synchronize()
    us = start.elapsed_time(end) * 1000 / 20
    print(f"perf: {us:.0f} us/iter")
    assert us < 1000, (
        f"bf16 transpose 仍在标量路径（{us:.0f}us ≥ 1ms）——"
        "tl::ascend::transpose 的 bf16 硬件实例化未生效")
    print("perf: PASS (<1ms，硬件路径)")


if __name__ == "__main__":
    main()
