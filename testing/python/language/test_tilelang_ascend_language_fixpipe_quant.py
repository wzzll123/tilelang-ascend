#!/usr/bin/env python3
"""fixpipe 随路 quant/relu 探针。

验证 L0C(fp32) -> GM(fp16/bf16) 的 T.copy:
1. DSL 不拦截 dtype 不一致的 copy_l0c_to_gm;
2. 生成 Fixpipe 且 quantPre=F322F16/F322BF16(随路转换,精度同 cast);
3. T.copy(..., enable_relu=True) 贯通到 FixpipeParams.reluEn。
"""
import torch

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

M, N, K = 64, 64, 64


def build(out_dtype, en_relu):
    @tilelang.jit(out_idx=[2], pass_configs=pass_configs, target="ascendc")
    def kernel():
        @T.prim_func
        def main(A: T.Tensor([M, K], "float16"),
                 B: T.Tensor([K, N], "float16"),
                 Out: T.Tensor([M, N], out_dtype)):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a_l1 = T.alloc_L1([M, K], "float16")
                b_l1 = T.alloc_L1([K, N], "float16")
                a_l0 = T.alloc_L0A([M, K], "float16")
                b_l0 = T.alloc_L0B([K, N], "float16")
                c_l0 = T.alloc_L0C([M, N], "float")
                with T.Scope("C"):
                    T.copy(A, a_l1)
                    T.copy(B, b_l1)
                    T.copy(a_l1, a_l0)
                    T.copy(b_l1, b_l0)
                    T.mma(a_l0, b_l0, c_l0, init=True, k_actual=K)
                    T.copy(c_l0, Out, enable_relu=en_relu)
        return main

    return kernel()


def run(out_dtype, en_relu, label):
    torch.manual_seed(0)
    tdt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[out_dtype]
    a = torch.randn(M, K, dtype=torch.float16).npu()
    b = (torch.randn(K, N, dtype=torch.float16) * 0.05).npu()
    if en_relu:
        a[0, :] = -a[0, :].abs()  # 第一行确保出负值
    got = build(out_dtype, en_relu)(a, b).cpu()
    ref = (a.cpu().double() @ b.cpu().double())
    if en_relu:
        ref = ref.clamp_min(0)
    ref = ref.to(tdt)
    d = (got.double() - ref.double()).abs()
    bad = int((d > 1e-2 + 1e-2 * ref.double().abs()).sum())
    print(f"{label}: max={d.max():.4f} bad={bad}")
    assert bad == 0, f"{label} failed"


if __name__ == "__main__":
    run("float16", False, "fp32->fp16 随路 quant")
    run("bfloat16", False, "fp32->bf16 随路 quant")
    run("float16", True, "fp32->fp16 随路 quant + relu")
    print("PASS: fixpipe 随路 quant/relu 均可用")
