#!/usr/bin/env python3
"""sync pass 流敏感 if 处理回归(#70 第一步:去 if 前置 PIPE_ALL)。

修复前: AscendSyncInsert 对每个运行期 if 前后各插一次 PipeBarrier<PIPE_ALL>
(conv strip 循环 T.Pipelined 展开后 23 个全停屏障,~600 cycle/个)。
修复后(安全形态): if 前不插——分支内依赖分析带完整历史进分支;
if 后暂恒插(merge 精化留待下轮)。

验证:
1. if 守卫的 copy->mma 链数值 == torch 参考;
2. 生成码: if 前的 PIPE_ALL 消失(总数 <= 分支数,即只剩后置)。
"""
import torch

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

M, N, K = 64, 64, 64


def build():
    @tilelang.jit(out_idx=[2], pass_configs=pass_configs, target="ascendc")
    def kernel():
        @T.prim_func
        def main(A: T.Tensor([M, K], "float16"),
                 B: T.Tensor([K, N], "float16"),
                 Out: T.Tensor([M, N], "float")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a_l1 = T.alloc_L1([M, K], "float16")
                b_l1 = T.alloc_L1([K, N], "float16")
                T.annotate_layout({a_l1: make_zn_layout(a_l1),
                                   b_l1: make_zn_layout(b_l1)})
                a_l0 = T.alloc_L0A([M, K], "float16")
                b_l0 = T.alloc_L0B([K, N], "float16")
                c_l0 = T.alloc_L0C([M, N], "float")
                with T.Scope("C"):
                    if cid >= 0:  # 运行期守卫 if(恒真,纯为造结构)
                        T.copy(A, a_l1)
                        T.copy(B, b_l1)
                        T.copy(a_l1, a_l0)
                        T.copy(b_l1, b_l0)
                        T.mma(a_l0, b_l0, c_l0, init=True, k_actual=K)
                        T.copy(c_l0, Out)
        return main

    return kernel()


def main():
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16).npu()
    b = (torch.randn(K, N, dtype=torch.float16) * 0.05).npu()
    k = build()
    got = k(a, b).cpu().double()
    ref = a.cpu().double() @ b.cpu().double()
    d = (got - ref).abs()
    bad = int((d > 1e-2 + 1e-2 * ref.abs()).sum())
    assert bad == 0, f"数值错误: max={d.max():.4f} bad={bad}"
    print(f"numeric: PASS (max={d.max():.2e})")

    src = k.get_kernel_source()
    n = src.count("PipeBarrier<PIPE_ALL>")
    print(f"PIPE_ALL 总数: {n}(修复前 if 前后各 1 = 2;修复后仅剩后置 1)")
    assert n <= 1, f"if 前置 PIPE_ALL 未消除(共 {n} 个)"
    print("codegen: PASS")


if __name__ == "__main__":
    main()
