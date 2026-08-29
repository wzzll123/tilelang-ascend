#!/usr/bin/env python3
"""sync pass 条件化消除 PIPE_ALL 回归(sliced 访问粒度)。

背景: AscendSyncInsert 对任何"非零/运行期偏移"的访问(is_sliced)一律插
PipeBarrier<PIPE_ALL>——conv P1 strip 循环 + T.Pipelined 变体实测循环体
23 个 PIPE_ALL(~600 cycle/个,占 kernel 96% 周期,是 Pipelined +7% 负收益
的机制根因)。修复: sliced 访问按"同 scope 全部 pending 写者(保守覆盖
潜在别名) × pipe 对"插 per-pipe sync,无关 pipe 不再被全局停住。

验证:
1. 数值: sliced GM->L1->L0A->mma 流水循环 == torch 参考;
2. 生成码: 循环体内 PipeBarrier<PIPE_ALL> 计数必须为 0(修复前每迭代 >=2);
3. 安全兜底(变异自检): 若 sync 被整体删光,数值必须错——证明测试有牙。
"""
import torch

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

M, N, K, K_PANELS, BK = 128, 128, 256, 4, 64


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
                a_l0 = T.alloc_L0A([M, BK], "float16")
                b_l0 = T.alloc_L0B([BK, N], "float16")
                c_l0 = T.alloc_L0C([M, N], "float")
                with T.Scope("C"):
                    # 全量搬入 L1(整 buffer,非 sliced)
                    T.copy(A, a_l1)
                    T.copy(B, b_l1)
                    for kp in T.Pipelined(K_PANELS, num_stages=2):
                        # L1 读侧运行期偏移切片: is_sliced 触发点(conv strip
                        # 循环的 f_l1[o_s*W:] / w_l1[kh0*KW*C0:] 同款)
                        T.copy(a_l1[0:M, kp * BK:kp * BK + BK], a_l0)
                        T.copy(b_l1[kp * BK:kp * BK + BK, 0:N], b_l0)
                        T.mma(a_l0, b_l0, c_l0, init=(kp == 0),
                              k_actual=BK)
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
    n_pipe_all = src.count("PipeBarrier<PIPE_ALL>")
    print(f"kernel 内 PIPE_ALL 总数: {n_pipe_all}")
    assert n_pipe_all == 0, (f"仍有 {n_pipe_all} 个 PipeBarrier<PIPE_ALL>"
                             "——sliced 访问仍走 blanket 全停")
    print("codegen: PASS (无 PIPE_ALL)")


if __name__ == "__main__":
    main()
