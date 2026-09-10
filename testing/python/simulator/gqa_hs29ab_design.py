#!/usr/bin/env python3
# NOTE: frozen fixture for test_gqa_hs29ab_variant.py -- snapshot of the GQA
# HS29 A+B design variant (per-kernel slot-credit preset + C0 preload reorder
# + no task-tail drain) that deadlocks FLAKILY on real A2 hardware while this
# simulator reports it healthy. Do NOT "fix" or sync it with the production
# design; its divergence is the test's purpose. Source:
# plugins-community/tilelang2ascendc-ops-generator/workflows/templates/
# archive_tasks/gqa/design/tile_level/gqa_hs29ab.py (2026-09-10).
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may obtain a copy of this software and associated documentation files (the "License"), under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------

# Tile-level TileLang design for GQA（cann-bench level4/gqa，BSND 布局），per-shape 版。
#
# 结构骨架见 design/block_level/gqa.py（两文件除本文件的 tile-level 填充外一致）。
# 要点：
#   - G 折叠进 M：task=(bz, nkv, s_blk, g_blk)，M 行 = bs×bg head-major
#     （行 r = g_local*bs + s_local），K/V tile 一次 load 被 bs*bg 行复用。
#   - ring 流水 prelaunch=2/ring_slots=3（FA archive 母本），三面 cross flag。
#   - causal：diffS 循环裁剪（kv_end）+ 对角块 UB 现算 mask（逐行 compare+select）。
#   - D>128：d_chunks 路独立 mma(init=True) 各落 ws_s chunk 槽，V1 相加
#     （规避 pto 后端 kv>64&dim>64 多 chunk mma 累加崩溃 M2）；PV 按 N chunk 写 ws_o。
#   - per-shape 编译：dim/bs/bg/block_m 编译期常量，buffer 精确分配；
#     最终交付物为手写泛化 AscendC kernel（ascendc/gqa.cpp，运行期 shape）。

import os
from dataclasses import dataclass

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout
from tvm.tir import BufferRegion
from tvm.ir import Range


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    # 手写同步波次：关 AUTO_SYNC，核内 pipe 同步全手写（C 侧 K/V 双槽预取
    # 槽级 flag 重叠 + V 侧逐字节复刻 AUTO_SYNC 生成码同步序列 + ring 顶
    # 定向 flag 替代 PIPE_ALL）。AUTO_SYNC 是全局开关不能 per-scope（wqbm
    # ISSUE-D27），故 C/V 两侧一并手写。蓝本：wqbm design/tile_level/
    # weight_quant_batch_matmul.py（C 侧双槽）+ gqa P4-only 生成码（V 侧复刻）。
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

# AIC<->AIV workspace 交接信号（ring_slots 组，按槽分 id）。
# 手写同步去掉了 ring 顶 T.barrier_all()（AUTO_SYNC 过渡态，阻断 P2 重叠）。
# 原单一 id 的 cross flag 是电平语义，C 核超前 V 核时同 id 被覆盖错位
# （ISSUE-HS3：kv_end>=3 ring 回绕后 V 读到错位的 ws_s/ws_p/ws_o）。
# 改为按 ring slot 分 id：同 slot 的生产者/消费者配对，ring_slots=prelaunch+1
# 保证 C 超前 V <= prelaunch 迭代、同 slot 复用前消费者已消费，电平不覆盖。
#   SIG_S: C1(t) 写 ws_s[t%3] -> V1(t) 读，id = t%3         (0/1/2)
#   SIG_P: V1(t) 写 ws_p[t%3] -> C2(t) 读，id = 3 + t%3     (3/4/5)
#   SIG_O: C2(t) 写 ws_o[now_k%3] -> V2(t) 读，id = 6 + now_k%3 (6/7/8)
def SIG_S_READY(slot): return slot        # 0/1/2
def SIG_P_READY(slot): return 3 + slot    # 3/4/5
def SIG_O_READY(slot): return 6 + slot    # 6/7/8

MAX_CORES = 20   # grid 固定物理核数；task_idx < block_num 守卫

MASK_NEG = -2 ** 30   # softmax 前 mask 填充值（FA 同款，非 -inf）

# 模拟构建开关（gqa_sim.py 置 1）：剔除模拟器 bridge 不可表达的数据依赖标量
# if（HS23 F217 全 mask 行保护，同步无关），生产/真机路径不受影响。
_SIM_BUILD = os.environ.get("GQA_SIM") == "1"


@dataclass(frozen=True)   # frozen: tilelang.jit 需要对 cfg 取 hash 做编译缓存键
class GQAConfig:
    batch: int
    heads_q: int
    heads_kv: int
    q_seq_len: int
    kv_seq_len: int
    dim: int
    causal: bool = False
    dtype: str = "float16"


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs, target='ascendc')
def gqa_fwd(cfg: GQAConfig):
    batch = cfg.batch
    heads_q = cfg.heads_q
    heads_kv = cfg.heads_kv
    q_seq_len = cfg.q_seq_len
    kv_seq_len = cfg.kv_seq_len
    dim = cfg.dim
    causal = cfg.causal
    dtype = cfg.dtype

    assert heads_q % heads_kv == 0, "N_q % N_kv != 0"
    group = heads_q // heads_kv
    assert not (causal and q_seq_len > kv_seq_len), "causal requires S <= S_kv"

    # C0 对齐（32B/elem：fp16/bf16→16）
    c0 = 16
    dim_align = ((dim + c0 - 1) // c0) * c0

    block_n = 128
    cube_k = min(dim_align, 128)
    d_chunks = (dim_align + 127) // 128
    prelaunch = 2
    ring_slots = prelaunch + 1
    accum_dtype = "float"

    # G 折叠进 M 的行预算：UB 驻留状态（acc_o 等）随 dim 增长，行预算随 dim_align
    # 降档以守住 UB/L0A/L1 容量（HS21 D 扩展：支持到 dim_align=512）。
    #   dim_align<=128 -> 128 (d_chunks=1); <=256 -> 64 (dc=2); <=512 -> 32 (dc=3/4)
    # D=384 (dc=3, bm=32): L1=416KB L0A_Q=24KB UB=95.6KB 全 OK（K/V 双槽保留）。
    # D=512 (dc=4, bm=32): K/V 双槽 L1=552KB 超限 -> kv_slots=1 单槽（L1=296KB OK）。
    if dim_align <= 128:
        m_budget = 128
    elif dim_align <= 256:
        m_budget = 64
    else:
        m_budget = 32
    # K/V L1 槽数：dim_align>=512 时双槽 L1 超 512KB，退化为单槽（关闭 P2 预取
    # 重叠，容量受限下的功能优先取舍）。槽索引 %kv_slots；单槽时恒为槽 0。
    kv_slots = 1 if dim_align >= 512 else 2
    bs = min(m_budget, q_seq_len)
    bg = min(group, max(1, m_budget // bs))
    block_m = ((bs * bg + 15) // 16) * 16   # M 凑 16 对齐 cube 分形
    bm2 = block_m // 2                       # 每个 AIV 子核处理的行数

    s_blks = (q_seq_len + bs - 1) // bs
    g_blks = (group + bg - 1) // bg
    block_num = batch * heads_kv * s_blks * g_blks
    used_core_num = MAX_CORES
    tasks_per_core = (block_num + used_core_num - 1) // used_core_num
    kv_loops = (kv_seq_len + block_n - 1) // block_n

    diff_s = kv_seq_len - q_seq_len   # causal 右下对齐偏移（非 causal 不使用）

    # BSND 布局 shape
    q_shape = [batch, q_seq_len, heads_q, dim]
    kv_shape = [batch, kv_seq_len, heads_kv, dim]
    out_shape = [batch, q_seq_len, heads_q, dim]

    tail_valid = kv_seq_len % block_n

    # epilogue 行带结构：bs >= bm2 时每半核是 band 内子段；bm2 % bs == 0 时每半核
    # 持整 band。两情形都给出静态 extent 的 per-band 拷贝；其余配置（不会出现于
    # 本任务的配置生成规则）由断拦截住。
    bands_per_half = bm2 // bs if bm2 % bs == 0 else 0
    sub_per_band = bs // bm2 if bs % bm2 == 0 else 0

    @T.prim_func
    def main(
        q: T.Tensor(q_shape, dtype),
        k: T.Tensor(kv_shape, dtype),
        v: T.Tensor(kv_shape, dtype),
        output: T.Tensor(out_shape, dtype),
        workspace_s: T.Tensor([block_num, ring_slots, d_chunks, block_m, block_n], accum_dtype),
        workspace_p: T.Tensor([block_num, ring_slots, block_m, block_n], dtype),
        workspace_o: T.Tensor([block_num, ring_slots, block_m, dim_align], accum_dtype),
        sm_scale: T.float32,
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # ---- L1 缓冲（Q 常驻 task 级；K/V 每 kv 块缓存；dim 维按 dim_align）----
            q_l1 = T.alloc_L1([block_m, dim_align], dtype)
            # P2：K/V L1 双缓冲（STAGES=2，专家 l1BTensor[2] 模式）。双槽 [kv_slots,
            # ...]，预取下一 KV 块到另一槽，GM→L1 的 MTE2 与当前块 mma 的 MTE1 重叠。
            # HS21：dim_align>=512 时 kv_slots=1 单槽（双槽 L1 超容量），P2 预取关闭。
            k_l1 = T.alloc_L1([kv_slots, block_n, dim_align], dtype)
            v_l1 = T.alloc_L1([kv_slots, block_n, dim_align], dtype)
            acc_s_l1 = T.alloc_L1([block_m, block_n], dtype)
            T.annotate_layout({
                q_l1: make_zn_layout(q_l1),
                k_l1: make_zn_layout(k_l1),
                v_l1: make_zn_layout(v_l1),
                acc_s_l1: make_zn_layout(acc_s_l1),
            })

            # Q 驻留 L0A（HS17 任务 1）：lhs_l0 分配 d_chunks 个 chunk 槽
            # （[d_chunks, block_m, cube_k]），ring 外一次性把 Q 的所有 d_chunks
            # 载入 L0A 驻留，ring 内 C1 不再每迭代重载 Q（Q 循环不变）。容量：
            # d_chunks*block_m*cube_k*2B = d128:1*128*128*2=32KB / d256:2*64*128*2
            # =32KB，与 p_l0（[block_m,block_n] 32KB）共存于 64KB L0A（m_budget
            # 不变量：d128 bm=128 / d256 bm=64，恰好各占 L0A 一半）。
            lhs_l0 = T.alloc_L0A([d_chunks, block_m, cube_k], dtype)
            rhs_l0 = T.alloc_L0B([cube_k, block_n], dtype)
            # C2 专用 L0：A=P [block_m, block_n]，B=V 切片 [block_n, cube_k]
            p_l0 = T.alloc_L0A([block_m, block_n], dtype)
            v_l0 = T.alloc_L0B([block_n, cube_k], dtype)
            acc_s_l0c = T.alloc_L0C([block_m, block_n], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_m, cube_k], accum_dtype)

            # ---- V 侧 UB：每个 AIV 子核处理 bm2 行（FA 清单 + causal 两项）----
            acc_o = T.alloc_ub([bm2, dim_align], accum_dtype)      # running O（驻留）
            acc_o_ub = T.alloc_ub([bm2, dim_align], accum_dtype)   # 当前块 O 贡献
            acc_o_half = T.alloc_ub([bm2, dim_align], dtype)
            sumexp = T.alloc_ub([bm2, 1], accum_dtype)       # running sum（驻留）
            m_i = T.alloc_ub([bm2, 1], accum_dtype)          # running max（驻留）
            alpha_ring = T.alloc_ub([ring_slots, bm2, 1], accum_dtype)
            sumexp_i_ring = T.alloc_ub([ring_slots, bm2, 1], accum_dtype)
            acc_s_ub = T.alloc_ub([bm2, block_n], accum_dtype)
            # d_chunks>1 时加和用的临时 buffer；d_chunks==1 时加和循环（T.serial(0)）
            # 不执行、永不引用，若仍分配则 codegen 因"未使用 buffer 无预分配地址"报错。
            # 故 d_chunks==1 时别名为 acc_s_ub（占位，不新分配、不被引用）。
            acc_s_tmp = (T.alloc_ub([bm2, block_n], accum_dtype)
                         if d_chunks > 1 else acc_s_ub)
            acc_s_half = T.alloc_ub([bm2, block_n], dtype)
            m_i_2d = T.alloc_ub([bm2, block_n], accum_dtype)
            alpha_2d = T.alloc_ub([bm2, dim_align], accum_dtype)
            mask_col = T.alloc_ub([block_n], accum_dtype)
            # causal UB 现算 mask：列索引向量 + uint8 比较掩码
            col_idx = T.alloc_ub([block_n], accum_dtype)
            # P4：对角块整 tile 向量化 mask（sub_per_band 情形，半核在单 band 内
            # s_local=h_start+h_i 线性）。行 bound 向量 + broadcast 成 2D +
            # 整 tile compare/select 各一次，替代 bm2 次串行 CompareScalar+Select。
            # bound_2d/col_2d 在泛化层复用 m_i_2d/acc_o_ub（V1 mask 点闲置）；
            # 设计层用独立 buffer，由 MEMORY_PLANNING 按 liveness 自动复用。
            bound_vec = T.alloc_ub([bm2], accum_dtype)          # 行 bound 向量
            col_2d = T.alloc_ub([bm2, block_n], accum_dtype)    # 列索引 broadcast
            bound_2d = T.alloc_ub([bm2, block_n], accum_dtype)  # 行 bound broadcast
            causal_cmp = T.alloc_ub([bm2, block_n], "uint8")    # 整 tile 比较掩码
            # bands_per_half 逐行路径用的单行掩码（select 的 selMask 只接受完整
            # Buffer，不支持 BufferRegion 行切片）
            causal_cmp_row = T.alloc_ub([1, block_n], "uint8")

            # HS29-A 变体（标定用）：per-kernel 一次性预置
            with T.Scope("C"):
                T.set_flag("mte1", "mte2", 0)
                T.set_flag("mte1", "mte2", 1)
                T.set_flag("mte1", "mte2", 2)
                T.set_flag("mte1", "mte2", 3)

            for local_idx in T.serial(tasks_per_core):
                task_idx = cid * tasks_per_core + local_idx
                # TIR 无 continue：用 if 守卫整个 task body
                if task_idx < block_num:
                    g_blk = task_idx % g_blks
                    s_blk = task_idx // g_blks % s_blks
                    nkv = task_idx // (g_blks * s_blks) % heads_kv
                    bz = task_idx // (g_blks * s_blks * heads_kv) % batch

                    s0 = s_blk * bs
                    hq0 = nkv * group + g_blk * bg
                    # causal 循环裁剪：本 task 可见的 kv 块数（静态循环 + 动态守卫）。
                    # 注意：TVM script 的 `if`（即便条件是编译期常量）会引入词法作用域，
                    # 块内定义的变量对块外不可见。故 kv_end 用单条条件表达式无条件赋值，
                    # 避免 "Undefined variable"（causal 为编译期常量，trace 时二选一）。
                    kv_end = (T.min(kv_loops, (s0 + bs - 1 + diff_s) // block_n + 1)
                              if causal else kv_loops)

                    # ---- C0：Q 常驻 L1（per-g splice；越界读编译器零填充）----
                    # 手写同步（关 AUTO_SYNC）。flag id 方案（每 HardEvent 独立 id
                    # 空间；蓝本 wqbm matmul + gqa P4-only 生成码）：
                    #   L1 槽信用（MTE1_MTE2=槽空闲 / MTE2_MTE1=槽数据就绪）：
                    #     K 槽 id 0/1，V 槽 id 2/3，Q 常驻 id 4（C0 自配对）。
                    #   C1 L0：MTE1_M id 2、M_FIX id 3（照抄 P4-only 生成码）。
                    #   C2 L0：MTE1_M id 7、M_FIX id 0（照抄 P4-only 生成码）。
                    #   预置信用在每 task C0 发、task 内消费、task 尾 drain——
                    #   task 内 set/wait 配对闭环，跨 task 无电平残留。
                    #   （HS29 的 per-kernel 预置/C0 重排引入真机 flaky 死锁，
                    #   已回退，见 GENERALIZE_DEBUG_LOG HS29 终态。）
                    with T.Scope("C"):
                        for gg in T.serial(bg):
                            if g_blk * bg + gg < group:
                                T.copy(q[bz, s0: s0 + bs, hq0 + gg, 0:dim],
                                       q_l1[gg * bs: (gg + 1) * bs, :])
                        # HS29-B 变体（标定用）：K/V preload 提到 Q wait 前
                        if kv_end > 0:
                            T.wait_flag("mte1", "mte2", 0)
                            T.copy(k[bz, 0: block_n, nkv, 0:dim], k_l1[0, :, :])
                            T.set_flag("mte2", "mte1", 0)
                            T.wait_flag("mte1", "mte2", 2)
                            T.copy(v[bz, 0: block_n, nkv, 0:dim], v_l1[0, :, :])
                            T.set_flag("mte2", "mte1", 2)
                        T.set_flag("mte2", "mte1", 4)
                        T.wait_flag("mte2", "mte1", 4)
                        # Q 驻留 L0A（HS17 任务 1）：ring 外一次性把所有 d_chunks
                        # 的 Q chunk 从 L1 载入 L0A 的对应 chunk 槽，ring 内 C1 复用
                        # 不再重载。MTE1 写 L0A -> ring 内首次 mma 读 L0A 的握手由
                        # C1 的 MTE1_M(2) 覆盖（Q 驻留写与 K 的 L0B 写同属 MTE1，
                        # 首次 mma 前的 set/wait MTE1_M(2) 同时等两者就绪）。
                        for cc in T.serial(d_chunks):
                            T.copy(q_l1[0, cc * cube_k], lhs_l0[cc, :, :])

                    # ---- V0：running 状态初始化 ----
                    with T.Scope("V"):
                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                        T.tile.fill(m_i, MASK_NEG)

                    # ---- 任务内 KV ring 流水 ----
                    for t in T.serial(kv_loops + prelaunch):
                        # ---------- 生产者：C1(t) / V1(t) ----------
                        if t < kv_end:
                            slot_prod = t % ring_slots

                            with T.Scope("C"):
                                # C1: S = Q @ K^T -> workspace_s[task_idx, slot_prod]
                                # P2 双槽重叠（手写）：K(t) 已在槽 t%kv_slots（ring 前/
                                # 上一轮预取）。先预取 K(t+1) 到槽 (t+1)%kv_slots（MTE2，
                                # 与下方 K(t)→L0B 的 MTE1 及 mma 重叠），再用槽
                                # t%kv_slots 做 mma。HS21：kv_slots==1 时关闭预取。
                                if kv_slots > 1 and t + 1 < kv_end:
                                    T.wait_flag("mte1", "mte2", (t + 1) % kv_slots)  # 槽空闲
                                    T.copy(k[bz, (t + 1) * block_n: (t + 2) * block_n,
                                             nkv, 0:dim], k_l1[(t + 1) % kv_slots, :, :])
                                    T.set_flag("mte2", "mte1", (t + 1) % kv_slots)   # 槽就绪
                                # HS21 单槽（kv_slots==1）：无预取，K(t) 不在片上，
                                # 此处同步加载 K(t) 到槽 0（C0 预载只覆盖 t==0）。
                                if kv_slots == 1 and t > 0:
                                    T.wait_flag("mte1", "mte2", 0)
                                    T.copy(k[bz, t * block_n: (t + 1) * block_n,
                                             nkv, 0:dim], k_l1[0, :, :])
                                    T.set_flag("mte2", "mte1", 0)
                                # 用槽 t%kv_slots 的 K(t)：等其 MTE2 就绪。
                                T.wait_flag("mte2", "mte1", t % kv_slots)
                                for cc in T.serial(d_chunks):
                                    # d_chunks>1 跨 chunk 反向握手（照抄 AUTO_SYNC d256
                                    # 生成码 kernel_fp16_d256_causal.cu L83-96）：
                                    #   cc+1 的 L0 加载（MTE1 写 rhs_l0）vs cc 的 mma
                                    #   （M 读 rhs_l0）-> M_MTE1；cc+1 的 mma（M 写
                                    #   acc_s_l0c）vs cc 的 copy L0C→GM（FIX 读 acc_s_l0c）
                                    #   -> FIX_M。仅 cc>0 wait / cc+1<d_chunks set（条件化
                                    #   防 d_chunks=1 残留；d_chunks=1 靠 ring 尾 barrier）。
                                    # HS17 任务 1（Q 驻留 L0A）：lhs_l0 在 ring 外一次性
                                    # 驻留（C0 后），ring 内不再重写——M_MTE1(4) 现在只
                                    # 保护 rhs_l0（K）的 cc+1 写 vs cc 读（Q 不再参与
                                    # L0 复用竞争）。
                                    if cc > 0:
                                        T.wait_flag("m", "mte1", 4)   # 等 cc-1 mma 读完 L0
                                    # Q 驻留：ring 内不再 copy Q，直接复用 lhs_l0[cc]。
                                    T.copy(k_l1[t % kv_slots, 0, cc * cube_k], rhs_l0, transpose=True)
                                    # L0 加载（MTE1）-> mma（M）：MTE1_M 握手（照抄 P4-only id 2）。
                                    # 注：首次迭代此握手同时等 Q 驻留写（ring 外 MTE1）与
                                    # 本次 K 的 L0B 写就绪。
                                    T.set_flag("mte1", "m", 2)
                                    T.wait_flag("mte1", "m", 2)
                                    if cc > 0:
                                        T.wait_flag("fix", "m", 5)    # 等 cc-1 FIX 读完 acc_s_l0c
                                    # 各 chunk 独立 mma（init=True）+ 立即落 ws_s
                                    # chunk 槽——规避多 chunk 累加的 M2 崩溃模式
                                    # Q 驻留：mma 读驻留的 lhs_l0[cc] 切片。
                                    T.mma(lhs_l0[cc, :, :], rhs_l0, acc_s_l0c, init=True)
                                    if cc + 1 < d_chunks:
                                        T.set_flag("m", "mte1", 4)    # 本 chunk mma 读完 L0
                                    # mma（M）-> L0C→GM（FIX）：M_FIX 握手（照抄 P4-only id 3）。
                                    T.set_flag("m", "fix", 3)
                                    T.wait_flag("m", "fix", 3)
                                    T.copy(acc_s_l0c,
                                           workspace_s[task_idx, slot_prod, cc, :, :])
                                    if cc + 1 < d_chunks:
                                        T.set_flag("fix", "m", 5)     # 本 chunk FIX 读完 acc_s_l0c
                                # HS19 位置 B（C1 跨迭代握手，去条件化 d_chunks==1 也发）：
                                # 本迭代 C1 的 mma 读 rhs_l0 完成（M_MTE1(3)）+ FIX 读
                                # acc_s_l0c 完成（FIX_M(3)），供 ring 尾 wait——替代原
                                # ring 尾 AIC 侧 PIPE_ALL 的跨迭代 L0B/L0C 复用兜底。
                                T.set_flag("m", "mte1", 3)
                                T.set_flag("fix", "m", 3)
                                # 槽 t%kv_slots 的 K(t) 所有 chunk 的 L0B 已读完（MTE1_M 已
                                # wait），释放槽 t%kv_slots 供后续预取复用。
                                T.set_flag("mte1", "mte2", t % kv_slots)
                                T.set_cross_flag("FIX", SIG_S_READY(slot_prod))   # C1(t) 完成 -> V1(t)

                            with T.Scope("V"):
                                # V1: scale/mask/online softmax -> workspace_p + meta
                                # 手写同步（复刻 P4-only 生成码 V1 序列）：ws_s GM→UB
                                # 的 MTE2 后 MTE2_V(4) 握手；V 管 op 链 pipe_barrier("V")
                                # 复刻 AUTO_SYNC 的 PIPE_V；cast 后 V_MTE3(5) 握手再写 P。
                                T.wait_cross_flag(SIG_S_READY(slot_prod))
                                h_start = vid * bm2
                                T.copy(m_i, alpha_ring[slot_prod, :, :])   # 旧 running max 进 slot
                                # HS19 位置 A（acc_s_ub）：排 V 管（V->MTE2）到此刻，
                                # 上一迭代 V1 的 acc_s_ub V 读（exp/reduce/cast）完成
                                # 才允许本迭代 MTE2 写 acc_s_ub。同迭代 set->wait 自配对
                                # （无跨迭代电平，不死锁），安全 id 1（id 0 helper）。
                                T.set_flag("v", "mte2", 1)
                                T.wait_flag("v", "mte2", 1)
                                T.copy(
                                    workspace_s[task_idx, slot_prod, 0,
                                                h_start: h_start + bm2, :],
                                    acc_s_ub,
                                )
                                # ws_s chunk0 的 GM→UB（MTE2）完成 -> 后续 V 计算可读 acc_s_ub。
                                # 握手必须在第一个依赖 ws_s MTE2 的 V 计算之前（d256 的
                                # d_chunks 加和 add 也依赖，不能在 mul 前才握手——ISSUE-HS2）。
                                T.set_flag("mte2", "v", 4)
                                T.wait_flag("mte2", "v", 4)
                                # d_chunks>1：各 chunk 分量和才是 S（QK 的 K 维分段）
                                for cc in T.serial(d_chunks - 1):
                                    T.copy(
                                        workspace_s[task_idx, slot_prod, cc + 1,
                                                    h_start: h_start + bm2, :],
                                        acc_s_tmp,
                                    )
                                    # chunk cc+1 的 GM→UB（MTE2）完成 -> add 读 acc_s_tmp。
                                    T.set_flag("mte2", "v", 4)
                                    T.wait_flag("mte2", "v", 4)
                                    T.tile.add(acc_s_ub, acc_s_ub, acc_s_tmp)
                                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                                # kv 尾块列 mask：PrimExpr 不能用 Python `and`，拆两层动态 if
                                if tail_valid != 0:
                                    if t == kv_loops - 1:
                                        T.tile.fill(mask_col, T.float32(MASK_NEG))
                                        T.tile.fill(
                                            BufferRegion(mask_col, [Range(0, tail_valid)]),
                                            T.float32(0.0),
                                        )
                                        for h_i in range(bm2):
                                            T.tile.add(acc_s_ub[h_i, :], acc_s_ub[h_i, :],
                                                       mask_col)
                                # causal 对角块 mask：UB 现算（无 GM mask 张量）。
                                # 块内末列超出首行可见界才需要 mask。
                                if causal:
                                    if (t + 1) * block_n - 1 > s0 + diff_s:
                                        if sub_per_band > 0:
                                            # P4 整 tile 向量化（sub_per_band：bs>=bm2，
                                            # 半核 bm2 行在单 band 内）。HS24：行 bound 用
                                            # band 内局部行号 s_local=(h_start+h_i)%bs——
                                            # 旧写法 s_local=h_start+h_i（不折返）在 bg>=2
                                            # 且半核落在 band>=1 时（如 S=48,bg=2,bs=bm2）
                                            # 少掩码 band>=1。单 band 半核满足
                                            # (h_start+h_i)%bs==(h_start%bs)+h_i，故线性
                                            # 序列起点取 s0+(h_start%bs)+diff_s。
                                            T.tile.createvecindex(
                                                col_idx, T.cast(t * block_n, "float32"))
                                            T.tile.createvecindex(
                                                bound_vec,
                                                T.cast(s0 + (h_start % bs) + diff_s, "float32"))
                                            # V 管内 RAW 屏障（对齐 AUTO_SYNC L117/119/121/123：
                                            # createvecindex 写 -> broadcast 读、
                                            # broadcast 写 -> compare 读、compare 写 ->
                                            # select 读，缺此致 mask 错位 ws_p 隔列 NaN）。
                                            T.pipe_barrier("V")
                                            # col_2d[r][c]=t*bn+c（沿行 broadcast）；
                                            # bound_2d[r][c]=bound_r（沿列 broadcast）
                                            T.tile.broadcast(col_2d, col_idx, axis=0)
                                            T.pipe_barrier("V")
                                            T.tile.broadcast(bound_2d, bound_vec, axis=1)
                                            T.pipe_barrier("V")
                                            # cmpmask[r][c] = (col_global <= bound_r)
                                            T.tile.compare(
                                                causal_cmp, col_2d, bound_2d, "LE")
                                            T.pipe_barrier("V")
                                            # 整 tile 一次 select：cmpmask 真保留 src0，
                                            # 假填 MASK_NEG
                                            T.tile.select(
                                                acc_s_ub, causal_cmp, acc_s_ub,
                                                T.float32(MASK_NEG),
                                                "VSEL_TENSOR_SCALAR_MODE")
                                        else:
                                            # bands_per_half（bs<bm2，decode/MTP）：
                                            # s_local=r%bs 周期折返，bound 非线性，
                                            # 保留逐行 compare+select（bs 小，非主收益面）。
                                            # selMask 用单行完整 Buffer（select 不支持
                                            # BufferRegion 行切片作 selMask）。
                                            T.tile.createvecindex(
                                                col_idx, T.cast(t * block_n, "float32"))
                                            T.pipe_barrier("V")   # createvecindex 写 -> compare 读
                                            for h_i in range(bm2):
                                                s_local = (h_start + h_i) % bs
                                                bound = T.cast(
                                                    s0 + s_local + diff_s, "float32")
                                                T.tile.compare(
                                                    causal_cmp_row, col_idx,
                                                    bound, "LE")
                                                T.pipe_barrier("V")   # compare 写 -> select 读
                                                T.tile.select(
                                                    acc_s_ub[h_i, :],
                                                    causal_cmp_row,
                                                    acc_s_ub[h_i, :],
                                                    T.float32(MASK_NEG),
                                                    "VSEL_TENSOR_SCALAR_MODE")
                                # online softmax：new_max = max(chunk_max, running_max)
                                # V 管 op 链 pipe_barrier("V") 复刻 AUTO_SYNC 的 PIPE_V。
                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                T.pipe_barrier("V")
                                T.tile.max(m_i, m_i, alpha_ring[slot_prod, :, :])
                                T.pipe_barrier("V")
                                T.tile.sub(alpha_ring[slot_prod, :, :], alpha_ring[slot_prod, :, :], m_i)
                                T.pipe_barrier("V")
                                T.tile.exp(alpha_ring[slot_prod, :, :], alpha_ring[slot_prod, :, :])  # alpha 落 ring
                                T.tile.broadcast(m_i_2d, m_i)
                                T.pipe_barrier("V")
                                T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)
                                T.pipe_barrier("V")
                                T.tile.exp(acc_s_ub, acc_s_ub)
                                T.pipe_barrier("V")
                                T.reduce_sum(acc_s_ub, sumexp_i_ring[slot_prod, :, :], dim=-1)
                                # HS19 位置 A（acc_s_half）：排 MTE3 管（MTE3->V）到此刻，
                                # 上一迭代 V1 的 ws_p copy MTE3 读 acc_s_half 完成才允许
                                # 本迭代 cast V 写 acc_s_half。同迭代 set->wait 自配对，
                                # 安全 id 2（id 0 helper、id 7 task 尾 C）。
                                T.set_flag("mte3", "v", 2)
                                T.wait_flag("mte3", "v", 2)
                                # P 写 workspace_p（dtype）。cast（V）-> P 写 GM（MTE3）：
                                # V_MTE3(5) 握手（照抄 P4-only id 5）。
                                T.copy(acc_s_ub, acc_s_half)
                                T.set_flag("v", "mte3", 5)
                                T.wait_flag("v", "mte3", 5)
                                T.copy(
                                    acc_s_half,
                                    workspace_p[task_idx, slot_prod,
                                                h_start: h_start + bm2, :],
                                )
                                T.set_cross_flag("MTE3", SIG_P_READY(slot_prod))  # V1(t) 完成 -> C2(t)

                        # HS19 位置 A（V1/V2 边界去 PIPE_ALL）：原 AIV-only
                        # T.pipe_barrier("ALL")（ISSUE-HS4 止血遗留）删除，由 V1 体内
                        # 两个定向自配对替代（V1 头 V_MTE2(1) 排 V->MTE2 覆盖
                        # acc_s_ub/alpha_2d 同址跨迭代 WAR；cast 前 MTE3_V(2) 排
                        # MTE3->V 覆盖 acc_s_half 跨迭代 WAR）。AIC 侧此位置无 hazard
                        # 且不排（P2 重叠保留，ISSUE-R4）。边界本身无屏障。

                        # ---------- 消费者：C2(t-2) / V2(t-2) ----------
                        if t >= prelaunch:
                            now_k = t - prelaunch
                            if now_k < kv_end:
                                slot_cons = now_k % ring_slots

                                with T.Scope("C"):
                                    # C2: O = P @ V -> workspace_o[task_idx, slot_cons]
                                    # P2 双槽重叠（手写）：V(now_k) 已在槽 now_k%2（预取）。
                                    # 先预取 V(now_k+1) 到槽 (now_k+1)%2（MTE2，与下方
                                    # V(now_k)→L0B 的 MTE1 及 mma 重叠），再用槽 now_k%2。
                                    T.wait_cross_flag(SIG_P_READY(slot_cons))
                                    T.copy(workspace_p[task_idx, slot_cons, :, :], acc_s_l1)
                                    # ws_p→acc_s_l1 的 GM→L1（MTE2）完成标记（id 5，
                                    # acc_s_l1 单槽专用），供下方 L1→L0A（MTE1）等待。
                                    T.set_flag("mte2", "mte1", 5)
                                    if kv_slots > 1 and now_k + 1 < kv_end:
                                        T.wait_flag("mte1", "mte2", 2 + (now_k + 1) % kv_slots)  # V 槽空闲
                                        T.copy(v[bz, (now_k + 1) * block_n: (now_k + 2) * block_n,
                                                 nkv, 0:dim], v_l1[(now_k + 1) % kv_slots, :, :])
                                        T.set_flag("mte2", "mte1", 2 + (now_k + 1) % kv_slots)   # V 槽就绪
                                    # HS21 单槽（kv_slots==1）：V(now_k) 同步加载到槽 0。
                                    if kv_slots == 1 and now_k > 0:
                                        T.wait_flag("mte1", "mte2", 2)
                                        T.copy(v[bz, now_k * block_n: (now_k + 1) * block_n,
                                                 nkv, 0:dim], v_l1[0, :, :])
                                        T.set_flag("mte2", "mte1", 2)
                                    # 等 acc_s_l1（id 5）与 V 槽 now_k%kv_slots 的 MTE2 就绪。
                                    T.wait_flag("mte2", "mte1", 5)
                                    T.wait_flag("mte2", "mte1", 2 + now_k % kv_slots)  # V 槽就绪
                                    for cc in T.serial(d_chunks):
                                        # d_chunks>1 跨 chunk 反向握手（同 C1，照抄
                                        # AUTO_SYNC d256 C2）：cc+1 L0 加载 vs cc mma
                                        # -> M_MTE1(id 6)；cc+1 mma vs cc FIX 读 acc_o_l0c
                                        # -> FIX_M(id 1)。仅 cc>0 wait / cc+1<d_chunks set。
                                        if cc > 0:
                                            T.wait_flag("m", "mte1", 6)   # 等 cc-1 mma 读完 L0
                                        T.copy(acc_s_l1, p_l0)
                                        T.copy(v_l1[now_k % kv_slots, 0, cc * cube_k], v_l0)
                                        # L0 加载（MTE1）-> mma（M）：MTE1_M 握手（照抄 P4-only id 7）。
                                        T.set_flag("mte1", "m", 7)
                                        T.wait_flag("mte1", "m", 7)
                                        if cc > 0:
                                            T.wait_flag("fix", "m", 1)    # 等 cc-1 FIX 读完 acc_o_l0c
                                        T.mma(p_l0, v_l0, acc_o_l0c, init=True)
                                        if cc + 1 < d_chunks:
                                            T.set_flag("m", "mte1", 6)    # 本 chunk mma 读完 L0
                                        # mma（M）-> L0C→GM（FIX）：M_FIX 握手（照抄 P4-only id 0）。
                                        T.set_flag("m", "fix", 0)
                                        T.wait_flag("m", "fix", 0)
                                        T.copy(
                                            acc_o_l0c,
                                            workspace_o[task_idx, slot_cons, :,
                                                        cc * cube_k: (cc + 1) * cube_k],
                                        )
                                        if cc + 1 < d_chunks:
                                            T.set_flag("fix", "m", 1)     # 本 chunk FIX 读完 acc_o_l0c
                                    # HS19 位置 B（C2 跨迭代握手，去条件化）：本迭代 C2 的
                                    # mma 读 v_l0 完成（M_MTE1(5)）+ FIX 读 acc_o_l0c 完成
                                    # （FIX_M(4)）+ L1→L0A MTE1 读 acc_s_l1 完成
                                    # （MTE1_MTE2(4)，acc_s_l1 单槽跨迭代复用），供 ring 尾
                                    # wait——替代原 ring 尾 AIC 侧 PIPE_ALL。
                                    T.set_flag("m", "mte1", 5)
                                    T.set_flag("fix", "m", 4)
                                    T.set_flag("mte1", "mte2", 4)
                                    # 释放 V 槽 now_k%kv_slots 供后续预取复用。
                                    T.set_flag("mte1", "mte2", 2 + now_k % kv_slots)
                                    T.set_cross_flag("FIX", SIG_O_READY(slot_cons))  # C2 -> V2

                                with T.Scope("V"):
                                    # V2: 读 meta（alpha/sumexp_i），merge 当前 O 贡献
                                    # 手写同步（复刻 P4-only V2）：ws_o GM→UB 的 MTE2 后
                                    # MTE2_V(1) 握手；V 管 op 链 pipe_barrier("V")。
                                    T.wait_cross_flag(SIG_O_READY(slot_cons))
                                    h_start = vid * bm2
                                    # HS19 位置 A（acc_o_ub）：排 V 管（V->MTE2）到此刻，
                                    # 上一迭代 V2 的 add 读 acc_o_ub 完成才允许本迭代
                                    # copy MTE2 写 acc_o_ub（模拟器标记的 add/v ->
                                    # copy_gm_to_ub/mte2 hazard）。同迭代 set->wait
                                    # 自配对，安全 id 2。
                                    T.set_flag("v", "mte2", 2)
                                    T.wait_flag("v", "mte2", 2)
                                    T.copy(
                                        workspace_o[task_idx, slot_cons,
                                                    h_start: h_start + bm2, :],
                                        acc_o_ub,
                                    )
                                    T.tile.broadcast(alpha_2d, alpha_ring[slot_cons, :, :])
                                    T.pipe_barrier("V")
                                    T.tile.mul(acc_o, acc_o, alpha_2d)
                                    # ws_o GM→UB（MTE2）完成 -> acc_o 加其贡献（V）。
                                    T.set_flag("mte2", "v", 1)
                                    T.wait_flag("mte2", "v", 1)
                                    T.pipe_barrier("V")
                                    T.tile.add(acc_o, acc_o, acc_o_ub)
                                    T.tile.mul(sumexp, sumexp, alpha_ring[slot_cons, :, :])
                                    T.pipe_barrier("V")
                                    T.tile.add(sumexp, sumexp, sumexp_i_ring[slot_cons, :, :])

                        # HS19 定向化（ring 尾去 PIPE_ALL）：原 AIC-only
                        # T.pipe_barrier("ALL") 分解为 5 个定向跨迭代消费者 drain
                        # （承重 hazard，HS19-1）：跨迭代 L0B/L0C/acc_s_l1 复用。
                        # 每个 wait 配对本迭代 C1/C2 内的 set（去条件化 d_chunks==1
                        # 也发）。wait 与 set 同条件守卫（电平语义：wait 未 set 的
                        # flag 会读陈旧电平）。只排 M读L0B + FIX读L0C + MTE1读
                        # acc_s_l1 的消费者完成，MTE2 预取（K(t+1)/V(t+1)）与 MTE1
                        # L1→L0A（Q 驻留）不排——P2 重叠保留。AIV 侧无 ring 尾 wait。
                        with T.Scope("C"):
                            if t < kv_end:  # 本迭代跑了 C1(t) -> drain 其消费者
                                T.wait_flag("m", "mte1", 3)   # mma 读 rhs_l0 完成
                                T.wait_flag("fix", "m", 3)    # FIX 读 acc_s_l0c 完成
                            if prelaunch <= t:
                                now_k = t - prelaunch
                                if now_k < kv_end:  # 本迭代跑了 C2(now_k)
                                    T.wait_flag("m", "mte1", 5)   # mma 读 v_l0 完成
                                    T.wait_flag("fix", "m", 4)    # FIX 读 acc_o_l0c 完成
                                    T.wait_flag("mte1", "mte2", 4)  # MTE1 读 acc_s_l1 完成

                    # ---- 收尾：归一化并 per-g 写出（越界/g 尾块由守卫与编译器处理）----
                    with T.Scope("V"):
                        # HS23（F217 全 mask 行保护）：golden 对全 mask 行（scores 全
                        # -inf）softmax 后置 0。本设计用有限 MASK_NEG=-2^30 掩码，全
                        # mask 行会得 mean(V)≠0。全 mask 行判定：running max m_i 从未
                        # 被有效 score 更新（保持 MASK_NEG 初值）。epilogue 把这些行
                        # 的 acc_o 置 0（仅 S>S_kv 违规输入可达；正常输入 m_i≠MASK_NEG，
                        # 为 no-op）。逐行标量判定 + Duplicate 置 0。
                        # 注：数据依赖标量 if 超出模拟器 bridge 能力（fail-closed
                        # UnsupportedSimOpError），模拟构建（GQA_SIM=1）时由 Python
                        # 层剔除本块——同步结构判定不受影响（V 管局部 op）。
                        if causal and not _SIM_BUILD:
                            for row in T.serial(bm2):
                                if m_i[row, 0] == MASK_NEG:
                                    T.tile.fill(acc_o[row, :], 0.0)
                            T.pipe_barrier("V")
                        T.tile.broadcast(alpha_2d, sumexp)
                        T.pipe_barrier("V")
                        T.tile.div(acc_o, acc_o, alpha_2d)
                        T.pipe_barrier("V")
                        T.copy(acc_o, acc_o_half)
                        T.set_flag("v", "mte3", 3)
                        T.wait_flag("v", "mte3", 3)
                        # 行 r = g_local*bs + s_local（head-major），本半核行区间
                        # [vid*bm2, vid*bm2+bm2)。
                        # 注意（编译 hang 修复）：原实现对 sub_per_band 用 Python range 双重
                        # 展开 + `seg_global//segs_per_half==vid` 嵌套 if 守卫，整除+vid 的
                        # 条件与 copy 运行期下界联立，触发 TVM IntervalSetEvaluator 在
                        # MinNode/VarNode 上无限递归（Simplify 卡死）。改为线性下界、
                        # 无 vid 整除守卫的写法：
                        if sub_per_band > 0:
                            # bs % bm2 == 0（bs >= bm2）：本半核 bm2 行落在单 band 内
                            # （bg<=2），g_local/s_local0 直接由 vid 线性算出，单次 copy。
                            g_local = (vid * bm2) // bs
                            s_local0 = (vid * bm2) % bs
                            if g_blk * bg + g_local < group:
                                T.copy(
                                    acc_o_half[:, 0:dim],
                                    output[bz,
                                           s0 + s_local0: s0 + s_local0 + bm2,
                                           hq0 + g_local, 0:dim],
                                )
                        elif bands_per_half > 0:
                            # bm2 % bs == 0（bm2 >= bs）：半核 vid 持 bands_per_half 个
                            # 整 band，每 band 一次 copy（下界 s0 固定，无 vid 整除守卫）。
                            for i_b in range(bands_per_half):
                                band = vid * bands_per_half + i_b
                                if band < bg:
                                    if g_blk * bg + band < group:
                                        T.copy(
                                            acc_o_half[i_b * bs: i_b * bs + bs, 0:dim],
                                            output[bz, s0: s0 + bs,
                                                   hq0 + band, 0:dim],
                                        )
                        else:
                            # 配置生成规则保证不会走到这里
                            T.tile.fill(acc_o_half, 0.0)

                    # ---- task 尾：跨 task 兜底（AIV-only）----
                    # HS19 定向化（task 尾去 PIPE_ALL，对齐 gqa.cpp 已验证形态）：
                    # 承重 hazard 在 AIV 侧——acc_o_half 跨 task WAR（本 task epilogue
                    # 的 copy_ub_to_gm MTE3 读 acc_o_half vs 下一 task epilogue cast
                    # 的 V 写 acc_o_half，ISSUE-5）。改 AIV-only 单个 MTE3_V(7) 自配对
                    # 排空（本 task output MTE3 读完成才进下一 task cast V 写）。
                    # AIC 侧无 acc_o_half hazard 且 ring 尾已排 AIC 管，AIC 跳过。
                    with T.Scope("V"):
                        T.set_flag("mte3", "v", 7)
                        T.wait_flag("mte3", "v", 7)

    return main
