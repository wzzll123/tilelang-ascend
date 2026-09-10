# 提案：sync_only 校验器补 GM 可见性窗口语义（MTE3 写 GM → MTE2 读 GM）

**状态**：已实施（2026-09-10）
**提出日期**：2026-09-10
**范围**：`tilelang/simulator/sync.py` 的静态校验（`validate_memory_synchronization`），可选扩展 `executor.py` 执行期语义
**背景代码基线**：本仓 `ascendc_pto` 分支 HEAD（≥ da774be4）

**实施结果**：在统一的 `hazard_check=error|warn|off` 策略下新增 GM/workspace RAW
窗口诊断、同相位 `MTE3_MTE2` 特例、跨 phase 读侧 MTE2/ALL drain 规则及 T1–T7
测试。全量 simulator 回归 612 passed。C1 三分支 clean；归档 MSD 设计已包含 D54
读侧 MTE2 barrier，测试中回退该 barrier 后 `xmax_sum` 负例会被正确检出。
跨核规则可校验显式诊断边；bridge 为避免把不同 core 的执行顺序错误串行化，仍不把
跨核 GM 访问加入 scheduler dependency。

2026-09-10 在 NPU3 `wzz_cann` 的 Ascend 910B3（逻辑卡 0、物理卡 2）重新进行了
独立 A/B：两份扩展从同一归档源码重新编译，broken 版本只删除 D54 读前的一个
`PipeBarrier<PIPE_MTE2>()`。输入为 M=256/N=7168/K=14336 的全 1 fp16 x、int8
weight、scale 和 offset，精确期望为 28672。fixed 连续 10/10 次所有元素均为
28672；broken 连续 10/10 次均出现 14336，输出均值在 21068–21632 间波动。
因此 A2 规则保留，但不设独立公开开关，并且在 A3 未有同等实证前不向 A3 外推。

---

## 1. 问题陈述

sync_only 静态校验目前对**同核跨 pipe 内存边**的 fence 判据是（`sync.py::validate_memory_synchronization`）：

> 程序序上 producer→consumer 之间存在 ① PIPE_ALL / ② 同 pipe FIFO / ③ 同 lane set/wait flag 对（`_has_local_flag_fence`）/ ④ mode-0 全核跨核屏障（`_has_cross_collective_fence`），即视为有序，放行。

这套判据对 **UB/L1/L0 上的边**与真机一致（pipe 有序 + flag 完成序足够）。但对 **GM（global scope）上的边**，真机实证表明它不成立——存在"程序序/fence 看似完备、真机仍读到陈旧数据"的**可见性窗口**：

- MTE3 写 GM 后，写数据需排干写 pipe 并经 L2 才对后续读可见；
- 消费端 MTE2 读 GM 可能命中**写入前已被填充的陈旧 L2 行**，或被提前发射；
- 真机二分实证：**跨相位的 GM MTE3→MTE2 边，set/wait flag 对（含 MTE3_MTE2）不保证可见，读端 `PipeBarrier<PIPE_MTE2>`（或 PIPE_ALL）才保证**。

校验器缺失这条语义 → 这类 bug sync_only 全部放行，只能真机/隐藏集暴露（已发生两次，见 §2）。

## 2. 真机证据链（必须精读，含完整二分数据）

来源：`plugins-community/tilelang2ascendc-ops-generator/workflows/templates/archive_tasks/weight_quant_batch_matmul/ascendc/GENERALIZE_DEBUG_LOG.md`

### 2.1 ISSUE-D53（nsub 尾 MTE3_MTE2，工程版 MSD）

- 场景：nsub 循环内 `copy_ub_to_gm`（MTE3 写 Y）→ 下一 nsub 迭代 `copy_gm_to_ub`（MTE2 读 scale/offset/bias/cs32）。
- 删 PIPE_ALL 后 bf16 split-K 多 nl case（13/17/20）**全局性数值错**。
- 逐 pipe 二分：`PipeBarrier<PIPE_V>` / `PIPE_MTE2` / `PIPE_MTE3` 单 pipe 均 17-18/20（**三单 pipe 组合 ≠ PIPE_ALL**——PipeBarrier 只做同 pipe 内排序，不建跨 pipe 依赖）。
- 修复：`SetFlag+WaitFlag<MTE3_MTE2>(2)` 跨 pipe 同步点（set+wait 同 nsub 迭代内相邻完成，pipeline barrier 模式）→ 20/20。
- **关键反差点**：DSL 编译产物（T.copy 序列）同位置**无此 flag 却真机 20/20**——生成码时序恰好有序。静态校验若对此误报，需要抑制手段（见 §4.4）。

### 2.2 ISSUE-D54（xsum 跨阶段 GM/L2 一致性，隐藏集 _56/_60/_91）

- 场景：阶段0 量化段 MTE3 写 `xmax_sum`→GM workspace；阶段2 dequant 段 MTE2 读回。**写读之间隔着多个 mode-0 全核跨核屏障**（SIG_VALL_Q / SIG_VQUANT / SIG_ACC）。
- 现象：dequant 读到恒 0（写入前的初始值），offset 外积项整体丢失，MERE 0.02-0.03。
- 逐 pipe 二分（读端插入，全真机）：
  | 插入物 | 结果 |
  |---|---|
  | `PipeBarrier<PIPE_V>` | 无效 |
  | `PipeBarrier<PIPE_MTE3>`（写侧排干） | **无效** |
  | `SetFlag<MTE3_MTE2>(7)`（set 在量化末 / wait 在 dequant 开头） | **无效** |
  | `PipeBarrier<PIPE_MTE2>`（读侧） | **有效** |
  | `PipeBarrier<PIPE_ALL>`（读侧） | 有效 |
- 根因定性：跨相位 GM 边的可见性需要**读侧 MTE2 排干**；写侧排干、跨 pipe flag、全核屏障**单独均不足**。
- 历史：gen 码原有 PIPE_ALL（隐含此防护），PIPE_ALL 清零战役中被删 → 隐藏集暴露。

### 2.3 两条数据点的统一经验规则

| 边的形态 | set/wait flag 对 | 全核屏障（mode-0） | 读侧 PIPE_MTE2 / PIPE_ALL |
|---|---|---|---|
| GM 边，set/wait 同相位相邻（D53） | **有效** | — | 有效 |
| GM 边，跨相位（写读间有跨核屏障，D54） | **无效** | **无效** | **有效（唯一）** |

> 注意 2.1 与 2.2 的张力：同为 MTE3→MTE2 GM 边，同相位相邻 flag 有效、跨相位 flag 无效。规则编码时必须区分"set 与 wait 之间是否隔着跨核 collective"。

## 3. 现状代码锚点（修复点）

- `tilelang/simulator/sync.py`
  - `validate_memory_synchronization`（~L90）：主校验循环。当前对 GM/UB 边一视同仁。
  - `_has_local_flag_fence`（~L228）：接受同 lane flag 对——**对 GM 边需收紧**。
  - `_has_cross_collective_fence`（~L186）：接受 mode-0 全核屏障作跨 pipe fence——**对 GM 边需拒绝**（D54 实证无效）。该函数注释自述是为"量化段 MTE3 写 GM → dequant MTE2 读"边加的白名单，而 D54 恰好证明这条白名单对 GM 是错的。
  - `_shared_buffer_name`（~L151）：已能取共享 buffer 名；**`BufferRegion.scope`（`program.py:297`）已携带 `MemoryScope`**，可直接判 GM（`MemoryScope` 的 global/gm 枚举值，以 `memory.py::_SHARED_SCOPES` 为准）——无需改 bridge 即可分类边的 scope。
- `tilelang/simulator/adapter.py`：`validate_sync` 解绑 auto_sync 的适配（手写同步 kernel 也走 hazard 检查）——新规则要在这里同步生效。
- 测试目录：`testing/python/simulator/`（已有 `test_cross_core_gm_barrier.py` 可作模式参考）。

## 4. 设计要求

### 4.1 新诊断类别

新增 `HazardDiagnostic.kind = "gm-visibility-window"`，与既有 `missing-pipe-synchronization` 区分（报文需说明"GM 边需要读侧 MTE2 排干，flag/屏障不足"并引用本提案 §2.3 表）。

### 4.2 核心规则（在 `validate_memory_synchronization` 中实现）

对每条跨 pipe 内存依赖边（producer.pipe != consumer.pipe），若共享 buffer `scope` 为 GM（global）：

1. 必须是**写→读边**（producer=MTE3 类写 / consumer=MTE2 类读，含 copy 方向推导；读→写 WAR 与写→写 WAW 的 GM 规则另列，本期可先只对 RAW 生效并在代码中 TODO 标注）。
2. 在 producer→consumer 的程序序区间内扫描 fence：
   - **接受**：`PipeBarrier<PIPE_ALL>`；或 `PipeBarrier<PIPE_MTE2>` 出现在**读侧**（consumer 所属 lane/core，且在最后一个跨核 collective 之后——若区间内无 collective 则只要求在 consumer 之前）。
   - **接受（同相位特例）**：区间内**无任何跨核 collective**，且存在匹配的 set/wait flag 对（src=MTE3，dst=MTE2）→ 放行（D53 数据点）。
   - **拒绝**：区间内存在跨核 collective 但读侧无 MTE2 排干 → 报 `gm-visibility-window`（D54 数据点）。
3. UB/L1/L0 scope 的边：维持现有判据不变（不允许本改动引入任何 UB 边回归）。

### 4.3 跨核 GM 边

跨核 GM 写→读（core A MTE3 写 → core B MTE2 读）：现有 mode-0 collective 放行逻辑对 GM 同样收紧——除 collective 外，要求 consumer 核读侧在 collective 之后有 `PIPE_MTE2`/`PIPE_ALL` 排干。

### 4.4 已知良性边的抑制通道

D53 记录：DSL 生成码存在真机有序的 GM 边（T.copy 数据流天然有序），新规则会保守误报。提供两级抑制：
- 不新增公开配置；诊断服从既有 `hazard_check: "error" | "warn" | "off"`。
- 诊断 metadata 中带 producer/consumer task_id 与 buffer 名，允许测试按 buffer 名 allowlist（测试用，不进生产配置）。

### 4.5 （可选，P2）执行期语义

`executor.py` 功能模拟可选建模：GM 写在进入"已排干"状态前，其它核/后续读的可见性未定义——读到未落地写产生 poison + 诊断。**本期不做也可**，sync_only 静态检查是主交付；做则单开 commit。

## 5. 测试计划（test-first，先写红测再实现）

位置 `testing/python/simulator/test_gm_visibility.py`：

| 用例 | 结构 | 期望 |
|---|---|---|
| T1-D53 复现 | nsub 循环：MTE3 写 GM(Y) → 下迭代 MTE2 读 GM(scale)，区间内仅 PIPE_V | error（gm-visibility-window） |
| T2-D53 修复态 | 同上 + 同相位相邻 set/wait MTE3_MTE2 flag（区间内无 collective） | clean |
| T3-D54 复现 | 相位0 MTE3 写 GM(workspace) → mode-0 全核屏障 ×2 → 相位2 MTE2 读同 buffer | error |
| T4-D54 修复态 | 同上 + 读侧（最后 collective 之后、读之前）PipeBarrier<PIPE_MTE2> | clean |
| T5-flag 跨相位无效 | 相位0 写 → set MTE3_MTE2 → 全核屏障 → wait → 读（D54 实测无效组合） | error |
| T6-UB 无回归 | 同结构但 buffer 为 UB scope + flag 对 | clean（现有行为不变） |
| T7-跨核 GM | core0 MTE3 写 GM → mode-0 屏障 → core1 MTE2 读，读侧无排干 | error；加读侧 PIPE_MTE2 后 clean |
| T8-既有套件回归 | 跑全部 `testing/python/simulator/` | 全绿（允许 T8 中既有 GM 边用例按 §4.4 allowlist 调整，逐一记录） |

**真机对拍（有条件则做）**：构造 T3 形态的微型 wqbm 变体（改 `wqbm_msd_vec.h` 读侧 barrier 种类），A2 真机验证"flag 跨相位无效 / PIPE_MTE2 有效"复现 D54 二分结论。无真机则以 D53/D54 已记录数据为准，不阻塞交付。

## 6. 验收标准

1. T1-T8 全绿；既有 `testing/python/simulator/` 套件无未解释回归。
2. wqbm 参考设计全量对拍：对 `weight_quant_batch_matmul` 的 MSD DSL 5 分支 + C1 DSL 3 分支跑 sync_only（hazard=error）：
   - 含 D53/D54 修复（MTE3_MTE2(2) flag、读侧 PIPE_MTE2）的当前态必须 clean；
   - **回退掉 D54 的读侧 PIPE_MTE2 后必须报 gm-visibility-window**（负例验证，证明规则咬住了真 bug 形态）。
3. 不修改 `ascend_sync_insert.cc` 等编译器同步插入 pass（纪律：该文件冻结）。
4. 文档：本提案 §2.3 经验规则表同步进 `docs/a2_a3_simulator_design.md` 的语义章节，标注"真机实证来源 D53/D54"。

## 7. 明确的非目标

- 不建模性能/时延（sync_only 只管正确性）。
- 不改 executor 的数值语义（§4.5 可选项除外）。
- 不处理 GM 的 WAR/WAW 边（代码留 TODO，本期只 RAW）。
- 不动 compiler 侧 AUTO_SYNC（冻结）。

## 8. 风险与注意

- **误报面**：规则是保守过近似（sound over-approximation），DSL 生成码可能新增少量 warn/error——用 §4.4 的 allowlist 逐个审，不许为消报而改宽规则。
- **scope 分类正确性**：`BufferRegion.scope` 的 GM 判定以 `memory.py::_SHARED_SCOPES` 与 bridge 实际标注为准；先在 T6/T7 里验证 scope 流转无损，再依赖它做规则分流。
- 证据全部来自 A2 (910B3) 真机；A3 (910C) 未单独验证，因此实现只在 A2 启用该经验规则。
