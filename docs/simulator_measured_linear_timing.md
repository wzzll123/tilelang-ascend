# 实测一次函数 timing model

目标：逐操作实测确定 `t_ns = a * x + b`，而不是调参逼近某个完整 kernel。
2026-10-21 用户取消端到端误差 ≤10% 的硬指标；仍须报告留出误差和适用边界。

## 第一批 A2 数据

910B3 / NPU6，CANN 9.0.0，FP16 MMA 与 FP32 Add/Reduce/Copy。
测量使用 GetSystemCycle 的 50 MHz 系统计数器（20 ns/tick），不是假设的 AI Core 频率。
输入驻留后计时，计时边界使用 PIPE_ALL，匹配空循环/空 drain 对照；
每个 count 预热3次、测量5次，count 为 0/16/64/256。

第一批共40个 shape/操作组合通过探针的数值与有限性检查：9 MMA、7 Add、
7 ReduceSum、7 ReduceMax、5 GM→UB、5 UB→GM。训练/留出在采集前固定。
Copy 是单核、双 AIV、重复热地址条件，不能称为冷 GM 或全芯片带宽标定。

下表仅列双槽重复操作的增量成本；时间为 ns，B 是一个逻辑 tensor 的处理字节数，
不是所有输入输出字节数的和。X 是补齐到 FP16 Cube 粒度的 MNK。

| 操作 | 拟合结果 | 留出最大绝对相对误差 |
|---|---|---:|
| FP16 MMA | `0.00013558305 * X + 0.13609` | 0.112% |
| FP32 Add | `0.0046069613 * B + 11.7512` | 0.055% |
| FP32 GM→UB（热地址） | `0.0065891558 * B - 1.4992` | 1.415% |
| FP32 UB→GM（热地址） | `0.0099595927 * B + 1.2370` | 0.899% |
| FP32 ReduceSum，末轴宽512 | `0.0023983639 * B + 66.9882` | 12.609% |
| FP32 ReduceMax，末轴宽512 | `0.0026868245 * B + 50.3650` | 12.020% |

MMA 的测量 X 范围为 262144–4194304；其他操作 B 范围8192–49152。
MMA 与 Add 各留出3个 shape，其余上述家族各留出1个 shape，不能将小样本误差
当成所有 shape 的精度保证。Add 留出组包括相同字节数、不同二维几何的对照。
拟合器明确拒绝静默外推；Reduce 宽128/256尚无独立训练组，不发布参数。

Reduce 已显现单直线的局限。helper 的 plan/tail/等待路径与行数和归约几何有关；
不能通过把输出尺寸当输入、混合不同宽度或把留出点放回训练来掩盖残差。
当前仅发布候选拟合及误差，尚未认定其适用于所有 Reduce shape。

## 单耗时模型与测量对照

按用户后续要求，不拆 latency/issue interval，不修改同 pipe 发射语义。
模型每个 tile 操作只有一个耗时；不同 pipe 的重叠仍由原执行队列和同步决定。

- `two_slot`：交替使用两个槽，不在每轮主动 drain。
- `chain`：单槽，每轮 drain，对照循环保留同样的 drain。

二者是不同协议下的增量 helper 成本，**不是裸指令 latency 和 issue interval**。
默认候选参数来自 two_slot，chain 仅作对照，不新增发射间隔模型。
曾采集 count=1 的20 ns精度诊断数据，保留为原始实验，不作为新的调度参数。

第二批 `linear-vector-v2` 新增 Sub/Mul/Div/Exp/Fill/AXPY，共30个组合，数值检查全过。
双槽模式留出最大误差分别为0.016%、5.674%、0.005%、0.032%、0.007%、1.252%。
Sub/Mul/Div 使用不同输入、原地更新；旧 `linear-vector-v1` 的同源二元输入仅保留
为特殊写法对照，不能与 v2 混合拟合通用参数。

第三批 `linear-cube-copy-v1` 新增 GM→L1、L1→L0A、L1→L0B、L0C→GM，各5个shape，
数值检查通过。前三者双槽留出误差3.03%、1.14%、1.18%。FIX FP32→FP32的
96×96实测990.24ns，通用字节数直线仅预测约320ns，留出误差67.69%。
这条 FIX 拟合**未通过适用性检查**，不能宣称为通用已验证参数。需要固定目标FA行宽
补测，区分布局影响；不能删除96×96的原始结果或把它放回训练掩盖问题。
拟合工具的 `status=fitted` 仅表示成功求出直线，不表示通过工程适用性验收。

随后按采集前固定的 N=128 测量 M=32/64/80/96/128，M=80留出。FIX 的行宽现在作为
测量 family 的一部分，不再把96列的结果套入128列模型。FP32→FP32 的留出误差
0.254%，FP32→FP16 为2.608%；对应输入字节数范围16384–65536。
96×96原始异常结果保留在独立行宽 family，当前数据不足，不发布该行宽参数。
同时测了 MMA init=True，按 MNK 拟合的留出误差0.018%。这批18项均通过数值检查，
累计108个通用/目标FA组合（不含弃用的同源输入对照）。原始数据为
`linear-fa-cube-v2`，汇总为实验目录中的 `fit_all_v5.json`。

## 实现与复现

`tilelang/simulator/linear_timing.py` 提供特征计算、OLS、参数序列化和适用域检查。
`scripts/fit_simulator_linear_costs.py` 读取原始 observations，严格只用 train 拟合，
按操作/dtype/归约宽度/测量模式分组，保存输入 SHA256 和逐点残差。
`scripts/measure_simulator_linear_costs.py` 是归档后的可复用测量入口，复用 expert FA
计时探针结构。必须先加载 CANN 环境并确认 Python/native library 来自目标构建。

可通过 `linear_timing.measured_timing_profile(report)` 构造实测 timing profile，交给
已有的 `SimulatorConfig.timing_profile`，没有新增发射参数或 CLI flag。每个任务只给
一个 duration，旧 scheduler 不变。缺失家族、域外尺寸、未实测 A3 明确拒绝，不静默
回退 PyPTO 或借用其他 dtype。混合输入/输出类型使用独立 family，例如
`float32_to_float16`，字节数自变量按输入 tensor 计算。

profile 在既有时间轴上用2个抽象 tick表示1 ns，以兼容现有2000 tick/us的 trace
换算。这只是数值网格，不代表 AI Core 工作在2 GHz。同步和标量目前保留原有1 tick
记账，不能称为已实测的同步指令延迟。

A2 默认已切换为 `measured-linear`：实测覆盖范围内使用参数，缺失家族或域外尺寸
显式标记 `timing_source=pypto-fallback`；命中为 `measured`，显式覆盖为 `override`，
同步/标量记账为 `bookkeeping`。严格的 `measured_timing_profile(report)` 不回退，
适合检查目标 kernel 是否完整覆盖。A3 默认仍为未标定的 PyPTO，不借用 A2 参数。

```bash
# 先确认物理卡6健康空闲；两个扫描串行执行。
ASCEND_RT_VISIBLE_DEVICES=6 python scripts/measure_simulator_linear_costs.py \
  --suite linear --output /path/to/compute
ASCEND_RT_VISIBLE_DEVICES=6 python scripts/measure_simulator_linear_costs.py \
  --suite copy --output /path/to/copy
python scripts/fit_simulator_linear_costs.py \
  /path/to/linear-v1/observations.json \
  /path/to/linear-reduce-v2/observations.json \
  /path/to/linear-copy-v1/observations.json --output /path/to/fit.json
python -m pytest testing/python/simulator/test_linear_timing.py -q
```

原始实验根位于服务器 Docker：
`/home/wenzhongzhen/benchmark/experiments/pypto-primitives-20261021/`。
各 run 保存 `bench_source.py`、`manifest.json`、生成 C++ 和原始 `observations.json`。
旧 observations 中 PyPTO/PTO 比较列不参与本次拟合，不能将旧 fallback 列当成完整 PTO 模型。

原始数据来自隔离 worktree `pypto-fast-cost-20261021`（基线 f8b502d5，Python层有
未提交的 bridge/unit-flag 修复）。精确 native library SHA256、包路径和探针源码指纹
以每个 run 的 manifest 为准。现有编译器主 checkout 未改动。

2026-10-21 Docker 回归：`786 passed, 3 subtests passed`（68.80s）。
本地无 native/PyTorch 环境，定向71项通过，完整回归以上述 Docker 结果为准。
新增实测 profile 和 ABI 特征映射后再次 Docker 回归：
`791 passed, 3 subtests passed`（67.90s）。

## 最终数据与参数覆盖

累计206个操作/shape组合通过探针数值与有限性检查，13份原始 observations 汇总为
`fit_all_v7.json`。发布33个双槽 family，参数、适用域、逐家族留出误差与13份输入
SHA256 固化在 `tilelang/simulator/measured_a2.py`，不依赖额外 JSON 安装路径。
报告 SHA256：`bf2297de112dd6a831e72cf2873d979f1141f4013c647e21db8343cea3056e31`。

覆盖 FP16 MMA（初始化/累加分别拟合），FP32 常用逐元素操作、小向量、BRCB、
row-expand、FP16/FP32 cast、mixed AXPY、FP16/FP32 GM↔UB、FP16 GM→L1、
L1→L0A/B（B转置单列）、FP32/FP16 FIX。FIX 区分128列连续输出与256列/512步长输出；
row-expand 按实际物理行步长区分，不拿逻辑连续字节数冒充连续访问。

数据不是所有操作都能被一条直线精确解释：mixed AXPY 留出误差13.50%，
FP16→FP32 cast 16.51%，反向12.61%，ReduceSum/Max 12.61%/12.02%，
row-expand mul 10.79%。这些残差保留，不以拟合成功冒充普适准确。
未覆盖 BF16、其他 Reduce 几何及其他路径/尺寸仍使用有标签的 fallback。
GM 参数来自重复驻留地址，不是多核共享带宽或冷缓存模型。

## FA expert R4 对照

原文件 `fa-expert-v3/code/ablations/r4_half_score_compact.py`，Q256/KV512，
非causal D128。源码 SHA256：
`108864cf04db6eb466cfd64be389ba5a781c4278a59c870b357503b5a2381264`。
预测采用严格实测 profile，所有计算/搬运任务均命中，无 fallback，未对 FA 调参。

| B,H,S,D | 预测 μs | 历史实机 μs | 相对误差 | 任务数 | 构建 s | 构建后执行 s |
|---|---:|---:|---:|---:|---:|---:|
| 4,2,3072,128 | 267.315 | 270.169 | −1.06% | 29,343 | 54.48 | 5.93 |
| 1,4,3072,128 | 163.054 | 166.259 | −1.93% | 17,625 | 34.04 | 3.11 |
| 1,8,8192,128 | 1815.294 | 1820.660 | −0.29% | 203,485 | 376.72 | 69.35 |

这是保留原20核任务分配的 **core0最大负载投影**，不是全20核联合模拟。
实机数字来自冻结的历史 NPU6 测量，不是本轮重新上卡，也不是 torch_npu FIA。
未建模共享GM带宽、编译器Vector mask设置的物理耗时；相近的总耗时不能证明各pipe准确。
本轮真实执行 sync-only，而非仅编译；不代表重新验证 FA 数值精度。
`execution_timeout_s=60` 未放大，但它约束调度循环，不是整个调用的墙钟时间；
8K的构建后执行总计仍为69.35秒，不能宣称整个调用满足60秒。

旧PyPTO同case0构建54.65秒、执行6.04秒，总计60.69秒；新模型总计60.41秒。
单次测量不能宣称提速，但没有显示一次函数计算本身引入明显退化。
大case耗时仍主要来自完整指令流展开，未解决这一独立可扩展性问题。

对照工件位于实验根的 `fa-measured-v1/case{0,1,2}.json`，含原始参数报告、
native库、参考结果和源码指纹；8K另存加载时的Python源码快照。
参考结果SHA256：`1caaeed9f244d9df0ece153738a2c35a8b9bc06bafbfe0a798514817cab1a351`；
native库SHA256：`9f90e18c3150d0521ce4bdf21a734e00f6f58e16f20276a9ea30042e95b6e535`。

复现完整FA对照：在上述隔离worktree、CANN环境及Python/native路径已绑定后运行
（环境设置见 [PyPTO baseline](simulator_pypto_fast_timing.md#docker-regression-environment)）：

```bash
EXP=/home/wenzhongzhen/benchmark/experiments/pypto-primitives-20261021
PY=/home/wenzhongzhen/kernel-tool/fla-kda-profile-env/bin/python
# 使用新输出文件，保留原记录；case 0/1/2分别对应上表。
$PY "$EXP/run_measured_fa.py" --fit "$EXP/fit_all_v7.json" \
  --case 2 --output "$EXP/fa-case2-repeat.json"
$PY -m pytest testing/python/simulator/ -q --disable-warnings
```

默认接入并格式检查后 Docker 完整模拟器回归：798项、3项子测试通过（68.29秒）。
按用户要求，只维护单核流水预测，不展开多核任务或新增共享带宽模型。
尚未实测 A3；更广 dtype/shape和精细 Reduce 模型是后续工作，
不能将当前参数当作通用周期精确模型。
