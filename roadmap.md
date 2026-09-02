# TileLang Ascend A2/A3 Simulator Roadmap

更新时间：2026-09-02

本路线图只覆盖 Ascend A2/A3（C220）。`shmem` 永久不支持，也不列为待办项；任何
shmem scope 或 intrinsic 都必须 fail fast。详细设计和验收原则见
[`docs/a2_a3_simulator_design.md`](docs/a2_a3_simulator_design.md)。

说明：只有已经实现并通过测试的事项才使用删除线。部分完成的阶段会拆成已完成和
待完成条目，不能因为已有骨架就把整个阶段标记为完成。

## 整体思路

### 目标与边界

模拟器同时提供两种能力，但共享同一份 SimIR：

1. **功能仿真**：在 CPU 上执行 kernel，验证数值、地址、容量、初始化、alias、tail
   和同步语义。
2. **性能 trace**：通过离散事件调度模拟 AIC/AIV pipe overlap、依赖和等待，输出
   Chrome/Perfetto trace 和统计数据。

功能结果必须正确，性能模型必须可解释。当前 timing profile 是未校准单位成本，不能
当作真机 latency。A2 与 A3 共用功能语义，但 timing profile 必须独立维护。

权威输入是 `OptimizeForTarget` 和最终 `tir.transform.Simplify()` 之后、
`device_codegen` 之前的 TIR。不要解析生成的 AscendC/PTO C++，也不要另建一条与原生
编译漂移的 lowering pipeline。

### 数据流

```text
TileLang PrimFunc / IRModule
  │
  ├─ LowerAndLegalize
  ├─ OptimizeForTarget
  └─ final Simplify
          │
          ▼
   final pre-codegen TIR
          │
          ▼
   TIR bridge（严格解析）
          │
          ▼
   KernelProgram / Task / BufferRegion
          │
          ├───────────────┐
          ▼               ▼
  FunctionalSimulator   DiscreteEventScheduler
  MemoryRuntime         flags/barriers/pipes
          │               │
          ▼               ▼
  NumPy golden/result   stats + Perfetto trace
```

bridge 必须 fail-closed：无法确定 operation、scope、shape、offset、layout 或同步语义时，
抛出带 operation、platform、lane/pipe 和 source span 的异常，不能跳过或猜测。

### 分层职责

| 文件 | 职责 | 接手注意事项 |
|---|---|---|
| `tilelang/engine/lower.py` | 共享最终 pre-codegen lowering | 原生编译与 simulator 必须共用 |
| `tilelang/jit/`、`tilelang/cache/` | `simulator=True` API 与 cache bypass | simulator 不应要求设备 binary |
| `tilelang/simulator/bridge.py` | final TIR → SimIR、operand 和 dependency 提取 | 未确认的签名必须 fail-closed |
| `tilelang/simulator/program.py` | `BufferSpec/Region`、Task、lane/pipe、DAG | 保持 backend-neutral 和不可变 |
| `tilelang/simulator/memory.py` | byte-addressed scope、view、capacity、poison | GM/workspace 共享，本地 scope 逐 core |
| `tilelang/simulator/hazard.py` | alias、越界和未初始化诊断策略 | `off/warn/error` 行为要一致 |
| `tilelang/simulator/executor.py` | NumPy 功能语义和 operation dispatch | 不支持的 variant 必须报错 |
| `tilelang/simulator/scheduler.py` | dependency、pipe FIFO、overlap、timeout | 调度顺序必须确定性 |
| `tilelang/simulator/sync.py` | local/cross flags 和 barriers | 等待必须成为显式 event |
| `tilelang/simulator/trace.py`、`stats.py` | trace 导出和 overlap-aware 统计 | 禁止简单累加并行 duration |
| `tilelang/simulator/profile.py` | A2/A3 topology 和 timing 参数 | 未校准值必须显式标注 |

### 不可破坏的语义约束

- `shmem` 永久不支持；不要实现空操作或近似替代。
- GM/workspace 在 core 间共享；L1/L0/UB/BT/LOCAL 必须逐 core 隔离。
- storage rewrite 后的 byte address、lifetime 和 alias 是内存真相，buffer 名只用于诊断。
- 读取未写入 byte 必须触发 poison/hazard；值恰好为 `0xff` 不能代表未初始化。
- scheduler 的依赖、pipe FIFO、flag/barrier 必须共同决定 start cycle。
- 功能执行与 timing 调度使用同一 Task DAG，但数值语义和周期参数保持解耦。
- AIC 与两个 AIV lane 共享部分资源但不是同一执行流；不能把所有 task 串行化。
- trace 中的 simulated cycle 必须携带 calibration 状态，不能伪装成测量数据。
- 不支持的 dtype、layout、mask、动态表达式或 intrinsic variant 必须明确报错。

### 添加一个 operation 的标准流程

每个 operation 按以下顺序落地，避免只写 NumPy 计算而漏掉地址、调度或 trace：

1. 从 `src/op/`、`src/transform/`、`src/target/` 和现有测试确认最终 TIR contract；记录
   参数顺序、读写方向、dtype、scope、layout、mask/tail 和同步要求。
2. 在 `classify_operation()` 中确定 AIC/AIV lane 和 pipe。
3. 在 bridge 中把 pointer/access_ptr、offset、shape、stride 和附加参数降成
   `BufferRegion` 与结构化 metadata。
4. 从读写 region 生成 RAW/WAR/WAW dependency；涉及 flag/barrier 时补同步 metadata。
5. 在 executor 中实现功能语义，并通过 MemoryRuntime 访问数据，不能绕过 bounds、
   poison 和 hazard。
6. 为 timing profile 定义 operation key；未校准时继续使用明确标注的 fallback。
7. 增加真实 TVM TIR bridge 测试、功能 golden、错误路径、tail/dtype/scope 测试和 trace
   断言。
8. 测试通过后再在本文件划掉对应条目；只完成一种签名时必须保留其它 variants。

### 当前实现状态与已知限制

当前已经存在一条可执行 vertical slice：

```text
真实 TVM PrimFunc
  → GM→UB strided copy
  → 一维 vector add 或二维 valid-rectangle tail unary/scalar/binary
  → UB→GM strided copy
  → 自动 RAW dependency
  → NumPy result + schedule stats
```

copy 已支持常量或 runtime affine `tvm_access_ptr` element offset、二维 valid rectangle
和 physical row stride；运行时 symbol 通过 `FunctionalSimulator(bindings=...)` 绑定。
GM→UB copy 在完整物理 destination view 可证明不越界时，会先用 literal `pad_value`
初始化物理 tile，再覆盖 valid rectangle；对越界 slice 禁用 pad 并保留诊断原因。
runtime integer 除仿射形式外，还支持 backend-neutral `SymbolicInt` 表达式树：
`min/max/mul/floordiv/floormod/truncdiv/truncmod/select`、整数 cast、比较和布尔组合。
动态 BufferSpec shape 会在 FunctionalSimulator 创建时根据 bindings 实例化。
vector binary 的普通形式当前支持一维显式 count；tail unary/scalar/binary 支持二维
valid rectangle，并保留原始 tail intrinsic 类型供 trace/debug。bridge 会读取最终
PrimFunc 的 `address_map/size_map`；dependency 与 FunctionalSimulator backing 均按
scope、core 和绝对 byte range 处理跨 buffer 的部分或完全 alias。
cast 已贯通 `dst/src/round_mode/count` contract；当前执行 `CAST_NONE/RINT/FLOOR/CEIL/
ROUND/TRUNC`，其中 ROUND 遵循 ties-away-from-zero；语义未充分确认的 CAST_ODD 明确拒绝。
fill 已贯通原生 intrinsic 和等价 call_extern contract，支持 literal scalar、runtime
count、pointer offset、poison 初始化和后续访问 dependency。
leaky_relu 已复用 scalar vector contract；axpy 将 destination 同时建模为 accumulator read
和 output write，因而保留 read-before-write 与 RAW/WAR/WAW dependency。
sigmoid、silu、sin 和 cos 已支持普通及显式/隐藏 scratch 签名；scratch 作为真实写 region
参与 alias 和 dependency，in-place silu 保留 source read 与 destination write。
pow 和 clamp/max/min 已支持普通与 scratch forms；pow 的 count 来自真实 access_ptr extent，
clamp 会验证 literal bounds 和 min/max 顺序。
broadcast 已支持 rank-1/2、两条二维广播轴、普通/scratch forms、shape legality 和动态
region；执行器严格按结构化 source/destination shape 广播。
compare/compare_scalar 已支持 EQ/NE/GT/GE/LT/LE 和 literal scalar，输出按仓库 golden
使用 little-endian bit packing；scalar BufferLoad form 仍 fail-closed。
tail reduction 已贯通仓库当前验证的 float32、axis 0、clear=true contract，支持
sum/max/min 对二维 valid rectangle 按列归约；workspace、axis 1 和 accumulate 仍待实现。
Cube copy 已有两条 GM→L1 路径：显式 RowMajor 路径按二维 stride 搬运；默认路径将有效
矩形按仓库 `ascend_layout.py` 的 zN 公式写入真实物理存储，并复现 tail tile 清零。默认
路径当前只接受 fractal/C0 对齐、tile-base 对齐的静态物理 tile；子 tile 拼接继续拒绝。

尚未支持任意 TIR Call 或未列入上述白名单的动态表达式、通用 mask、完整 software
pipeline、其余 Cube copy/MMA/fixpipe，以及后续 P3–P8 operation。
JIT adapter 的静态 schedule 可用，但普通 tensor 调用仍应 fail-closed，直到 bridge
能为实际 kernel 生成完整可执行 operands。

### 开发和验证环境

当前 macOS ARM 环境已在 `3rdparty/tvm/build` 构建 CPU-only TVM；构建产物不提交。
它可以验证 Memory/SimIR/scheduler/真实 TVM TIR bridge，不能验证 CANN codegen、A2/A3
真机功能或 timing calibration。

运行 simulator 回归：

```bash
cd /Users/wzz/Desktop/Research/kernel-tool/cannbot-skills/tilelang-ascend
PYTHONPATH=3rdparty/tvm/python \
  /Users/wzz/miniconda3/bin/python -m pytest testing/python/simulator -q
```

当前基线为 `173 passed`。TVM 在 Python 3.13 下会产生 parser deprecation warnings；这些
不是 simulator failure。完整 lowering/JIT 测试需要 Linux、CANN、构建后的
`libtilelang`，最终 timing 还需要分别在 A2/A3 真机校准。

### 接手顺序建议

接手者先运行上述 173 个测试并阅读最近提交，再按本文件“下一批工作”推进。优先维持
端到端 vertical slice：每增加一种 TIR form，都要让它贯穿 bridge、memory、executor、
scheduler 和测试，而不是先铺大量不可执行的 operation 名称。推荐顺序是：

1. 通用 mask、剩余 cast round mode 和 vector forms；
2. reduction；
3. Cube copy、MMA 和 fixpipe；
4. 后续 P3–P8。

## P0：Lowering、SimIR 与运行时骨架

- [x] ~~创建 `feat/a2-a3-simulator` 开发分支。~~
- [x] ~~统一最终 pre-codegen Ascend TIR lowering 入口，原生编译和模拟器共用 pass
  序列。~~
- [x] ~~接入 `tilelang.compile(..., simulator=True, sim_config=...)` 和
  `@tilelang.jit(..., simulator=True)`。~~
- [x] ~~定义 A2/A3 配置、拓扑、lane、pipe、task、core program 和 kernel program。~~
- [x] ~~实现 SimIR DAG 校验、未知依赖检查和确定性任务顺序。~~
- [x] ~~实现常量 `For`、`IfThenElse`、`LetStmt`、`blockIdx.x/y`、AIC/AIV resource
  scope 的 TIR bridge。~~
- [x] ~~采集 PrimFunc 参数 buffer、`Allocate` 和 `Block.alloc_buffers`。~~
- [x] ~~未知 operation、未知 memory scope 和 shmem fail-closed。~~
- [x] ~~构建 macOS CPU-only TVM，并用真实 `tvm.tir.PrimFunc` 验证 bridge。~~
- [x] ~~以 backend-neutral `AffineInt` 表示 runtime symbol、常量加减和常量倍乘，并在
  FunctionalSimulator 中绑定 region shape/offset/stride。~~
- [x] ~~以 backend-neutral `SymbolicInt` 支持 min/max、非仿射乘法、floor/trunc
  div/mod、select、整数 cast、比较和布尔表达式，并按 runtime bindings 实例化动态
  BufferSpec。~~
- [ ] 覆盖其它最终 TIR 中实际出现的动态表达式，未知节点继续 fail-closed。
- [ ] 将 `BufferStore`、pointer arithmetic、`address_of`、`tvm_access_ptr` 和切片统一
  降为可执行 `BufferRegion`。
- [ ] 为最终 pre-codegen TIR 增加稳定的 IR snapshot 测试。
- [ ] 在 Linux+CANN 环境验证完整 TileLang lowering；Mac 只负责 CPU/TVM 层测试。

## P1：Vector MVP

- [x] ~~实现 GM/workspace 共享地址空间和逐 core 的 L1、L0A、L0B、L0C、UB、BT、
  LOCAL 地址空间。~~
- [x] ~~实现 byte-addressed memory、连续/strided view、bounds、capacity、poison 和
  read-before-write 检查。~~
- [x] ~~实现 storage-rewrite 地址、非重叠 lifetime 复用和显式 alias。~~
- [x] ~~从真实 PrimFunc `address_map/size_map` 导入 planned storage，使不同 buffer 名的
  部分/完全物理 alias 共享 backing 并产生 RAW/WAR/WAW dependency。~~
- [x] ~~实现显式 SimIR 的 GM→UB→vector add→UB→GM 数值链路。~~
- [x] ~~验证双 AIV core 和非整块 tail（19 元素拆分为 10+9）。~~
- [x] ~~从真实 TIR 自动提取简单三参数 `copy_gm_to_ub` 的 src/dst/extent 并执行。~~
- [x] ~~解析 GM↔UB `tvm_access_ptr` 的常量 element offset、二维 valid rectangle、
  physical stride、mask/pad 元数据，并执行 strided copy。~~
- [x] ~~实现基本二元执行器：add、sub、mul、div、min、max。~~
- [x] ~~自动提取真实 TIR 的一维 vector add 操作数，并完成
  `PrimFunc→SimIR→GM→UB→add→UB→GM→NumPy`。~~
- [x] ~~实现普通一维 binary 的 sub/mul/div/min/max 执行语义。~~
- [x] ~~按真实 tail pass contract 解析 operation tag 和二维 valid rectangle，贯通
  tail unary、scalar、binary 的 bridge、dependency 与数值执行。~~
- [x] ~~实现 scalar vector forms：adds、subs、muls、divs、mins、maxs。~~
- [ ] 实现普通二维 vector、通用 mask 和其它 tail variants。
- [x] ~~补齐 GM↔UB copy 的 symbolic affine runtime extent 和动态 element offset。~~
- [x] ~~实现完整物理 destination view 的 GM→UB literal pad 写入，并对越界 slice
  禁用 pad。~~
- [x] ~~实现上述白名单内的非仿射 runtime region，并用真实 TIR 覆盖
  `min + floormod/floordiv + Select`。~~
- [x] ~~实现 abs、exp、relu、sqrt、rsqrt、reciprocal 和 ln 的基础 NumPy 功能语义。~~
- [x] ~~按真实 `dst/src/round_mode/count` contract 实现 `CAST_NONE` 与 `CAST_RINT`，
  覆盖 float32→float16 和 float32→int32。~~
- [x] ~~实现 fill 的 literal scalar、runtime count、pointer offset、poison 初始化和
  dependency 语义。~~
- [x] ~~实现 leaky_relu 和 in-place accumulator 语义的 axpy。~~
- [x] ~~实现 sigmoid、silu、sin、cos 的普通与 scratch forms。~~
- [x] ~~实现 pow 和 clamp/max/min 的普通与 scratch forms。~~
- [x] ~~实现 CAST_FLOOR、CAST_CEIL、CAST_ROUND ties-away-from-zero 和 CAST_TRUNC。~~
- [ ] 实现 duplicate；确认 CAST_ODD，并为已有 unary/cast 补齐 dtype/异常值语义。
- [x] ~~实现 rank-1/2 broadcast、两条二维广播轴和普通/scratch forms。~~
- [x] ~~实现 compare/compare_scalar 的六种 mode、literal scalar 和 packed mask。~~
- [x] ~~实现 select tensor-tensor/tensor-scalar 的普通与 scratch forms，并将 packed
  mask 纳入显式 read dependency。~~
- [x] ~~实现 tail_compare/tail_compare_scalar 的二维有效矩形、逐行 packed mask、
  物理 stride 和 literal scalar。~~
- [x] ~~实现 tail_select tensor/scalar、可选 tmp、逐行 mask 解包和有效矩形写入。~~
- [x] ~~实现 tail_broadcast 两条二维广播轴、可选 scratch、valid rectangle 和
  AscendC 32-byte source row stride。~~
- [ ] 实现 compare_scalar BufferLoad、select BufferLoad/CMPMASK mode 和其它
  tail/mask variants。
- [x] ~~实现真实 `tail_reduce` 的 float32 axis-0 clear=true sum/max/min，并覆盖
  valid rectangle、dependency 和非法 dim/clear contract。~~
- [x] ~~实现普通 float32 reduce 的 sum/max/min、axis 0/-1、clear=true row tmp 和
  dependency。~~
- [x] ~~实现普通 reduce clear=false 的 dst accumulator merge，以及 PTO column 单
  output tmp、row main/output 双 tmp dependency。~~
- [ ] 实现 narrow physical row，以及 whole/block reduction。
- [ ] 覆盖 float16、bfloat16、float32、int/uint 常用 dtype 的舍入、溢出和饱和语义。
- [ ] 增加动态 shape、动态 tail、misalignment、partial write 和非法 copy path 测试。

## P2：Cube MVP

- [x] ~~按仓库 `ascend_layout.py` 公式实现 zN/nZ 的 C0/fractal storage size、物理
  index 和 NumPy pack/unpack codec。~~
- [x] ~~实现显式 RowMajor `copy_gm_to_l1_linear` 的 valid rectangle、scope/dtype
  contract 和 32-byte row alignment。~~
- [x] ~~实现默认 zN `copy_gm_to_l1` 的 aligned physical tile/base 路径、tail clear
  和真实物理 pack。~~
- [ ] 实现非对齐/子 tile GM→L1、L1→L0A/L0B 和 L0C→GM 的合法 copy 路径。
- [ ] 实现 `gemm_v0`。
- [ ] 实现显式 `mma`，支持 init、accumulate 和多 K tile 累加。
- [ ] 实现基础 fixpipe 和 L0C 输出。
- [ ] 支持 A/B transpose，以及 NZ、ZN、fractal 等核心 layout。
- [ ] 使用 NumPy/PyTorch golden 覆盖 M/N/K tail、多 core 分块和 pipeline overlap。

## P3：完整搬运、Vector 与 Reduction

- [ ] 覆盖最终 TIR 中全部合法 A2/A3 copy 方向。
- [ ] 实现 UB→L1，以及 L0C→UB 的 GM workspace/CV handoff 路径。
- [ ] 完成 vector operation family、standalone transpose 和完整 reduction family。
- [ ] 实现 gather、gather-mask、gatherb 所需的底层地址与 mask 语义。
- [ ] 建立 operation × dtype × scope × alignment 支持矩阵。
- [ ] 对每条 memory path 增加越界、容量、alias 和 address-range hazard 测试。

## P4：Bias、Fixpipe 与量化

- [ ] 实现 BT bias 和 bias-initialized MMA。
- [ ] 完成 fixpipe 选项、relu/fused output 行为。
- [ ] 实现 fp32→fp16/bfloat16 转换。
- [ ] 实现仓库实际暴露的 integer quant/dequant、rounding 和 saturation 语义。
- [ ] 增加 bit/ULP golden、边界值、BT capacity 和 MMA/fixpipe 配对协议测试。

## P5：索引、排序与 TopK

- [ ] 完成 gather、gather-mask 和 gatherb variants。
- [ ] 实现 sort、sort32、init_sort_buf 和 merge_sort。
- [ ] 实现 TopK。
- [ ] 验证重复值、稳定性/index 语义、奇数 valid count、动态 valid length 和 scratch
  buffer bounds。

## P6：Convolution 与 Im2Col

- [ ] 实现 `im2col`。
- [ ] 实现显式 im2col + MMA convolution。
- [ ] 接入仓库中存在的高层 convolution lowering 形式。
- [ ] 验证 stride、dilation、padding、C0/channel block、spatial tail 和多 C1 累加。
- [ ] 将 A2/A3 硬件限制编码为明确 legality check。

## P7：Atomic 与 Persistent

- [ ] 实现 atomic 操作和 output seeding。
- [ ] 对跨 core 冲突提供确定性的序列化模型与 trace 解释。
- [ ] 实现 Persistent kernel 的 work distribution、residency 和 liveness 模型。
- [ ] 覆盖 atomic contention、初始化规则、终止条件和 deadlock。

## 同步、流水与调度

- [x] ~~实现 dependency DAG 和 `(core, lane, pipe)` FIFO 离散事件调度。~~
- [x] ~~允许不同 pipe overlap，并统计 makespan、utilization、wait 和 per-core completion。~~
- [x] ~~实现 local/cross set-wait flag、auto flag、barrier_all 和 pipe_barrier。~~
- [x] ~~将同步等待输出为显式 trace wait event。~~
- [x] ~~实现 max cycles、wall timeout 和基础 deadlock 报告。~~
- [x] ~~从真实 TIR 的同名重叠 `BufferRegion` 自动生成 RAW、WAR 和 WAW dependency。~~
- [x] ~~将 dependency 扩展到 storage alias、跨 buffer 物理地址和保守动态 region。~~
- [ ] 将 hazard 诊断扩展到带 source span 的动态精确 address range。
- [ ] 执行 software-pipeline prologue、steady state、epilogue、stage/ring index 和 wrap。
- [ ] 检查 ring-slot reuse、in-flight memory hazard 和 flag 配对协议。
- [ ] 将 deadlock 报告扩展到 outstanding producer、flag ID、memory region 和 source span。

## Trace 与性能诊断

- [x] ~~导出 Chrome/Perfetto JSON complete events。~~
- [x] ~~为 core/lane/pipe、operation、wait reason 和未校准 profile 输出基础元数据。~~
- [x] ~~提供 overlap-aware makespan、pipe utilization、wait cycles 和 core completion
  统计。~~
- [ ] 输出 dependency/flag flow events、queue depth、live local-memory 和 active-core counter。
- [ ] 统计各 memory path bytes、operation counts、peak local-memory 和 hazard counts。
- [ ] 标记 critical path、load imbalance、copy/compute overlap 和主要 stall 原因。
- [ ] 增加 trace schema/version 回归测试和 trace comparison 工具。

## P8：A2/A3 Timing Calibration 与硬化

- [x] ~~A2/A3 分离配置，并将当前成本明确标记为 `uncalibrated-unit-cost`。~~
- [ ] 建立 copy、vector、reduce、cube、fixpipe、sync、atomic 微基准集。
- [ ] 在 A2 和 A3 上分别采集 latency、throughput、bandwidth 和 contention 数据。
- [ ] 建立独立的、带版本和来源的 A2/A3 timing profile。
- [ ] 建模 multi-core wave、DDR/NoC contention 和 pipeline resource 限制。
- [ ] 用 held-out kernels 报告绝对误差、排序相关性和误差区间。
- [ ] 在性能模型通过验证门槛前，禁止把模拟 cycle 当作真实 latency 或 autotuner
  objective。

## 测试与交付门槛

- [x] ~~纯 simulator 测试可在无 CANN、无 NPU、无 `torch_npu` 的 CPU host 运行。~~
- [x] ~~当前测试基线：173 passed，覆盖 memory、scheduler、sync、trace、functional
  executor、真实 TIR bridge 和 shmem rejection。~~
- [ ] 每个 operation 必须有正向、错误路径、dtype、shape/tail、scope 和 trace 测试。
- [ ] PTO 是第一验证目标；随后补齐 AscendC intrinsic parity。
- [ ] 在 Linux+CANN CI 运行 lowering/JIT 集成测试。
- [ ] 在 A2/A3 真机分别完成 functional 对比和 timing calibration。
- [ ] 发布 machine-readable operation coverage table，列出精确支持和不支持的 variants。

## 下一批工作

1. 实现 L1→L0A/L0B 的合法 layout 转换和真实物理存储。
2. 实现 L0C→GM/fixpipe 基础输出路径。
3. 接入 `gemm_v0`，再实现显式 `mma` 的 init/accumulate/K-tile。
4. 并行补齐 compare/select mask variants 与 reduction dtype/whole/block variants。
