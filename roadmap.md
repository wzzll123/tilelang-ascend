# TileLang Ascend A2/A3 Simulator Roadmap

更新时间：2026-09-01

本路线图只覆盖 Ascend A2/A3（C220）。`shmem` 永久不支持，也不列为待办项；任何
shmem scope 或 intrinsic 都必须 fail fast。详细设计和验收原则见
[`docs/a2_a3_simulator_design.md`](docs/a2_a3_simulator_design.md)。

说明：只有已经实现并通过测试的事项才使用删除线。部分完成的阶段会拆成已完成和
待完成条目，不能因为已有骨架就把整个阶段标记为完成。

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
- [ ] 支持动态标量表达式与运行时 shape/stride/offset 求值。
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
- [x] ~~实现显式 SimIR 的 GM→UB→vector add→UB→GM 数值链路。~~
- [x] ~~验证双 AIV core 和非整块 tail（19 元素拆分为 10+9）。~~
- [x] ~~从真实 TIR 自动提取简单三参数 `copy_gm_to_ub` 的 src/dst/extent 并执行。~~
- [x] ~~解析 GM↔UB `tvm_access_ptr` 的常量 element offset、二维 valid rectangle、
  physical stride、mask/pad 元数据，并执行 strided copy。~~
- [x] ~~实现基本二元执行器：add、sub、mul、div、min、max。~~
- [x] ~~自动提取真实 TIR 的一维 vector add 操作数，并完成
  `PrimFunc→SimIR→GM→UB→add→UB→GM→NumPy`。~~
- [ ] 补齐 sub/mul/div/min/max、二维 tail/mask 和 scalar vector forms。
- [ ] 补齐 GM↔UB copy 的 symbolic runtime extent、动态 offset 和 pad 写入语义。
- [ ] 实现 fill、duplicate、cast、abs、exp、relu、silu、sqrt、rsqrt、reciprocal、
  sigmoid、sin、cos、ln、pow、clamp 和 axpy。
- [ ] 实现 broadcast、compare、compare_scalar、select 和 tail/mask 语义。
- [ ] 实现基础 sum/max/min reduction 和 whole/block reduction。
- [ ] 覆盖 float16、bfloat16、float32、int/uint 常用 dtype 的舍入、溢出和饱和语义。
- [ ] 增加动态 shape、动态 tail、misalignment、partial write 和非法 copy path 测试。

## P2：Cube MVP

- [ ] 实现 GM→L1、L1→L0A/L0B 和 L0C→GM 的合法 copy 路径。
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
- [ ] 将 dependency/hazard 扩展到 storage alias、跨 buffer 物理地址和动态 region。
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
- [x] ~~当前测试基线：71 passed，覆盖 memory、scheduler、sync、trace、functional
  executor、真实 TIR bridge 和 shmem rejection。~~
- [ ] 每个 operation 必须有正向、错误路径、dtype、shape/tail、scope 和 trace 测试。
- [ ] PTO 是第一验证目标；随后补齐 AscendC intrinsic parity。
- [ ] 在 Linux+CANN CI 运行 lowering/JIT 集成测试。
- [ ] 在 A2/A3 真机分别完成 functional 对比和 timing calibration。
- [ ] 发布 machine-readable operation coverage table，列出精确支持和不支持的 variants。

## 下一批工作

1. 支持 copy 的 symbolic runtime extent、动态 offset 和 pad 写入语义。
2. 补齐 vector 二维 tail/mask、scalar 和 unary forms。
3. 将 TIR dependency/hazard 扩展到 storage alias 和物理地址。
4. 完成 P1 cast、broadcast、compare/select 和基础 reduction。
