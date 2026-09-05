# TileLang Ascend A2/A3 Simulator Roadmap

更新时间：2026-09-05

本路线图只覆盖 Ascend A2/A3（C220）。`shmem` 永久不支持，也不列为待办项；任何
shmem scope 或 intrinsic 都必须 fail fast。详细设计和验收原则见
[`docs/a2_a3_simulator_design.md`](docs/a2_a3_simulator_design.md)。

说明：只有已经实现并通过测试的事项才使用删除线。部分完成的阶段会拆成已完成和
待完成条目，不能因为已有骨架就把整个阶段标记为完成。

## 进度口径

- **完整 roadmap：约 71%**。checkbox 裸计数约 76%，但未完成的完整
  operation families、pipeline、atomic/persistent、convolution 和 A2/A3 timing calibration
  权重更高，因此采用保守工作量加权值。
- **可用功能模拟 MVP：100%**。已有 TIR→NumPy、CPU Torch/NumPy tensor adapter、
  内存/hazard、调度/trace，以及核心 vector、reduction 和 half GEMM vertical slices；
  macOS 已验证实际 `elementwise_add` example 在 A2/A3 完成完整 lowering 和数值执行。
  后续复杂 example 指令覆盖属于完整 roadmap，并仍需在 Linux+CANN 环境交叉验证。

进度只在功能、错误路径和回归测试同时落地后上调；未校准 timing 不计入功能完成度。

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

### EasyASC 语义复用策略

不要重新发明 EasyASC 已在 A2/CANNSIM/真机验证过的执行语义。EasyASC 当前有效实现是
`../plugins-community/ops-easyasc-dsl/easyasc/simulator/`；`simulator_v2/` 和 A5-only
micro/MX/SIMT 路径不属于本项目当前范围。复用边界如下：

| EasyASC 来源 | TileLang 落点 | 复用方式 |
|---|---|---|
| `util.py` 的 NZ/ZZ、stride、footprint | `layout.py`、`memory.py` | 移植物理公式和边界 golden |
| `pipe_cube.py` | `bridge.py`、`executor.py` | 移植 GM/L1/L0/MMAD/fixpipe 功能语义 |
| `pipe_vec.py`、`cast_rounding.py` | `executor.py` | 移植 mask/repeat/round/saturate 语义 |
| `hazard.py`、`physical_alias.py` | `hazard.py`、`memory.py` | 移植实际 touched ranges 与 ownership hazard |
| `trace.py` 和 trace analyzer | `trace.py`、`stats.py` | 对齐事件字段、merge、makespan/utilization |
| `timing/a2_cycle_model.json` | `profile.py` | 保留来源和 calibration 标记后导入 A2 参数 |

不能照搬 EasyASC bridge：EasyASC 输入是 DSL 已发射 instruction stream，TileLang 的权威
输入是 final pre-codegen TIR。每个移植语义必须增加独立 physical-coordinate golden，能运行
EasyASC 时再增加 differential test；禁止只用同一 codec pack 后再 unpack 的自洽测试。
A3 没有对应的 EasyASC timing profile，功能语义可在 contract 相同时共享，性能参数不能从
A2 猜测。

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
createvecindex 和 arith_progression 已支持 pointer offset、float16/float32 与常用 16/32-bit
整数 dtype；first/difference/count 可使用 runtime integer binding，结果按目标 dtype 执行
转换和整数 wraparound，并验证 intrinsic template dtype。
bitwise AND/OR/XOR/NOT 与 scalar left/right shift 已支持 int16/int32/uint16/uint32、pointer
offset 和显式 count；XOR scratch 参与 alias/dependency，执行保持定宽 wraparound，并区分
signed 算术右移和 unsigned 逻辑右移。literal/runtime shift 均按 dtype 位宽验证范围。
leaky_relu 已复用 scalar vector contract；axpy 将 destination 同时建模为 accumulator read
和 output write，因而保留 read-before-write 与 RAW/WAR/WAW dependency。
mul_add_dst 已支持 A2/A3 的 float16/float32、pointer offset 和显式 count；destination 同样
作为 accumulator read 与 output write 建模，并验证三个 operand dtype 一致。
sigmoid、silu、sin 和 cos 已支持普通及显式/隐藏 scratch 签名；scratch 作为真实写 region
参与 alias 和 dependency，in-place silu 保留 source read 与 destination write。
round 已支持 A2/A3 float16/float32、显式/隐藏 scratch 和 ties-away-from-zero，并拒绝
后端 workspace policy 未允许的非浮点 dtype。
gather 已支持 flat UB source、int32/uint32 字节 offset、base address、pointer slice 和
显式/隐藏 scratch；执行前验证 32B pointer alignment、元素对齐、dtype 和源索引范围。
gatherb 已支持每 repeat 八个 32B DataBlock、目标 block/repeat stride 和连续 offset
读取；静态/动态 repeat、模板 dtype、pointer/offset alignment 与源 block bounds 均验证。
gather_mask 已支持七种 fixed pattern 的稳定压缩与 custom uint32 元素索引；destination
未选尾部确定性清零，显式/隐藏 scratch、模板/dtype/capacity/source-bounds 均验证。
brcb_experiment 已按 EasyASC `pipe_vec.py` 的 A2 语义实现：每个 repeat 固定消耗 8 个
连续 src 元素，把元素 b 广播进 32B 单位偏移 `repeat*dst_rep_stride + b*dst_blk_stride`
的完整 32B block；dst footprint 按 EasyASC 保守跨距建模并参与 RAW/WAR/WAW，未寻址的
间隙保持 poison，32B 对齐、模板/src-dst dtype、静态 repeat/stride 均验证。
row_expand_{mul,sub,div}_experiment 已支持语言层的 (tag, dst, src0, src1[, tmp])
contract：256B 行、1..255 行、32B 对齐行窗，行 i 取 src1 第 i 块首元素做标量；
tmp 形式按 codegen 的 brcb(src1→tmp) 端态把广播 scratch 建模为本任务写 region，
后续 tmp 读者经 WAW/RAW 正确排序。
exp_experiment 已按 AscendC `exp_mask` 的实际 final ABI 支持二维 strided window：每行处理
64 个 fp32 或 128 个 fp16 元素，保留原 2D UB 的物理 row stride，支持对齐 column offset
和 in-place 执行；仓库语言层实际 kernel 已在 A2/A3 与 Torch golden 匹配。
sub/abs/mins_experiment 已按 codegen 发出的显式 count ABI 贯通真实 TIR bridge 和 NumPy
执行，复用普通 tensor-tensor、unary、tensor-scalar vector 语义，覆盖 A2/A3、
float16/float32、依赖和负 count 拒绝路径。
reducesum_experiment 已支持带可选/注入 scratch 的最终 ABI，只写 `dst[0]` 并对 count
范围做静态/运行时验证；sum_experiment 已支持 `outter × inner` 行布局和每行前 `n`
元素求和，显式使用 inner row stride，padding 不参与计算。两者覆盖 A2/A3 和
float16/float32。
topk 已支持 workspace 注入后的最终八参数 ABI、float16/float32、静态 K/max count 和
静态或 runtime actual count；输出遵循降序交错 `(value, index)` 格式，重复值保持原始
index 顺序，scratch 作为真实写 region 参与 dependency，并验证 repeat、extent、scope、
dtype 和 runtime count。
sort32 已对齐 TileLang 真机 golden 和 EasyASC A2 语义：每个 32 元素 block 独立稳定降序，
float32 使用两槽 value/index-bitcast 编码，float16 使用四槽 value/reserved/index-low/
index-high 编码；支持多个 repeat、int32/uint32 index，并验证 UB scope、32B 对齐、dtype、
footprint 和 repeat 范围。
sort 已支持 workspace 注入后的最终六参数 ABI、float16/float32 和静态/runtime actual
count；只读取有效前缀并产生全局稳定降序的数值型 `(value, index)` 交错输出，aligned
capacity 尾部不进入排序，scratch 参与 dependency，并在运行前验证 repeat 与
`ceil(actual_num/32)` 一致。
merge_sort 已支持 float32 的两槽 value/index pair 和 float16 的四槽
value/reserved/index-low/index-high 记录（两者每条记录均为 8B）、2/3/4 路输入，以及
AscendC 无 scratch 和 PTO 有 scratch 两种最终 ABI；float16 payload 的原始位型会被完整
保留，语言层也会按四槽记录计算 block length。各 source footprint 独立参与 RAW，输出
保持跨 source 的稳定降序，并拒绝非降序输入、非法 block length、extent、scope、
alignment、dtype 和 template；32B aligned source/destination offset 可执行，scratch 按
仓库 workspace 公式校验容量，NaN 因缺少可靠 A2/A3 排序 golden 而 fail-closed。实际
`examples/sort/example_merge_sort.py` 已支持 A2/A3 simulator，并对 2/3/4 路完整
value/index 记录做精确比较。
init_sort_buf 的 ABI 已修复并实现：语言层 `init_sort_buf(buffer, num, rsv)` 的发射
顺序已对齐硬件包装层 `InitSortBuf(src, eleNum, rsv=0)`（此前 `(ptr, rsv, num)` 会使
codegen 把 `rsv` 当作 `eleNum` 传入并丢弃 `num`）；模拟器按
`src/tl_templates/ascend/common.h` 的语义在 float32 UB workspace 的 int32 视图上交错
填充（偶数 lane 写 fp32 -inf 位型、奇数 lane 写 -1），仅覆盖完整 64 元素（256B）块，
`num % 64` 尾部不写入并保持 poison 诊断；UB scope、template dtype、静态 num/extent、
WAW/RAW 依赖和 `num < 64` 的 no-op 语义均已测试，float16 等其它 dtype 继续
fail-closed。
UB→GM atomic_add 已支持 lowered DMA 五参数 ABI、float16/float32、二维有效矩形和 GM
row stride；destination 同时作为 accumulator read 和 atomic write 建模，因此同一 GM
region 的跨 core 更新会在 DAG/trace 中确定性串行化。目标必须预先初始化，未初始化读取、
非 32B 行宽、scope/dtype/template 错误均明确失败。
L0C→GM atomic_add 已支持 lowered DMA 五参数 ABI、L0C fractal layout 解码、有效尾块和
GM row stride，并覆盖 fp32→fp32/fp16 与 int32→int32。转换发生在加法之前，destination
同样作为 accumulator 和 atomic write 参与依赖分析；bfloat16 因当前 NumPy host 表示/codec
尚未落地而继续 fail closed。
transpose 已支持静态 rank-2 UB tile 的非方形转置和常用 B8/B16/B32 dtype，并验证
source/destination shape 互换、dtype、scope、whole-buffer 和双维度 32B 对齐契约。
reinterpretcast 已按零拷贝元数据操作实现：执行时将目标 UB allocation 重绑到源
allocation，后续不同 dtype 的 region 通过同一原始字节双向读写；bridge 按 core 维护
active alias 图，使转换后的 RAW/WAR/WAW dependency 与运行时别名一致，并验证 whole-buffer、
等字节数、目标 dtype 和 UB scope。
pow 和 clamp/max/min 已支持普通与 scratch forms；pow 的 count 来自真实 access_ptr extent，
clamp 会验证 literal bounds 和 min/max 顺序。
broadcast 已支持 rank-1/2、两条二维广播轴、普通/scratch forms、shape legality 和动态
region；执行器严格按结构化 source/destination shape 广播。storage rewrite 将局部 buffer
展平后，bridge 仍按 broadcast 的逻辑二维宽度恢复 source/destination row stride；runtime
float scalar 参数保持浮点绑定，不再被错误截断为整数。
compare/compare_scalar 已支持 EQ/NE/GT/GE/LT/LE、literal scalar 和 AscendC 六参数
indexed scalar-buffer form，输出按仓库 golden 使用 little-endian bit packing；scalar
buffer 的单元素 region 会参与 bounds、poison 和 RAW dependency。
select 已支持 tensor-tensor、literal tensor-scalar 和 indexed BufferLoad scalar，覆盖普通/
显式 scratch forms、`VSEL_TENSOR_*` 与 `VSEL_CMPMASK_SPR`；mask、tensor source 和
scalar source region 都参与 bounds、poison 与 dependency，执行器按同一 packed predicate
选择结果。
tail reduction 已贯通仓库当前验证的 float32、axis 0、clear=true contract，支持
sum/max/min 对二维 valid rectangle 按列归约。axis 1、workspace 与 accumulate 经
2026-09-04 复核确认为上游 contract 缺失：传播 pass 仅在 axis 0/-2、clear=true、
float32、无 tmp、非累加时重写 tail_reduce（`ascend_tail_mask_propagation.cc`
supported_contract），C++ helper 亦只实现该路径（`(void)dim; (void)clear;`）——这些
variants 在上游暴露实现前继续 fail-closed。
deprecated AscendC `block_reduce_{sum,max,min}` 已支持 float16/float32、静态 mask、多个
repeat 及显式 source/destination stride；每个 32B block 独立归约，partial block mask、
跨 repeat footprint 和依赖均按真实访问建模。`wholereduce{sum,max,min}` 也已支持同一
mask/repeat/stride contract；max/min 覆盖 `ORDER_ONLY_VALUE` 和 value/index bitcast 输出。
普通 row reduce 已支持 trailing `physical_cols` narrow-row contract：二维 slice offset 会
保留物理 row stride，并严格检查 axis=-1、logical≤physical、32B row alignment、单个
256B repeat 和 clear=true。
Cube copy 已有两条 GM→L1 路径：显式 RowMajor 路径按二维 stride 搬运；默认路径将有效
矩形按仓库 `ascend_layout.py` 的 zN 公式写入真实物理存储，并复现 tail tile 清零。默认
路径接受 fractal/C0 对齐的静态物理 tile：tile-base（或整 tile 环形槽）偏移作为 primary
copy 清零整个物理 tile；非整 tile 偏移按 codegen 的 splice / vertical-merge 语义建模，
锚点须落在同一 zN 视图的分形行边界上，只写有效矩形触碰的分形行、逐 fractal 列带分段
写入，不清零也不破坏同 tile 内先前的 DMA 数据。
UB→UB copy 已支持 op lowering 的最终 8 参 ABI（模板 `<dstT, srcT, len>` 随操作名传
递）：同 dtype 按二维 tile/row stride 连续或跨行搬运，异 dtype 按包装层方向表执行
CAST_NONE（五种 widening 组合）或 CAST_RINT（整型目标舍入到最近偶数、浮点目标 IEEE
最近偶数转换），bfloat16 继续 fail-closed；分类为 AIV/VECTOR pipe。UB→L1 copy 已支持
Nd2Nz 语义（half、fractal/C0 对齐静态 tile，AIV/MTE3），复用 zN 物理打包路径。
L1→L0 已支持 tile-base 路径：L0A 执行 zN→zZ，L0B 执行 zN→nZ；transpose 路径利用
zN(A) 与 nZ(Aᵀ) 的物理等价关系，并保留 L1 写入到 L0 读取的 RAW dependency。
L1→L0 还支持 32-byte 对齐的 L1 source window：从 zN/nZ 物理 offset 反解逻辑 origin，
每个 C0 片段使用真实二维 physical stride，跨 C0 窗口由多段 source regions 拼接；实际
memory access 和 dependency 均不再退化为完整 L1 tile。
standalone L0C→GM 已支持 accumulator fractal 到 RowMajor GM 的 valid rectangle、
float32→float16 转换和 ReLU。`unitFlag=0` 可独立执行；`unitFlag=0b11` 可与同 core、同
L0C region、有效列数一致的 MMA release 配对，fixpipe source footprint 只读取有效列。
显式 half→float32 MMA 已支持 L0A/L0B 物理 payload 解码、K-tail、`init=true` 覆盖和
`init=false` 累加；已有 GM→L1→L0A/L0B→MMA→L0C→GM 的真实 final-TIR NumPy golden，
整条功能模拟路径不调用 BiSheng。显式 MMA 已支持静态、16 列粒度的 partial `n_actual`，
并将 L0B/L0C footprint 收窄到有效列；还支持 MMA `0b10` hold、累加 MMA `0b11`
release、fixpipe `0b11` consume 的配对链，pair task id/role 会进入 trace metadata。
bf16/fp32 input 等其它 dtype 仍 fail-closed。整数 `BufferLoad` 形式的 `n_actual`
会保留为稳定绑定名
（如 `nact[0]`），在 `FunctionalSimulator(bindings=...)` 创建时解析并检查范围和对齐。
高层 half→float32 `gemm_v0` 也可直接执行，支持 A/B transpose、K-tail、init/accumulate，
并已打通 GM→L1→gemm_v0→L0C→GM，同时复现模板的 kL0Size、N tiling 和 L0A/L0B slot
capacity legality check。当 `gemm_v0` 有多个 K/N step 时，bridge 会展开每步 L0A/L0B
MTE1 load 和 MATRIX stage，
模拟双 slot ping-pong reuse dependency 与 copy/compute overlap；内部 stage 已物化真实
L0A/L0B payload，逐 K tile 累加并按 N tile 写入对应 L0C column band。内部 load 的 L1
窗口已拆为精确 zN C0 source regions。各 stage 已有独立的 `gemm_v0.load_a/load_b/mma`
timing key、transfer bytes/math ops 和 MTE1↔M event annotation；周期仍使用可替换的
`uncalibrated-unit-cost` profile，annotation 尚不是显式硬件 flag task。
实际 `examples/gemm/example_gemm.py` 已提供 `--simulator --platform A2/A3` 入口；单 tile
以及多 M/N tile、两段 K 累加均与 PyTorch 结果匹配。final TIR 中循环相关的
`init=(k == 0)` 会按每个静态展开迭代解析，不再要求源表达式本身是 literal。
`gemm_v0` 还支持仓库 AscendC contract 中 `transpose_B=true` 的静态 partial `n_actual`：
只加载/计算 16 列对齐的输出前缀，并在多 K tile 下保持正确 zN stride 和 L0C 累加；
非 transpose-B partial-N 仍拒绝；transpose-B 的整数 `BufferLoad n_actual` 已支持，并让
符号 L1→L0B、MMA 和 L0C footprint 在运行时使用同一个绑定值。
显式 `mma` 与高层 `gemm_v0` 还支持仓库已有真机 contract 的 int8×int8→int32 路径；
layout codec 按 32B C0 自动采用 int8 的 32-element K block，执行使用 int32 matmul 与
累加。测试覆盖 A2/A3、K-tail、init/accumulate、transpose，以及多 K-stage 的 L1 source
window 与 L0A/L0B ping-pong 展开，并对非法 int8 accumulator dtype 保持 fail closed。
BT bias vertical slice 已打通：float32 RowMajor L1 通过 64B 对齐的 `copy_l1_to_bt` 写入
逐 core BT，随后 half→float32 `mma_bias` 从 BT 初始化 L0C 并沿 M 维广播 bias。
`copy_l1_to_bt` 在 MTE1、`mma_bias` 在 MATRIX pipe，bias region 作为真实 read operand
参与 RAW dependency；模板中被保留但不控制初始化来源的 `init` 两种取值均已覆盖。
`examples/flash_attention/flash_attn_bhsd.py` 所需的 Cube vertical slice 已完整执行：
GM→zN L1、L1→L0A/L0B、half→float32 MMA，以及 L0C→RowMajor GM 的 NO_QUANT
fixpipe；A2/A3 都通过 PyTorch 数值对比。这里的“其余 Cube/fixpipe variants”专指该
FA 不会发射的 quant、非 RowMajor、L0C→UB 和其它 dtype/layout contract，不能理解为
FA 主链仍靠跳过执行通过。
`examples/convolution/im2col_conv2d.py` 进一步打通 float16 C0=16 的
GM→L1→im2col→L0A→MMA→L0C→GM 链，覆盖 padding 和 M tail，并在 A2/A3 simulator
与 PyTorch conv2d 对齐。

尚未支持任意 TIR Call 或未列入上述白名单的动态表达式、通用 mask、完整 software
pipeline、其余 Cube copy/MMA/fixpipe variants，以及后续 P3–P8 operation。
JIT adapter 已支持 CPU Torch/NumPy tensor 调用；当 bridge 无法为实际 kernel 生成完整
可执行 operands 时继续 fail-closed。

### 开发和验证环境

当前 macOS ARM 环境已有 `3rdparty/tvm/build` 的 CPU-only TVM，并在仓库根目录 `build/`
构建了 `libtilelang_module.dylib`；构建产物不提交。
它可以验证 Memory/SimIR/scheduler/真实 TVM TIR bridge，不能验证 CANN codegen、A2/A3
真机功能或 timing calibration。

macOS Command Line Tools 26.x 需要显式使用 SDK 中的 libc++ headers。重新配置/编译时：

```bash
export CPLUS_INCLUDE_PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
cmake -S . -B build -G Ninja \
  -DTVM_PREBUILD_PATH="$PWD/3rdparty/tvm/build" \
  -DTVM_SOURCE_DIR="$PWD/3rdparty/tvm" \
  -DUSE_ASCEND=OFF -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_LLVM=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tilelang_module -j8
```

运行前设置本地 TVM/TileLang 路径，可避免依赖系统安装：

```bash
export PYTHONPATH="$PWD:$PWD/3rdparty/tvm/python${PYTHONPATH:+:$PYTHONPATH}"
export TVM_LIBRARY_PATH="$PWD/3rdparty/tvm/build"
export TILELANG_LIBRARY_PATH="$PWD/build"
```

运行 simulator 回归：

```bash
cd /Users/wzz/Desktop/Research/kernel-tool/cannbot-skills/tilelang-ascend
PYTHONPATH=.:3rdparty/tvm/python \
TVM_LIBRARY_PATH="$PWD/3rdparty/tvm/build" \
TILELANG_LIBRARY_PATH="$PWD/build" \
  /Users/wzz/miniconda3/bin/python -m pytest testing/python/simulator -q
```

当前基线为 `519 passed`。TVM 在 Python 3.13 下会产生 parser deprecation warnings；这些
不是 simulator failure。本地 CPU 已能运行 simulator lowering/JIT；NPU codegen 和真机
交叉验证仍需要 Linux+CANN，最终 timing 还需要分别在 A2/A3 真机校准。

### 接手顺序建议

接手者先运行上述 519 个测试并阅读最近提交，再按本文件“下一批工作”推进。优先维持
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
- [x] ~~接通公开 CPU Torch/NumPy tensor adapter：按最终 PrimFunc 参数映射 GM 输入，
  自动读取 out_idx、绑定输入动态 shape、返回同类 CPU tensor，并在功能执行后写 trace；
  `examples/elementwise/elementwise_add.py` 已增加 `--simulator`/A2/A3/trace 入口。~~
- [x] ~~定义 A2/A3 配置、拓扑、lane、pipe、task、core program 和 kernel program。~~
- [x] ~~实现 SimIR DAG 校验、未知依赖检查和确定性任务顺序。~~
- [x] ~~实现常量 `For`、`IfThenElse`、`LetStmt`、`blockIdx.x/y`、AIC/AIV resource
  scope 的 TIR bridge。~~
- [x] ~~采集 PrimFunc 参数 buffer、`Allocate` 和 `Block.alloc_buffers`。~~
- [x] ~~未知 operation、未知 memory scope 和 shmem fail-closed。~~
- [x] ~~构建 macOS CPU-only TVM，并用真实 `tvm.tir.PrimFunc` 验证 bridge。~~
- [x] ~~在 macOS CPU 环境端到端运行仓库 `examples/elementwise/elementwise_add.py`：
  A2/A3 的 `128×256` 以及 A2 默认 `1024×1024` 均通过 Torch golden。~~
- [x] ~~在 macOS CPU 环境端到端运行仓库 `examples/flash_attention/flash_attn_bhsd.py`：
  A2/A3 默认 `B=1,S=128,H=1,D=512` 均通过 Torch golden，最大绝对误差
  `0.00048828125`；example 已增加 `--simulator`、`--platform` 和 `--trace` 入口。~~
- [x] ~~在 macOS CPU 环境端到端运行实际 merge_sort、GEMM 和 cann-bench gather：
  merge_sort 的 2/3/4-way 完整 value/index 记录、GEMM 的单 tile 与多 tile/多 K-step、
  gather 的 16 行循环均在 A2/A3 与 Torch golden 匹配。~~
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
- [x] ~~实现 AscendC `exp_experiment` 的二维 strided masked window：fp32 每行 64 元素、
  fp16 每行 128 元素，覆盖多行、in-place、column offset、physical stride、A2/A3、
  dtype/extent/alignment/cross-row 错误路径和实际语言层 final TIR。~~
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
- [x] ~~实现 createvecindex/arith_progression 的 offset、常用 dtype、runtime
  first/difference/count、wraparound、dependency 和 template dtype 验证。~~
- [x] ~~实现 bitwise AND/OR/XOR/NOT 与 scalar left/right shift 的 offset、count、
  int16/int32/uint16/uint32 定宽语义、scratch dependency 和 shift 范围验证。~~
- [x] ~~实现 leaky_relu 和 in-place accumulator 语义的 axpy。~~
- [x] ~~实现 float16/float32 mul_add_dst 的 offset、count、dtype validation 和 in-place
  accumulator dependency/执行语义。~~
- [x] ~~实现 sigmoid、silu、sin、cos 的普通与 scratch forms。~~
- [x] ~~实现 A2/A3 float16/float32 round 的普通/scratch forms、dtype validation 和
  ties-away-from-zero 语义。~~
- [x] ~~实现 sub/abs/mins_experiment 的显式 count ABI、A2/A3 float16/float32 NumPy
  语义、memory dependency 和负 count validation。~~
- [x] ~~实现 reducesum_experiment 的可选 scratch/count/单结果 ABI，以及 sum_experiment
  的 outter/inner/n 行归约、padding 排除、template/dtype/extent validation。~~
- [x] ~~实现 gather 的 flat-source 字节 offset、base address、scratch、alignment、dtype、
  source-bounds、poison 和 dependency 语义。~~
- [x] ~~实现 gatherb 的八个 32B block/repeat、目标 block/repeat stride、连续 offset、
  template/control/alignment/source-bounds validation、poison 和 dependency 语义。~~
- [x] ~~实现 gather_mask 七种 fixed pattern 与 custom uint32 index 的稳定压缩、tail-zero、
  scratch/template/dtype/capacity/source-bounds validation 和 dependency 语义。~~
- [x] ~~实现静态 rank-2 UB transpose 的非方形 tile、常用 B8/B16/B32 dtype、shape/scope/
  alignment validation、poison 和 dependency 语义。~~
- [x] ~~实现 UB reinterpretcast 的零拷贝 allocation rebinding、跨 dtype 原始字节双向
  alias、按 core RAW/WAR/WAW dependency 和 size/type/scope validation。~~
- [x] ~~实现 pow 和 clamp/max/min 的普通与 scratch forms。~~
- [x] ~~实现 CAST_FLOOR、CAST_CEIL、CAST_ROUND ties-away-from-zero 和 CAST_TRUNC。~~
- [ ] 为 duplicate 补齐 TileLang builtin/language/codegen contract 后实现模拟；当前树只有
  sync 配置和 simulator 白名单，不能猜测 ABI。确认 CAST_ODD，并为已有 unary/cast
  补齐 dtype/异常值语义。
- [x] ~~实现 rank-1/2 broadcast、两条二维广播轴和普通/scratch forms。~~
- [x] ~~实现 compare/compare_scalar 的六种 mode、literal scalar 和 packed mask。~~
- [x] ~~实现 select tensor-tensor/tensor-scalar 的普通与 scratch forms，并将 packed
  mask 纳入显式 read dependency。~~
- [x] ~~实现 tail_compare/tail_compare_scalar 的二维有效矩形、逐行 packed mask、
  物理 stride 和 literal scalar。~~
- [x] ~~实现 tail_select tensor/scalar、可选 tmp、逐行 mask 解包和有效矩形写入。~~
- [x] ~~实现 tail_broadcast 两条二维广播轴、可选 scratch、valid rectangle 和
  AscendC 32-byte source row stride。~~
- [x] ~~实现 compare_scalar indexed BufferLoad，保留单元素 scalar region、RAW dependency
  和六种 mode 的 packed-mask golden。~~
- [x] ~~实现 select indexed BufferLoad 和 `VSEL_CMPMASK_SPR`，覆盖 source type 0/2、
  optional scratch、scalar region/dependency 和 packed-mask golden。~~
- [ ] 实现其它 tail/mask variants。
- [x] ~~实现真实 `tail_reduce` 的 float32 axis-0 clear=true sum/max/min，并覆盖
  valid rectangle、dependency 和非法 dim/clear contract。~~
- [x] ~~实现普通 float32 reduce 的 sum/max/min、axis 0/-1、clear=true row tmp 和
  dependency。~~
- [x] ~~实现普通 reduce clear=false 的 dst accumulator merge，以及 PTO column 单
  output tmp、row main/output 双 tmp dependency。~~
- [x] ~~实现 deprecated AscendC block reduce sum/max/min 的 float16/float32 静态
  mask/repeat/stride contract，覆盖 partial block、跨 repeat footprint 和错误路径。~~
- [x] ~~实现 whole reduce sum/max/min 的 float16/float32 静态 mask/repeat/stride，
  覆盖 only-value、value/index bitcast、partial mask、跨 repeat footprint 和错误 order。~~
- [x] ~~实现 narrow physical row reduction 的 slice offset/stride 和 axis、宽度、对齐、
  单 repeat 宽度、clear contract 验证。~~
- [ ] 覆盖 float16、bfloat16、float32、int/uint 常用 dtype 的舍入、溢出和饱和语义。
- [ ] 增加动态 shape、动态 tail、misalignment、partial write 和非法 copy path 测试。

## P2：Cube MVP

- [x] ~~按仓库 `ascend_layout.py` 公式实现 zN/nZ 的 C0/fractal storage size、物理
  index 和 NumPy pack/unpack codec。~~
- [x] ~~实现显式 RowMajor `copy_gm_to_l1_linear` 的 valid rectangle、scope/dtype
  contract 和 32-byte row alignment。~~
- [x] ~~实现默认 zN `copy_gm_to_l1` 的 aligned physical tile/base 路径、tail clear
  和真实物理 pack。~~
- [x] ~~实现 tile-base L1→L0A/L0B 的 EasyASC A2 ZZ/NZ physical payload、transpose
  语义、容量检查和跨级 RAW dependency。~~
- [x] ~~完成第一轮 EasyASC Cube layout audit，修正 L0A fractal 内顺序，并用独立 raw
  offset golden 区分 L0A ZZ 与 L0B NZ。~~
- [ ] 对 EasyASC A2 `pipe_cube.py` 的 copy/MMAD/fixpipe 建立逐项 parity/differential
  测试；contract 不同的项记录差异而不是强行套用。
- [x] ~~实现 standalone L0C→RowMajor GM 的固定 accumulator layout、valid rectangle、
  dtype conversion、ReLU 和 `unitFlag=0` fixpipe。~~
- [x] ~~实现 32-byte 对齐、可映射为逻辑 origin 的 sliced L1→L0A/L0B source window。~~
- [x] ~~为单 C0 zN/nZ source window 建立二维 strided region，收窄实际 memory access 和
  hazard footprint。~~
- [x] ~~将跨 C0 sliced L1→L0 拆成多段 strided source regions，并用于功能读取和
  dependency footprint。~~
- [x] ~~实现非对齐/子 tile GM→L1：非整 tile 偏移的 copy_gm_to_l1 按 codegen
  splice / vertical-merge 语义建模——锚点须为同一 (dstM, dstN) zN 视图的分形行边界，
  逐 fractal 列带分段写入有效矩形触碰的分形行（`dst_regions` 参与 RAW/WAR/WAW），
  不清零先前 DMA 数据；整 tile 偏移（含环形槽 base）保持 primary clear 语义。覆盖
  单带/多带 golden、无 clobber、未写行 poison、错位偏移和容量错误路径。~~
- [ ] 实现其它 L0C→GM variants。
- [x] ~~实现 half→float32 `gemm_v0` 功能语义，覆盖 A/B transpose、K-tail、
  init/accumulate 和无 BiSheng 端到端路径。~~
- [x] ~~将多 step `gemm_v0` 展开为 MTE1/MATRIX trace stages，建模双 slot reuse dependency
  和 copy/compute overlap，并保持单次功能结果提交。~~
- [x] ~~为 `gemm_v0` 内部 stages 物化双 slot L0A/L0B payload，执行 K-tile accumulate 和
  N-tile L0C column-band 写入；覆盖四种 transpose 组合。~~
- [x] ~~为 `gemm_v0` 内部 K/N stages 建立精确 zN C0 source regions 和 offset golden。~~
- [x] ~~为 `gemm_v0` 内部 stages 提供独立 timing key、工作量 metadata 和 MTE1↔M
  event annotation，并保持 calibration 状态进入 trace。~~
- [ ] 校准 `gemm_v0` 的 A2/A3 stage timing；按实际 lowering contract 将 event annotation
  升级为显式硬件 flag/wait task。
- [x] ~~实现显式 half→float32 `mma`，支持 K-tail、init 和 accumulate。~~
- [x] ~~实现多 MMA task 对同一 L0C 的 K 分块累加与 RAW dependency golden。~~
- [x] ~~实现显式 MMA 及 transpose-B `gemm_v0` 的静态、16 列粒度 partial `n_actual`，
  收窄 L0B/L0C footprint，并覆盖 init/accumulate 与多 K tile。~~
- [x] ~~实现 explicit MMA 与 transpose-B `gemm_v0` 的整数 BufferLoad `n_actual`，通过
  simulator bindings 解析符号 footprint，并在执行前校验范围和 16 列粒度。~~
- [x] ~~实现显式 `mma` 与高层 `gemm_v0` 的 int8×int8→int32 功能路径，覆盖 K-tail、
  init/accumulate、transpose 和多 K-stage 展开。~~
- [ ] 实现 bf16/fp32 input 等其它硬件支持的 MMA dtype，并明确各模式的精度语义。
- [x] ~~实现 MMA/fixpipe `unitFlag` 配对协议，覆盖 `0b10` hold、`0b11` release/consume、
  partial-N source footprint、同 L0C region/列数校验和 trace pair metadata。~~
- [ ] 实现 quant、非 RowMajor 输出和其它 fixpipe variants。
  说明（2026-09-04 调研）：当前仓库没有任何 quant fixpipe 发射点——
  `tl_templates/ascend/common.h` 的 `copy_l0c_to_gm` 硬编码
  `ScaleGranularity::NO_QUANT`，语言层与 codegen 亦无 quant op；GM 输出 layout 仅
  `LayoutGM = layout::RowMajor`。两者在语言层/codegen 暴露实际 TIR contract 前无法
  按可信源实现，继续 fail-closed。
- [x] ~~支持显式 MMA/gemm_v0 的 A/B transpose，以及 zN/nZ、L0A zZ、L0B nZ、L0C
  fractal 等当前计算链实际使用的核心 layout。~~
- [x] ~~使用 NumPy/PyTorch golden 覆盖 M/N/K tail、多 core 分块和 pipeline overlap；
  实际 GEMM example 另覆盖 4 core、两段 K 累加。~~

## P3：完整搬运、Vector 与 Reduction

- [ ] 覆盖最终 TIR 中全部合法 A2/A3 copy 方向。
- [x] ~~实现 UB→L1 的 `copy_ub_to_l1` Nd2Nz 路径（half、fractal/C0 对齐静态 tile、
  AIV/MTE3 分类、zN 物理打包、模板 dtype 与 scope 校验）。~~
- [ ] 实现 L0C→UB 的 GM workspace/CV handoff 路径；`copy_l0c_to_ub` 目前会被
  reduction 改写为 `copy_l0c_to_gm` 且 codegen 无独立 handler，需先确认实际 contract。
- [ ] 完成 vector operation family、standalone transpose 和完整 reduction family。
- [x] ~~实现 gather 的 byte offset/base address、gather-mask 的 7 种 fixed pattern 与
  uint32 custom index、gatherb 的每 repeat 8 个 32B block 和 destination strides；覆盖
  A2/A3、dtype、边界、alignment、dependency 与 runtime repeat。~~
- [ ] 建立 operation × dtype × scope × alignment 支持矩阵。
- [ ] 对每条 memory path 增加越界、容量、alias 和 address-range hazard 测试。

## P4：Bias、Fixpipe 与量化

- [x] ~~实现 float32 RowMajor L1→BT 的 64B 对齐 copy，以及 half→float32
  bias-initialized MMA 的 M 维广播、RAW dependency 和 init 参数兼容语义。~~
- [ ] 支持非 64B 整倍数 bias copy 的真实向上取整 DMA footprint，并补齐其它合法
  bias dtype/shape variants。
- [x] ~~完成当前仓库 `NO_QUANT + RowMajor` fixpipe 的 ReLU on/off、valid rectangle、
  fp32→fp32/fp16 与 int32→int32 输出。~~ 其它 fused/quant/layout 选项等待语言层暴露。
- [x] ~~实现 standalone L0C→GM 和 UB→UB 的 fp32→fp16 转换。~~ bfloat16 host 表示与
  位级 contract 尚未落地，继续 fail-closed。
- [ ] 实现仓库实际暴露的 integer quant/dequant、rounding 和 saturation 语义。
- [ ] 增加 bit/ULP golden、边界值、BT capacity 和 MMA/fixpipe 配对协议测试。

## P5：索引、排序与 TopK

- [x] ~~完成仓库公开语言层实际暴露的 gather、gather-mask 和 gatherb variants；实际
  cann-bench gather final TIR 已在 A2/A3 执行。storage rewrite 后的 reinterpret view
  使用 access_ptr 逻辑 dtype 并按位物化，避免循环间 alias 泄漏。~~
- [x] ~~实现 sort32 的多 repeat 32-element block 排序、float16/float32 物理 value/index
  编码、alignment/footprint/dtype validation 和 dependency。~~
- [x] ~~实现 sort 的最终 workspace ABI、静态/runtime actual count、float16/float32
  有效前缀稳定排序、数值型 value/index 输出和 runtime contract validation。~~
- [x] ~~实现 merge_sort 的 float32 两槽和 float16 四槽 8B 记录、2/3/4-way、
  AscendC/PTO 两种最终 ABI、多源 dependency、稳定合并、float16 payload 位型保留和
  输入/extent/scope/alignment/dtype validation；修复语言层 float16 block length 计算。~~
- [x] ~~确认并修复 init_sort_buf ABI 后实现：语言层发射顺序对齐硬件包装层
  `InitSortBuf(src, eleNum, rsv=0)`（原 `(template, buffer, rsv, num)` 会使 codegen 将
  `rsv` 当作 `eleNum` 传入并丢弃 `num`）；模拟器按 `common.h` 语义在 float32 UB
  workspace 的 int32 视图上交错填充（偶数 lane fp32 -inf 位型、奇数 lane -1，仅覆盖
  完整 64 元素块，尾部保持 poison），并验证 UB scope、template dtype、静态 num/extent、
  WAW/RAW 依赖和 read-before-write 错误路径。float16 缺少可靠 golden，继续 fail-closed。~~
- [x] ~~实现 TopK 的最终 workspace ABI、静态/runtime actual count、float16/float32
  稳定降序 value/index 输出、dependency 和错误路径。~~
- [x] ~~验证重复值、稳定性/index 语义、奇数 valid count、动态 valid length 和 scratch
  buffer bounds：TopK/sort32/sort/merge_sort 均覆盖重复值稳定顺序与 payload/index，sort
  覆盖 37/33 元素及 runtime actual_num；TopK/sort 按 A2/A3 AscendC workspace 公式、
  merge_sort 按 PTO workspace 公式检查最小 scratch，merge_sort 另覆盖 32B aligned
  source/destination offset。NaN 排序在
  缺少 A2/A3 golden 时明确 fail-closed。~~

## P6：Convolution 与 Im2Col

- [x] ~~实现 float16/float32、单 C0 的 `im2col`（分别为 C0=16/8），覆盖
  padding、stride、dilation、M tail、zN L1 输入与 zZ L0A 输出，并按 A2 已验证
  contract 拒绝非零 posM/posK。~~
- [x] ~~实现显式 im2col + MMA + NO_QUANT RowMajor fixpipe convolution，并由实际
  `examples/convolution/im2col_conv2d.py` 在 A2/A3 simulator 做端到端数值验证。~~
- [x] ~~实现 fp32 C0=8 的 im2col + fp32 MMA：修正 L0B nZ 编码，处理 K=72 的
  zN L1 尾部分形 footprint，并由 CPU JIT 回归在 A2/A3 验证双 C1 累加的完整 conv2d
  与 NumPy reference 一致。~~
- [ ] 接入仓库中存在的高层 convolution lowering 形式。
- [ ] 对实际完整 convolution kernel 验证 stride、dilation、非对称 padding 和 spatial
  tail；当前这些 im2col 参数已在 bridge golden 覆盖，完整端到端只覆盖了 padding 和多 C1。
- [ ] 将 A2/A3 硬件限制编码为明确 legality check。

## P7：Atomic 与 Persistent

- [x] ~~实现 UB→GM atomic_add 的 float16/float32、二维有效矩形、GM stride、目标 seeding
  要求和 read-modify-write 功能语义。~~
- [x] ~~实现 L0C→GM atomic_add 的 layout 解码、有效尾块、GM stride，以及
  fp32→fp32/fp16、int32→int32 转换。~~
- [ ] 在 host bfloat16 表示与 layout codec 落地后补齐 L0C→GM atomic_add 的
  fp32→bfloat16 路径。
- [x] ~~对同一 GM region 的跨 core atomic 冲突提供确定性的 DAG 序列化，并在 trace 中
  验证前一原子写结束后下一写启动。~~
- [ ] 实现 Persistent kernel 的 work distribution、residency 和 liveness 模型。
- [ ] 覆盖 atomic contention、初始化规则、终止条件和 deadlock。

## 同步、流水与调度

- [x] ~~实现 dependency DAG 和 `(core, lane, pipe)` FIFO 离散事件调度。~~
- [x] ~~允许不同 pipe overlap，并统计 makespan、utilization、wait 和 per-core completion。~~
- [x] ~~实现 local set/wait 的单 latch、event 0..7、source/destination pipe 作用域，以及
  manual/auto intrinsic metadata 解码。~~
- [x] ~~按 EasyASC 语义实现 mode-2 cross flag：Cube→Vector 双 lane 独立 fan-out、
  Vector→Cube 双 lane join、方向隔离和 15 个 outstanding phase 上限。~~
- [x] ~~实现 lane-local barrier_all、pipe-local pipe_barrier，并验证同步不会错误阻塞无关
  lane/pipe。~~
- [x] ~~实现 cross flag mode 0：所有 AIC 或所有 core 上的双 AIV lane 按 phase 做全局
  collective，每个 waiter 独立推进消费 phase。~~
- [x] ~~实现 cross flag mode 1：同一 core/group 的两个 AIV lane 按 phase 做 collective，
  并拒绝 Cube 参与及同一 wait namespace 的 mode 歧义。~~
- [x] ~~将同步等待输出为显式 trace wait event。~~
- [x] ~~实现 max cycles、wall timeout 和基础 deadlock 报告。~~
- [x] ~~跟进主线 `e9f9456` 对自动同步优化的回滚：真实两核 IfThenElse lowering 在每个
  core 恢复 if 前后 `auto_barrier(PIPE_ALL)`；bridge 保留该任务，scheduler 按 lane
  drain 前序 pipe 并阻止后续任务越过。A2/A3 数值与 barrier schedule 回归均通过。~~
- [x] ~~区分“用于确定性功能执行的内存 dependency”和“硬件真实同步”：对同 lane
  跨 pipe 的 RAW/WAR/WAW 边，要求存在匹配 local set/wait flag 或 `PIPE_ALL`。使用同一份
  实际 FA 对比验证：buggy `90bb652` 在 A2/A3 均报告 MTE2→MTE1 missing synchronization，
  修复版 `e9f9456` 在 A2/A3 数值通过；避免 NumPy DAG 静默替编译器补同步。~~
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
- [x] ~~当前 simulator 核心测试基线：527 passed，覆盖 memory、scheduler、sync、trace、
  functional executor、真实 TIR bridge、跨 pipe 同步 hazard 和 shmem rejection；另有
  2 个依赖 native module 的 JIT 集成测试文件，覆盖自动同步回滚以及 FA 使用的二维
  row broadcast/runtime float scalar。~~
- [ ] 每个 operation 必须有正向、错误路径、dtype、shape/tail、scope 和 trace 测试。
- [ ] PTO 是第一验证目标；随后补齐 AscendC intrinsic parity。
- [ ] 在 Linux+CANN CI 运行 lowering/JIT 集成测试。
- [ ] 在 A2/A3 真机分别完成 functional 对比和 timing calibration。
- [ ] 发布 machine-readable operation coverage table，列出精确支持和不支持的 variants。

## 下一批工作

1. ~~`elementwise_add.py`、`flash_attn_bhsd.py`、`im2col_conv2d.py`、2/3/4-way
   `example_merge_sort.py` 和独立
   `example_gemm.py` 已在 A2/A3 simulator 端到端通过；merge 校验完整 value/index 记录，
   GEMM 覆盖单 tile 与多 tile/多 K-step，卷积覆盖显式 im2col/MMA/fixpipe。实际
   cann-bench gather 也已在 A2/A3 通过。~~ 下一步优先扩展多 C1、stride/dilation/
   asymmetric padding 的实际卷积矩阵，然后继续高频二维 vector/mask variants。
2. ~~排序族 offset、重复值稳定性、奇数/runtime valid length、scratch bounds 已补齐；
   NaN 在没有可靠 A2/A3 golden 时 fail-closed。~~ `init_sort_buf` 的 float16 workspace
   预填仍等待可靠 golden。
3. fixpipe quant variants 等待语言层/codegen 暴露实际 TIR contract（当前仓库仅有
   `NO_QUANT` 路径）；bf16/fp32 input MMA 等待可靠的位级精度 contract。
4. 当前仓库能发射的主流 NO_QUANT RowMajor Cube/fixpipe 路径已纳入实际 FA/GEMM/conv
   验收。剩余 copy variants 以真实 final TIR 出现为准；`copy_l0c_to_ub` 须先确认被
   workspace reduction 改写前后的实际 contract，再实现 CV handoff。
5. 在可获得 A2/A3 测量数据后校准 `gemm_v0` stage timing，并验证显式 flag contract。
