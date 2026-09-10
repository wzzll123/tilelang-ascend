# WQBM D53/D54 同步问题复盘

**状态**：已按官方内存一致性模型修正（2026-09-10）

> 文件名为历史名称。最终实现不存在 `gm_visibility` 配置，也不存在
> `gm-visibility-window` 诊断。

## 结论

WQBM D53/D54 是 MTE 异步流水的完成序/同步链问题，不应泛化成普通 MTE GM
访问的“GM/L2 缓存一致性”。官方文档明确区分：

- A2/A3 L2 是核外共享缓存，不存在每核私有 L2 副本问题；
- 普通 MTE3 写 GM、普通 MTE2 读 GM 不经过 Scalar DCache；
- DCCI 适用于 Scalar DCache、SIMT DCache，以及支持缓存的 NDDMA 路径；
- MTE/Fixpipe 指令顺序发射但访存异步完成，依赖必须由 pipe fence、跨流水
  notify/wait 或核间同步建立。

因此模拟器只使用统一的 `hazard_check=off|warn|error`，输出
`missing-pipe-synchronization`。只有确认经过私有 Cache 的访问路径，未来才应进入
独立的 cache-coherence 模型。

## D54 的同步链

修复态的抽象依赖为：

```text
MTE3 write workspace
  -> CrossCoreSetFlag<mode=0, PIPE_MTE3>
  -> CrossCoreWaitFlag
  -> PipeBarrier<PIPE_MTE2>
  -> MTE2 read workspace
```

校验器把它解释成传递可达链：

```text
producer pipe -> completed collective -> consumer-pipe fence -> consumer pipe
```

缺少最后的 consumer-pipe fence 时，collective 只完成到达/等待关系，不能独自证明
后续消费 pipe 的访问已经被约束，因而报告普通 missing synchronization。直接匹配的
`MTE3_MTE2` set/wait 或 `PIPE_ALL` 也构成有效路径。

## 真机 A/B

环境：NPU3 的 `wzz_cann`，Ascend 910B3，物理卡 2。两份扩展从同一归档源码重新
编译，broken 版本只删除 D54 读前的一个 `PipeBarrier<PIPE_MTE2>()`。

输入为 M=256/N=7168/K=14336，全 1 fp16 x、int8 weight、fp16 scale 和 offset；
精确期望输出为 28672。

| 版本 | 结果 |
|---|---|
| 保留 `PIPE_MTE2` | 连续 10/10 次所有元素均为 28672 |
| 删除 `PIPE_MTE2` | 连续 10/10 次错误；出现 14336，均值在 21068–21632 间波动 |

该实验确认同步点确实必要，但不能单独证明存在私有 Cache 副本或“GM/L2 visibility”
机制。后续若要进一步区分具体硬件路径，应使用只改变同步原语的最小 kernel，而不是
把 WQBM 时序现象推广成所有 GM RAW 的规则。

## 回归要求

`testing/python/simulator/test_gm_visibility.py` 保留历史文件名，覆盖：

1. collective 单独存在：报 `missing-pipe-synchronization`；
2. collective 后接 consumer `PIPE_MTE2`：通过；
3. 直接匹配 `MTE3_MTE2` set/wait：通过；
4. 无关 pipe barrier：仍报错；
5. 跨核 producer/consumer 必须同时具备共同 collective 和 consumer-pipe fence；
6. A2/A3 使用同一通用同步规则；
7. 全量 simulator 测试保持通过。

## 非目标

- 不修改编译器 `ascend_sync_insert.cc`；
- 不宣称 `PIPE_MTE2` 是 Cache invalidation；
- 不用经验性的延迟值模拟真机竞争；
- 不为该问题增加公开配置开关。
