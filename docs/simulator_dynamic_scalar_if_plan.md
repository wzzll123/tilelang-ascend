# 模拟器动态标量控制流支持提案

> 2026-09-10，GQA 战役 HS29（FAI 对齐同步瘦身）期间沉淀。现状：模拟器 bridge
> 对**数据依赖标量 if** fail-closed 报 `UnsupportedSimOpError`，全量与
> sync_only 双模式都不支持，迫使 design 层加逃逸开关（GQA_SIM=1 剔除 HS23
> F217 分支）——逃逸降低了被模拟 kernel 与生产 kernel 的一致性。

## 1. 触发案例

GQA 的 HS23（F217 全 mask 行保护）epilogue：

```python
for row in T.serial(bm2):
    if m_i[row, 0] == MASK_NEG:      # 读 UB 标量做分支条件
        T.tile.fill(acc_o[row, :], 0.0)
```

`gqa_sim.py` 全 11 个结构 case 中 4 个 causal case（causal_spb/causal_kv3/
causal_var/decode_causal）因此在 bridge 层被拒：

```
tilelang.simulator.errors.UnsupportedSimOpError:
dynamic if condition is not supported by the first A2/A3 simulator bridge:
m_i[row] == T.float32(-1073741824)
```

这类"读 tensor 值决定分支"的结构在真实算子里并不罕见：全 mask 行保护、
early-exit（softmax max 不再更新时跳 rescale）、稀疏/变长路径、NaN 护栏。
每撞一次就要在 design 里开一个逃逸开关，被模拟的 kernel 就离生产 kernel
远一分——**模拟器的核心价值（sync 结构忠实）被自身缺口侵蚀**。

## 2. 分两模式的支持方案

### 2.1 全量模式：直接求值（应该容易）

全量模式 executor 本来就维护真实数值（numpy）。动态 if 的条件是标量
PrimExpr（BufferLoad/比较/算术），bridge 只需：

1. 允许 `IfThenElse` 节点的条件含 BufferLoad（当前拒掉的形态）；
2. 在 executor 的 `_if` 处理里，对条件表达式按既有标量求值路径求值
   （buffer 读走真实存储），按结果选择分支执行。

sync/hazard/flag 语义**不需要任何特判**——分支内的 set/wait/barrier 按
实际执行路径走，和真机完全一致。这本来就是"模拟器逐指令解释执行"的
题中之意，bridge 的 fail-closed 只是没实现，不是语义矛盾。

### 2.2 sync_only 模式：分支态合并（需要设计）

sync_only 跳过数值计算，条件**无值可求**。选项：

- **(a) 配置项 `dynamic_if: "error" | "then" | "else" | "both"`**（默认
  `"error"` 保持 fail-closed 哲学）：
  - `"then"/"else"`：强制走指定分支。适用于"分支稀有触发"的保护性代码
    （如 F217：正常输入永不进 then）——验证主路径选 `"else"`，验证保护
    路径选 `"then"`，两次运行覆盖双侧。
  - `"both"`：顺序执行两个分支并合并抽象状态（flag 电平/poison 标记取
    并集）。语义上偏保守：分支互斥的 flag 配对可能被误报不平衡，需要
    按"每分支独立账目、合并时取能配平的解释"处理——**复杂度高，建议
    二期**，一期只落地 `"then"/"else"`。
- **(b) 标量污染标记传播**：sync_only 已维护 poison/initialized 元数据，
  可把"读了 poison 标量的分支条件"建模为 nondet 并等价于 (a) 的 both。
  同样二期。

一期范围：**全量模式直接求值 + sync_only 的 `dynamic_if="then"/"else"`
配置**。

## 3. 附带缺口（同一提案顺带登记，不承诺一期）

- `T.tile.createvecindex` 未建模（gqa_sim.py 注释：causal sub_per_band 的
  createvecindex 路径"模拟器暂未建模"）——数值类 op，sync_only 下只需
  维护 initialized/poison 标记，工作量小，建议一并补。
- 其他 vector op 的 sync_only 标记覆盖建议顺手普查一遍（fail-closed 会让
  未建模 op 在 sync_only 下显形，普查成本就是跑一次结构覆盖集）。

## 4. 验收（test-first）

1. **红**：`testing/python/simulator/test_dynamic_if.py`
   - 最小 kernel：UB 标量读作 if 条件，then/else 各含一个 set_flag/wait_flag
     + copy。全量模式：按输入数据走正确分支，flag 配对判定与真机一致；
   - sync_only `dynamic_if="then"` / `"else"`：分别覆盖两分支，无死锁、
     flag 账目平衡；默认 `"error"` 仍 fail-closed（回归网）。
2. **实战验收**：GQA design 去掉 GQA_SIM=1 逃逸（保留 m_i[row,0] 修复），
   `gqa_sim.py` 全 11 case 在 sync_only（`dynamic_if="else"` 与 `"then"`
   各跑一遍 causal 组）+ 全量模式全绿。
3. 回归：`testing/python/simulator/` 全套既有测试不塌。
4. commit 到 tilelang-ascend ascendc_pto 分支；告知用户推 fork。

## 5. 环境/约束

- 仓：/home/wenzhongzhen/tilelang-ascend（ascendc_pto 分支）；
  改动面 `tilelang/simulator/`（bridge 的 IfThenElse 处理 + executor `_if`
  + config.py 加 `dynamic_if` 字段）+ `testing/python/simulator/`。
- 纯 CPU 测试，不需要 NPU 卡。
- 默认行为不变（`dynamic_if="error"`），全量模式行为不变（只是少拒）。
