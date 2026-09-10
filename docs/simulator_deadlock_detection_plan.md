# 模拟器死锁检测提案（从"超时兜底"升级为"死锁证明"）

> 实现状态（2026-09-10）：L1 已完成并默认开启，scheduler 在全局无 task 可推进时
> 立即报告 `DEADLOCK (global no-progress)`；L2 已完成 task/FIFO 依赖的 wait-for
> 环证明，并附有限长度最近事件。flag/cross-flag 阻塞会报告具体 id、方向和参与
> lane，跨 flag producer 的 lane 级闭环重建仍可继续增强。L3 已实现为独立
> `flag_balance_check="off"|"warn"|"error"`，默认 off。GQA 实证表明正确的循环
> 流水会在 kernel 结束时合法保留 slot-free 信用，因此原提案“所有 flag 必须归零”
> 会误报，不能作为默认正确性条件。

> 2026-09-10，GQA 战役 HS29 死锁定位期间沉淀。现状：模拟器只有
> `execution_timeout_s` 兜底——转够时间抛 `SimulationLimitError` 附 pending
> 任务清单。**无法区分"真死锁"与"单纯慢"**，也不报告卡在哪个 flag 上。
> 当日实证：multitask（96 task）sync_only 超时、pending 全是 AIC，与真机
> aicore timeout 互为印证后才敢判死锁——模拟器本可以**直接给出证明和
> 定位**，把小时级的真机盲猜压到分钟级。

## 1. 背景与问题

sync 调试是 kernel 战役最高频内环（wqbm D10/D19/D29、GQA HS2/HS4/HS29）。
死锁类 bug 的真机信号是 aicore timeout（无位置信息），模拟器的当前信号是
执行超时（有 pending 清单，无等待点信息、无死锁/慢的区分）。模拟器持有
全部 lane 的完整状态（flag 电平、pipe 在飞窗口、lane PC、pending 原因），
**具备做确定性死锁证明的一切信息**，缺的只是判定逻辑。

## 2. 方案：三层检测（按实现成本排序）

### L1 无进展检测（全局停顿 = 死锁充分条件）

每个模拟宏步检查全局可推进性：

- 所有 pending lane 都阻塞在 wait（flag / cross-flag / barrier）上；
- 且各 pipe 的在飞窗口为空（没有任何在飞 copy/mma/vector op 还能完成并
  触发 set_flag）。

两者同时成立 ⇒ 系统再无任何可发生的事件 ⇒ **结构性死锁，立即终止**。
（注意：pipe 在飞非空时不能判——在飞 op 完成后可能 set flag 解除阻塞，
那是"慢"不是死锁。这正好区分今天的两类超时。）

### L2 wait-for 环检测（死锁证明 + 定位报告）

阻塞时建立等待图：

- 节点 = lane（cube-N / vec-N）；
- 边：lane X 等 flag F（HardEvent, id）→ F 的 set 责任 lane Y
  （同 lane 的未来指令 / cross-flag 的对侧 lane）；
- Y 自身也阻塞 ⇒ 沿其等待边继续走。

图上出现环 ⇒ 死锁证明。报告内容（把今天手工排查想要的信息直接给出）：

```
DEADLOCK (wait-for cycle):
  cube-7  waits MTE1_MTE2 id=0  (slot-free credit, level=0)
          producer: cube-7 itself @ future C1 release — but cube-7 is blocked
  vec-3   waits CrossFlag id=5  (SIG_P_READY slot 2)
          producer: cube-7 @ C2 — blocked
  cycle: cube-7 -> vec-3 -> cube-7
  last 8 events per lane in cycle: ...
```

附带每个环上 lane 的最近事件轨迹（最后 N 条 set/wait/copy），直接指向
Hazard 位置。

### L3 电平守恒审计（kernel 边界，可选诊断）

显式启用时校验 flag 残留并报告 id、电平和最后 set task。该结果是协议审计
信号，不是普适死锁证明：循环流水可能故意在结尾恢复预置信用。只有调用方明确
要求“该协议结束必须归零”时才应使用 `error`；默认关闭以避免误报。

## 3. 实现要点

- 改动面：`tilelang/simulator/executor.py`（主循环加 L1 停顿判定 + L2 图
  构建）、`tilelang/simulator/sync.py`（flag 电平账目 API + L3 审计）、
  `tilelang/simulator/config.py`（`deadlock_detect: bool = True` 开关，
  默认开；`deadlock_history_limit` 控制报告长度；L3 使用独立
  `flag_balance_check` 策略）。
- 与 sync_only 的关系：L1/L2/L3 全部**只依赖 flag/pipe/lane 状态**，不依赖
  数值——sync_only 模式原生受益（这也是它最该在的地方）。
- 判定必须**零误报**：L1 只在"全部 lane 阻塞 + 全部 pipe 空闲"时触发；
  L2 只在环闭合时触发。慢 kernel 绝不许报死锁（回归网兜底）。
- `execution_timeout_s` 保留为最后兜底（防止检测逻辑自身漏判）。

## 4. 验收（test-first）

1. **红**：`testing/python/simulator/test_deadlock_detect.py`
   - **阳性（必须报）**：三个变异 kernel——(a) 删一个 wait 配对的 set
     （单 lane 自锁）；(b) cross-flag 方向接反（AIC↔AIV 互等环）；
     (c) 槽信用少预置一级（HS29 型：tpc≥N 时跨 task 信用断供）。
     三者都必须在秒级内以 L1/L2 报出，且报告含正确的 flag id 与环。
   - **阴性（不许报）**：既有全套正确 kernel（含 GQA gqa_sim 11 case、
     长 ring 大 shape 慢 case）在 `deadlock_detect=True` 下零误报；
     `execution_timeout_s` 调小仍只对"真慢"抛 SimulationLimitError。
   - **L3 审计**：人为制造电平残留（多 set 不 wait）→ kernel 结束审计
     报出残留 id/电平/位置。
2. **实战验收**：把 HS29 的死锁版 gqa design（per-kernel 预置 + 96-task
   multitask）喂给实现了 L2 的模拟器——报告应直接指出断供的信用 id，
   与真机 case 2 形态互为印证。
3. 回归：`testing/python/simulator/` 全套既有测试不塌。
4. commit 到 tilelang-ascend ascendc_pto 分支；告知用户推 fork。

## 5. 环境/约束

- 仓：/home/wenzhongzhen/tilelang-ascend（ascendc_pto 分支）；纯 Python
  层，无需重编 C++，测试不需要 NPU 卡。
- 默认行为变化：`deadlock_detect=True` 默认开——只会把"原来超时才能
  发现的事"提前报出，不改变任何通过路径的行为。
- 与 [[simulator_dynamic_scalar_if_plan]]（动态标量 if 提案）相互独立，
  可分别落地；建议本提案优先（直接服务当前 HS29 类调试）。
