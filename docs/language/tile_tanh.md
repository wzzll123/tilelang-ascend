# T.tile.tanh

逐元素计算双曲正切，映射到 A2/A3 的 `AscendC::Tanh(dst, src, tmp, count)`。

```python
T.tile.tanh(dst, src, tmp=None)
```

`dst` 和 `src` 必须是元素数相同、dtype 相同的连续 UB buffer 或连续切片，
支持 `float16`、`float32`，字节偏移必须按 32B 对齐。源数据保持不变。
源、目的和临时空间不得重叠；前端保守拒绝使用同一底层 buffer 的操作数。
非连续切片、不能证明安全的动态地址和 PTO 后端明确报错。

`tmp` 是可选 UB scratch arena，不是数值输入。省略时编译器按官方
`GetTanhMaxMinTmpSize` 的最大值分配：FP32 为 `max(N * 4, 256)` bytes，
FP16 为 `4 * max(N * 2, 256)` bytes。显式 arena 可采用任意固定宽度标量
dtype，lowering 以 `uint8` 视图传入；至少需要 FP32 256B、FP16 1024B。
该大小关系依据 asc-devkit `ec0f29df44b54a4460ac04fb4ee6a21613ccd967`
的 `impl/adv_api/tiling/math/tanh_tiling_impl.cpp`，不沿用 Sigmoid 的大小估计。

Tanh 作为 `PIPE_V` 的 self-contained helper 参与指令选择与 mask legalization。
接口内部设置 counter mask，退出前恢复 normal 模式与 full mask；后续 Vector
指令的 mask 由编译器按这一终态继续维护。临时空间按写入/clobber 建模。

```python
with T.Scope("V"):
    T.copy(x, src_ub)
    T.tile.tanh(dst_ub, src_ub)
    T.copy(dst_ub, y)
```
