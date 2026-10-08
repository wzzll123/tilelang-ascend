# First automatic prefetch hardware check

Compiler commit: `1aeedc56`, branch `codex/a2a3-auto-schedule-fa128`.
Docker worktree: `/home/wenzhongzhen/tilelang-ascend-wt/auto-schedule-fa128`.
Case0 passed the existing designer `verification_tilelang.py` on physical A2
card6: FP16 `(B,H,Sq,Skv,D)=(1,4,128,128,128)`, noncausal, CPU FP64 golden and
the original CANN-aligned comparison. The original result JSON is preserved as
`fa128-prefetch-1aeedc56-case0-precision.json`; its legacy atol/rtol fields are not
the executed acceptance thresholds.

Process settings: `FA_STEP=auto128`, `ASCEND_RT_VISIBLE_DEVICES=6`,
`ASCEND_LAUNCH_BLOCKING=1`, isolated cache
`/tmp/fa128-prefetch-cache-1aeedc56`. No NPU clock change or other-process stop.
Card6 was healthy and had no running NPU processes immediately before launch.

Verified Python:
`/home/wenzhongzhen/tilelang-ascend-wt/auto-schedule-fa128/tilelang/__init__.py`.
Verified loaded native library:
`build-auto-schedule/libtilelang_module.so`, SHA256
`4ce6649be570c447d58f5fa1e5b236eb1a1473a6a9f88a010de4e4578eddd98b`.
TVM loaded from `/home/wenzhongzhen/tilelang-ascend/build/tvm/libtvm.so`;
Z3 `4.15.4`, existing isolated Python3.12 environment. Source and TVM pins,
CANN/Bisheng installation are unchanged from the archived baseline provenance.

The actual cached `kernel.cu` SHA256 is
`6305fdb18dfeeeeb0de04a146cc2430de54337075e2811d3b94c894baac226d7`.
It contains `epoch_wave < 3`, guarded K/V loads for `epoch_wave < 2`, and slot
offsets `epoch_wave * 8192`. This establishes that the real verification used
the new prefetch wave, not a sequential solver fallback. In this two-iteration
case the optimizer simplified the two-slot modulo expression to the epoch.

This is one precision run, not the fourteen-case gate, repeated numerical
stability, explicit finite-output validation, or a performance measurement.
Case42 (three iterations, first slot reuse) and case10 (causal) are separate
follow-up checks; their results are not claimed here.
