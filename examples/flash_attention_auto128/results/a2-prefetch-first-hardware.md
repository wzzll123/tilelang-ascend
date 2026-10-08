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
Follow-up single precision runs also passed:

| Case | B,H,Sq,Skv,D | Mode | Coverage |
| --- | --- | --- | --- |
| 42 | 2,4,192,192,128 | noncausal | three iterations, first slot reuse |
| 10 | 2,8,512,512,128 | causal | cid-dependent trip count and repeated reuse |

Both original JSON reports are preserved alongside case0. Their real cached
sources also contain the new wave: case42 has `epoch_wave < 4`; case10 has
`epoch_wave < floordiv(floormod(cid,16),2)+2`. These are not sequential fallback
passes.

The remaining cases `2,6,7,12,13,14,15,16,36,37,41` subsequently passed the same
original verification script, serially on card6. All fourteen fixed cases now
have one real-device precision pass; each unmodified JSON is archived here as
`fa128-prefetch-1aeedc56-caseN-precision.json`. The serial fail-fast shell loop
terminated with exit code 0, and all fourteen reports were independently read
back and checked for `ok: true` and `device: npu`.

Follow-up CPU-only cache inspection matched each fixed case uniquely against
its first three parameter shapes plus the causal query-row code. Full cache
keys and both source/parameter SHA256 values are recorded in
`prefetch-native-cache-audit.csv`. All fourteen sources contain a wave loop,
16384-element K/V allocations and 8192-element slot pitches. This excludes
sequential fallback in those cached sources. Parameter inspection used
`pickletools.genops`, not `pickle.load`, so no serialized objects were executed.
Cases13/15 have identical source hashes but different parameter shapes/hashes;
source hashes alone cannot identify a case. Cache inspection by itself is not
proof of historical loading; here it is paired with the explicit fixed cache
in the actual validation environment above.

All fourteen cases subsequently passed twenty real-device launches each, with
an explicit finite-output gate and the original precision comparator on every
launch. The serial fail-fast process terminated with exit code0; all fourteen
reports were independently checked and archived as
`fa128-prefetch-1aeedc56-caseN-stability20.json`. This is fixed-input stability
over 280 launches, not proof for arbitrary inputs or future compiler changes.

The three-round prefetch performance measurement has now completed: all84
original reports and336 profiler CSV files are archived. The three-round
median ratio geometric mean is0.107698, so the >=0.90 gate FAILS; see
`a2-prefetch-three-round-performance.md`. This is not completion of the
migration or the performance requirement. The wider multistage compiler
work remains local and is not part of this frozen `1aeedc56` hardware build.
