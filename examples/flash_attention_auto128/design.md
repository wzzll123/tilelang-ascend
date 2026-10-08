# FA128 automatic-scheduler correctness baseline

## Algorithm and interface

Standard MHA, contiguous BHSD, positive batch/heads/Q/KV lengths, FP16 and
head dimension 128. Bottom-right causal mode requires Sq <= Skv, matching
the existing task golden's finite-row domain. Output has Q's exact shape.
No host padding, crop, tensor substitution or data-dependent dispatch.
Configuration is metadata only; runtime inputs are Q/K/V and sm_scale.

For each 32-row Q tile, iterate over 64-column KV tiles. Keep FP32 maximum m,
denominator l and unnormalized output o across iterations. Given scaled scores
s, update m' = max(m, rowmax(s)), alpha = exp(m-m'), p = exp(s-m'),
l' = alpha*l + rowsum(p), and o' = alpha*o + FP16(p) @ V. Finally return
FP16(o/l). Invalid KV columns are masked before rowmax and zeroed after exp.
The persistent FP32 states are single-version, not scratch to be rotated.

For causal mode, the KV trip count is now
`min(ceildiv(Skv,64), ceildiv((q_tile+1)*32+Skv-Sq,64))`.
Only columns beyond every valid Q row's allowed prefix are skipped; the
existing per-row causal and KV-tail masks remain. Noncausal keeps the full
KV trip count. The compiler certifies that this cid-derived bound is identical
for the Cube and both Vector participants, bounded, and free of integer wrap.
It does not use tensor values or introduce a per-participant branch.

## API, ownership and synchronization

Reference: examples/developer_mode/flash_attn_bshd_developer.py (MIT).
The kernel uses alloc_shared/alloc_fragment, gemm_v0, copy, tile broadcast,
compare/select, vector arithmetic, reduce_max/reduce_sum and CreateVecIndex.
No hand flags/barriers, physical addresses or ring indices. The opt-in
auto_schedule path manages task scopes, dependencies, event allocation,
ready/release and drain. Old pipeline/sync rewriting must not also run.

One-dimensional grid B*H*ceildiv(Sq,32), threads=2. Inline division/modulo
decode batch, head and Q tile, avoiding scalar Let tasks. L1 Q/K/V/P belong
to Cube; softmax/output vector data belong to Vector. Existing lowering
creates GM workspaces for C/V transfers; A2/A3 do not share these local
buffers. This is automatic source-level CV allocation, not a hand workspace
signature or a claim of direct physical L0C-to-UB transfer.

Data paths: GM Q/K/V -> L1 -> generated L0A/L0B -> L0C; score/output L0C
-> generated GM workspace -> UB; FP16 probabilities UB -> generated GM
workspace -> L1. Final output UB -> GM. No scalar GM data accesses.

## Tiling and tails

32x64 QK and 32x128 PV tiles, K widths 128 and 64 respectively, fit the
supported single-chunk GEMM decomposition. Explicit buffers are deliberately
disjoint for this baseline. L0C score/output total 24 KiB before alignment;
generated L0A scratch 12 KiB and L0B scratch 32 KiB. Capacity legality is
checked by the compiler's A2/A3 plan; these are not A5 capacity assumptions.
Actual physical layout, injected UB scratch and event assignments must be
read from generated diagnostics, not inferred only from source lifetimes.

Q/KV counts use ceildiv; copy lowering clips valid GM rectangles and clears
base-zero L1 padding. Empty Vector-1 output copies issue no native DMA but
still participate in surrounding synchronization. Valid Q rows never depend
on padded Q output. Nonzero ring-base padding is not certified by this design.

At full tiles, per Q/KV iteration user-input reads are 16 KiB K + 16 KiB V;
workspace hops are 8 KiB score, 4 KiB probability, 16 KiB output each written
and read. Q is 8 KiB once per Q tile. These extra GM transactions explain why
correctness does not imply competitive performance. There is no per-element
GM indexing; task decode uses metadata div/mod once per Q tile.

## Validation and current limits

Use the existing designer verification_tilelang.py / evaluate_tilelang.sh,
CPU FP64 golden and original precision rules. First actual A2 card6 case0,
[1,4,128,128] FP16, passed. The fixed fourteen-case run is a separate gate;
that single pass is not all-case correctness, stability or performance proof.
CPU primitive/integration regressions are supplementary, not a hardware gate.
A3 has native/platform regression only, no hardware validation.

The first fourteen-case A2 precision run subsequently completed: every fixed
case matched the original verifier. Raw report:
results/a2-card6-precision-14cases.json. This is one run, not the stability or
performance gate. Fixed case42 has Sq=Skv=192, so it does not exercise arbitrary
Q/KV tails for the current 32x64 tiling.

Final required gates remain: finite outputs, all fourteen cases, repetitions,
baseline/candidate three-run warmup3/repeat20 comparison, geomean ratio >=.90,
and msprof overlap analysis. No tolerance or golden may be changed to obtain
passing results. The current 32x64, single-version KV baseline does not claim
that coordinated multiversion pipelining or performance acceptance is finished.

Explicit dispatcher domain is one FP16 dim128 path; all other dtypes/dims fail.
No hand-tuned/PyTorch fallback. Default task selector remains unchanged;
FA_STEP=auto128 explicitly imports this entry. The final runtime signature has
sm_scale before generated workspaces; do not reuse legacy workspace indices.
