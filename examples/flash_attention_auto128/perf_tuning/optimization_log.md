# Automatic FA128 optimization log

## Baseline

The production single-version32x64 kernel passed14 original A2 precision cases.
Three real profiler rounds failed the performance gate: geomean v06/auto0.090009.
Raw84 reports and CSV are in ../results; no Python wall-time performance claim.

## Part A: candidates

- P0 host preprocessing (optimization-guide §2.12): no change applicable;
  the kernel accepts original BHSD inputs and writes exact-shaped output, with
  no host pad, transpose, cast, crop or tensor substitution in the entry.
- #1 causal KV pruning: applicable; the last allowed key for a Q tile is
  `min(Skv-1, q_end-1+Skv-Sq)`. Existing bounded participant-uniform clock
  certification makes the equivalent prefix count safe for both C/V programs.
  This changes no layout, tiling, buffers, API, dtype or synchronization policy.
- #2 joint versions/stage skew (guide §2.2, §2.3, §3.1): applicable later, but
  requires actual compiler reconstruction, carried-edge and capacity proofs.
  Do not reuse the guide's hand flags or old T.Pipelined rewrite: user requires
  automatic scheduling and no manually indexed rings. Persistent m/l/o stay1.
- #3 tile granularity/fixed-core (guide §2.9, §2.13): candidates later; require
  supported loop/control reconstruction and updated local-capacity proof.
- Split-K/full-load/coalesced metadata (§2.1/2.4/2.5): no change this iteration;
  K=128/64 fits the supported GEMM decomposition, Q already resident, no scales.
- Vectorization/fusion/sparse-gather (§2.6/2.7/2.8/2.15): no change this iteration;
  existing whole-tile operations; not a paged/sparse scalar gather or scan.
- Config disabling (§2.10): prohibited substitution for this auto-schedule goal.
- UB/select/stride/D2H/dtype/reorder (§2.11/2.12.1/2.14/2.16/2.17): no change;
  mask mode, allocation count, contiguous FP16 input and exact output unchanged.
- Cross interval (§3.2): no manual synchronization batching in this task.

[ORDER-PLAN] #1 clock certificate and pruning -> #2 executable joint-version
pipeline -> #3 measured tiling/task adjustments. No call-domain expansion.

## Iteration1: causal bound

[ORDER-CHECK] #1 prerequisite: compiler uniform clock certificate3d9d3bae and
termination overflow fixc835bf50, local35 targeted tests; proceed.
[IMPL-#1] Original auto128 kernel is the reference. Keep its online-softmax
equations, mask, GEMM init, storage, prefix and suffix; shorten only the causal
serial loop. T.serial semantics checked in api-schedule-sync §1. Legacy skill
instructions about four old passes/direct physical CV access are superseded by
the approved opt-in compiler path and actual A2/A3 GM workspace architecture.
[DISPATCH-COVERAGE-AUDIT] Same FP16 dim128 MHA domain, positive sizes and
causalSq<=Skv; noncausal fullKV prefix unchanged; uncovered_input:none.
[HOST-METADATA-AUDIT] No host tensor operation introduced; cfg is metadata only.
[SELF-CHECK] Only #1 changes numerical work executed. End count is ceildiv of
the possible allowed-key prefix, clamped to total KV tiles; tail masks retained.
[RESULT-#1] Test-first production lowering/native-source regressions: initially
the2 causal tests failed on constant extent, then all4 A2/A3 causal/noncausal
tests passed. Generated mode2 flags and MMA remain; no PIPE_ALL was introduced.
Actual A2 card6 case10, (2,8,512,128) FP16 causal, passed the unchanged original
CPU FP64/native CANN benchmark verifier. First profiler round:231.31us,
warmup3/repeat20/no-boost/no-sol, versus earlier auto median381.47us (39.36%
lower,1.649x). This is a single new round versus a historical three-round median,
not the final alternating comparison or all-case acceptance. Reference v06
median36.30us remains much faster. No full-pipeline completion claimed.
Compilerae4b1a35 and isolated cache /tmp/fa128-auto-causal-cache-ae4b1a35;
Python/TVM/native/CANN/card identities unchanged from the frozen baseline.
Raw case10 precision/performance JSON is in ../results, profiler CSV remains in
/tmp/perf_tilelang_401509_5l6ycctf. Other causal cases and repeated profiling pending.

### Completed causal follow-up

Original verification subsequently passed all3 remaining causal cases36/37/41,
including the unchanged CPU FP64/native precision comparison. Together with
case10, all4 affected fixed causal cases pass one numerical run. Finite-output
and repeated numerical stability gates remain separate and incomplete.
The conditional profile job finished exit0 with3 rounds, warmup3/repeat20,
no-boost/no-sol, same compiler/cache/card6. Raw3-case precision and12 profiler
JSON files are retained under ../results/fa128-causal-ae4b1a35-*.

| Case | Earlier auto median us | Pruned auto median us | Earlier/pruned | Duration reduction |
|---:|---:|---:|---:|---:|
| 10 | 381.47 | 231.39 | 1.6486 | 39.34% |
| 36 | 11469.99 | 6021.96 | 1.9047 | 47.50% |
| 37 | 5833.10 | 3195.14 | 1.8256 | 45.22% |
| 41 | 11399.69 | 5863.30 | 1.9442 | 48.57% |

Earlier measurements are the frozen prior three-round auto baseline, not newly
alternated comparisons. Pruned round values respectively:
10[231.26,231.39,231.70],36[6019.40,6021.96,6024.06],
37[3195.14,3196.76,3194.54],41[5864.48,5861.18,5863.30].
No speed claim for the10 noncausal cases or updated14-case acceptance score;
current results still leave a large gap to v06 and are not task completion.

## Compiler prerequisite for iteration2

`bd9ae47d` adds an executable logical-epoch stage-wave reconstruction component.
All8 focused tests pass, including actual mode2 emitter bundles and zero/short
epochs, wrong single-slot reuse rejection, guard/flag coupling and overflow.
It is not yet connected to production FA: joint storage eligibility/version
selection, phased ring protocols and actual hardware checks remain required.
The helper does not invent full overwrite proofs or silently enable old passes.

## Iteration2: production input-prefetch waves

The prerequisite paragraph above records the state at `bd9ae47d`.
At `1aeedc56`, certified K/V input lifecycles, joint capacity validation,
physical rings, per-edge event counts, phased synchronization and executable
waves are connected to the compiler entry. Persistent softmax state is not
rotated. Explicit hints stay on their existing validated path; automatic solver
failure reports a sequential new-path fallback, never legacy synchronization.

Native A2/A3 generation and compiler/solver/policy/report regressions passed:
61 tests and11 subtests. The first A2 card6 real precision check, case0, passed
the original CPU FP64/CANN-aligned script. Its actual cached native source has
the prefetch wave and8192-element K/V slot offsets; it is not a solver fallback.
Exact loaded-library/source hashes and the raw report are archived in
`results/a2-prefetch-first-hardware.md`.

This iteration has **no measured performance result yet**. Case42(first reuse)
and case10(causal) are follow-up precision checks, followed by the remaining
fixed cases and alternating profiler comparisons. Input prefetch alone is not
claimed to complete broader C/V overlap or to satisfy the90% performance gate.
