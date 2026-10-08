# A2/A3 automatic scheduling migration — FA128

## Scope and acceptance

FP16 standard MHA, head dimension 128, online softmax with FP32 state and output
accumulation. Noncausal, bottom-right causal, and Q/KV tails are required. The
final kernel must not contain manually selected flag IDs, barriers, physical
addresses, or ring-slot indices. Optional Stage and buffer-version declarations
are permitted. BF16, GQA, and other head dimensions are outside this milestone.

Fixed zero-based IDs in `general_case.jsonl`:
`0, 2, 6, 7, 10, 12, 13, 14, 15, 16, 36, 37, 41, 42`.
All must pass existing FP64-golden checks, finite-output checks, and repeated
real-device launches. Performance acceptance is the geometric mean of
`manual_baseline_us / automatic_us >= 0.90`, with every case reported.
A3 generation/regression results are not A3 hardware validation.

## Implementation status

This is an **incomplete migration**, not a usable FA auto-scheduler release.
`TL_ASCEND_AUTO_SCHEDULE` defaults off. When enabled, the entry validates target
and configuration, extracts lowered tasks, then fails closed because TIR
materialization and unified synchronization are not implemented. It must never
quietly execute the old pipeline and present that result as automatic scheduling.

Implemented, with CPU tests:

- `T.Stage` and `T.annotate_buffer_versions` metadata on the existing TVM/TIR.
- Conservative lowered-operation extraction and physical storage interning.
  Literal contiguous scalar/vector byte ranges are exact. Copy and MMA pointer
  counts do not prove physical ranges for strided/fractal/padded operations;
  those footprints stay unknown until helper/layout semantics are established.
- Actual TIR `LetStmt` condition snapshots at the original position, including
  inside loops; extraction refers to the real bindings without duplicate work.
- Unscoped scalar tasks resolve only in an unambiguous single-core context.
  Mixed C/V control requires per-core normalization, not an invented C owner.
  Scalar Vars and local-buffer identities are private to their execution core.
- Same-iteration and carried RAW/WAR/WAW analysis, including uncertain regions.
- Full-overwrite owner eligibility and iteration/counter clock descriptions;
  initial reads, read-modify-write and partial initialization cannot rotate.
- Standalone Z3 modulo constraints with true dependence distances separate
  from version-dependent storage reuse. Explicit constraints cannot be dropped.
- JSON analysis reports and independent constraint validation.
- Bounded-instance asynchronous alias contracts accepting only explicit
  execution-order proofs; inferred hazards and estimated times do not prove
  completion. This is not yet connected to the physical planner.
- Conservative A2/A3 compiler allocation budgets, checked against the existing
  C++ planner's reservations, not copied from A5.
- Native PassContext key registration and verified runtime fingerprint utility.
- Bounded affine old-TVM byte-region/Z3 proof adapter with intermediate integer
  overflow checks. Unsupported nodes or solver unknown retain conflicts. It is
  not yet connected to automatic dependence pruning.
- Flat single-core local event TIR emitter, independently tested through the
  simulator on A2/A3. Removing a generated wait exposes a memory hazard. This
  is not a loop/cross-core/software-pipeline materializer or full compiler path.
  Event IDs are never reused in this slice; a textually prior wait is not proof
  it executed before another asynchronous source set. Exhaustion is explicit.
- FP16 single-tile GEMM decomposition into actual L1-to-L0A, L1-to-L0B and
  MMA TIR tasks using explicit caller-owned scratch, without hidden flags.
  Supported static aligned M/N/K <=128, kL0Size=128, FP32 C. Init and the
  transpose-B actual-N behavior follow the original numerical helper. Operand
  and scratch capacities are checked. Scratch allocation/placement/lifetimes
  and automatic pass integration are still pending.

Still required, in order:

1. Decompose composite GEMM and cross-core copy into actual pipe tasks. Reject
   `gemm_v0/v1`, `copy_ub_to_l1`, and `copy_l0c_to_ub` for now: their hidden
   MTE1/M or cross-core GM-hop work cannot be modeled as a single atomic pipe.
2. Connect the verified condition snapshots and region proofs to the supported
   task/control-flow reconstruction and helper-specific physical byte regions.
   Duplicate common mixed-core scalar control into the actual C/V programs
   while preserving provably equivalent guard/owner epochs.
3. Connect owner/full-footprint proof, per-core cost/storage model and solver to
   real loops; materialize versions, slot selection, prologue/body/epilogue.
4. Generate local events and A2/A3 mode-2 cross-core ready/release protocols,
   with per-slot credits, two-AIV joins, flag allocation and final drains.
5. Prove guarded and carried asynchronous lifetimes, then connect safe reuse
   contracts to memory planning. Do not use lexical lifetimes or solver time
   estimates as safety proofs.
6. Replace the old pipeline/sync rewrites on the opt-in path, retaining operation
   lowering and existing numerical templates. Never run both synchronization
   implementations on one function.
7. Implement `fa_auto128.py`, explicit model selection, and the existing script
   compatibility changes; run all correctness/sync/stability/performance gates.

## Workspaces and provenance

Compiler base: fork `ascendc_pto`, commit
`10f3f81ccd669efab048114f2d141aab44bf53e4`.
Development branch: `codex/a2a3-auto-schedule-fa128`.

Upstream design reference: `tile-ai/tilelang`, fixed commit
`47976f210de3598444003e9b05de8204dfa7ff88` (MIT).
Relevant sources are `src/ascend/transform/auto_schedule.cc`,
`src/ascend/transform/auto_schedule/`,
`tilelang/ascend/transform/z3_scheduler.py`,
`src/ascend/transform/materialize_schedule_units.cc`,
`normalize_control_flow_for_schedule.cc`, `insert_sync.cc`, and
`merge_ub_allocations.cc`. Python models here adapt those concepts and cyclic
constraints rather than replacing the fork's TVM with TIRX or copying A5's
mode-4 protocol/capacities. Preserve MIT attribution if code is reused later.

Local worktree: `tilelang-ascend-wt/auto-schedule-fa128` beside the original fork.
Docker worktree: `/home/wenzhongzhen/tilelang-ascend-wt/auto-schedule-fa128`.
Keep the Docker main checkout and every frozen ladder implementation untouched.
Use process-local Python/native paths and a separate build, not global installs.

## Testing and environment identity

New unit suites are in `testing/python/transform/test_ascend_schedule_*.py` and
`testing/python/language/test_ascend_auto_schedule_annotations.py`.
The pure model/solver suites can run without importing TileLang/TVM. Real parser
and lowering-entry tests require the fork's existing TVM and a rebuilt TileLang
library. Install `z3-solver` only in an isolated environment for solver tests.

Before each actual kernel run, execute
`scripts/ascend_schedule_environment.py` in the same process environment with
explicit `--compiler-root`, `--compiler-library`, `--tvm-source`, `--tvm-library`,
`--platform A2`, `--require-auto-schedule`, case/shape/card/tiling/pass-config
arguments. It verifies loaded paths and prints native hashes, Python/package
paths, commits/dirty state, and available toolkit/Bisheng versions. An unavailable
version is not a successful environment check; fill that gap before acceptance.

The original local TVM runtime is commit
`6dcf3f0d3169e693b3f5f0525556a40d23eaf703`, TVM `0.17.dev0`.
The local independent build uses that explicitly selected runtime. The initial
AppleClang build needed the C++ standard include path
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1`;
this is a local CMake parameter, not a repository-wide compiler flag.

Latest local milestone check: 118 new tests plus 104 selected existing
simulator/platform regressions, **222 passed**, 50 subtests, zero skips.
This used the independent rebuilt native library and the existing older TVM.
Warnings are older TVM parser deprecations under Python 3.13, not failing tests.
Native compilation succeeded; hardware compilation/execution is not included.

Docker first-milestone validation also passed all 222 selected tests using its
independent native build, toolkit 9.0.0, Python 3.12.13 and the same clean TVM commit.
Loaded native SHA256: `0778c7e71891d09820fbb75517e8e414bceedfcb1597bfc7b095bd453ed51ee1`.
Loaded TVM SHA256: `2df7999bbed69490640f5bc50166a5a650ac543bdea22514878e86ae31ccd36e`.
Docker isolated solver environment:
`/tmp/a2a3-auto-schedule-fa128-venv.JsRZee` (Z3 4.15.4).
Local Z3 is 5.1.0; both passed the first-milestone constraint suite. The exact
local version has no binary wheel in the Docker ARM index and its source build
failed, so Docker uses an available wheel, not a global installation change.

Second local milestone: condition reconstruction, byte proofs, local-event
emission and conservative transfer/MMA extraction passed **270 tests**, 66
subtests, zero skips, with the same native library. Both A2/A3 local event
mutation checks execute sync-only, not just compile, and reject deleted waits.
This does not establish full FA scheduling, cross-core protocols or hardware
correctness. YAPF 0.40.2, Ruff 0.6.5, codespell 2.3.0 and diff checks passed.

The second milestone was subsequently transferred to the independent Docker
worktree at `f6f2b6a7f1f6a0e83e2a7d0a67e8751b6fa09c6a`: **270 passed**, 66
warnings, 11.21 seconds, no skips or failures. The worktree was clean. This
confirms the same suites with Docker's Z3 4.15.4 and older TVM/native runtime;
the warnings are Python 3.12/older-TVM parser deprecations.

Execution-domain follow-up: **274 local tests**, 66 subtests passed. Pure C and
pure V scalar producers resolve correctly; explicitly scoped mixed-core scalar
and local storage stays separate; ambiguous mixed-core controls are rejected.

GEMM decomposition follow-up: **284 local tests**, 66 subtests passed. Ten new
tests verify real intrinsic arguments, unsupported families/capacities, and
the extracted MTE1/MTE1/M tasks feeding dependency analysis and event emission.
This is not a numerical or NPU GEMM/FA acceptance result. A runtime symbolic
`n_actual` is preserved for code generation, but the current simulator's direct
L1-to-L0 copy dimension decoder still requires concrete dimensions; that gap
must be addressed before dynamic-tail sync-only acceptance.

Function-level GEMM follow-up: **290 local tests**, 66 subtests passed. The
enabled entry now decomposes supported GEMMs before condition normalization
and task extraction. Each callsite receives distinct compact 2D L0A/L0B
allocations without physical addresses; loops, branches, Stage annotations and
function attributes are preserved. Six additional tests cover function
rewriting, idempotence, name collisions, enabled-entry fail-closed behavior,
and native A2/A3 memory-planning/source generation after LowerTileOp. The late
buffer flattening discovers scratch absent from the initial shape collection.
Source-only probes are intentionally not launched: loop-carried scratch reuse,
slot materialization and unified synchronization still need implementation.

The function-level milestone `d12d0e33` was fast-forwarded into the clean
independent Docker worktree: **290 passed**, 66 warnings, 11.94 seconds. No NPU
kernel was launched and the main Docker checkout was not modified.

Loop-event follow-up: **303 local tests**, 66 subtests passed. A new local
emitter handles an unconditional flat serial-loop body with same-iteration
and distance-one dependencies. Each cross-pipe edge owns ready/release
channels, with a single initial credit and a matching final drain; it does not
rely on counter accumulation to protect repeated notifications. Thirteen tests
include real A2/A3 sync-only execution for 0/1/2/5 iterations, a removed wait
reported as a memory hazard, and removed drain reported as runtime flag
accounting failure even for an empty loop. This remains a materialization
building block, not the general guarded, versioned or cross-core pipeline.
Distances greater than one are explicitly rejected until slot/cycle ownership
materialization can establish their protocol. Full FA hardware gates remain
unrun; neither this conservative handshake nor its performance is accepted as
the final FA schedule.

Iteration-version materialization follow-up: **312 local tests**, 66 subtests
passed. Certified single iteration owners can now expand compact physical
Allocate storage and shift every lowered access_ptr by the owner-relative
epoch modulo version count, without changing the per-access extent. Nine tests
cover nonzero loop minima, ownership/eligibility rejection, outside-owner
accesses, raw-handle escape, and native A2/A3 allocation/source generation of
dynamic slots. Capacity and access bounds are checked; fractal tiles require
aligned dimensions and whole-version pitches are 32-byte aligned. Counter
owners, mixed-dtype aliases, logical BufferLoad/Store and Block allocation
views still require additional materialization work and currently fail closed.
This is not yet wired as a complete executable automatic-schedule path: slot
release events must be derived from the same certified owner clock before
versioned kernels can be accepted for execution. No hardware or performance
acceptance claim follows from source-generation tests.

Version-clock/event integration follow-up: **322 local tests**, 66 subtests
passed. The local loop emitter now assigns ready/release channels per slot and
selects them with the same `(iteration - loop_min) % count` clock as physical
addresses. Every slot is initialized and drained, including unused slots in
empty/short loops. Ten tests actually execute the materialized double-buffer
TIR with generated synchronization on A2/A3 simulator configurations for
0/1/2/3/5 iterations and a nonzero loop minimum. A carried reuse edge must have
distance equal to the selected slot count; other distances are not discarded.
The current emitter requires a uniform slot count within its flat local slice;
mixed owner clocks, guarded tasks, cross-core protocols and whole-function
automatic materialization are still outstanding. This result is CPU sync-only,
not NPU FA correctness or performance acceptance.

Reproduce CPU validation inside `wzz_cann`:

```bash
cd /home/wenzhongzhen/tilelang-ascend-wt/auto-schedule-fa128
export PYTHONPATH=/home/wenzhongzhen/tilelang-ascend-wt/auto-schedule-fa128:/home/wenzhongzhen/tilelang-ascend/3rdparty/tvm/python
export TVM_LIBRARY_PATH=/home/wenzhongzhen/tilelang-ascend/build/tvm
export TILELANG_LIBRARY_PATH=/home/wenzhongzhen/tilelang-ascend-wt/auto-schedule-fa128/build-auto-schedule
export LD_LIBRARY_PATH="$TILELANG_LIBRARY_PATH:$TVM_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
/tmp/a2a3-auto-schedule-fa128-venv.JsRZee/bin/python -m pytest -q --disable-warnings \
  testing/python/transform \
  testing/python/language/test_ascend_auto_schedule_annotations.py \
  testing/python/simulator/test_scheduler.py \
  testing/python/simulator/test_sync_only.py \
  testing/python/simulator/test_simulator_memory.py \
  testing/python/simulator/test_deadlock_detect.py \
  testing/python/language/test_tilelang_ascend_language_auto_platform.py
```

## Docker validation gates — not yet run

Task root:
`/home/wenzhongzhen/kernel-tool/cannbot-skills-0817/plugins-community/tilelang2ascendc-ops-generator/workflows/templates/archive_tasks/flash_attention`.
New kernel destination: `design/tile_level/ladder/fa_auto128.py` under that root.
Designer scripts live under the same plugin's
`skills/tilelang2ascend-tilelang-designer/scripts`.

Use the existing `verification_tilelang.py` / `evaluate_tilelang.sh`, then real
`sim_sync_tilelang.py` execution with hazard/balance/deadlock checks and the
unchanged 60-second execution timeout. Compilation alone is not a sync pass.
Then use `perf_tilelang.sh` / `perf_tilelang.py` and `msprof_tilelang.sh`.
Only add necessary compiler-path, implementation selection, or rectangular-shape
options; do not invent another scoring framework or relax precision thresholds.

Resample v06 (and any proven better frozen per-case reference); historical
`LADDER.md` wall-clock numbers are not acceptance data. Prefer idle healthy card
6, continue avoiding cards 0 and 3, recheck usage before launches. Run baseline
and candidate serially/interleaved on the same card: at least three full runs,
three warmups and twenty measured repetitions each, no boosting. Preserve raw
results and spread. On a suspected hang stop additional launches; do not reset
the card or stop unrelated processes.

No correctness, sync-only, stability, performance, or A3 platform acceptance
result is currently claimed. Commit verified milestones independently; do not
push automatically or mix other agents' main-checkout modifications.
