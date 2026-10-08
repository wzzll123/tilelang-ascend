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

The slot integration milestone `325fcae2` was transferred to the clean
independent Docker worktree: **322 passed**, 66 warnings, 12.44 seconds.

CV workspace task follow-up: **326 local tests**, 66 subtests passed. The
enabled extraction path reuses AscendWorkspaceReduction's existing numerical
lowering before task extraction: UB-to-L1 becomes V/MTE3 GM write followed by
C/MTE2 GM read; L0C-to-UB becomes C/FIX GM write followed by V/MTE2 GM read.
Four native A2/A3 tests check both directions, generated workspace parameters,
shared-storage RAW edges, and the enabled entry reaching its materializer
diagnostic rather than treating composites as one atomic pipe operation. The
wrapper skips composite-free input because the native pass refreshes workspace
attributes even when rerun without composites. Numerical helper templates,
row pitches, dtype handling and core/vid offsets remain native-pass owned.

An attempted static simulator hazard acceptance check did **not** report these
unfenced GM-hop probes. It is not counted as a passing negative test or evidence
of safe synchronization. Explicit mode-2 protocols and real execution negative
tests remain required. No event is emitted by workspace lowering, and inferred
GM data edges must not be treated as runtime execution fences. When the full
entry becomes executable, its physical-workspace lowering must replace (not
run twice with) the later legacy workspace pass. Dual-vector/tail numerical
acceptance and FA hardware gates remain unverified.

Mode-2 flat-slice follow-up: the A2/A3 emitter now combines real local events
and source-pipe mode-2 sets with scalar cross waits, preserving original C/V
resource scopes. It requires an explicit two-Vector participant group, flat
unguarded tasks, distance-zero dependencies and distinct caller-owned channels.
Both workspace directions now actually execute sync-only with both Vector
lanes; native source-generation verifies mode 2 and the empty wait-pipe ABI.
Deleting a cross set deadlocks. Deleting a cross wait reports uninitialized
GM consumption when the producer is deliberately delayed; this is an
adversarial scheduling test, not a hardware timing prediction. Correct
synchronization passes both normal and delayed-producer schedules.

This integration exposed missing simulator operand metadata for native
WorkspaceReduction's short GM-to-L1/FIX-copy ABI. The bridge now supplies the
template dimensions and C++ defaults (including zero-tail meaning a full tile)
while preserving its existing explicit final ABI. A one-task local emitter
also needed to avoid the old TVM's prohibited single-element SeqStmt.

The standard migration suites pass **327 tests**, 66 subtests. Adding the
complete `test_tir_bridge.py` gives **742 passed, 1 failed**: the pre-existing
`test_row_expand_validates_rows_dtype_alignment_and_tag` expects `256-byte`,
but the existing diagnostic says `32-byte-aligned physical row stride of
128..4080 elements`. An isolated probe loading the exact pre-change HEAD
bridge reproduced that failure. Excluding that known test yields 742 passed,
1 deselected, 66 subtests; this is not claimed as a fully green bridge suite.
No NPU binary execution or FA acceptance is claimed. Guarded/cyclic cross-core
owner protocols and whole-function scheduled materialization remain pending.

Mode-2 loop-owner follow-up: unguarded single-slot CV loops now emit both local
and cross ready/release handshakes, scoped initial credits and final drain.
Both workspace directions execute on A2/A3 simulator configurations with two
Vector participants, nonzero loop minima and 0/1/2/5 iterations, under normal
and delayed-producer timing. Missing cross sets deadlock; missing final drain
fails runtime flag accounting, including empty loops (local residual credits).
These are real sync-only runs, not compilation-only tests.

Conservative dependency analysis also emits carried copies of same-epoch
edges. The emitter does not add a second credit protocol for those copies:
source(i) -> source(i+1) is actual source-pipe FIFO and the generated
source(i+1) -> target(i+1) fence proves the carried edge. This execution-path
proof avoids a false protocol cycle; reverse reuse edges remain fenced.
Guard/counter owners, mixed version counts and cyclic multiversion cross
channels are not accepted yet. The enabled whole-function entry still fails
closed until reconstruction and alias contracts are integrated.

The expanded suites remain **742 passed, 1 deselected**, 66 subtests; the one
deselected row-expand diagnostic test is the verified pre-existing failure
described above. No NPU FA hardware/performance result is claimed.

GM workspace/per-edge version follow-up: generated scratch GM parameters can
now expand their buffer_map capacity without changing parameter order,
auto_gm_indices or storage identity. Ordinary GM inputs are rejected. Physical
slot offsets use the original pointer integer width. Thread-axis bounds and
native copy helper spans prove Vector shard accesses when native access_ptr
extent advertises a whole-core tile; unknown helpers still fail closed.

The mode-2 owner emitter accepts independently certified slot counts per edge,
initializes/drains every ready/release slot, and uses the same owner-relative
iteration modulo as GM addresses. This permits a double-buffered GM exchange
alongside single-version local storage. Both directions execute on A2/A3 CPU
simulator configurations, with two Vector lanes, 0/1/2/5 epochs and normal or
delayed producers. The test supplies conservative full-workspace footprints
for its one-core fixture to analyze distance-two reuse: automatically deriving
these layout-aware certificates from general lowered TIR is still pending.
No arbitrary carried edge is changed into a timing assumption. Longer carried
duplicates are discharged through actual repeated source-pipe FIFO plus an
emitted shorter-distance fence; unsupported reuse distances are rejected.

Expanded regression: **744 passed, 1 deselected**, 66 subtests. The deselected
test remains the verified pre-existing row-expand diagnostic mismatch. This
milestone does not enable the full-function automatic entry or establish FA
hardware correctness/performance. Counter/guard clocks, complete extraction
proofs, scheduled reconstruction and safe alias contracts remain outstanding.

Automatic workspace-envelope follow-up: materialization records a TIR-native
storage pitch/count/iteration-owner certificate. The workspace proof adaptor
checks parameter capacity, generated-scratch indices, loop clock bounds, dtype,
the canonical slot offset and native row-major copy spans. Proven accesses get
conservative whole-slot byte envelopes for carried-dependency analysis. Unknown
helpers, changed pitches, wrong slot formulas and unsupported guards retain
their original unknown footprints, with diagnostics. Full overwrite, actual
synchronization and physical alias safety are deliberately separate proofs.

The two-direction/double-Vector/double-buffer execution regression no longer
manually supplies workspace Access footprints: certificates are derived from
the transformed copy TIR. Normal and delayed-producer runs still pass across
empty, short and wrapping loops. Three negative proof tests verify that missing
slot selection, incorrect pitches and unknown helpers do not remove conflicts.
Shared helper-span logic is used by both physical materialization and proof.

Expanded local regression: **747 passed, 1 deselected**, 66 subtests. The known
row-expand diagnostic mismatch remains separately documented. This covers
generated row-major GM scratch copies, not every on-chip layout or dynamic
counter/guard owner. The full automatic entry, FA kernel and NPU correctness /
stability / performance acceptance are still outstanding.

Executable-entry follow-up: enabled scheduling now returns materialized TIR
for supported unconditional flat local/CV epochs. It validates defined storage,
core accessibility and static per-core capacity, emits actual local/mode-2
fences and stores a JSON feasibility report on the function. Report costs are
explicit unit costs, **not calibrated NPU performance predictions**. Unsupported
loops, guards, scalar bindings or version hints still fail closed; this does
not enable the complete FA schedule yet.

Entry placement is after numerical/tail/workspace/safe-access legalization,
still before pipeline rewriting and flattening. This avoids changing the
analyzed operation stream after synchronization materialization. On the enabled
path target optimization requires the completed-materialization marker, skips
legacy cross-core/software pipeline and sync passes, and disables lexical
storage reuse. Native memory planning uses disjoint linear addresses until
asynchronous alias contracts are integrated. The disabled path retains its
existing pass sequence and configuration.

Four additional A2/A3 tests execute the automatic entry without manual flags:
vector fill/store passes true sync-only and native source generation after
target optimization; FP16 GEMM plus FIX GM store produces the expected CPU
numeric output. Both CV workspace directions also execute through the automatic
entry under normal and delayed producers. Spies reject any legacy pipeline or
sync pass invoked on the enabled path. Expanded local regression: **751 passed,
1 deselected**, 66 subtests, with the same previously verified row-expand test
failure excluded. These are CPU/native-source results, not NPU FA acceptance.
Whole-function loop/guard reconstruction, calibrated optimization and safe
alias admission remain necessary for the original FA128 performance goal.

Single-local-owner follow-up: the executable entry now accepts one unconditional
serial loop containing all tasks, with one physical version and no nested
control. Original allocation/core/thread wrappers remain intact. Same-iteration
and distance-one RAW/WAR/WAW stay in the report; redundant longer fences between
the same task pair are discharged by source-pipe FIFO plus the shortest emitted
fence, rather than being emitted as a second handshake. Initial credits and
final drain execute even for an empty owner.

Test-first validation covered A2/A3 with loop minimum 3 and extents 0/1/2/5;
all execute true sync-only with balance checking. Removing the final drain is
rejected in every case. Expanded local regression: **759 passed, 1 deselected**,
66 subtests (the existing row-expand diagnostic mismatch above). This is still
a conservative single-version local loop, **not** complete FA software
pipelining. Mixed C/V owner integration, guarded execution, scalar/control
reconstruction and joint version materialization remain outstanding; no NPU
correctness or performance acceptance is claimed.

Shared-CV-owner follow-up: automatic entry also reconstructs a sole static
serial C/V owner with one physical version. Both UB-to-L1 and L0C-to-UB numeric
workspace paths execute for extents 0/1/2/5 on A2/A3, with mode-2 ready/release
and final drain. The extent-two variants pass native target optimization and
generate actual mode-2 calls. The owner validator proves that a two-Vector
participant context dominates initialization, the loop and drain; detecting
that context somewhere in the function is not sufficient. Body tasks must
occur exactly once in matching explicit C/V scopes, with no hidden control or
allocation. Nonuniform bounds and wrongly nested participant scopes fail
closed; removed drain is rejected during execution. The expanded suite remains
**759 passed, 1 deselected**, now including these additional loop checks in
the existing four workspace tests.

This is still sequentially reconstructed single-version execution, not a
jointly optimized multiversion FA pipeline. Guards, scalar controls, multiple
epochs, online accumulated state and safe alias admission are not yet wired
into a complete FA driver. No NPU FA result or speed ratio has been measured.

Executable UB version-selection follow-up: the local owner driver now proves
full overwrite from actual native `Fill` count/offset/dtype and full-mask
`copy_ub_to_gm` source geometry. Pointer extent is not an overwrite proof.
For one compact scratch UB allocation it feeds true dependencies, storage reuse
constraints and A2/A3 capacity into Z3, then physically applies the selected
version count. Other allocations and persistent state remain fixed. Bare
integer-one hints opt out; supported explicit counts two and three are honored.
Counter clocks, multiple scratch candidates and unproven helpers are not yet
version-materialized (unsupported explicit hints fail closed).

The physical rewriter handles dominating opaque root allocation blocks,
preserves sibling allocations and records on-chip version certificates. Inner,
conditional or unrelated sibling allocations and logical block views are
rejected. Scalar index and owner-relative clock arithmetic must be proven not
to overflow before slots can be treated as disjoint. Fresh task/owner identities
are re-extracted after rewriting; synchronization uses the same owner-relative
slot count. Unversioned same-pipe hazards remain FIFO dependencies even when
their distance differs from the ring count. Alias annotations are validated
before eligibility filtering, including unsupported scratch owners.

A2/A3 tests cover automatic two-version selection, one-version opt-out and
explicit two/three versions: native allocation/source generation, full numeric
execution (five epochs starting at three, output seven), and sync-only balance.
Misleading full pointer extent with partial fill and read-before-fill owners
cannot be accepted for scratch rotation. These tests exposed and fixed
simulator fill scalar binding: loop environment substitution precedes literal
evaluation. Unit costs still mean feasibility, **not** calibrated performance.
Expanded local regression: **773 passed, 1 deselected**, 66 subtests; the
previously documented row-expand diagnostic mismatch remains the exclusion.
Whole-FA guard/scalar reconstruction, coordinated C/V multiversion owners and
NPU correctness/stability/performance acceptance remain incomplete.

Whole-function phase follow-up: automatic entry now reconstructs an
unconditional prefix, one regular static serial owner, and suffix. Boundary
RAW/WAR/WAW use whole-storage envelopes and one-shot notifications outside the
complete initialization/body/drain protocol. Prefix-to-suffix protection is
retained for zero owner iterations. Persisted accumulator allocations remain
single-version. Every helper shares a function-level event registry: local
pools are per core/destination pipe, cross IDs are globally owned, and lexical
wait submission does not permit reuse. Channel exhaustion remains an error.

The admission validator checks complete lexical task occurrence, unique node
identity, matching noncontradictory core scopes, a common two-Vector context,
core-neutral mixed-owner cuts and allocation domination. This matters because
native nested AIC/AIV conditions do not behave like simulator lane replacement.
The boundary wrapper is attached to the whole protocol, never to the repeated
For or an enclosing SeqStmt that also contains prefix operations.

A2/A3 tests execute persistent numeric accumulation for 0/1/3 iterations and
shared C/V FIX/workspace/Vector/store phases, with sync-only balance and native
target/source generation. Removed waits are detected (balance or hazard);
this is not a universal timing-race claim. Negative scopes, repeated task
nodes, unrelated allocation lifetimes and nested participant groups fail
closed. Expanded local regression: **781 passed, 1 deselected**, 66 subtests.
This remains single-version three-phase reconstruction: mutable scalar
controls, guarded owners, coordinated multiversion FA scheduling and original
14-case NPU performance acceptance are not yet complete.

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

### Developer-task placement and GEMM owner lifetime follow-up

The new entry now assigns missing C/V scopes from extracted physical calls,
without invoking legacy CombineCV. Mixed owners remain core-neutral; only their
operations are scoped. Single-core owners keep a flat body with their scope
outside the owner, so initialization and drain share that domain. Existing
matching scopes are preserved; contradictory scopes and repeated node identities
remain errors. Immediate tensor scalar operands fail closed: merely labeling the
enclosing call PIPE_V does not model its scalar GetValue or hidden barrier.

GEMM decomposition can allocate its freshly created, distinct L0A/L0B scratch at
an unconditional common root outside loop owners. The registry is the actual
created buffers, not a name-prefix heuristic. User allocations are not hoisted;
conditional roots fail closed. Aggregate capacity and physical dependency checks
still apply. Hoisted scratch reuse retains distance-one MMA-to-MTE1 WAR edges.
Unconditional sequence expansion is flattened without crossing scopes or guards.

A2/A3 tests execute generated GEMM/FIX loops for 1/2/5 epochs in full numeric and
sync-only modes, check correct CPU output and flag balance, reject removed-wait
mutants, and generate native source with exposed MMA tasks and real events.
Expanded local regression: **795 passed, 1 deselected**, 66 subtests; the same
pre-existing row-expand diagnostic exclusion remains. This is not NPU validation.

An actual AST-loaded Developer FA128 example was lowered without importing its
top-level Torch/NPU launch code. This exposed a bound nested-loop IV being treated
as unallocated memory; that diagnosis is fixed and tested. General row loops and
tensor-scalar operations are still not admitted. Whole-FA guards, tails,
coordinated version selection, event reuse and the 14-case hardware acceptance
remain open; these unit milestones do not satisfy final delivery.

Docker CPU follow-up at **19329738492ec8e511f2f3e74d09289d1b598aa1**:
the expanded suite (including `test_tir_bridge.py` with the documented exclusion)
reported **795 passed, 1 deselected**, 348 warnings, in **43.39 seconds**.
The independent worktree remained clean. It loaded the worktree Python package
and `build-auto-schedule` native library with the explicit TVM/venv paths above;
no NPU launch, global Python install change, main-checkout modification, or push.

### Next real-FA gate: native VID broadcast/count rewriting

Source inspection found a concrete mismatch in `src/transform/ascend_vid_reduction.cc`:
UB allocations and access-pointer extents are halved, but `tl.ascend_broadcast`
explicit shape operands are not. `BroadcastOpCodegen` consumes those operands.
Replacing the example's row scalar loops with broadcasts is therefore not yet a
safe workaround. Before admitting the automatic FA fixture, add native tests and
fix the following together:

- Axis-one `[M,N] <- [M]` / `[M,1]`: change both row counts to the actual mapped
  shapes; first slice requires full zero-offset operands and static even M.
- Axis-zero `[M,N] <- [N]` / `[1,N]`: only destination rows are distributed;
  preserve the replicated column source before building reduced buffer mappings.
- Preserve workspace byte arenas. Broadcast workspace is not row data: the
  aligned FP32 axis-one heuristic in `allocate_tmp_buffer.cc` needs 256 bytes
  independent of M, so blindly halving that arena is not justified.
- Reject unsupported slices/contradictory storage roles; prove shape and pointer
  span agreement, check generated native arrays and distinct row values on both
  Vector lanes. Threads-one behavior must remain unchanged.

Causal primitive counts need their own native tests: `createvecindex` is absent
from the tile-size allowlist, and scalar-mode `select` places its count before
dtype strings, not at the final argument. Do not use generic last-argument
halving, introduce a host mask as a hidden fallback, or launch a candidate with
these unverified shape/count rewrites. These are source-confirmed admission gaps,
not an observed NPU failure or a completed native fix.

Native broadcast follow-up now implements the regular full-operand slice:
axis-one row geometry and pointer spans halve consistently; axis-zero column
sources and explicit/injected broadcast workspace remain replicated. Source
layout must actually be `[M]`/`[M,1]` or `[N]`/`[1,N]`, not merely have the same
element count. Slices, odd/symbolic rows, noncompact bases and conflicting
replicated/sharded broadcast or elementwise roles fail closed. Threads-one is
unchanged. Nineteen native tests cover these contracts, A2/A3 generated arrays,
the 256-byte arena and different values on both Vector lanes. This is native
source plus CPU numeric validation, not NPU validation.
Expanded local regression: **814 passed, 1 deselected**, 66 subtests, 8.20 seconds.

The local module build target is `cmake --build build-auto-schedule --target
tilelang_module -j 4`. The unrelated macOS aggregate shared-library target has
unresolved TVM link symbols; the actual loaded module target builds successfully.
Do not treat an aggregate build failure as permission to silently load an old
module. Causal create-index/select count handling remains open. A separate
placement follow-up must test single-core scopes inside participant attributes:
the simulator chooses Vector lanes when it encounters `resource_scope`, so a
scope outside `thread_extent` does not demonstrate correct per-lane execution.

Docker native broadcast milestone **cdb4f87472b56c09c7e5db922162c5beec4e672b**
was rebuilt as `tilelang_module` and validated in the independent worktree:
**814 passed, 1 deselected**, 348 warnings, **44.14 seconds**, clean checkout.
No NPU was launched. The single-core placement follow-up now inserts its scope
inside the deepest common launch/participant binding while remaining outside
the flat owner. Separate participant groups without a common binding fail closed.
A real automatic-entry CPU test fills lane-private UB using each participant's
index and observes distinct values one/two on both Vector lanes. This fixes a
previous admission/bridge mismatch; it does not waive the upcoming full-FA tests.

### First whole regular FA128 integration (not final ladder acceptance)

`testing/python/transform/fa128_auto_fixture.py` adapts the existing Developer
FA online-softmax example to full vector broadcasts, without manual scopes,
flags, barriers, physical addresses or slot indices. Its fixed shape is BHSD
`[1,1,128,128]`, block M/N=32/64, two KV epochs, noncausal with no tails.
`test_ascend_schedule_fa128.py` runs real LowerAndLegalize/OptimizeForTarget for
A2 and A3, generates exposed MMA/native mode-two events without PIPE_ALL,
checks finite numeric output against CPU FP64 golden, and truly executes
sync-only with the unchanged 60-second timeout and error-level flag balance.
Removed cross sets deadlock; removed cross waits read uninitialized GM
workspace at runtime. The static local-memory checker does not diagnose these
GM-hop mutants, so the tests do not claim it does.

This fixture first failed because the function-level destination event pool
exhausted. No event limit was increased. A distance-zero A->B protocol's release
already proves B(i)->A(i+slot_count), so an exactly matching reciprocal carried
protocol is redundant. Local and mode-two emitters now discharge only this exact
endpoint/slot match, retaining all analyzer hazards in the report. Different
slot counts/endpoints and carried-only protocols remain. Drain mutation tests
remove the actual suffix, rather than assuming the old protocol count.

The first standalone CPU observation was approximately 0.029 seconds for true
sync-only execution, approximately 0.34 seconds for the numeric run plus golden
comparison, finite output and max absolute difference 3.7e-5. These are not
hardware timing or a baseline comparison. Expanded local regression:
**823 passed, 1 deselected**, 66 subtests, 13.52 seconds.

The final ladder module, causal/tail support, coordinated automatic multiversion
pipeline, fourteen-case stability and NPU performance acceptance are still open.
Designer integration must export the existing config/factory/runtime-scale
interface and explicit `auto128` selection. Generated workspaces change signature
indices: do not reuse runners' hardcoded 4/5/6 workspace indices if index four
is sm_scale. Existing verification helpers also need top-level/input-attribute
causal propagation, using the existing bottom-right golden convention, not a
new scoring rule. All required implementation/test selections must be explicit.

Docker regular FA follow-up at **899f94f0b4ba15b4e73e492098372480260e97ef**:
**823 passed, 1 deselected**, 944 warnings, 66.45 seconds, clean independent
checkout. No NPU launch or remote push.

### Causal coordinate/count integration

Native VID now rewrites full compact CreateVecIndex destinations using the
actual lane capacity and rebases the first value by `vid * lane_capacity`.
Replicated column sources retain their complete count and origin. Select's
scalar-constant and tensor variants use their actual count argument (six, or
seven with optional workspace), not the trailing dtype strings. Packed uint8
masks must partition on byte boundaries. Immediate tensor-scalar variants,
nonzero-offset slices, odd/unproven counts and incompatible capacities fail
closed. Select workspace stays replicated when present; real AscendC lowering
normally removes its unused explicit workspace before VID.

The FA integration fixture now constructs causal coordinates and masks entirely
in the kernel, using the existing right-bottom convention `j <= i + Skv - Sq`.
It selects a finite negative score before maximum reduction and zero probability
after exponentiation for masked elements. No host mask or hand synchronization.
Full numeric/sync-only/native tests cover both platforms and `(Sq,Skv)`
`(128,128)`, `(64,128)`, `(128,256)`, causal and noncausal. Primitive tests observe
different global row coordinates on both Vector lanes and packed-bit Select
results, not only changed argument counts. Expanded local regression:
**842 passed, 1 deselected**, 66 subtests, 52.51 seconds.

Existing `model.py` and `fa_kvend_runner.py::golden_fa` do not sanitize fully
masked causal rows; no root case JSONL contains causal `Sq > Skv`. This fixture
explicitly rejects that boundary rather than substituting a zero-output golden.
Final support/precision policy must not be inferred from a new scoring rule.
Q/KV tails, final ladder integration, automatic coordinated versions/pipeline,
and all fourteen hardware stability/performance gates remain incomplete.

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

### Q/KV tail integration (CPU/native milestone, not hardware acceptance)

The integration fixture uses ceiling-divided Q/KV tile counts and masks invalid
KV columns before maximum reduction and after exponentiation. Existing
bottom-right causal masks already exclude out-of-range KV columns for valid Q
rows. GM-to-L1 copies remain single-version, base-zero destinations, so the
native helper clears the entire template FRACTAL extent for partial tiles. This
does not certify partial copies into nonzero ring slots.

Q tails with at most sixteen valid rows leave Vector 1 with a zero-row output
DMA. The simulator previously attempted to resolve that empty shifted GM view,
causing a bounds error despite no memory being accessed. Full and sync-only
execution now skip certified empty copy rectangles, after preserving any full
GM-to-UB padding initialization. General memory bounds checks are unchanged.
The native UB-to-GM helper returns before DMA when either valid dimension is
zero; surrounding flags and both participants still execute normally.

Test-first regressions reproduce both the empty-view error and the native
helper issuing an empty DMA. A host C++ stub executes the actual helper text,
observing no DMA for zero rows/columns and one DMA each for 1/15/16 rows.
FA coverage includes `(Sq,Skv)` `(32,65)`, `(64,130)`, `(1,65)`, `(16,65)`,
`(17,65)`, `(33,130)`, in addition to the previous aligned shapes, on A2/A3
with causal/noncausal masks. Each checks generated native code, CPU FP64
numerical reference, true sync-only execution with a 60-second timeout, and
missing-cross-set/wait mutations. Expanded local regression: **871 passed,
1 deselected**, 66 subtests, 104.27 seconds. The deselected preexisting bridge
alignment test remains documented; no checker or precision threshold was
relaxed. These results do not constitute the required fourteen-case hardware
precision, stability or performance acceptance.

### First actual A2 execution

The automatic ladder entry is now installed at the specified task's
`design/tile_level/ladder/fa_auto128.py`. The task's existing
`model_new_tilelang.py` adds only an explicit `FA_STEP=auto128` import branch;
its default and previous ladder versions are unchanged. A tracked copy of the
kernel lives in `examples/flash_attention_auto128/fa_auto128.py`. It supports
BHSD tensors using a flattened batch/head/Q-tile grid, runtime `sm_scale`,
bottom-right causal masking and Q/KV tails. This initial 32x64 version is a
correctness baseline; coordinated multiversion/performance acceptance remains
open. Its native signature is Q/K/V/Output/sm_scale followed by generated
workspaces: legacy manually supplied workspace indices 4/5/6 must not be used.

On NPU3's `wzz_cann`, physical card 6 was healthy and had no running processes.
The actual existing `verification_tilelang.py` ran case **0**, FP16
`[1,4,128,128]`, with `FA_STEP=auto128`, `ASCEND_RT_VISIBLE_DEVICES=6` and
`ASCEND_LAUNCH_BLOCKING=1`. It reported **PASS / matched**, against CPU FP64
golden and original-dtype native using its unchanged cannbench comparison.
No timing, repetition stability, remaining case or A3 hardware result is
claimed by this single smoke result. Output: `/tmp/a2a3-auto128-case0.json`
inside Docker. The fixed fourteen-case precision run was subsequently launched
using the same script; it is not recorded as passed until terminal evidence.

Environment fingerprints for this execution:

- Compiler checkout: `cb896f6c35a9328d108e7ceeb9c181c89e064c48`.
- Python: `/tmp/a2a3-auto-schedule-fa128-venv.JsRZee/bin/python` (3.12.13),
  using system torch 2.10.0+cpu plus torch_npu; NPU availability was True.
- Package: `/home/wenzhongzhen/tilelang-ascend-wt/auto-schedule-fa128/tilelang`.
- Native module SHA256:
  `4ce6649be570c447d58f5fa1e5b236eb1a1473a6a9f88a010de4e4578eddd98b`.
- TVM library SHA256:
  `2df7999bbed69490640f5bc50166a5a650ac543bdea22514878e86ae31ccd36e`.
- Installed ladder source SHA256:
  `fbe8d1805e6865f6a66f89c7506ec948e9c83fa16d13579f38dfb9408e5c87d1`.
- CANN 9.0.0; Bisheng 2026-04-25 clang15.0.5; architecture dav-2201.
- Passes: auto_schedule, auto_cv_combine, auto_cross_core_sync enabled.

Independent worktree dependency preflight must include actual device headers,
not just successful host module build or CPU tests. Two initial attempts stopped
at compilation, before kernel execution: missing CATLASS, then missing shmem.h
(unconditionally included by common.h even though this kernel does not use
SHMEM). Both dependencies were independently checked out from clean local
object repositories at the checkout's exact gitlinks: CATLASS
`769cd40a8716b28650b6bebb08db4834eea4462f`, SHMEM
`157eecf7d2102c7282269db55e54362e90dbac7b`. Main-checkout dependencies and
global Python were not changed. This initializes required build headers only;
it does not extend the task's SHMEM feature scope.

For reproduction, source `/usr/local/Ascend/ascend-toolkit/set_env.sh`; set
PYTHONPATH to the independent checkout and existing TVM Python directory,
TVM_LIBRARY_PATH to `/home/wenzhongzhen/tilelang-ascend/build/tvm`,
TILELANG_LIBRARY_PATH to the independent `build-auto-schedule`, and prepend
both library directories to LD_LIBRARY_PATH. Use the original designer
`verification_tilelang.py <task-root> --case 0 --json <report>` with the
explicit selector and card environment above. Do not switch global installs.
