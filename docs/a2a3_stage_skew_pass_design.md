# A2/A3 executable stage-skew reconstruction

## Purpose and placement

`stage_skew.materialize_staged_owner` is a Python older-TIR reconstruction
component, not a new C++/FFI pass or configuration. It consumes an already
synchronized regular owner with complete dependence analysis and certified
storage versions. The production driver now calls it for certified K/V input
prefetch (stages0/1), with joint capacity feasibility and phased ring admission.
More general C/V stage partitioning and intermediate-storage admission remain
open; support in this reconstruction component alone does not enable them.

The intended call position is inside the enabled AscendAutoSchedule path after
physical version materialization and owner ready/release insertion, before
target flattening/code generation. Boundary fences, initial credits and drains
remain outside the transformed loop. The legacy pipeline/sync path stays off.
Pass-design skill dependency and IR guidance informed this component; the
user-approved Python migration overrides its generic new-C++-pass template.

## Executable transformation

Input pseudo IR:

```
seed slot credits
for logical_epoch in [min, min+T):
  load K       # complete wait / operation / set bundle
  read K
  load V
  read V
drain slot credits
```

With load stages0 and compute stages1, output pseudo IR:

```
seed slot credits
for wave in [0, T+1):
  if 0 <= wave < T:
    load K/V bundles at logical_epoch=min+wave
  if 1 <= wave < T+1:
    compute bundles at logical_epoch=min+wave-1
drain slot credits
```

Buffer addresses and all flag selectors are substituted together. Thus compute
uses `(wave-1)%versions`, not `wave%versions`. Conditions guard the whole bundle,
not only its numerical instruction. T0/T1 execute no speculative DMA or flag
operation for an invalid logical epoch. The wave extent T+1 must fit the native
integer type; the existing participant-uniform clock proof remains mandatory.

The reconstruction now admits any complete assignment of nonnegative integer
stages with minimum0 and maximumS>0, including holes in stage numbering. The
wave extent is T+S, each bundle remains guarded by `s <= wave < T+s`, and its
logical epoch remains `min+wave-s`. Empty stages add no operations. Overflow
uses the certified maximum T plus S, not just T+1.

## Ordering proof

For analyzed edge a(e)->b(e+d), execution waves are e+s[a] and e+d+s[b].
Require `s[a] <= d+s[b]`. When equal, require source before target in the actual
wave's bundle order. This rejects one-slot WAR reuse when prefetch moves ahead
of consumption. The test verifies a two-slot WAR can admit the same schedule.
Estimated duration does not establish this order; no inferred edge is dropped.

For a producer at stage0 and last consumer at stageS, a reuse distance S is
not sufficient under this ascending-stage wave order: the next producer is
before the old consumer in the same wave. The carried WAR proof rejects it;
distance S+1 may be admitted. The caller must still certify the actual number
of physical slots, complete writes/read envelopes, all true dependencies and
event ready/release/seed/drain protocols. This helper does not infer versions
or certify those protocols merely from a nonnegative stage assignment.

Within each stage, original bundle order is retained. Only flat unconditional
scoped task/event bundles are accepted; unknown operations, orphan flags,
duplicate identities, incomplete stage assignments and nested control fail.
Caller must supply complete true memory dependencies and storage certificates:
this helper cannot prove eligibility from pointer counts, task names or timing.
It also does not validate the numerical semantics of user-supplied operations.

## Tests and remaining integration

Test-first7 tests initially failed on missing implementation; all8 current
tests pass. They enumerate0/1/2/3 trips, verify exact operation counts/logical
epochs, flag-slot substitution and next-load-before-current-compute order;
negative cases cover single-slot WAR, orphan wait and wave overflow. The8th
test consumes actual `emit_mode2_loop_sync` output and checks guarded generated
cross flags while seeds/drains remain outside the owner. These are compiler
reconstruction tests, not a production FA hardware speedup claim.

The multi-stage extension was test-first: ten new cases failed at the old
two-stage admission gate before implementation. The reconstruction suite now
has23 tests, including stages0/1/2 with3 slots and0/2/3 with4 slots, T0/1/2/5,
group-dependent clocks, boolean/negative/unanchored stages, same-wave reuse
conflicts and T+S overflow. Together with the existing uniform-clock suite,
41 tests pass locally. The six-file regression spanning stage reconstruction,
uniform clocks, phased protocols, L1 prefetch, materializer and physical buffer
versions also passes:113 tests, with old TVM/Python AST deprecation warnings.
`examples/flash_attention_auto128/run_local_compiler_tests.sh` pins the verified
local Python/TVM/native paths and appends the isolated solver package path
without shadowing Conda's setuptools; it prints the actual library bindings.
Existing actual-mode2 bundle coverage is retained.
The initial multi-stage cases are TIR reconstruction tests, not hardware
acceptance of a new FA schedule. Follow-up source-generation tests in
`test_ascend_schedule_cv_stage_waves.py` use the actual workspace lowering:
FIX→Vector MTE2→Vector operation→Vector MTE3→Cube MTE2→Cube MTE1,
with stages0/1/2 and two three-version generated GM parameters. Both A2 and A3
generate native sources for T0/1/2/5. Version plans are explicitly supplied
for this fixture; this does not certify production automatic discovery,
collective complete-write eligibility or numerical FA correctness. The L0C
source has an external initialization precondition; no such kernel was launched.
The fixture supplies16 cross IDs for its12-channel three-slot protocol; it is
not production-default event-capacity admission. Supplying the production
eight-ID pool is tested to fail with event-capacity exhaustion. Native checks
also verify logical stage1/2 pointer shifts and guards, modulo3 slot selection
for longer owners, and legal constant-slot folding for a one-trip owner.

These native tests exposed a bundle-boundary bug: SeqStmt construction flattens
adjacent task/event sequences inside a shared C/V scope. The owner emitter now
wraps each operation and its waits/sets in `tl.ascend.task_bundle` metadata.
Stage reconstruction preserves that exact boundary, not an inferred adjacency.
The attribute generates no hardware work. Removing the marker reproduces the
fail-closed orphan-synchronization error; removing a carried physical WAR edge
is separately rejected by complete dependency validation.

Eight-file regression after this bundle fix:125 tests passed, including native
workspace lowering and existing production prefetch/phase paths. The old TVM
parser emits17744 Python AST deprecation warnings. No new simulator feature or
real-device numerical result is claimed for this native-only fixture; the
ongoing hardware profiler uses the frozen `1aeedc56` checkout, not this change.

Before using wider stages in production: certify each intermediate buffer's
complete writes/read envelopes including tails; select stages and counts jointly
under A2/A3 capacities; preserve Q/m/l/o single-version; analyze carried
dependencies to the maximum selected count; pass edge-specific counts to phase
protocols; prove all slot ready/release/seed/drain paths; check native source and
actual-device FA precision/profiling. None of these obligations is discharged
by simply increasing max_stage. Do not represent the standalone helper as
completed automatic software pipelining.

## Group-relative generated-workspace coverage

`workspace_epoch.certify_workspace_epoch_write` returns a typed writer
partition certificate, not an `Access.full_overwrite` hint or a versioning
authorization. A Cube writes one complete group tile; the two Vector
participants write complementary half-tile intervals. The version pitch is
the entire generated GM allocation, while coverage is relative to one group.
Conflating these two sizes would incorrectly authorize arbitrary group reads.

The first admitted geometry is an unconditional generated compact GM view,
one dominating `blockIdx.x`, two `threadIdx.x` participants, exact group/shard
offsets, complete native DMA dimensions and a source allocation on the actual
ancestor path. All resource scopes must agree, including nested scopes.
Different buffer views sharing the same data variable are rejected. FIX is
restricted to a complete 16-aligned compact L0C tile at source base zero;
MTE3 requires a 32-byte-aligned UB base and contiguous local row pitch.
Address intermediates and participant indices must fit their native integer
types. The FIX boolean template argument is `enRelu`; both recognized values
preserve coverage. It is not an atomic mode.

Test-first writer coverage:26 tests pass, including actual native-lowered
single/three-group C/V cases. New negatives reproduced seven validation gaps
before the first fix, then two more row-pitch/address-overflow gaps before the
second fix. A further narrowing source-pointer cast reproduced a third
validation gap: TVM clips Cast bounds to the destination range, so the
preconversion bounds must separately fit that range. Remaining negatives
include partial DMA, conditional execution,
unknown participants, source role/geometry/base and ordinary user GM input.
This module is not yet used for production GM version admission: every reader
still needs the same group partition proof and all endpoints need a common
owner clock and lifetime protocol. No new numerical or hardware result is
claimed for this isolated writer certificate.

Nine-file regression including this certificate:151 tests pass locally, with
17744 existing TVM/Python AST deprecation warnings. The local test wrapper now
includes this suite by default; the actual library paths remain printed.

The subsequent reader certificate revalidates the writer against the current
function, requires matching group/participant axes and the exact same serial
owner, verifies complete GM→UB/GM→L1 geometry and a dominating aligned local
destination. The aggregate `certify_workspace_epoch` audits every pointer use:
one writer must precede all readers, while direct load/store, unknown helpers,
RMW, multiple writers and escaped data or parameter handles are rejected.
Native endpoints must be direct Evaluate statements, not calls hidden in
expression-level conditionals. Blocks with initialization need a separate
execution proof; a reduction init is not an every-epoch writer. Bounds with
TVM infinity sentinels are rejected before native integer range checks.

The workspace suite now has70 cases. It includes16 actual FA lowering inputs
across A2/A3, Q128/130, KV192/130 and causal/noncausal, with three generated
workspaces each. It observes the real scheduler input by suppressing only the
scheduling entry in the test; native workspace lowering, GEMM decomposition
and core normalization still run. This is not a full scheduler or hardware
test. All three certificates and the existing participant-uniform owner-clock
proof pass in each input. Source-level `enRelu` spellings0/1 and true/false are
accepted consistently. Group-dependent causal loop extents do not imply a
participant-dependent clock or a guarded DMA endpoint.

Production integration still needs a typed partition lifecycle planner rather
than L1's whole-allocation `full_overwrite` projection, a common joint stage /
version plan, explicit accounting of host-allocated GM workspace bytes, and
the existing mode2 ready/release/seed/drain reconstruction. This certificate
alone does not authorize a physical version rewrite or claim hardware speedup.

The nine-file compiler regression after reader/all-endpoint integration passes
195 tests, with65232 existing TVM/Python AST deprecation warnings. No simulator
path was added or used in these native-input tests.

## Typed workspace lifecycle planning

`workspace_planning.discover_workspace_epochs` now maps the complete partition
certificate to exactly one extracted writer and all readers, preserving task
and Access metadata. It checks roles, native helper core/pipe assignments,
shared GM storage domain, exclusive owner and participant-uniform clock.
Candidates sharing the owner share one clock certificate. Both the current
function and extraction are checked for explicit Stage/version constraints;
these remain on the existing validated hint path, not unconstrained discovery.
Already address-planned input requires the asynchronous alias contract.

`plan_workspace_epochs` is a separate typed partition planner. It does not
route collective writes through L1's full-world `Access.full_overwrite` proof.
Reader stage cannot precede writer stage. For ascending stage-wave execution,
the slot count exceeds the writer-to-last-reader stage span, with minimum2.
The returned iteration EpochPlan undergoes the existing physical rewriter's
pointer-span/index/all-use preflight; the temporary rewritten IR is discarded.
The original world version pitch and group coverage byte count remain separate.

Nineteen test-first discovery/planning cases cover stage spans1/2/4, invalid
stages/counts, escaped parameters, incorrect execution metadata, private GM,
explicit constraints, stale-extraction constraints and physical-address input.
The16 real FA native-input cases additionally discover all three candidates,
share their owner clock, and preflight two-slot physical materialization for
each workspace independently. That per-workspace test is not a globally
executable FA stage assignment. Ten-file regression passes214 tests.

The planner has not yet been installed in the production entry. Joint local
capacity, host-GM workspace allocation accounting, complete dependencies and
event-channel feasibility must select one common schedule before actual rings,
mode2 protocols and waves are emitted. GM bytes cannot be charged against L1
or UB hardware capacity. No new hardware performance result is attributed to
these local changes; the frozen prefetch build's three-round performance gate
has failed at0.107698, with all84 original reports archived separately.

## Joint solver workspace accounting

The joint solver and result validator now distinguish `StorageCost.scope=GM`
from local hardware scopes. GM version selection remains bounded by each
storage's max_versions; a caller-supplied host allocation budget additionally
bounds the sum of world-pitch bytes times selected counts. A GM entry in local
`capacities` is rejected. Omitting the host budget does not invent device DRAM
capacity: reports explicitly record selected bytes, a null budget and
`device_available_memory_verified=false`. Local capacities are unchanged.
Minimum-allocation checks, Z3 constraints and injected-result validation use
the same separate workspace rule. Byte sizes must be nonnegative integers;
NaN/fractional/bool sizes cannot bypass budget comparisons in fallback mode.

Test-first coverage includes local/GM separation, exact and insufficient
budgets, joint II/version selection under different budgets, immutable carried
state dependencies and invalid numeric inputs. No new public TileLang pass
configuration or simulator configuration was added.

## Production FA partition evidence

Read-only analysis of actual native FA input B1/H1/Q128/KV192/D128 identifies
owner tasks4…34. The existing conservative dependency model rejects a direct
QK C0 / softmax V1 / PV C2 / output V3 split: correction and shared tmp_ub
carried edges span V3→V1, and unversioned GM carried WAR ties violate the
ascending-stage wave order. This is an admission result under current
footprint proofs, not a hardware measurement or a claim that every conservative
edge has been independently established as a numerical dependency.

A smaller candidate passed read-only physical TIR and d0…2 dependency checks
for KV192/130 with causal/noncausal: QK/score tasks4…8 at stage0, the remaining
softmax→PV→output epoch at stage1, only score GM workspace at two versions.
The m/l/o, correction, shared temporary and other GM workspaces remain
single-version. The reader clock stays participant-uniform within each group.
No events were emitted and no kernel was launched for this candidate yet.

Combining K/V input prefetch could instead use loads at0, QK at1 and the
remaining epoch at2. Under ascending-wave ordering it requires at least K2,
V3 and score-GM2 slots; joint local capacity and event allocation must prove
admission. These are next production integration candidates, not completed
automatic partition selection or achieved hardware performance.

Twelve-file compiler/solver/report regression after workspace accounting:
255 tests passed, with65232 existing TVM/Python AST deprecation warnings.

## Production Cube-prefix overlap integration

The previously read-only two-stage candidate is now installed in the
unconstrained production entry via `score_pipeline.py`. Candidate discovery
uses typed complete Cube-to-Vector GM coverage, not FA names or task IDs.
The pure Cube prefix through the earliest complete workspace writer is stage0;
the entire remaining epoch is stage1. Only that workspace is physically
rotated. The full d0…2 dependency set, same-wave ordering, local allocation
capacity and mode2 channel protocol are checked before admitting the wave.
Online softmax carried state stays single-version. This is a checked
two-stage candidate, not general optimal A5 schedule search.

No eligible candidate retains the previous L1-prefetch path. An infeasible
score candidate emits an explicit diagnostic and uses the new sequential
materializer, not legacy auto-sync. Explicit stage/version constraints retain
their separate validated path. Unit costs establish feasibility only.

Compiler regression covers actual native A2/A3 FA inputs with Q/KV tails and
causal guards, unmocked entry selection, zero/one-trip seed/drain and rejection
of missing carried edges, wrong slots and reversed same-wave dependencies.
The normal target pipeline is also checked with outer memory planning enabled:
the automatic path overrides lexical reuse, and final `address_map` offsets
agree with generated UB `GetWithOffset` allocations. Physical Vector-subblock
sizes are used to prove score and score_recv byte intervals disjoint.

This integration is Python-only. Docker deployment must use a committed
compiler identity and a new isolated cache, recording the reused native
library's original build identity and hash separately. The frozen1aeedc56
hardware precision/performance results above do not validate this candidate.
Numerical acceptance and performance remain real-device gates; simulator
execution is not a prerequisite for this work.

## Larger-tile admission, separate from the frozen hardware campaign

The example configuration now accepts `block_m` and `block_n`, preserving
32x64 defaults and the existing JIT/task interface. M must be an integer
multiple of32 (two Vector participants with16-row Cube geometry); N an integer
multiple of16. These are algorithm tiling values, not event IDs, slots,
addresses or hand scheduling. No new compiler or simulator switch is added.

Test-first native-input checks admit64x128 with Q/KV length130, causal and
noncausal on both A2/A3. The production automatic entry still selects the
certified two-stage workspace wave. Final generated UB allocation intervals
are checked disjoint and bounded by the platform's real capacity; unknown
allocation dtypes cannot silently escape the assertion. Invalid geometries
are rejected before lowering. Larger tiles have no real numerical or timing
validation yet; native generation is not hardware acceptance. The Docker
2c83b27d campaign remains frozen32x64 until all three rounds finish.
