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
New multi-stage cases are TIR reconstruction tests, not native A2/A3 or hardware
acceptance of a new FA schedule.

Before using wider stages in production: certify each intermediate buffer's
complete writes/read envelopes including tails; select stages and counts jointly
under A2/A3 capacities; preserve Q/m/l/o single-version; analyze carried
dependencies to the maximum selected count; pass edge-specific counts to phase
protocols; prove all slot ready/release/seed/drain paths; check native source and
actual-device FA precision/profiling. None of these obligations is discharged
by simply increasing max_stage. Do not represent the standalone helper as
completed automatic software pipelining.
