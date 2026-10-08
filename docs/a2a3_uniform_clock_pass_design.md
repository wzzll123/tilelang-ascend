# A2/A3 participant-uniform owner clocks

## Scope and placement

This validation component of AscendAutoSchedule permits bounded immutable
core-group-dependent loop counts. It does not itself choose pipeline stages,
rotate buffers or alter numerical operations. No additional configuration.

Placement follows the user-approved migration: within `schedule_module` at the
end of LowerAndLegalize, after LowerTileOp, workspace reduction, safe-memory
legalization and Simplify, before target pipeline/flattening. The enabled path
does not also execute legacy pipeline/sync rewriting. Unlike the generic skill
template, this is a Python older-TIR adaptation, not a new C++ pass or FFI API.

Inputs are the original PrimFunc, exact serial For identity, and dominating
thread_extent attributes. Outputs are a bounded Analyzer context and minimum/
maximum trip count. Consumers: CV owner validation, prefix/body/suffix sync,
and iteration-version allocation. All three must certify the same original
owner; none may infer bounds from unrelated sibling thread attributes.

## IR and proof

Input/output pseudo IR is unchanged:

```
thread_extent blockIdx.x = groups:
  thread_extent threadIdx.x = 2:
    prefix
    for epoch in [0, min(kv_tiles, ceildiv((cid % q_tiles + 1)*32 + Skv-Sq, 64))):
      Cube and Vector tasks
    suffix
```

Previously this was rejected solely because the extent was not IntImm. The new
certificate accepts IntImm and dominating blockIdx.x expressions composed of
Add/Sub/Mul/Min/Max, positive-divisor FloorDiv/FloorMod and lossless integer
Cast. Tensor loads, calls, participant IDs and unbound parameters are rejected.
Parameterized runtime clocks without certified parameter bounds remain outside
this component; adding them requires an explicit immutable range contract.

Every intermediate is checked for integer overflow before bounding the final
extent. A final Min cannot justify an overflowing multiplication. Extent must
be nonnegative; absolute and relative owner indices must fit their integer
types. The unique actual owner ancestor path supplies group ranges; duplicate
or conflicting thread bindings invalidate the proof. Other semantic gates
(guards, phase ownership, allocations and participant count) remain active.

## Implementation and dependencies

`uniform_clock.py::certify_uniform_clock` returns a frozen UniformClock record.
It traverses the owner's ancestor path without rewriting TIR, binds the one
dominating blockIdx.x range, validates expressions recursively, and checks
owner bounds. `buffer_versions` uses this context before binding the loop
index, fixing its former extent-before-thread-range ordering. Existing pointer
footprint, slot-offset and allocation-budget checks still run afterward.

The emitter retains the original loop extent. Empty and short loops still use
the established initialization/drain protocol. This certificate is not a proof
of operation first-write coverage, alias safety, guard equivalence or version
reuse; those remain separate prerequisites. Low-level emitters are not general
untrusted entrypoints: callers must perform the clock validation.

## Tests and acceptance

Test-first: the initial twelve certificate tests failed on the missing module;
the version binding-order regression separately failed with owner-overflow.
After implementation, the clock and existing version suites passed34 tests,
including A2/A3 native source generation preserving the dynamic owner and
generated events. Negative cases cover vid, load, call, unbound variable,
intermediate overflow, zero divisor, negative extent, sibling range and
ambiguous thread bindings. No simulator implementation changed.

This establishes the compiler building block, not hardware FA pruning speedup,
the full coordinated software pipeline, or the0.90 performance gate. Those need
actual-device correctness and profiler measurements after integrating the
causal FA bound. Generic pass-skill dependency/IR/negative-test guidance informed
the implementation; existing user-selected pipeline placement takes precedence.
