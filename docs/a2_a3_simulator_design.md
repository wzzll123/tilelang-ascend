# A2/A3 Functional Simulator and Performance Trace Design

Status: implementation in progress on `feat/a2-a3-simulator`.

The initial foundation is implemented: shared final-TIR lowering, `simulator=True` JIT
plumbing, A2/A3 profiles, SimIR program validation, a first fail-closed TIR bridge, static
discrete-event scheduling, local/cross flag and barrier modeling, byte-addressed local memory,
hazard checks, storage aliases, and Chrome/Perfetto trace export. Functional operation
executors and the remaining roadmap phases are not implemented yet.

## 1. Purpose

This document defines a CPU-hosted simulator for TileLang-Ascend kernels targeting
Ascend A2 and A3. The simulator has two related responsibilities:

1. **Functional simulation**: execute a lowered kernel without CANN or NPU hardware and
   detect numerical, bounds, initialization, memory-lifetime, and synchronization errors.
2. **Performance trace simulation**: run a parameterized discrete-event model of the AIC/AIV
   pipelines and export an explainable Perfetto/Chrome trace.

The performance model is intended to compare schedules and explain stalls. It is not a
cycle-accurate replacement for on-device profiling, and simulated cycles must not be reported
as measured kernel latency.

The supported hardware scope is deliberately limited to the common C220 execution model used
by A2 and A3. A2 and A3 share functional semantics but use separate timing profiles.

## 2. Scope and non-goals

### In scope

- A2 and A3 platforms.
- Both Developer and Expert programming styles after they have been lowered to TIR.
- Cube and Vector execution, legal local-memory spaces, GM workspace, explicit and automatic
  synchronization, software pipelines, functional checks, hazard checks, trace generation,
  and calibrated timing profiles.
- The full operation roadmap in [Section 10](#10-phased-operation-coverage), including copy,
  vector, reduction, GEMM, MMA, bias, fixpipe, quantization, convolution/im2col, sort, TopK,
  gather, transpose, atomics, and Persistent kernels.

### Explicitly out of scope

- **shmem is not supported and is not planned.** This includes any device-side shared-memory
  or cross-core shmem API and its visibility/coherence protocol. The simulator must fail with
  `UnsupportedSimOpError` if such an operation reaches the bridge.
- A5/C310, CPU, CUDA, and HIP simulation.
- Bisheng, PTO, or AscendC source interpretation.
- Exact instruction issue, cache, NoC, DDR contention, thermal, or frequency simulation.
- Treating simulated cycles as autotuner latency before a separately approved validation gate.

TileLang's Ascend storage-scope names such as `shared.l1`, `shared.ub`, and `shared.bt` describe
AICore local memories. They are **not** the shmem feature excluded above and remain in scope.
GM workspace created by `AscendWorkspaceReduction` is also in scope.

## 3. Confirmed compiler integration point

The current Ascend lowering path in `tilelang/engine/lower.py` is:

```text
PrimFunc / IRModule
  -> attach npu_platform
  -> LowerAndLegalize(mod, target)
  -> OptimizeForTarget(mod, target, platform)
  -> device_codegen(mod, target, platform)
```

`LowerAndLegalize` currently performs, among other work, buffer-scope inference, VID reduction,
parallel-to-vector lowering, layout inference, tile-op lowering, tail-mask propagation,
workspace reduction, and safe-memory legalization.

`OptimizeForTarget` currently performs the passes that are essential to simulation fidelity:

```text
PlanAndUpdateBufferAllocationLocation
CrossCorePipeline
CombineCV
PipelinePlanning
InjectSoftwarePipeline
AscendLowerOpaqueBlock
NarrowDataType / ConfigIndexBitwidth
Flatten2DBuffer / FlattenBuffer
VectorizeLoop
AscendStorageRewrite
UnrollLoop and TIR cleanup
AscendMemoryPlanning
AscendSyncInsert
AscendSyncInsertVS
```

Therefore, the authoritative simulation input is the module **after `OptimizeForTarget` and a
final `tir.transform.Simplify()`, but before `device_codegen`**. At this point software pipelines
have been expanded, local memory has been planned, and both explicit and automatically inserted
synchronization remain visible in TIR.

The implementation should first extract a reusable lowering function, tentatively:

```python
def lower_ascend_ir(
    func_or_mod: tir.PrimFunc | tvm.IRModule,
    target: str | Target,
    platform: str,
) -> tuple[tvm.IRModule, list[KernelParam]]:
    """Return the final, simplified pre-codegen Ascend TIR."""
```

Both native compilation and simulation must call this function so that their pass sequences
cannot drift. A diagnostic option may retain the post-`LowerAndLegalize` module, but it must not
be used for authoritative timing or synchronization simulation.

The bridge consumes TIR intrinsics and `call_extern` operations. It does not parse emitted C++.
Unknown operations must produce an error containing the TIR operation name, source span when
available, platform, target model, and the current lane/pipe; silently skipping an operation is
forbidden.

## 4. Public API and JIT integration

Simulation is an execution backend, not a compilation target. Keeping `target="pto"` or
`target="ascendc"` preserves the same target-dependent passes and operation contracts used by
native compilation.

The proposed API is:

```python
from tilelang.simulator import SimulatorConfig

kernel = tilelang.compile(
    func,
    out_idx=[-1],
    target="pto",
    platform="A2",
    simulator=True,
    sim_config=SimulatorConfig(
        trace_path="trace.json",
        hazard_check="error",
    ),
)
```

The same options should be accepted by `@tilelang.jit`. `platform` must resolve to `A2` or `A3`;
all other platforms fail before simulation begins.

`SimulatorKernelAdapter` must follow `BaseKernelAdapter` calling conventions:

- accept CPU `torch.Tensor` values and scalar dynamic-shape arguments;
- preserve `out_idx`, `workspace_idx`, and `auto_gm_indices` behavior;
- allocate automatic outputs and GM workspace using the same shape contract as the native
  adapter;
- preserve single-output and multiple-output return conventions;
- expose `get_simulator_ir()`, `last_trace`, and `last_stats`;
- skip C++ code generation, Bisheng compilation, Cython/ctypes host wrappers, `torch_npu`, and
  native binary caching.

Simulation configuration belongs in its cache key if a simulator kernel is cached. Native
kernel cache entries and simulator entries must never alias. The existing profiler and autotuner
must reject the simulator adapter until simulated-cycle consumption is explicitly designed and
validated.

## 5. Proposed package layout

```text
tilelang/simulator/
  __init__.py
  config.py                 # public SimulatorConfig and validation
  errors.py                 # unsupported op, bounds, hazard, deadlock, timeout
  program.py                # backend-neutral simulation IR
  bridge.py                 # final Ascend TIR -> simulation IR
  expression.py             # scalar TIR expression evaluator
  memory.py                 # GM and per-core local memory objects/views
  runtime.py                # grid/core/lane execution and functional state
  scheduler.py              # discrete-event scheduler and resource queues
  sync.py                   # flags, cross flags, barriers, ownership transfer
  hazard.py                 # address-range visibility and lifetime checks
  trace.py                  # Perfetto/Chrome JSON and summary statistics
  ops/
    copy.py
    vector.py
    reduce.py
    cube.py
    data_movement.py
    complex.py
  timing/
    model.py
    a2.json
    a3.json

tilelang/jit/adapter/
  simulator.py

testing/python/simulator/
  test_lowering_boundary.py
  test_bridge.py
  test_jit_adapter.py
  test_memory.py
  test_copy.py
  test_vector.py
  test_reduce.py
  test_cube.py
  test_sync.py
  test_pipeline.py
  test_trace.py
  test_timing.py
  test_complex_ops.py
  test_unsupported.py
```

## 6. Simulation IR

The bridge normalizes the broad TIR surface into a small, versioned simulation IR. Suggested
top-level objects are:

```python
@dataclass(frozen=True)
class KernelProgram:
    platform: Literal["A2", "A3"]
    target_model: Literal["pto", "ascendc"]
    params: tuple[SimParam, ...]
    core_program: CoreProgram
    metadata: Mapping[str, object]

@dataclass(frozen=True)
class SimTask:
    task_id: int
    op: str
    lane: Literal["cube", "vector"]
    pipe: str
    reads: tuple[MemoryRegion, ...]
    writes: tuple[MemoryRegion, ...]
    attrs: Mapping[str, object]
    dependencies: tuple[int, ...]
    source: SourceLocation | None
```

The IR must also represent allocation, views, scalar assignments, `For`, `IfThenElse`, and sync
operations. Expressions need integer, unsigned, float, and boolean semantics plus common TIR
operators such as min/max, floor division, modulo, casts, and select. Dynamic values are bound
from invocation arguments.

The bridge should use an explicit operation registry. Each entry defines:

- accepted TIR op names and signatures;
- lane and pipe classification;
- functional implementation;
- read/write region derivation;
- legality checks by dtype, scope, shape, alignment, and platform;
- timing-model key and trace metadata.

A bridge schema version must be included in serialized diagnostics and trace metadata so changes
to lowering or intrinsic signatures cannot be mistaken for timing changes.

## 7. A2/A3 execution model

The runtime models a logical core ID (`cid`) with Cube and Vector work selected from the lowered
program. The function attribute `npu_cv_ratio` and lowered `cid`/`vid` expressions are the source
of truth for whether a kernel uses a 1:1 or 1:2 Cube-to-Vector mapping. The runtime must not infer
lane membership solely from a fixed physical-core count.

The initial resource model is:

```text
Cube lane:   MTE2 -> MTE1 -> M -> FIX
Vector lane: MTE2 -> V -> MTE3
Control:     S and synchronization resources
```

Each pipe is an ordered queue. Different pipes can overlap when data dependencies, visibility,
flags, barriers, and resource constraints permit it. One functional operation may expand into
multiple scheduled tasks when the lowered contract exposes distinct movement and compute stages.

The runtime should use deterministic scheduling and deterministic tie-breaking. Re-running the
same program, inputs, configuration, and timing-profile revision must produce byte-equivalent
functional outputs, event ordering, statistics, and trace ordering.

## 8. Memory and correctness model

The functional runtime models:

- GM parameters and GM workspace;
- `shared.l1` (L1);
- L0A, L0B, and `wmma.accumulator` (L0C);
- `shared.ub` (UB);
- `shared.bt` (BT/bias table);
- scalar/local state required by lowered control flow.

Every buffer has a byte-addressed backing store plus dtype, shape, strides, physical-layout
metadata, owning core/lane, allocation lifetime, and visibility state. Logical tensor operations
may use PyTorch or NumPy for efficient numerical evaluation, but all accesses must first resolve
to physical byte ranges. This prevents logical tensor evaluation from hiding overlap, stride,
layout, or out-of-bounds bugs.

Uninitialized GM outputs, workspace, and local allocations are poisoned. Reads of poison are
reported independently of whether the resulting floating-point value happens to be NaN. An
explicit configuration option may seed an output from a caller-provided tensor for valid
read-modify-write and atomic cases; it is disabled by default.

Checks are independently configurable as `off`, `warn`, or `error` where appropriate:

- out-of-bounds and misaligned access;
- local-memory capacity and overlapping live allocations;
- uninitialized reads and incomplete output writes;
- read-before-visible, write-after-read, and write-after-write hazards;
- illegal memory paths and cross-lane ownership transfers;
- invalid dtype, shape, layout, or instruction constraints;
- unmatched flags, event reuse, barrier participation, deadlock, and timeout.

Physical layout support is staged. Early operations may use an explicitly documented canonical
layout, but NZ/ZN/fractal conversion must be implemented before the corresponding GEMM, MMA,
fixpipe, transpose, or convolution feature is marked complete.

## 9. Synchronization and software pipelines

The simulator must interpret both user-authored synchronization and the automatic intrinsics
left by `AscendSyncInsert` and `AscendSyncInsertVS`:

- `set_flag` / `wait_flag`;
- automatic set/wait flags;
- `set_cross_flag` / `wait_cross_flag`;
- automatic cross flags;
- `pipe_barrier`, including `ALL`;
- synchronization produced by `CrossCorePipeline` and `CombineCV`.

An asynchronous write progresses through `issued`, `in_flight`, `completed`, and `visible`
states. Completion frees an execution resource; visibility permits a dependent reader. These
states must remain distinct because a flag or barrier can alter visibility without changing the
functional result.

`T.Pipelined` is not interpreted as a high-level construct in authoritative simulation. The
`PipelinePlanning` and `InjectSoftwarePipeline` passes have already expanded it. The simulator
executes the resulting prologue, steady state, epilogue, stage-index expressions, and ring-buffer
reuse. Tests must include enough iterations to wrap every ring; one-tile smoke tests are not
sufficient pipeline validation.

Deadlock detection should report blocked tasks, outstanding producers, awaited flag/event IDs,
memory regions, core/lane/pipe, and source locations instead of only reporting a timeout.

## 10. Phased operation coverage

No row in this table describes current simulator support. It is the required implementation
roadmap.

| Phase | Required operation coverage | Required validation |
|---|---|---|
| P0: lowering and runtime skeleton | scalar expressions, `For`, `IfThenElse`, allocation/views, `cid`/`vid`, GM and local-memory skeleton, unsupported-op audit | final pre-codegen IR snapshot tests; deterministic bridge; unsupported intrinsic fails closed |
| P1: vector MVP | legal `T.copy` paths needed by GM-L1-L0 and GM-UB-GM flows; fill; cast; basic vector add/sub/mul/div, min/max, abs, exp, relu, silu; basic sum/max/min reduction; barriers and local flags | identity, vector arithmetic, tail, dynamic shape, poison, bounds, capacity, flag and barrier tests; Perfetto trace schema |
| P2: cube MVP | `gemm_v0`; explicit L1-to-L0A/L0B copy; `mma`; L0C-to-GM fixpipe; init versus accumulate; transpose operands; core NZ/ZN/fractal layouts | PyTorch golden across M/N/K tails, multi-K accumulation, multi-core partitioning, and pipeline overlap |
| P3: complete movement and arithmetic | every legal A2/A3 copy direction including UB-to-L1 and L0C-to-UB virtual CV paths through GM workspace; full supported vector families; broadcast, compare/select; complete reductions; standalone transpose | operation-by-operation dtype/scope/alignment matrix; workspace and CV handoff; address-range hazard tests |
| P4: cube data formats | BT bias and bias-initialized MMA; fixpipe options; fp32-to-fp16/bfloat16 and supported integer quant/dequant paths; relu/fused output behavior | bit/ULP-aware golden tests, saturation/rounding boundaries, bias capacity, paired MMA/fixpipe protocol |
| P5: indexed and ordering ops | gather, gather-mask/gatherb variants used by the repository, sort/sort32, merge-sort where exposed, and TopK | duplicate values, stable/index semantics where specified, odd valid counts, dynamic valid length, scratch-buffer bounds |
| P6: convolution | `im2col`, explicit im2col-plus-MMA convolution, and higher-level convolution lowering present in the repository | stride/dilation/padding, C0/channel blocks, spatial tails, multi-C1 accumulation, A2/A3 hardware restrictions represented as legality checks |
| P7: global execution features | atomic operations, including output seeding and deterministic conflict serialization; Persistent kernel scheduling and residency/liveness model | cross-core atomic contention, initialization rules, persistent work distribution, termination and deadlock tests |
| P8: timing calibration and hardening | separate A2/A3 operation tables, bandwidth/latency/resource calibration, multi-core waves and contention refinements | microbenchmark corpus, held-out kernels, ranking correlation, error bands, profile-version regression tests |

The copy milestone means all **legal A2/A3 paths represented by the lowered IR**, not arbitrary
cross-level access. Illegal paths remain errors. Operation support is complete only when its
functional semantics, physical address mapping, legality checks, hazards, timing key, trace
metadata, and positive/negative tests all exist for each advertised dtype and variant.

PTO support is the first validation target because it is the active C220 path for this project.
The bridge should remain codegen-independent; AscendC target parity is required before declaring
the A2/A3 simulator generally complete. Target-specific intrinsic forms may use separate bridge
adapters that lower to the same simulation IR.

## 11. Discrete-event timing model

Functional execution and timing scheduling share the same dependency graph but maintain separate
state. For a task:

```text
start = max(
    pipe_available,
    operand_visible,
    explicit_dependency_ready,
    flag_or_barrier_release,
    modeled_resource_available,
)
end = start + timing_profile.estimate(task)
```

Initial models should be simple and inspectable:

- copy: startup latency plus bytes/effective bandwidth, alignment, and transaction penalties;
- vector: repeat count divided by issue width plus operation latency;
- reduction/sort/TopK: parameterized data-size and stage models;
- GEMM/MMA: issued instruction/tile count and dtype-specific throughput plus startup;
- barrier/flag/atomic: explicit latency and participant/resource rules;
- kernel: resource-limited waves based on core assignment and modeled local-memory usage.

`a2.json` and `a3.json` contain only measured or deliberately provisional values with provenance:

```json
{
  "schema_version": 1,
  "platform": "A3",
  "profile_revision": "unreleased",
  "calibration": "uncalibrated",
  "source": "placeholder",
  "parameters": {}
}
```

If an A3 profile temporarily derives from A2, trace and statistics must say
`"calibration": "a2-derived"`; they must not imply A3 measurement. Every calibrated parameter
should link to a repository microbenchmark, device/compiler version, measurement method, and
sample date. Profile changes require regression results on held-out kernels.

The initial acceptance goal is explainability and correct relative ordering for selected schedule
variants, not absolute-cycle accuracy. Absolute timing error and ranking correlation must be
reported separately.

## 12. Trace and statistics

The primary artifact is Chrome Trace Event Format, directly loadable by Perfetto. Recommended
mapping:

- process (`pid`): simulated logical/physical core group;
- thread (`tid`): `cid/lane/pipe`, for example `cid3/cube/M` or `cid3/vec1/V`;
- category: `copy`, `vector`, `reduce`, `cube`, `sync`, `wait`, `atomic`, or `control`;
- complete events (`ph: "X"`): scheduled operations with simulated start and duration;
- flow events: producer-to-consumer dependencies and flag handoffs;
- counters: outstanding operations, queue depth, live local-memory bytes, and active cores.

Every operation event should include operation name, dtype, logical and physical region, bytes or
tile shape, pipeline stage/ring slot, source location, timing-model key, and profile revision.
Wait events should identify the exact dependency that blocked progress.

`last_stats` and an optional JSON summary should report at least:

- makespan in simulated cycles;
- busy and idle cycles per pipe;
- wait cycles by reason;
- Cube/Vector and copy/compute overlap ratios;
- per-core completion cycles and load imbalance;
- bytes moved by memory path;
- operation counts;
- peak local-memory use;
- hazard/warning counts;
- timing-profile and simulator schema revisions.

Trace comparison tooling should compare makespan, critical path, utilization, waits, and overlap,
not sum durations from concurrently active pipes.

## 13. Test strategy and acceptance gates

All simulator tests must run on CPU-only hosts without CANN, `torch_npu`, Bisheng, or NPU
hardware. Native codegen tests remain separate.

Each phase has four test layers:

1. **Bridge unit tests** over hand-built and compiler-produced final TIR.
2. **Functional unit tests** for dtype, shape, scope, layout, tails, dynamic values, and errors.
3. **Scheduler tests** with synthetic tasks for dependencies, overlap, barriers, flags, deadlock,
   and deterministic ordering.
4. **End-to-end kernel tests** against PyTorch/NumPy golden outputs and invariant-based trace
   assertions.

The initial end-to-end ladder is:

1. GM to UB to GM identity.
2. Vector add with aligned and non-divisible tails.
3. GM to L1, `gemm_v0`, L0C to GM.
4. Explicit L1 to L0A/L0B, multiple `mma` accumulations, paired fixpipe.
5. Explicit and automatic set/wait flags.
6. Cube-to-GM-workspace-to-Vector handoff with cross flags.
7. Two-stage and three-stage pipelines with ring wrap.
8. Multi-core uneven work distribution.

Negative tests are mandatory for unsupported shmem, unknown intrinsics, illegal memory paths,
capacity overflow, misalignment, out-of-bounds access, poison reads, incomplete writes, bad flag
pairs, ring reuse, deadlock, and timeout.

An operation cannot move from experimental to supported until:

- its contract is traced to repository source and tests;
- the bridge fails closed for unsupported variants;
- functional positive and negative tests pass on CPU;
- its trace identifies lane, pipe, regions, and timing key;
- A2 and A3 differences are either tested or explicitly marked uncalibrated;
- documentation lists remaining semantic or timing limitations.

## 14. Delivery sequence

The recommended development sequence follows the phase table:

1. Refactor the shared pre-codegen lowering boundary and add IR snapshots.
2. Add the simulator adapter, configuration, strict bridge, memory skeleton, and CPU invocation.
3. Deliver P1 functional vector simulation and valid Perfetto traces.
4. Deliver P2 Cube/GEMM/MMA/fixpipe simulation and pipeline scheduling.
5. Expand movement, vector, reduction, layout, bias, and quantization coverage.
6. Add indexed/order operations and convolution/im2col.
7. Add atomic and Persistent execution models.
8. Calibrate A2 and A3 independently, publish validation error, and harden regression tooling.

Each pull request should update a machine-readable operation coverage table and include the exact
unsupported variants. Until the final acceptance gate is reached, user-facing output must state
that the simulator is experimental and that hardware validation remains authoritative.
