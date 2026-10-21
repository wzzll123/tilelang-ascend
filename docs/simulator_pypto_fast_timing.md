# PyPTO fast CostModel baseline

The A3 default and A2's explicitly labeled fallback use the CANN/PyPTO fast
CostModel at revision `49587f393680c6bc814b00e7ed4b99956a1aa979`. This imported
baseline is **not a hardware-calibrated prediction**. A2 now prefers the
[measured linear model](simulator_measured_linear_timing.md) within its measured
families and size domains; missing coverage is labeled `pypto-fallback` in task metadata.

The parameter table and numerical implementation are in
`tilelang/simulator/pypto_timing.py`. Sources are
`framework/src/cost_model/simulation/arch/PipeSimulatorFast.cpp` and
`framework/src/cost_model/simulation/arch/A2A3/PostSimulatorA2A3.h` in
<https://gitcode.com/cann/pypto>.

All 40 coefficient entries and the inherent-latency fallback table are retained.
For a table hit, let `B` be dtype width times the minimum product of the last two
dimensions across input/output operands. Let `C` be the minimum product of their
leading dimensions. The imported formula is:

```text
cycles = C * (truncate((a * B * parallelRatio * 1e-6 + b) * 2000) + 1)
```

The implementation reproduces upstream float32 intermediate arithmetic, dtype
ratios and integer truncation. Original `COPY_IN` operations add 1400 cycles;
original `COPY_OUT` operations add 300, once per operation, not per batch.
Unknown opcodes use the upstream repeat-count/inherent-latency fallback.
The numerical constants, including the factor 2000, are not device measurements
made by this project.

TileLang operation names and memory scopes are mapped to PyPTO opcodes. Scratch
regions are excluded from operands. Flattened MMA views use the bridge's logical
M/N/K dimensions to reconstruct operand shapes, then use the **same PyPTO
formula**, not the previous MMA-repeat estimate. One-dimensional views are
represented as `(1, N)` because upstream expects at least two dimensions.
Unresolved operand metadata uses the profile's declared fallback cost; nonpositive
upstream costs are clamped to one to satisfy the scheduler's positive-duration
contract. Explicit per-operation overrides remain supported.

This change imports costs only: it does not modify synchronization, race checking,
execution queues, or functional values. PyPTO's optional cache machine and PV
simulator are not included. Existing explicit `pto_fallback_timing_profile()`
remains a compatibility API but is no longer the default or mixed into the new
path. Synchronization instructions retain the existing one-cycle accounting.

`testing/python/simulator/test_pypto_timing.py` freezes reference values obtained
from compiled upstream C++, including every coefficient entry. These tests check
implementation parity, not hardware accuracy. The imported module retains CANN
Open Software License Agreement Version 2.0; see
`licenses/CANN-Open-Software-License-2.0.txt`.

## Docker regression environment

The isolated test checkout is
`/home/wenzhongzhen/tilelang-ascend-wt/pypto-fast-cost-20261021` in NPU3's
`wzz_cann` container, based on `f8b502d5`. It reuses the existing native build;
only Python and tests change. The main checkout and its dirty TVM submodule
are not modified. Reproduce inside the container with:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/wenzhongzhen/tilelang-ascend-wt/pypto-fast-cost-20261021
export PYTHONPATH="$PWD:$PWD/3rdparty/tvm/python"
export TVM_LIBRARY_PATH="$PWD/build/tvm:$PWD/build"
export TILELANG_LIBRARY_PATH="$PWD/build"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
/home/wenzhongzhen/kernel-tool/fla-kda-profile-env/bin/python -m pytest \
  testing/python/simulator -q --disable-warnings --maxfail=3
```

An additional standalone comparison against compiled upstream C++ covered
5,544 combinations of opcode, dtype, shape and batch count, with zero mismatches.
It used `-ffp-contract=off` and checked both coefficient and fallback paths.
This is formula parity, not an NPU latency calibration.

Docker full-suite result (2026-10-21): **773 passed, 3 subtests passed** in
72.88 seconds. The only obsolete timing assertion found in the first run
hard-coded six cycles for vector-add; it now checks independent-pipe scheduling
using the selected task durations, while retaining the numeric-output check.

## UnitFlag geometry correction for expert-v3

UnitFlag validation now tracks physical 512-byte L0C blocks instead of requiring
one MMA width to equal one FIX width. The UnitFlag feature described in
asc-devkit's `mmad_compute_key_features/UnitFlag.md` gives the corresponding
readable/writable-bit protocol: MMA 2 retains writable state, MMA 3 publishes
readable state, and FIX 3 consumes/resets it. Two adjacent MMA results can feed
one combined FIX, and a wide MMA can feed separate disjoint FIX reads. Missing
producers, repeated consumption, overwriting unread data and incomplete final
drain remain errors. Physical aliases share the same block state.

Ready and reuse edges are labeled `unit_flag_dependencies` and
`hardware_dependencies`. The static checker accepts these validated UnitFlag
edges; ordinary inferred memory edges still do not create execution ordering.
Tests are in `testing/python/simulator/test_unit_flag_blocks.py`.

This is a conservative **tile-atomic** scheduling abstraction. It does not yet
allow FIX to begin while its producing MMA completes other 512-byte blocks, so
it can overestimate time. FIX mode 2 remains explicitly unsupported by the ABI
bridge. This is not complete fine-grained UnitFlag performance modeling.

## First frozen-R4 prediction

On 2026-10-21 the frozen R4 source (`108864cf04db6eb466cfd64be389ba5a781c4278a59c870b357503b5a2381264`)
passed real sync-only execution for FP16 noncausal MHA `[4, 2, 3072, 128]`.
The diagnostic expands core 0 only **after** lowering the original 20-core
kernel, preserving its task-distribution arithmetic. Core 0 has the largest
number of equal-size query tasks. This is an independent-core prediction,
not a simulation of shared bandwidth across 20 cores.

The predicted makespan is 867,033 model cycles, or 433.5165 microseconds using
the formula's nominal 2,000 cycles/microsecond scale. The frozen NPU6 msprof
observation is 270.169 microseconds: signed error **+60.46%**. No new NPU
timing or numerical correctness experiment was performed for this comparison.

Construction took 54.65 seconds and the complete run 60.69 seconds. Execution
kept its 60-second timeout. Replacing per-task subtraction of the entire
completed-key view with direct dependency membership probes removed a
quadratic scheduler hotspot without changing dependency semantics.

The cost audit contains 6,505 coefficient-table calls and 5,960 fallback calls
(excluding synchronization). Nine fallback opcodes remain: AXPY,
BRCB_EXPERIMENT, COPY_UB_TO_UB, FILL, MAX, MUL_ADD_DST, and the three
ROW_EXPAND_*_EXPERIMENT operations. AUTO_BARRIER uses the declared unit
fallback; compiler-managed Vector mask-setting instruction timing is not
modeled by the semantic projection. These results are not full cost coverage
or a ten-percent-accuracy claim.

Artifacts in the Docker are under
`/home/wenzhongzhen/benchmark/experiments/fa-expert-v3/results/pypto-cost-prediction-20261021/`;
`case0_core0_fixed.json` records the successful run. The original all-core
construction timed out at 300 seconds and produced no valid prediction.
