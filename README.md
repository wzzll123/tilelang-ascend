# Agent-Evolving Compiler

An agent-driven compiler laboratory for Huawei Ascend A2/A3.

Coding agents start from real kernels and observable failures, inspect TIR and generated code, reproduce behavior in a CPU-only functional simulator, add focused regressions, implement compiler fixes, and submit each verified improvement as an upstream issue and pull request.

```text
real kernel or failure
          ↓
TIR / generated-code inspection
          ↓
A2/A3 functional simulation and trace
          ↓
minimal regression
          ↓
compiler, codegen, or runtime fix
          ↓
NPU validation
          ↓
upstream issue and pull request
```

## Why this repository exists

Ascend kernel debugging normally crosses several layers: DSL, TIR passes, memory planning, automatic synchronization, C++ or PTO code generation, BiSheng, and hardware execution. A failure may appear as a compile error, wrong result, memory hazard, deadlock, or severe performance regression.

This repository turns those failures into a continuous compiler-evolution loop. Every change should provide:

- a concrete failure or missing capability;
- a minimal and executable regression;
- a narrowly scoped implementation;
- simulator or NPU evidence;
- an upstream issue and reviewable pull request.

## A2/A3 functional simulator

The simulator executes lowered TIR with NumPy-backed representations of Ascend memory and pipeline behavior. It is designed to validate functional semantics without invoking BiSheng.

Supported areas include:

- GM, UB, L1, L0A, L0B, L0C, and workspace memory;
- DMA copies, padding, physical strides, slices, and tail tiles;
- vector arithmetic, reductions, broadcasts, transpose, sort, TopK, and atomics;
- Cube MMA, accumulation, bias initialization, fixpipe, conversion, and ReLU;
- explicit and compiler-inserted pipeline synchronization;
- local flags, cross-core flags, barriers, persistent kernel waves, and active-core scheduling;
- FP16, BF16, FP32, and common integer data paths;
- Flash Attention and explicit im2col convolution execution.

The simulator is a functional and debugging model, not a cycle-accurate performance model. SHMEM is not supported in the current implementation, but remains a long-term roadmap item for distributed memory, communication, synchronization, and trace modeling.

See [roadmap.md](./roadmap.md) for the detailed coverage matrix, completed work, validation cases, known limitations, and remaining tasks.

## Trace and diagnostics

Trace records the relationship between execution, memory, and synchronization instead of only reporting the final mismatch.

It can expose:

- RAW, WAR, and WAW memory hazards;
- unmatched or incorrectly ordered set/wait flags;
- local and cross-core synchronization dependencies;
- pipeline ownership and dependency chains;
- synchronization deadlocks and the participating operations;
- physical memory ranges and exact strided overlap;
- active-core utilization and persistent-wave scheduling;
- the operation that first diverges from expected semantics.

This makes the simulator useful both for compiler development and for debugging kernels whose hardware execution merely returns an incorrect result.

## Extended compiler capabilities

- Native FP32 GEMM on the AscendC backend through the correct Catlass v2 L1 → L0A/L0B load path.
- AscendC `T.tile.im2col` for NC1HWC0 L1 → L0A convolution tile extraction.
- End-to-end im2col → MMA → L0C → fixpipe → GM convolution simulation.
- BF16 block transpose through the width-compatible hardware path.
- Dynamic-tail and physical-layout handling throughout the A2/A3 memory hierarchy.
- Functional execution of Flash Attention, GEMM, convolution, reduction, atomic, sorting, and persistent-kernel paths.

## Representative upstream work

### New capabilities

- [FP32 GEMM on AscendC](https://github.com/tile-ai/tilelang-ascend/pull/1760)
- [AscendC im2col tile extraction](https://github.com/tile-ai/tilelang-ascend/pull/1759)
- [BF16 hardware block transpose](https://github.com/tile-ai/tilelang-ascend/pull/1772)

### Compiler and code-generation fixes

- [Correct physical row widths for 2D subregion copies](https://github.com/tile-ai/tilelang-ascend/pull/1756)
- [Preserve floor semantics for negative division and modulo](https://github.com/tile-ai/tilelang-ascend/pull/1757)
- [Resolve memory-planning kills at scope exits](https://github.com/tile-ai/tilelang-ascend/pull/1755)
- [Reject copies with more than two active source dimensions](https://github.com/tile-ai/tilelang-ascend/pull/1754)
- [Lower scalar BF16-to-FP32 casts through AscendC](https://github.com/tile-ai/tilelang-ascend/pull/1758)
- [Flatten multidimensional scalar-buffer indices correctly](https://github.com/tile-ai/tilelang-ascend/pull/1761)
- [Clear partial GM-to-L1 copies at reused ring-slot bases](https://github.com/tile-ai/tilelang-ascend/pull/1770)

Each contribution is paired with an issue that documents the observed failure, expected behavior, reproduction boundary, and validation plan.

## Development principles

1. Start from evidence, not an assumed backend behavior.
2. Reproduce the smallest failing semantic path.
3. Compare against Ascend documentation, installed CANN headers, upstream code, and real NPU results.
4. Keep fixes independent and reviewable.
5. Run the repository's Ruff and clang-format checks before pushing.
6. Do not claim hardware or dtype coverage that has not been verified.
7. Preserve simulator limitations explicitly instead of silently approximating unsupported behavior.

## Current focus

- Ascend A2/A3 and the AscendC backend;
- simulator fidelity for compiler-generated TIR;
- synchronization, memory-layout, tail-tile, Cube, fixpipe, and convolution correctness;
- converting real operator failures into upstream compiler improvements.

## Long-term roadmap

- SHMEM put/get and symmetric-memory semantics;
- cross-core and cross-device communication ordering;
- SHMEM-aware memory hazards, synchronization deadlocks, and trace flows;
- communication/computation overlap and topology-aware timing calibration;
- multi-device functional examples and differential validation against real A2/A3 systems.
