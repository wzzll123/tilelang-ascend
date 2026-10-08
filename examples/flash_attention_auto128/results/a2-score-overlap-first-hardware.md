# Cube-prefix workspace overlap: first A2 hardware result

Compiler identity: `2c83b27dc1763f908f7b1a2412ad6ad41532186f` on the
independent `auto-schedule-fa128` Docker worktree. The dirty Docker main
checkout and frozen v06 baseline were not modified. The deployment fast-forward
verified an empty native-source/submodule diff from frozen `1aeedc56`.

The reused native module SHA256 is
`4ce6649be570c447d58f5fa1e5b236eb1a1473a6a9f88a010de4e4578eddd98b`;
this is the existing build's identity, not a claim of a new native build.
TVM native SHA256 remains
`2df7999bbed69490640f5bc50166a5a650ac543bdea22514878e86ae31ccd36e`.
Python 3.12.13, torch 2.10.0+cpu/torch_npu, Z3 4.15.4 and CANN9.0.0.
Actual package/library bindings were printed and matched the independent
checkout and pinned shared TVM. Physical A2 card6 was healthy and idle.

Fresh cache: `/tmp/fa128-score-2c83b27d-cache.fim7S9`. Explicit task selector
`FA_STEP=auto128`, card selector `ASCEND_RT_VISIBLE_DEVICES=6`, blocking
launches. The ladder kernel was unchanged, SHA256
`bd5bbaa5aba60dc802129b071da0708fcea256fb5ab7c1182e3ec737136ac164`.
The original verifier SHA256 was
`fe881e92db75fb3a8e5f13f6e47fe5124af72c269ad117a19ddbdd231858c1d2`.

Case0 `[1,4,128,128]` FP16 passed a single original designer comparison and
finite-output check. Its original JSON is archived alongside this document.
The original comparator uses CPU FP64 golden plus original-dtype native;
the JSON's legacy atol/rtol metadata does not describe this CANN comparison.

The serial fourteen-case twenty-launch run then started with the tracked
`run_card6_score_precision.sh` environment wrapper. At this checkpoint,
cases0,42 and10 passed all20 comparisons with finite outputs. Case10 retained
the `True` causal input. Remaining cases are **not yet recorded as passed**;
the live execution must reach terminal evidence and reports must be read back.
No performance result, arbitrary-input stability or A3 hardware result is
claimed. This is direct hardware validation, without simulator execution.

Read-back of the new cache's native `kernel.cu` (cache key
`e046fc54eaf3bdac759e916177b90f0574d8b52873544ae68c679b61f2321c9a`)
also confirms the pipeline actually reached generated code: four epoch waves
for three data epochs, score GM offsets alternating two world-sized slots,
mode2 FIX ready notifications and next-stage Vector waits. The generated UB
allocations are `score: GetWithOffset<float>(1024,28800)` and
`score_recv: GetWithOffset<float>(1024,32896)`: adjacent, disjoint 4096-byte
regions. This source inspection is not a timing/overlap measurement.

The same cache's actual runtime parameter metadata identifies case42 Q shape
`[2,4,192,128]`, automatic GM parameter indices `[5,6,7]`, and workspace shapes
`[96,32,64]`, `[48,32,64]`, `[48,32,128]`. The first allocation therefore has
196608 float elements: exactly two 98304-element world slots used by the
generated offsets, not the old single-slot size. This read-only audit loaded
`params.pkl` with cloudpickle and `auto_gm_idx.pkl` with JSON, matching the
real KernelCache serialization despite the latter's filename extension.

Reproduce inside Docker after checking healthy idle card6:

```bash
TILELANG_CACHE_DIR=/tmp/fa128-score-2c83b27d-cache.fim7S9 \
    bash /tmp/fa128-score-2c83b27d-precision.sh
```

The wrapper intentionally refuses to overwrite prior case reports. For a new
run, use a fresh isolated cache and new report names; do not delete user caches
or restart an already-live validation loop. Stop on failure or timeout, and do
not reset devices or launch subsequent cases after a suspected hang.
