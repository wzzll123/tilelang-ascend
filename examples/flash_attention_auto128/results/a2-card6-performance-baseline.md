# A2 card6: first automatic FA128 performance baseline

This is a failed performance gate, not the completed automatic-pipeline migration.
Both implementations separately passed all fourteen original precision cases.
Those single precision runs do not establish repeated numerical stability or
explicit finite-output checking. Profiler repetitions do not compare outputs.

## Measurement identity

- Automatic compiler: `607b87c4`, independent `build-auto-schedule`.
- Frozen v06 compiler: `e47171e32f6e64ff613cc6f398ec4c32a7c76d9f` plus audited
  binary-mask dependency patch SHA256
  `2d17655a54500418ff4bb6c989553c785def3539fb6a4586cad5f05470ccd977`.
  These are distinct compiler snapshots, not a same-compiler comparison.
- Automatic native SHA256:
  `4ce6649be570c447d58f5fa1e5b236eb1a1473a6a9f88a010de4e4578eddd98b`.
- Reference native SHA256:
  `6c5740dca57791696c5165a01942d8163d4078cce5acd451b67499f095bb1c21`.
- Shared TVM commit `6dcf3f0d3169e693b3f5f0525556a40d23eaf703`, library SHA256
  `2df7999bbed69490640f5bc50166a5a650ac543bdea22514878e86ae31ccd36e`.
- NPU3 `wzz_cann`, physical A2 card6, toolkit9.0.0, Python3.12.13,
  isolated solver environment `/tmp/a2a3-auto-schedule-fa128-venv.JsRZee`.
- Original designer `perf_tilelang.py`, `--warmup 3 --repeat 20 --no-boost --no-sol`.
  Serial order: v06 then auto128 for each of three complete rounds. No frequency
  boost or device reset. Explicit `FA_STEP`; private implementation-specific caches.
- Frozen v06 source SHA256:
  `b345fc6813b7ede3d710e4fb89dd45ee51a64674009987b2433c44bffe606f77`.
- Auto source SHA256:
  `fbe8d1805e6865f6a66f89c7506ec948e9c83fa16d13579f38dfb9408e5c87d1`.

## Results

Each input number is the original harness's sum of per-kernel median durations,
not Python wall time. The following table uses the median across three rounds.
All raw round values and profiler directory references are retained in
`fa128-perf-607-v06e471-round{1,2,3}-{v06,auto128}/case_N.json`.

| Case | v06 us | auto us | v06 / auto |
|---:|---:|---:|---:|
| 0 | 8.41 | 9.32 | 0.9024 |
| 2 | 39.13 | 382.45 | 0.1023 |
| 6 | 65.19 | 760.40 | 0.0857 |
| 7 | 104.06 | 1471.05 | 0.0707 |
| 10 | 36.30 | 381.47 | 0.0952 |
| 12 | 398.01 | 5846.46 | 0.0681 |
| 13 | 703.27 | 11497.59 | 0.0612 |
| 14 | 1334.64 | 22801.42 | 0.0585 |
| 15 | 703.70 | 11503.59 | 0.0612 |
| 16 | 193.42 | 2928.40 | 0.0660 |
| 36 | 537.97 | 11469.99 | 0.0469 |
| 37 | 316.73 | 5833.10 | 0.0543 |
| 41 | 597.21 | 11399.69 | 0.0524 |
| 42 | 10.91 | 33.34 | 0.3272 |

Geometric mean: **0.090009**. Required >=0.90: **FAIL**.
No cases are excluded. `summarize_perf.py` reproduces the table and all three
round values and rejects missing cases, incorrect repeat settings or invalid
durations. All 84 reports have one corresponding `kernel_details.csv`, archived
separately in `fa128-three-rounds-kernel-details.tar.gz`.

## Interpretation and next compiler work

The current automatic kernel uses 32x64 tiles and single-version KV/workspace
storage; executable tasks retain lexical order. Causal loops still traverse
all KV tiles and mask irrelevant columns. Coordinated buffer versions,
participant-uniform causal trip counts and true stage-skew scheduling remain
unfinished. These are verified implementation limitations, not a claim that
one particular pipe accounts for the measured slowdown. Pipe-level attribution
still requires the planned msprof overlap analysis.

Implement these compiler capabilities and retest on actual hardware. Do not
replace this result with a hand-synchronized kernel, enable the old pipeline
silently, change precision thresholds, or report solver ordering as executed
software pipelining. A3 remains without hardware validation.
