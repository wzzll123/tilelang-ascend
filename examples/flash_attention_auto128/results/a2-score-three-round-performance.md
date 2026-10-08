# Score-workspace pipeline: complete three-round A2 comparison

**Performance FAIL: geometric mean v06/auto = 0.104285, required >=0.90.**
Precision/stability for the same frozen automatic build passed14 fixed cases
with20 finite/CANN-aligned comparisons each. This does not complete the goal.

Compiler build identity `2c83b27dc1763f908f7b1a2412ad6ad41532186f`, physical A2
card6. Independent worktree/native bindings, fresh cache and source fingerprints
are recorded in `a2-score-overlap-first-hardware.md`. No simulator was used.
The frozen v06 reference is the same e471/native/dependency snapshot used in
the previous input-prefetch campaign. The original designer profiler ran
adjacent same-case v06 then auto, three complete rounds, warmup3/repeat20,
no boost or SOL. Neither compiler nor ladder changed during measurement.

All84 original JSONs passed identity, exact case set, warmup/repeat and positive
finite-duration checks. The local report reader reproduces the server result:

```bash
python examples/flash_attention_auto128/summarize_perf.py \
  examples/flash_attention_auto128/results --prefix fa128-score-2c83b27d
```

| case | v06 median us | auto median us | v06/auto |
|---:|---:|---:|---:|
|0|7.96|10.19|0.7812|
|2|39.25|393.39|0.0998|
|6|66.32|781.98|0.0848|
|7|104.46|1494.17|0.0699|
|10|37.14|242.00|0.1535|
|12|395.27|5944.78|0.0665|
|13|702.25|11611.51|0.0605|
|14|1338.34|22930.36|0.0584|
|15|704.03|11606.85|0.0607|
|16|196.55|2977.44|0.0660|
|36|536.81|6121.74|0.0877|
|37|322.30|3289.87|0.0980|
|41|599.69|5919.86|0.1013|
|42|10.91|35.17|0.3102|

Every case regresses. For case14, automatic rounds are
22933.36/22930.36/22930.02us; v06 rounds1339.85/1338.34/1333.35us.
All per-case three-round values remain in the unchanged original JSONs.

Original artifact archive: `fa128-score-2c83b27d-perf-rounds123.tar.gz`,
60891bytes, containing84 JSONs and336 original profiler CSVs at their original
relative paths. SHA256 matched Docker/host/local:
`c5ce1e62ed01612dcada5b42c243bea75c5546e791277e5c38e1eb8cc60b4845`.

The emitted two-stage overlap alone has not closed the gap. These are kernel
durations, not pipe-level measurements; they do not establish whether Vector
work, DMA, scalar dispatch or synchronization dominates. The next original
msprof case14 A/B collection is separate from these completed measurements.
Larger64x128 tiles and combined input/score composition are compiler-only
candidates, not hardware-tested improvements or explanations of this result.
A3 remains not real-device verified. Additional non-square/tail hardware
coverage and the full migration/performance acceptance remain open.
