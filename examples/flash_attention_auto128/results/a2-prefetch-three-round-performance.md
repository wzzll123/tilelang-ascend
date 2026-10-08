# A2 card6: frozen K/V-prefetch FA128, three-round comparison

**Performance gate: FAIL.** Geometric mean of `v06_us / auto_us`, using each
case's three-round median, is **0.107698**, below the required0.90. Every fixed
case is included; there is no selected-case acceptance claim.

The actual serial profiler wrapper completed with exit0. Each round alternated
v06→auto128 for the same case on physical A2 card6, warmup3 and repeat20.
It used the original designer `perf_tilelang.py`, `--no-boost --no-sol` and
`ASCEND_RT_VISIBLE_DEVICES=6`. No clocks were changed, other processes stopped,
or devices reset. `--no-boost` also skips the original combined cache-clear
helper, identically for both arms. Before starting, all14 auto cases had
passed20 finite-output/precision launches each using the original comparator.

| case | v06 median, us | auto median, us | v06 / auto |
| ---: | ---: | ---: | ---: |
| 0 | 8.50 | 9.60 | 0.8854 |
| 2 | 39.88 | 382.97 | 0.1041 |
| 6 | 67.42 | 760.58 | 0.0886 |
| 7 | 105.31 | 1469.30 | 0.0717 |
| 10 | 37.50 | 231.36 | 0.1621 |
| 12 | 393.49 | 5847.90 | 0.0673 |
| 13 | 711.33 | 11494.93 | 0.0619 |
| 14 | 1332.68 | 22799.81 | 0.0585 |
| 15 | 703.49 | 11498.55 | 0.0612 |
| 16 | 194.58 | 2931.18 | 0.0664 |
| 36 | 534.25 | 6023.30 | 0.0887 |
| 37 | 320.57 | 3197.46 | 0.1003 |
| 41 | 596.99 | 5862.94 | 0.1018 |
| 42 | 11.04 | 33.22 | 0.3323 |

The84 unmodified JSON reports remain under
`fa128-prefetch-1aeedc56-round{1,2,3}-{v06,auto128}/`. Every report's case ID,
warmup/repeat schedule and positive finite duration were checked. The archive
`fa128-prefetch-1aeedc56-perf-rounds123-20261009.tar.gz` additionally preserves
336 original profiler CSV files (four per run) at their original relative
paths. It was downloaded and independently SHA256 checked:
`707d6ea03a96d56cf633b23ce9b2aa8a71f48b9a40ace207662fd0153331450a`.

Reproduce the summary without renaming reports or changing statistics:

```bash
python examples/flash_attention_auto128/summarize_perf.py \
  examples/flash_attention_auto128/results --prefix fa128-prefetch-1aeedc56
```

## Frozen identities and interpretation

This measurement used Docker compiler commit`1aeedc56`, not the newer local
workspace-certificate commits. Its FA source remains32×64, FP16 GEMM input,
FP32 online-softmax/output accumulation, with certified two-slot K/V prefetch.
The wrapper pins the independent compiler/baseline packages and loaded native
libraries, prints bindings before every run, and preserves the existing
default model entry through explicit `FA_STEP=v06/auto128` selection.
See `../run_card6_prefetch_perf.sh`, `prefetch-native-cache-audit.csv`, the
original precision/stability JSON and `a2-prefetch-first-hardware.md` for
shape/dtype mapping and environment provenance.

| component | frozen identity |
| --- | --- |
| auto compiler | `1aeedc56`, `build-auto-schedule/libtilelang_module.so` |
| auto native SHA256 | `4ce6649be570c447d58f5fa1e5b236eb1a1473a6a9f88a010de4e4578eddd98b` |
| auto FA SHA256 | `bd5bbaa5aba60dc802129b071da0708fcea256fb5ab7c1182e3ec737136ac164` |
| v06 compiler | `e47171e32f6e64ff613cc6f398ec4c32a7c76d9f` plus frozen binary-mask dependency patch |
| v06 native SHA256 | `6c5740dca57791696c5165a01942d8163d4078cce5acd451b67499f095bb1c21` |
| v06 FA SHA256 | `b345fc6813b7ede3d710e4fb89dd45ee51a64674009987b2433c44bffe606f77` |
| TVM native SHA256 | `2df7999bbed69490640f5bc50166a5a650ac543bdea22514878e86ae31ccd36e` |
| original profiler SHA256 | `c54714b2875f19c9610a91e0dff4d84936aaa6ab1e99f3e400913be36cd6b70e` |

The repeated timings confirm a large, reproducible gap. K/V prefetch alone is
not sufficient; the wider C/V stage schedule is still absent from this build.
Kernel-duration CSV cannot independently attribute the gap to a particular
pipe or distinguish synchronization from tiling/vector-lowering overhead.
Do not claim that one bottleneck or a hypothetical overlap explains it without
instruction/pipe-level profiling. Neither this failed performance gate nor
the separate local native-input proofs completes the migration. A3 remains
not hardware verified.
