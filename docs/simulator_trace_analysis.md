# Simulator trace analysis

`scripts/analyze_sim_trace.py` reads complete Chrome trace JSON from TileLang or
EasyASC. It needs only Python's standard library; it does not load TileLang, TVM,
NumPy, CANN, or a device runtime.

```bash
python scripts/analyze_sim_trace.py trace_sim_cycles.json --top 12
python scripts/analyze_sim_trace.py trace_sim_cycles.json --json > analysis.json
python testing/python/simulator/test_trace_analysis.py
```

The script adapts the user-provided Huawei EasyASC utilization analyzer. Its
original copyright and CANN Open Software License v2.0 notice are retained;
the license text is in `scripts/licenses/CANN_OPEN_SOFTWARE_LICENSE_V2.txt`.

## Input and units

The reader supports EasyASC's `core_idx/lane_name/pipe_name` fields, native
TileLang `core-N` and `lane/pipe` tracks, and normalized numeric Chrome IDs with
`original_pid/original_tid` or thread-name metadata. Unknown tracks fail instead
of silently merging all work into `runtime.operation`.

TileLang's `cycle_begin/cycle_end` coordinates take precedence over normalized
display timestamps. EasyASC `time_domain=cycle` also reports cycles. Untagged
Chrome numeric timestamps are microseconds, regardless of `displayTimeUnit`.
Mixed time domains fail. Simulator cycles are not hardware time; no clock-rate
conversion is performed. Legacy JSON field names ending in `_cycles` use the
reported `time_unit` throughout.

## Utilization and overlap

Active time is the **union** of intervals on each physical track. Overlapping
events on the same track produce a warning rather than utilization above 100%.
`active/span` uses that track's first-to-last-event span; `active/makespan` uses
the entire input trace. Scalar tracks and synchronization/stall events are
excluded from active utilization by default, but still contribute to the full
trace time boundary. `--include-s`, `--include-sync`, and `--include-stall` retain
their original utilization meanings.

Overlap always considers useful work, not synchronization markers or waits.
For each core separately, the tool intersects the interval unions for M-MTE1,
M-FIX, and M-V. It reports intersection length divided by active M, active peer,
and full makespan. Vector lanes are reported individually; `M-V(any)` intersects
M with the **union** of Vector lanes, never their sum. It never pairs different
cores. Missing tracks are omitted rather than invented as zero-utilization
resources.

## Wait diagnostics

Use the **complete trace**, even when the displayed timeline hides set/wait and
S. The analyzer resolves explicit `wait -> blocked_task -> sync_producers` links
and reports flag ID, source/destination pipe, receiver, and producer task IDs.
Memory-conflict arrows are not treated as synchronization.
Task IDs must be globally unique within the trace; duplicate IDs fail rather
than attributing a wait to an unrelated core's operation.

The report contains leading wait-channel groups, a separate cross-core list,
and longest individual wait windows. Cross-core waits on S remain in this
analysis when S is hidden from the utilization table.

- `raw_sum_window_cycles`: sum before merging potentially overlapping windows.
- `window_union_cycles`: union for one receiver/channel group.
- `uncovered_window_cycles`: union minus receiver active intervals. Local waits
  subtract that pipe's work; cross-core waits subtract all compute/copy work on
  the receiving lane, because already-issued work may continue.

These are **diagnostic modeled windows**, not exclusive hardware stalls or
critical-path attribution. Do not sum different groups or compare their sum
with kernel latency. Missing producer records stay unresolved. A trace without
wait records reports `unavailable`, not "zero waiting"; filtered traces cannot
establish that the original execution did not wait.

`--json` emits plain JSON to stdout and errors to stderr. Output preserves the
source scope/model metadata: a core-0 projection is not a full-grid validation,
and a relative timing model is not a measured performance profile.

## Source-fragment readiness

`fragment_readiness` automatically checks `copy_gm_to_ub` reads against recorded
GM writes (`dst` and `dst_regions`), without using `memory_dependencies` as
synchronization. It intersects exact byte intervals from buffer name, byte
offset, source dtype, shape and byte strides. Strided row gaps are not treated
as written bytes. For each source byte it selects the latest completed writer,
so old ring-slot contents and partially overwritten versions are not reused.

The report distinguishes whole-source readiness from earlier partial readiness.
Each milestone includes newly ready bytes, cumulative coverage, producer task
IDs and byte intervals (at most 64 shown, with an explicit truncation flag).
Candidates are ordered by consumer start time; `--top` limits displayed
candidates, not the total count. `whole_copy_start_lag_cycles` is the gap after
**all** source bytes are ready. `first_fragment_lead_cycles` is the interval
before the existing copy starts during which its first fragment is ready;
it is **not** a predicted latency reduction.

This is a source-only opportunity detector, not a safe rescheduling proof. A
candidate requires splitting the copy, per-fragment synchronization, checking
destination reuse and pipe/control constraints, and preserving full-row
dependencies of downstream reductions such as softmax. No automatic kernel
rewrite or performance gain is claimed.

Coverage is limited to concrete dense 1D or row-strided 2D regions of named GM
buffers, with at most 4096 rows. Unknown layouts, symbolic offsets, padded reads
whose valid extent differs from the source shape, and unknown writer regions
are not optimistically analyzed. Read/write overlap or concurrently conflicting
writers suppress candidates. External inputs or missing coverage produce
`incomplete_source`, not an uninitialized-memory verdict. Missing writes and
physical aliases under different buffer names cannot be inferred from the
trace; status counts expose unsupported and incomplete cases.

### Case 4 / 42 check (2026-10-20)

Both complete restored-model core-0 traces automatically identify
`c0-vector0-82` as the first candidate: it reads 16 x 512 FP32 values from `ws_s`
at cycle 5448, while four FIX writes produce successive 128-column bands:

| Producer | Ready cycle | Newly ready bytes | Cumulative coverage |
| --- | ---: | ---: | ---: |
| `c0-cube-31` | 2330 | 8192 | 25% |
| `c0-cube-47` | 3368 | 8192 | 50% |
| `c0-cube-63` | 4406 | 8192 | 75% |
| `c0-cube-75` | 5443 | 8192 | 100% |

Thus the whole-copy lag is only 5 model cycles; the first fragment is ready
3118 cycles before the existing copy starts. Prefetching the first 128 columns
is a protocol/splitting candidate, not permission to advance the full 512-column
copy. The complete traces contain 320 / 624 candidate reads respectively
(including separate Vector lanes and repeated iterations, not unique code sites).
No synchronization, scheduler or timing-model behavior is changed by analysis.

In the Docker experiment directory
`/home/wenzhongzhen/benchmark/fia_tl_trace_opt_20261019`, inputs are
`best_restored_pto_20261020_case{4,42}/trace_sim_cycles.json`; saved reports are
`fragment_readiness_case{4,42}.json`. Reproduce with this repository's script:

```bash
python scripts/analyze_sim_trace.py "$TRACE_ROOT/best_restored_pto_20261020_case4/trace_sim_cycles.json" --json
python scripts/analyze_sim_trace.py "$TRACE_ROOT/best_restored_pto_20261020_case42/trace_sim_cycles.json" --json
```

Set `TRACE_ROOT` to that experiment directory. Timings above are uncalibrated
relative model cycles, not NPU measurements.
