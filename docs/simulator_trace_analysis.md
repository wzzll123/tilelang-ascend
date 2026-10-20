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
