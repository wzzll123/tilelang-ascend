#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See scripts/licenses/CANN_OPEN_SOFTWARE_LICENSE_V2.txt for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Analyze EasyASC/TileLang simulator traces without importing either runtime.

Derived from the user-provided EasyASC analyze_sim_trace.py (Huawei, 2026).
The original CANN Open Software License notice above applies to this file.
Overlap is an interval intersection, not a sum of parallel lane durations.
Wait windows are diagnostic model records, not exclusive hardware stall time.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Sequence

SYNC_OPS = frozenset([
    "bar_all",
    "event_set",
    "event_setall",
    "event_wait",
    "event_release",
    "cube_ready",
    "vec_ready",
    "wait_vec",
    "wait_cube",
    "intracore_allvec_ready",
    "intracore_allvec_wait",
    "allcube_ready",
    "allcube_wait",
    "allvec_ready",
    "allvec_wait",
    "setflag",
    "waitflag",
    "set_flag",
    "wait_flag",
    "auto_set_flag",
    "auto_wait_flag",
    "set_cross_flag",
    "wait_cross_flag",
    "auto_set_cross_flag",
    "auto_wait_cross_flag",
    "pipe_barrier",
    "barrier_all",
    "auto_barrier",
    "sync_all",
])

PIPE_ORDER = {
    "S": 0,
    "MTE2": 1,
    "MTE1": 2,
    "M": 3,
    "FIX": 4,
    "V": 5,
    "MTE3": 6,
}

LANE_ORDER = {
    "cube": 0,
    "vec": 1,
    "vec0": 1,
    "vec1": 2,
    "runtime": 3,
}


@dataclass(frozen=True)
class TraceEvent:
    core_idx: int
    lane_name: str
    pipe_name: str
    opname: str
    cat: str
    ts: float
    dur: float
    time_domain: str
    args: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def end(self) -> float:
        return self.ts + self.dur

    @property
    def track_key(self) -> tuple[int, str, str]:
        return self.core_idx, self.lane_name, self.pipe_name


@dataclass(frozen=True)
class TrackStats:
    core_idx: int
    lane_name: str
    pipe_name: str
    label: str
    event_count: int
    active_event_count: int
    active_cycles: float
    first_ts: float
    last_end: float
    makespan: float

    @property
    def span(self) -> float:
        return max(self.last_end - self.first_ts, 0.0)

    @property
    def active_over_span(self) -> float:
        if self.span <= 0.0:
            return 0.0
        return self.active_cycles / self.span

    @property
    def active_over_makespan(self) -> float:
        if self.makespan <= 0.0:
            return 0.0
        return self.active_cycles / self.makespan


@dataclass(frozen=True)
class GroupStats:
    core_idx: int | None
    label: str
    core_count: int
    track_count: int
    event_count: int
    active_event_count: int
    avg_active_cycles: float
    avg_span: float
    avg_active_over_span: float
    avg_active_over_makespan: float


def _as_float(value: Any, field: str, event_idx: int) -> float:
    try:
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("expected a finite nonnegative number")
        return number
    except (TypeError, ValueError) as exc:
        raise ValueError(f"event {event_idx} has invalid {field}: {value!r}") from exc


def _as_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def load_timed_events(path: str) -> tuple[list[TraceEvent], str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return parse_events(data)


def parse_events(data: dict[str, Any]) -> tuple[list[TraceEvent], str]:
    raw_events = data.get("traceEvents")
    if not isinstance(raw_events, list):
        raise ValueError("trace JSON does not contain a traceEvents list")

    names = {
        (e.get("pid"), e.get("tid")): e.get("args", {}).get("name", "")
        for e in raw_events
        if isinstance(e, dict) and e.get("name") == "thread_name"
    }
    metadata = data.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    events: list[TraceEvent] = []
    for idx, item in enumerate(raw_events):
        if not isinstance(item, dict) or item.get("ph") != "X":
            continue
        args = item.get("args", {})
        if not isinstance(args, dict):
            args = {}
        ts = _as_float(item.get("ts"), "ts", idx)
        dur = _as_float(item.get("dur", 0.0), "dur", idx)
        domain = str(args.get("time_domain", ""))
        if "cycle_begin" in args or "cycle_end" in args:
            ts = _as_float(args.get("cycle_begin"), "cycle_begin", idx)
            end = _as_float(args.get("cycle_end"), "cycle_end", idx)
            if end < ts:
                raise ValueError(f"event {idx} ends before it starts")
            dur, domain = end - ts, "cycle"
        elif domain == "cycle" or metadata.get("timestamp_unit") == "simulator_cycle":
            if domain != "cycle" and metadata.get("chrome_mapping"):
                # TileLang's normalized Chrome export maps cycle -> display ns.
                if "1 simulator cycle = 1 display ns" not in metadata["chrome_mapping"]:
                    raise ValueError("unknown simulator-cycle Chrome mapping")
                ts, dur = ts * 1000, dur * 1000
            domain = "cycle"
        else:
            domain = domain or "us"  # Chrome numbers are microseconds, regardless of displayTimeUnit.
        track = str(
            args.get("original_tid",
                     names.get((item.get("pid"), item.get("tid")), item.get("tid", ""))))
        lane, separator, pipe = track.rpartition("/")
        pipe_name = str(args.get("pipe_name", pipe if separator else item.get("cat", ""))).upper()
        lane_name = str(args.get("lane_name",
                                 lane if separator else "runtime")).replace("vector", "vec")
        core = _as_int(args.get("core_idx", args.get("core_id")), default=-1)
        if core < 0:
            match = re.fullmatch(r"core-(\d+)", str(args.get("original_pid", item.get("pid", ""))))
            if match:
                core = int(match.group(1))
        runtime_control = args.get(
            "lane_name") == "runtime" and pipe_name == "S" and "core_idx" in args
        if pipe_name not in PIPE_ORDER or ((core < 0 or lane_name == "runtime") and
                                           not runtime_control):
            raise ValueError(
                f"event {idx} has no supported core/lane/pipe identity; do not merge unknown tracks"
            )
        opname = str(args.get("opname", item.get("name", "")))
        events.append(
            TraceEvent(
                core_idx=core,
                lane_name=lane_name,
                pipe_name=pipe_name,
                opname=opname,
                cat=str(item.get("cat", args.get("cat", pipe_name))),
                ts=ts,
                dur=dur,
                time_domain=domain,
                args=dict(args),
            ))

    unit = _detect_unit(events, "us")
    if unit == "mixed trace units":
        raise ValueError("mixed time domains cannot share an overlap/makespan denominator")
    return events, unit


def _detect_unit(events: Sequence[TraceEvent], display_unit: str) -> str:
    domains = set(event.time_domain for event in events if event.time_domain)
    if domains == set(["cycle"]):
        return "cycles"
    if not domains:
        return display_unit
    if len(domains) == 1:
        return next(iter(domains))
    return "mixed trace units"


def event_track_included(event: TraceEvent, include_s: bool) -> bool:
    return include_s or event.pipe_name != "S"


def event_is_active(event: TraceEvent, include_sync: bool, include_stall: bool) -> bool:
    name = event.opname or ""
    cat = event.cat or ""
    if not include_stall and (cat.lower() in {"stall", "wait"} or name.startswith(
        ("stall:", "WAIT "))):
        return False
    return include_sync or (cat.lower() != "sync" and name.lower() not in SYNC_OPS)


def group_label(event: TraceEvent, group_by: str) -> str:
    if group_by == "lane-pipe":
        return event.lane_name + "." + event.pipe_name
    if event.pipe_name == "MTE2" and event.lane_name in ("cube", "vec0", "vec1"):
        side = "cube" if event.lane_name == "cube" else "vec"
        return side + "." + event.pipe_name
    return event.pipe_name


def compute_makespan(events: Sequence[TraceEvent]) -> float:
    if not events:
        return 0.0
    first_ts = min(event.ts for event in events)
    last_end = max(event.end for event in events)
    return max(last_end - first_ts, 0.0)


def compute_track_stats(
    events: Sequence[TraceEvent],
    makespan: float,
    group_by: str,
    include_s: bool,
    include_sync: bool,
    include_stall: bool,
) -> list[TrackStats]:
    by_track: defaultdict[tuple[int, str, str], list[TraceEvent]] = defaultdict(list)
    for event in events:
        if event_track_included(event, include_s):
            by_track[event.track_key].append(event)

    stats: list[TrackStats] = []
    for (core_idx, lane_name, pipe_name), track_events in by_track.items():
        first_ts = min(event.ts for event in track_events)
        last_end = max(event.end for event in track_events)
        active_events = [
            event for event in track_events
            if event_is_active(event, include_sync=include_sync, include_stall=include_stall)
        ]
        active_cycles = interval_length(
            merge_intervals([(event.ts, event.end) for event in active_events]))
        label = group_label(track_events[0], group_by)
        stats.append(
            TrackStats(
                core_idx=core_idx,
                lane_name=lane_name,
                pipe_name=pipe_name,
                label=label,
                event_count=len(track_events),
                active_event_count=len(active_events),
                active_cycles=active_cycles,
                first_ts=first_ts,
                last_end=last_end,
                makespan=makespan,
            ))
    return sorted(
        stats, key=lambda stat: (stat.core_idx, _label_sort_key(stat.label), stat.lane_name))


def summarize_group(core_idx: int | None, label: str, tracks: Sequence[TrackStats]) -> GroupStats:
    if not tracks:
        return GroupStats(core_idx, label, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
    core_count = len(set(track.core_idx for track in tracks))
    count = float(len(tracks))
    return GroupStats(
        core_idx=core_idx,
        label=label,
        core_count=core_count,
        track_count=len(tracks),
        event_count=sum(track.event_count for track in tracks),
        active_event_count=sum(track.active_event_count for track in tracks),
        avg_active_cycles=sum(track.active_cycles for track in tracks) / count,
        avg_span=sum(track.span for track in tracks) / count,
        avg_active_over_span=sum(track.active_over_span for track in tracks) / count,
        avg_active_over_makespan=sum(track.active_over_makespan for track in tracks) / count,
    )


def summarize_core_pipe(track_stats: Sequence[TrackStats]) -> list[GroupStats]:
    groups: defaultdict[tuple[int, str], list[TrackStats]] = defaultdict(list)
    for stat in track_stats:
        groups[(stat.core_idx, stat.label)].append(stat)
    rows = [
        summarize_group(core_idx, label, tracks) for (core_idx, label), tracks in groups.items()
    ]
    return sorted(
        rows,
        key=lambda row: (row.core_idx
                         if row.core_idx is not None else -1, _label_sort_key(row.label)))


def summarize_pipe_across_cores(core_pipe_rows: Sequence[GroupStats]) -> list[GroupStats]:
    grouped: defaultdict[str, list[GroupStats]] = defaultdict(list)
    for row in core_pipe_rows:
        grouped[row.label].append(row)

    rows: list[GroupStats] = []
    for label, per_core_rows in grouped.items():
        count = float(len(per_core_rows))
        rows.append(
            GroupStats(
                core_idx=None,
                label=label,
                core_count=len(per_core_rows),
                track_count=sum(row.track_count for row in per_core_rows),
                event_count=sum(row.event_count for row in per_core_rows),
                active_event_count=sum(row.active_event_count for row in per_core_rows),
                avg_active_cycles=sum(row.avg_active_cycles for row in per_core_rows) / count,
                avg_span=sum(row.avg_span for row in per_core_rows) / count,
                avg_active_over_span=sum(row.avg_active_over_span for row in per_core_rows) / count,
                avg_active_over_makespan=sum(row.avg_active_over_makespan for row in per_core_rows)
                / count,
            ))
    return sorted(rows, key=lambda row: _label_sort_key(row.label))


def _label_sort_key(label: str) -> tuple[int, int, str]:
    if "." in label:
        lane, pipe = label.rsplit(".", 1)
        return LANE_ORDER.get(lane, 99), PIPE_ORDER.get(pipe, 99), label
    return 0, PIPE_ORDER.get(label, 99), label


def format_number(value: float) -> str:
    if abs(value - round(value)) < 0.0005:
        return str(int(round(value)))
    return f"{value:.3f}"


def format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(row: Sequence[str]) -> str:
        return "  ".join(
            cell.rjust(widths[idx]) if idx > 0 else cell.ljust(widths[idx])
            for idx, cell in enumerate(row))

    lines = [render_row(headers), "  ".join("-" * width for width in widths)]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def group_row(row: GroupStats, include_core: bool = False) -> list[str]:
    cells: list[str] = []
    if include_core:
        cells.append(str(row.core_idx))
    cells.extend([
        row.label,
        str(row.core_count) if not include_core else str(row.track_count),
        str(row.track_count) if not include_core else str(row.event_count),
        format_number(row.avg_active_cycles),
        format_number(row.avg_span),
        format_percent(row.avg_active_over_span),
        format_percent(row.avg_active_over_makespan),
    ])
    return cells


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union half-open intervals; touching endpoints do not create overlap."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def interval_length(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def intersection_length(left: Sequence[tuple[float, float]],
                        right: Sequence[tuple[float, float]]) -> float:
    """Linear intersection of two sorted interval unions."""
    i, j, total = 0, 0, 0.0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def compute_overlaps(events: Sequence[TraceEvent], makespan: float) -> list[dict[str, Any]]:
    active = [event for event in events if event_is_active(event, False, False)]
    by_track: defaultdict[tuple[int, str, str], list[tuple[float, float]]] = defaultdict(list)
    for event in active:
        by_track[event.track_key].append((event.ts, event.end))
    tracks = {key: merge_intervals(value) for key, value in by_track.items()}
    rows = []
    for core in sorted({event.core_idx for event in events}):
        matrix = tracks.get((core, "cube", "M"))
        if not matrix:
            continue
        pairs = [("M-MTE1", tracks.get((core, "cube", "MTE1"))),
                 ("M-FIX", tracks.get((core, "cube", "FIX")))]
        vectors = [(lane, value)
                   for (idx, lane, pipe), value in tracks.items()
                   if idx == core and pipe == "V" and lane in {"vec", "vec0", "vec1"}]
        pairs += [(f"M-V({lane})", value) for lane, value in sorted(vectors)]
        if vectors:
            pairs.append(
                ("M-V(any)", merge_intervals([span for _, spans in vectors for span in spans])))
        for label, other in pairs:
            if other is None:
                continue
            overlap = intersection_length(matrix, other)
            active_a, active_b = interval_length(matrix), interval_length(other)
            rows.append({
                "core_idx": core,
                "pair": label,
                "overlap_cycles": overlap,
                "active_a_cycles": active_a,
                "active_b_cycles": active_b,
                "overlap_over_active_a": overlap / active_a if active_a else 0,
                "overlap_over_active_b": overlap / active_b if active_b else 0,
                "overlap_over_makespan": overlap / makespan if makespan else 0
            })
    return rows


def analyze_waits(events: Sequence[TraceEvent], top: int = 10) -> dict[str, Any]:
    """Resolve explicit wait -> blocked operation -> actual sync producers.

    Never infer synchronization from memory_dependency arrows. Cross waits on
    S remain visible to this analysis even when S is omitted from utilization.
    """
    operations = {}
    for event in events:
        if event.cat.lower() not in {"wait", "stall"} and "task_id" in event.args:
            task_id = event.args["task_id"]
            if task_id in operations:
                raise ValueError(
                    f"ambiguous task_id {task_id!r}; wait attribution requires globally unique IDs")
            operations[task_id] = event
    tracks: defaultdict[tuple[int, str, str], list[tuple[float, float]]] = defaultdict(list)
    lanes: defaultdict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        if event_is_active(event, False, False) and event.pipe_name != "S":
            tracks[event.track_key].append((event.ts, event.end))
            lanes[(event.core_idx, event.lane_name)].append((event.ts, event.end))
    track_busy = {key: merge_intervals(value) for key, value in tracks.items()}
    lane_busy = {key: merge_intervals(value) for key, value in lanes.items()}
    groups = {}
    longest = []
    for event in events:
        if event.cat.lower() not in {"wait", "stall"} and not event.opname.startswith(
            ("WAIT ", "stall:")):
            continue
        blocked_id = event.args.get("blocked_task")
        blocked = operations.get(blocked_id)
        args = {**event.args, **(blocked.args if blocked else {})}
        producer_ids = args.get("sync_producers", [])
        if isinstance(producer_ids, str):
            producer_ids = [producer_ids]
        producers = []
        for task_id in producer_ids:
            source = operations.get(task_id)
            producers.append({
                "task_id": task_id,
                "resolved": source is not None,
                "operation": source.opname if source else None,
                "lane": source.lane_name if source else None,
                "pipe": source.pipe_name if source else None,
                "end_cycle": source.end if source else None
            })
        cross = bool(blocked and "cross" in blocked.opname.lower()) or "cross" in str(
            event.args.get("stall_reason", "")).lower()
        scope = "receiver_lane" if cross else "receiver_pipe"
        busy = (
            lane_busy.get((event.core_idx,
                           event.lane_name), []) if cross else track_busy.get(event.track_key, []))
        reason = event.args.get("stall_reason", event.cat)
        src = str(
            args.get("src_pipe") or "+".join(sorted({p["pipe"] for p in producers if p["pipe"]})) or
            "unknown").upper()
        dst = str(args.get("dst_pipe") or event.pipe_name).upper()
        flag = args.get("flag_id")
        key = (event.core_idx, event.lane_name, event.pipe_name, str(reason), src, dst, str(flag),
               str(args.get("mode")), str(args.get("phase")))
        if key not in groups:
            groups[key] = {
                "core_idx": event.core_idx,
                "lane": event.lane_name,
                "pipe": event.pipe_name,
                "reason": reason,
                "src_pipe": src,
                "dst_pipe": dst,
                "flag_id": flag,
                "mode": args.get("mode"),
                "phase": args.get("phase"),
                "busy_subtraction_scope": scope,
                "cross_core": cross,
                "examples": [],
                "count": 0,
                "windows": [],
                "busy": busy
            }
        group = groups[key]
        group["count"] += 1
        group["windows"].append((event.ts, event.end))
        if len(group["examples"]) < 3:
            group["examples"].append({
                "blocked_task": blocked_id,
                "blocked_operation": blocked.opname if blocked else None,
                "start_cycle": event.ts,
                "end_cycle": event.end,
                "producers": producers
            })
        longest.append({
            "core_idx":
                event.core_idx,
            "lane":
                event.lane_name,
            "pipe":
                event.pipe_name,
            "reason":
                reason,
            "flag_id":
                flag,
            "src_pipe":
                src,
            "dst_pipe":
                dst,
            "start_cycle":
                event.ts,
            "end_cycle":
                event.end,
            "window_cycles":
                event.dur,
            "uncovered_window_cycles":
                event.dur - intersection_length([(event.ts, event.end)], busy),
            "blocked_task":
                blocked_id,
            "blocked_operation":
                blocked.opname if blocked else None,
            "producers":
                producers,
            "detail":
                event.args.get("detail")
        })
    rows = []
    for group in groups.values():
        windows, busy = group.pop("windows"), group.pop("busy")
        merged = merge_intervals(windows)
        group.update({
            "raw_sum_window_cycles": interval_length(windows),
            "window_union_cycles": interval_length(merged),
            "uncovered_window_cycles": interval_length(merged) - intersection_length(merged, busy)
        })
        rows.append(group)
    rows.sort(key=lambda row: -row["uncovered_window_cycles"])
    longest.sort(key=lambda row: -row["window_cycles"])
    return {
        "status":
            "available" if rows else "unavailable",
        "meaning":
            "diagnostic modeled wait windows; uncovered windows are NOT exclusive hardware stalls or critical-path attribution; do not sum groups",
        "groups":
            rows[:top],
        "cross_core_groups": [row for row in rows if row["cross_core"]][:top],
        "longest":
            longest[:top]
    }


def analyze_document(document: dict[str, Any],
                     path: str = "",
                     group_by: str = "pipe",
                     include_s: bool = False,
                     include_sync: bool = False,
                     include_stall: bool = False,
                     top: int = 10) -> dict[str, Any]:
    events, unit = parse_events(document)
    if not events:
        raise ValueError("trace contains no timed events with ph == 'X'")
    makespan = compute_makespan(events)
    stats = compute_track_stats(events, makespan, group_by, include_s, include_sync, include_stall)
    core_rows = summarize_core_pipe(stats)
    report = build_json(path, events, unit, makespan, summarize_pipe_across_cores(core_rows),
                        core_rows)
    report["overlaps"] = compute_overlaps(events, makespan)
    report["waits"] = analyze_waits(events, top)
    report["warnings"] = []
    active: defaultdict[tuple[int, str, str], list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        if event_is_active(event, False, False):
            active[event.track_key].append((event.ts, event.end))
    for key, intervals in sorted(active.items()):
        if interval_length(intervals) > interval_length(merge_intervals(intervals)) + 1e-9:
            report["warnings"].append(
                f"same-track overlap on {key}: busy time uses interval union; check nested events or scheduling"
            )
    metadata = document.get("metadata", {})
    report["source_metadata"] = metadata
    if report["waits"]["status"] == "unavailable":
        report["warnings"].append(
            "No wait records available; filtered/absent wait data does not prove zero waiting. Use the complete trace."
        )
    if unit == "cycles":
        report["warnings"].append(
            "Simulator/model cycles are not measured hardware time; no frequency conversion is applied."
        )
    if unit != "cycles":
        report["warnings"].append(
            "Legacy *_cycles fields use the reported time_unit, not necessarily cycles.")
    return report


def configure_tool_logging(log: logging.Logger) -> None:
    """Standalone message-only logging; JSON mode writes directly to stdout."""
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


LOG = logging.getLogger("analyze_sim_trace")


def print_report(path: str, events: Sequence[TraceEvent], unit: str, makespan: float,
                 pipe_rows: Sequence[GroupStats], core_rows: Sequence[GroupStats]) -> None:
    if not LOG.handlers:
        configure_tool_logging(LOG)
    LOG.info("Trace: %s", path)
    LOG.info("Timed events: %s", len(events))
    LOG.info("Time unit: %s", unit)
    LOG.info("Total time taken: %s %s", format_number(makespan), unit)
    LOG.info("")

    LOG.info("Average utilization by pipe across cores")
    LOG.info(
        render_table(
            ["pipe", "cores", "tracks", "avg_active", "avg_span", "active/span", "active/makespan"],
            [group_row(row) for row in pipe_rows],
        ))
    LOG.info("")

    LOG.info("Utilization by core and pipe")
    LOG.info(
        render_table(
            [
                "core", "pipe", "tracks", "events", "avg_active", "avg_span", "active/span",
                "active/makespan"
            ],
            [group_row(row, include_core=True) for row in core_rows],
        ))


def build_json(path: str, events: Sequence[TraceEvent], unit: str, makespan: float,
               pipe_rows: Sequence[GroupStats], core_rows: Sequence[GroupStats]) -> dict[str, Any]:

    def encode(row: GroupStats) -> dict[str, Any]:
        out: dict[str, Any] = {
            "pipe": row.label,
            "core_count": row.core_count,
            "track_count": row.track_count,
            "event_count": row.event_count,
            "active_event_count": row.active_event_count,
            "avg_active_cycles": row.avg_active_cycles,
            "avg_span": row.avg_span,
            "avg_active_over_span": row.avg_active_over_span,
            "avg_active_over_makespan": row.avg_active_over_makespan,
        }
        if row.core_idx is not None:
            out["core_idx"] = row.core_idx
        return out

    return {
        "trace_path": path,
        "timed_event_count": len(events),
        "time_unit": unit,
        "total_time": makespan,
        "average_by_pipe": [encode(row) for row in pipe_rows],
        "by_core": [encode(row) for row in core_rows],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report utilization, overlap and diagnostic waits from EasyASC/TileLang Chrome traces."
    )
    parser.add_argument("trace_json", help="Path to simulator Chrome trace JSON")
    parser.add_argument(
        "--group-by",
        choices=("pipe", "lane-pipe"),
        default="pipe",
        help="Use pipe labels, splitting MTE2 into cube.MTE2 and vec.MTE2, "
        "or split all labels by lane and pipe. Default: pipe")
    parser.add_argument(
        "--include-s",
        action="store_true",
        help="Include PipeS/control tracks. They are omitted by default.")
    parser.add_argument(
        "--include-sync",
        action="store_true",
        help="Count sync events such as event_wait/bar_all as active time.")
    parser.add_argument(
        "--include-stall", action="store_true", help="Count stall:* events as active time.")
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of tables.")
    parser.add_argument(
        "--top", type=int, default=10, help="Maximum wait groups and longest waits to report.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure_tool_logging(LOG)
    args = parse_args(argv)
    try:
        if args.top <= 0:
            raise ValueError("--top must be positive")
        with open(args.trace_json, encoding="utf-8") as stream:
            document = json.load(stream)
        report = analyze_document(document, args.trace_json, args.group_by, args.include_s,
                                  args.include_sync, args.include_stall, args.top)
        if args.json:
            print(json.dumps(report, indent=2, allow_nan=False))
        else:
            print(
                f"Trace: {args.trace_json}\nTime unit: {report['time_unit']}\nTotal time: {format_number(report['total_time'])}"
            )
            for title, key in (("Average utilization by pipe", "average_by_pipe"),
                               ("Utilization by core and pipe", "by_core")):
                print("\n" + title)
                print(
                    render_table(
                        ["core", "pipe", "active", "span", "active/span", "active/makespan"], [[
                            str(row.get("core_idx", "avg")), row["pipe"],
                            format_number(row["avg_active_cycles"]),
                            format_number(row["avg_span"]),
                            format_percent(row["avg_active_over_span"]),
                            format_percent(row["avg_active_over_makespan"])
                        ] for row in report[key]]))
            print("\nSame-core overlap (V(any) is the union of Vector lanes, not their sum)")
            print(
                render_table(["core", "pair", "overlap", "/active A", "/active B", "/makespan"], [[
                    str(row["core_idx"]), row["pair"],
                    format_number(row["overlap_cycles"]),
                    format_percent(row["overlap_over_active_a"]),
                    format_percent(row["overlap_over_active_b"]),
                    format_percent(row["overlap_over_makespan"])
                ] for row in report["overlaps"]]))
            print("\nDiagnostic wait windows (NOT exclusive stalls; do not sum groups)")
            print(
                render_table([
                    "core", "lane/pipe", "reason", "src->dst", "flag", "count", "union", "uncovered"
                ], [[
                    str(row["core_idx"]), row["lane"] + "/" + row["pipe"], row["reason"],
                    row["src_pipe"] + "->" + row["dst_pipe"],
                    str(row["flag_id"]),
                    str(row["count"]),
                    format_number(row["window_union_cycles"]),
                    format_number(row["uncovered_window_cycles"])
                ] for row in report["waits"]["groups"]]))
            print(
                "\nCross-core wait windows (reported separately so local windows cannot hide them)")
            print(
                render_table(
                    ["core", "receiver", "source", "flag", "count", "union", "uncovered"], [[
                        str(row["core_idx"]), row["lane"] + "/" + row["pipe"], row["src_pipe"],
                        str(row["flag_id"]),
                        str(row["count"]),
                        format_number(row["window_union_cycles"]),
                        format_number(row["uncovered_window_cycles"])
                    ] for row in report["waits"]["cross_core_groups"]]))
            for row in report["waits"]["longest"]:
                print(
                    f"wait {row['blocked_task']} flag={row['flag_id']} [{row['start_cycle']}, {row['end_cycle']}): "
                    f"producers={','.join(str(p['task_id']) for p in row['producers']) or 'unresolved'}"
                )
            for warning in report["warnings"]:
                print("Warning: " + warning)
        return 0
    except Exception as exc:
        LOG.error("error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
