# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Standalone tests: no TileLang, TVM, NumPy or NPU runtime is required."""
import importlib.util
import json
import random
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "analyze_sim_trace.py"
spec = importlib.util.spec_from_file_location("trace_analysis_tool", SCRIPT)
tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)


def event(task, pipe, start, end, lane="cube", core=0, name="mma", cat="operation", **args):
    return {
        "name": name,
        "ph": "X",
        "cat": cat,
        "pid": f"core-{core}",
        "tid": f"{lane}/{pipe}",
        "ts": start,
        "dur": end - start,
        "args": {
            "task_id": task,
            "cycle_begin": start,
            "cycle_end": end,
            **args
        }
    }


class TraceAnalysisTests(unittest.TestCase):

    def analyze(self, events, **kwargs):
        return tool.analyze_document({"traceEvents": events, **kwargs})

    def test_union_and_intersection(self):
        self.assertEqual(tool.merge_intervals([(0, 4), (2, 8), (8, 9)]), [(0, 9)])
        self.assertEqual(tool.intersection_length([(0, 4), (6, 9)], [(2, 8)]), 4)

    def test_native_tracks_and_overlap(self):
        report = self.analyze([event("m", "m", 0, 10), event("l", "mte1", 5, 15)])
        row = next(r for r in report["overlaps"] if r["pair"] == "M-MTE1")
        self.assertEqual(row["overlap_cycles"], 5)
        self.assertEqual(row["overlap_over_active_a"], .5)
        self.assertAlmostEqual(row["overlap_over_makespan"], 1 / 3)

    def test_no_cross_core_pairing(self):
        report = self.analyze([event("m", "m", 0, 10), event("f", "fix", 0, 10, core=1)])
        self.assertFalse(any(r["pair"] == "M-FIX" for r in report["overlaps"]))

    def test_two_vector_lanes_union_not_sum(self):
        report = self.analyze([
            event("m", "m", 0, 10),
            event("v0", "v", 0, 8, lane="vector0"),
            event("v1", "v", 2, 10, lane="vector1")
        ])
        rows = {r["pair"]: r for r in report["overlaps"]}
        self.assertEqual(rows["M-V(any)"]["overlap_cycles"], 10)
        self.assertEqual(rows["M-V(vec0)"]["overlap_cycles"], 8)

    def test_same_track_union_and_overlap_warning(self):
        report = self.analyze([event("a", "m", 0, 10), event("b", "m", 5, 15)])
        self.assertEqual(report["average_by_pipe"][0]["avg_active_over_makespan"], 1)
        self.assertTrue(any("overlap" in text for text in report["warnings"]))

    def test_easyasc_compatibility_and_filters(self):
        events = [{
            "name": "mmad",
            "ph": "X",
            "cat": "M",
            "ts": 0,
            "dur": 10,
            "args": {
                "core_idx": 0,
                "lane_name": "cube",
                "pipe_name": "M",
                "time_domain": "cycle"
            }
        }, {
            "name": "setflag",
            "ph": "X",
            "cat": "sync",
            "ts": 10,
            "dur": 1,
            "args": {
                "core_idx": 0,
                "lane_name": "cube",
                "pipe_name": "M",
                "time_domain": "cycle"
            }
        }]
        report = self.analyze(events)
        self.assertEqual(report["time_unit"], "cycles")
        self.assertEqual(report["average_by_pipe"][0]["avg_active_cycles"], 10)

    def test_normalized_ids_and_cycle_coordinates(self):
        e = event("m", "m", 0, 100)
        e.update(pid=1, tid=2, ts=0, dur=.1)
        e["args"].update(original_pid="core-3", original_tid="cube/m")
        report = self.analyze([e], displayTimeUnit="ns")
        self.assertEqual(report["total_time"], 100)
        self.assertEqual(report["by_core"][0]["core_idx"], 3)

    def test_native_metadata_thread_names(self):
        e = {
            "name": "mma",
            "ph": "X",
            "cat": "operation",
            "pid": 0,
            "tid": 7,
            "ts": 0,
            "dur": 10,
            "args": {
                "core_idx": 0,
                "time_domain": "cycle"
            }
        }
        report = self.analyze([{
            "name": "thread_name",
            "ph": "M",
            "pid": 0,
            "tid": 7,
            "args": {
                "name": "cube/m"
            }
        }, e])
        self.assertEqual(report["average_by_pipe"][0]["pipe"], "M")

    def test_chrome_numeric_units_ignore_display_unit(self):
        e = {
            "name": "mma",
            "ph": "X",
            "cat": "M",
            "ts": 0,
            "dur": 2,
            "args": {
                "core_idx": 0,
                "lane_name": "cube",
                "pipe_name": "M"
            }
        }
        report = self.analyze([e], displayTimeUnit="ns")
        self.assertEqual(report["time_unit"], "us")
        self.assertEqual(report["total_time"], 2)

    def test_wait_resolves_flag_and_producer_and_subtracts_busy(self):
        events = [
            event("work", "m", 0, 8),
            event("set", "mte1", 9, 10, name="set_flag", src_pipe="mte1", flag_id=2),
            event(
                "blocked",
                "m",
                10,
                11,
                name="wait_flag",
                src_pipe="mte1",
                dst_pipe="m",
                flag_id=2,
                sync_producers=["set"]),
            event(
                "blocked#wait",
                "m",
                0,
                10,
                name="wait",
                cat="wait",
                blocked_task="blocked",
                stall_reason="local flag")
        ]
        report = self.analyze(events)
        row = report["waits"]["groups"][0]
        self.assertEqual(row["flag_id"], 2)
        self.assertEqual(row["window_union_cycles"], 10)
        self.assertEqual(row["uncovered_window_cycles"], 2)
        self.assertEqual(report["waits"]["longest"][0]["producers"][0]["task_id"], "set")

    def test_scalar_cross_wait_not_hidden_from_analysis(self):
        report = self.analyze([
            event("producer", "fix", 0, 10, name="set_cross_flag", flag_id=3),
            event(
                "consumer",
                "s",
                10,
                11,
                lane="vector0",
                name="wait_cross_flag",
                flag_id=3,
                sync_producers=["producer"]),
            event(
                "consumer#wait",
                "s",
                0,
                10,
                lane="vector0",
                name="wait",
                cat="wait",
                blocked_task="consumer",
                stall_reason="cross flag"),
            event("v", "v", 0, 6, lane="vector0")
        ])
        self.assertFalse(any(row["pipe"] == "S" for row in report["average_by_pipe"]))
        self.assertEqual(report["waits"]["groups"][0]["uncovered_window_cycles"], 4)

    def test_filtered_trace_does_not_claim_zero_waits(self):
        report = self.analyze([event("m", "m", 0, 10)], metadata={"display_filter": "compute only"})
        self.assertEqual(report["waits"]["status"], "unavailable")
        self.assertTrue(any("wait" in text.lower() for text in report["warnings"]))

    def test_invalid_numbers_rejected(self):
        for value in (float("nan"), float("inf"), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.analyze([event("m", "m", 0, value)])

    def test_touching_pipes_do_not_overlap(self):
        report = self.analyze([event("m", "m", 0, 10), event("f", "fix", 10, 20)])
        self.assertEqual(report["overlaps"][0]["overlap_cycles"], 0)

    def test_explicit_easyasc_runtime_control(self):
        report = self.analyze([
            event("m", "m", 0, 10), {
                "ph": "X",
                "name": "control",
                "cat": "S",
                "ts": 10,
                "dur": 2,
                "args": {
                    "core_idx": -1,
                    "lane_name": "runtime",
                    "pipe_name": "S",
                    "time_domain": "cycle"
                }
            }
        ])
        self.assertEqual(report["total_time"], 12)
        self.assertFalse(any(row["pipe"] == "S" for row in report["average_by_pipe"]))

    def test_wait_window_union_not_sum(self):
        report = self.analyze([
            event("blocked", "m", 12, 13, name="wait_flag", flag_id=2),
            event("w1", "m", 0, 10, name="wait", cat="wait", blocked_task="blocked"),
            event("w2", "m", 5, 12, name="wait", cat="wait", blocked_task="blocked")
        ])
        group = report["waits"]["groups"][0]
        self.assertEqual(group["raw_sum_window_cycles"], 17)
        self.assertEqual(group["window_union_cycles"], 12)

    def test_missing_producer_not_replaced_with_memory_edge(self):
        report = self.analyze([
            event("memory-only", "mte1", 0, 9),
            event(
                "blocked",
                "m",
                10,
                11,
                name="wait_flag",
                sync_producers=["missing"],
                memory_dependencies=["memory-only"]),
            event("w", "m", 0, 10, name="wait", cat="wait", blocked_task="blocked")
        ])
        producers = report["waits"]["longest"][0]["producers"]
        self.assertEqual(producers[0]["task_id"], "missing")
        self.assertFalse(producers[0]["resolved"])

    def test_include_sync_is_only_a_utilization_option(self):
        events = [
            event("m", "m", 0, 10),
            event("sync", "m", 10, 12, name="set_flag"),
            event("f", "fix", 10, 12)
        ]
        document = {"traceEvents": events}
        report = tool.analyze_document(document, include_sync=True)
        self.assertEqual(
            next(row for row in report["by_core"] if row["pipe"] == "M")["avg_active_cycles"], 12)
        self.assertEqual(report["overlaps"][0]["overlap_cycles"], 0)

    def test_unknown_track_rejected(self):
        with self.assertRaises(ValueError):
            self.analyze([{"ph": "X", "name": "mma", "cat": "operation", "ts": 0, "dur": 1}])

    def test_ambiguous_task_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous task_id"):
            self.analyze([event("same", "m", 0, 10), event("same", "m", 0, 10, core=1)])

    def test_mixed_time_domains_rejected(self):
        a = event("a", "m", 0, 5)
        b = {
            "ph": "X",
            "name": "copy",
            "cat": "MTE1",
            "ts": 0,
            "dur": 1,
            "args": {
                "core_idx": 0,
                "lane_name": "cube",
                "pipe_name": "MTE1",
                "time_domain": "us"
            }
        }
        with self.assertRaises(ValueError):
            self.analyze([a, b])

    def test_cli_json_standalone(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": [event("m", "m", 0, 10)]}))
            process = subprocess.run(
                [sys.executable, str(SCRIPT), str(path), "--json"],
                capture_output=True,
                text=True,
                check=True)
            self.assertEqual(json.loads(process.stdout)["total_time"], 10)


def region(buffer="ws", shape=(2, 8), offset=0, strides=(32, 4), dtype="float32"):
    return dict(
        buffer=buffer,
        scope="gm",
        shape=list(shape),
        dtype=dtype,
        byte_offset=offset,
        strides_bytes=list(strides),
        core_id=None)


class FragmentReadinessTests(unittest.TestCase):

    def analyze(self, events):
        return tool.analyze_document({"traceEvents": events})["fragment_readiness"]

    def fixture(self):
        writes = [
            event(
                f"f{i}",
                "fix",
                i * 10,
                i * 10 + 5,
                name="copy_l0c_to_gm",
                dst=region(shape=(2, 2), offset=i * 8)) for i in range(4)
        ]
        read = event("read", "mte2", 40, 45, lane="vector0", name="copy_gm_to_ub", src=region())
        return writes, read

    def test_four_column_fragments_not_whole_copy_early(self):
        writes, read = self.fixture()
        report = self.analyze(writes + [read])
        row = report["candidates"][0]
        self.assertEqual(row["source_bytes"], 64)
        self.assertEqual(row["whole_source_ready_cycle"], 35)
        self.assertEqual(row["whole_copy_start_lag_cycles"], 5)
        self.assertEqual([m["cumulative_fraction"] for m in row["milestones"]], [.25, .5, .75, 1.])
        self.assertEqual(row["milestones"][0]["producer_tasks"], ["f0"])
        self.assertEqual(row["milestones"][0]["byte_intervals"], [[0, 8], [32, 40]])
        self.assertFalse(row["legal_reschedule_proven"])

    def test_holes_are_not_filled_by_bounding_box(self):
        writes, read = self.fixture()
        report = self.analyze(writes[:1] + [read])
        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["status_counts"]["incomplete_source"], 1)

    def test_latest_ring_slot_write_wins(self):
        writes, read = self.fixture()
        writes += [event("new", "fix", 36, 39, dst=region())]
        report = self.analyze(writes + [read])
        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["status_counts"]["single_ready_time"], 1)

    def test_partial_overwrite_keeps_latest_version(self):
        writes, read = self.fixture()
        writes += [event("new", "fix", 36, 39, dst=region(shape=(2, 2), offset=0))]
        row = self.analyze(writes + [read])["candidates"][0]
        self.assertEqual(row["milestones"][-1]["producer_tasks"], ["new"])
        self.assertEqual(row["milestones"][-1]["bytes"], 16)
        self.assertNotIn("f0", str(row["milestones"]))

    def test_inflight_or_concurrent_overwrite_is_not_ready(self):
        for start, end in [(38, 43), (42, 44)]:
            writes, read = self.fixture()
            writes += [event("racing", "fix", start, end, dst=region())]
            report = self.analyze(writes + [read])
            self.assertEqual(report["candidate_count"], 0)
            self.assertEqual(report["status_counts"]["overlapping_write"], 1)

    def test_symbolic_or_layout_regions_are_unknown(self):
        writes, read = self.fixture()
        writes += [event("unknown", "fix", 36, 39, dst={**region(), "byte_offset": "dynamic"})]
        report = self.analyze(writes + [read])
        self.assertEqual(report["status_counts"]["unknown_writer_region"], 1)
        read["args"]["src"]["strides_bytes"] = [32, 8]
        report = self.analyze([read])
        self.assertEqual(report["status_counts"]["unsupported_source_region"], 1)

    def test_source_dtype_and_buffer_identity(self):
        writes, read = self.fixture()
        read["args"]["src"] = region(dtype="float16", strides=(32, 2))
        row = self.analyze(writes + [read])["candidates"][0]
        self.assertEqual(row["source_bytes"], 32)
        read["args"]["src"]["buffer"] = "other"
        self.assertEqual(self.analyze(writes + [read])["candidate_count"], 0)

    def test_dst_regions_and_unsorted_trace(self):
        writes, read = self.fixture()
        for write in writes:
            write["args"]["dst_regions"] = [write["args"].pop("dst")]
        row = self.analyze([read] + list(reversed(writes)))["candidates"][0]
        self.assertEqual(row["whole_source_ready_cycle"], 35)

    def test_concurrent_completed_writers_are_ambiguous(self):
        writes, read = self.fixture()
        writes += [
            event("racing", "mte3", 30, 37, lane="vector0", dst=region(shape=(2, 2), offset=24))
        ]
        report = self.analyze(writes + [read])
        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["status_counts"]["ambiguous_writer_order"], 1)

    def test_region_validation_and_large_rows_fail_closed(self):
        for r in [
                region(shape=(5000, 8)),
                region(offset=-1),
                region(strides=(16, 4)),
                region(dtype="int4")
        ]:
            self.assertIsNone(tool.region_intervals(r))
        self.assertEqual(tool.region_intervals(region(shape=(8,), strides=(4,))), [(0, 32)])

    def test_interval_subtraction_retains_gaps(self):
        self.assertEqual(
            tool.split_coverage([(0, 10), (20, 30)], [(2, 5), (8, 22)]),
            ([(2, 5), (8, 10), (20, 22)], [(0, 2), (5, 8), (22, 30)]))

    def test_padding_metadata_is_not_full_source_read(self):
        writes, read = self.fixture()
        read["args"]["copy"] = {"valid_cols": 3}
        report = self.analyze(writes + [read])
        self.assertEqual(report["status_counts"]["unsupported_source_region"], 1)

    def test_fake_memory_edge_does_not_change_readiness(self):
        writes, read = self.fixture()
        before = self.analyze(writes + [read])
        read["args"]["memory_dependencies"] = ["f3"]
        self.assertEqual(self.analyze(writes + [read]), before)

    def test_equal_ready_times_do_not_imply_streaming(self):
        writes, read = self.fixture()
        for write in writes:
            write["args"]["cycle_begin"] = 0
            write["args"]["cycle_end"] = 5
        report = self.analyze(writes + [read])
        self.assertEqual(report["status_counts"]["single_ready_time"], 1)

    def test_interval_subtraction_matches_byte_set_oracle(self):
        rng = random.Random(42)
        for _ in range(100):
            unions = [
                tool.merge_intervals([tuple(sorted(rng.sample(range(32), 2)))
                                      for _ in range(5)])
                for _ in range(2)
            ]
            a, b = [{i for lo, hi in spans for i in range(lo, hi)} for spans in unions]
            matched, rest = tool.split_coverage(*unions)
            self.assertEqual({i for lo, hi in matched for i in range(lo, hi)}, a & b)
            self.assertEqual({i for lo, hi in rest for i in range(lo, hi)}, a - b)


if __name__ == "__main__":
    unittest.main()
