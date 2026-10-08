"""Fail-closed integrity checks for the actual FA performance report reader."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("fa_perf_summary",
                                              Path(__file__).with_name("summarize_perf.py"))
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def _reports(root, prefix="fa128-perf-607-v06e471"):
    for round_id in (1, 2, 3):
        for implementation in ("v06", "auto128"):
            directory = root / f"{prefix}-round{round_id}-{implementation}"
            directory.mkdir()
            for case_id in summary.CASE_IDS:
                report = dict(
                    case_index=case_id,
                    warmup=3,
                    repeat=20,
                    elapsed_us=round_id * (10 if implementation == "v06" else 20))
                (directory / f"case_{case_id}.json").write_text(json.dumps(report))
    return root / f"{prefix}-round3-auto128/case_42.json"


def test_all_cases_use_three_round_medians(tmp_path):
    _reports(tmp_path)
    rows = summary.summarize(tmp_path)
    assert [row[0] for row in rows] == list(summary.CASE_IDS)
    assert all(row[1:4] == (20, 40, 0.5) for row in rows)


def test_prefetch_campaign_uses_same_statistic_without_renaming_reports(tmp_path):
    prefix = "fa128-prefetch-1aeedc56"
    _reports(tmp_path, prefix)
    rows = summary.summarize(tmp_path, prefix=prefix)
    assert all(row[1:4] == (20, 40, 0.5) for row in rows)
    with pytest.raises(ValueError, match="Incomplete/unexpected"):
        summary.summarize(tmp_path)


def test_incomplete_round_is_not_a_performance_pass(tmp_path):
    last = _reports(tmp_path)
    last.unlink()
    with pytest.raises(ValueError, match="Incomplete/unexpected"):
        summary.summarize(tmp_path)


@pytest.mark.parametrize("field,value", [("case_index", 0), ("repeat", 5), ("warmup", 0),
                                         ("elapsed_us", -1), ("elapsed_us", float("nan"))])
def test_invalid_measurement_is_rejected(tmp_path, field, value):
    last = _reports(tmp_path)
    report = json.loads(last.read_text())
    report[field] = value
    last.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        summary.summarize(tmp_path)
