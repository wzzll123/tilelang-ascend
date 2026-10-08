# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""External designer report-CLI contract; no imports or kernel launches from that SDK."""

import argparse
import ast
import json
import logging
import os
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize("output_mode", ["directory", "legacy", "conflict"])
def test_report_cli(tmp_path, monkeypatch, output_mode):
    source = os.environ.get("FA_PERF_SCRIPT")
    if source is None:
        pytest.skip("Set FA_PERF_SCRIPT to the external designer perf_tilelang.py")
    tree = ast.parse(Path(source).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "main")
    isolated = ast.Module(body=[main], type_ignores=[])
    namespace = dict(logging=logging, argparse=argparse, Path=Path, json=json, sys=sys,
                     WORKDIR=tmp_path, logger=logging.getLogger("report-test"),
                     _load_module=lambda *a: None,
                     _get_input_groups=lambda *a: [[], []],
                     perf_case=lambda task, groups, idx, args, **kw:
                         dict(case_index=idx, elapsed_us=idx + 1),
                     _fmt_result=lambda r: str(r))
    exec(compile(isolated, "perf-report-main", "exec"), namespace)
    args = ["perf_tilelang.py", str(tmp_path), "--case", "0", "--case", "1"]
    folder, single = tmp_path / "reports", tmp_path / "legacy.json"
    if output_mode in ("directory", "conflict"):
        args.extend(["--json-dir", str(folder)])
    if output_mode in ("legacy", "conflict"):
        args.extend(["--json", str(single)])
    monkeypatch.setattr(sys, "argv", args)
    if output_mode == "conflict":
        with pytest.raises(SystemExit) as error:
            namespace["main"]()
        assert error.value.code == 2
    else:
        namespace["main"]()
        if output_mode == "directory":
            assert sorted(p.name for p in folder.iterdir()) == ["case_0.json", "case_1.json"]
            for idx in (0, 1):
                assert json.loads((folder / f"case_{idx}.json").read_text())["case_index"] == idx
        else:
            assert json.loads(single.read_text())["case_index"] == 1
