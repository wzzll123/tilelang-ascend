#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Fit resident-data bench observations without native TileLang or NumPy.

Input records must explicitly mark train/holdout. Never calibrate against a
whole-kernel target. Preserve raw input hashes and per-shape validation errors.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

module_path = Path(__file__).resolve().parents[1] / "tilelang/simulator/linear_timing.py"
spec = importlib.util.spec_from_file_location("_simulator_linear_timing", module_path)
linear = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = linear
spec.loader.exec_module(linear)


def fit_observations(paths):
    groups = defaultdict(list)
    inputs = []
    names = set()
    for path in paths:
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs.append({"path": str(path.resolve()), "sha256": digest})
        for record in json.loads(path.read_text()):
            if record["name"] in names:
                raise ValueError(f"duplicate observation: {record['name']}")
            names.add(record["name"])
            if record.get("correctness") != "passed":
                raise ValueError("cannot fit failed correctness observations")
            if record.get("split") not in {"train", "holdout"}:
                raise ValueError("explicit train/holdout membership required")
            op, dtype = record["op"], record["dtype"]
            m, n, k = record["shape"]
            shape = tuple(
                record.get("timing_shape", (m, n, k) if op in {"mma", "mma_init"} else (m, n)))
            x = linear.timing_feature(op, shape, dtype)
            # Reduction helpers with distinct widths are NOT silently pooled.
            geometry = (f"last_axis_width_{n}" if op.startswith("reduce_") else record.get(
                "geometry", f"row_elements_{n}" if op == "copy_l0c_gm" else "contiguous"))
            for mode in ("two_slot", "chain"):
                groups[(op, dtype, geometry, mode)].append({
                    "name": record["name"],
                    "split": record["split"],
                    "feature": x,
                    "observed_ns": record[mode]["ns_per_op"],
                    "count_sweep_r_squared": record[mode]["r_squared"],
                })
                completion = record[mode].get("single_op_net_completion_ns")
                if completion is not None:
                    groups[(op, dtype, geometry, "single_net_" + mode)].append({
                        "name": record["name"],
                        "split": record["split"],
                        "feature": x,
                        "observed_ns": completion,
                        "counter_tick_ns": record[mode]["single_op_counter_tick_ns"],
                    })
    families = []
    evidence = "sha256:" + hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    for (op, dtype, geometry, mode), points in sorted(groups.items()):
        item = {
            "operation": op,
            "dtype": dtype,
            "geometry": geometry,
            "measurement_mode": mode,
            "observations": points
        }
        training = [(p["feature"], p["observed_ns"]) for p in points if p["split"] == "train"]
        try:
            cost = linear.fit_linear_cost(
                training,
                feature="padded_mnk" if op in {"mma", "mma_init"} else "bytes",
                evidence=evidence)
        except ValueError as error:
            item.update(status="insufficient_or_invalid_fit", reason=str(error))
            families.append(item)
            continue
        for point in points:
            try:
                prediction = cost.predict_ns(point["feature"])
            except ValueError:
                point["validation"] = "outside_measured_domain"
                continue
            point["predicted_ns"] = prediction
            point["relative_error"] = prediction / point["observed_ns"] - 1
        held_out = [
            abs(p["relative_error"])
            for p in points
            if p["split"] == "holdout" and "relative_error" in p
        ]
        item.update(
            status="fitted",
            cost=cost.to_dict(),
            held_out_count=len(held_out),
            held_out_max_abs_relative_error=max(held_out) if held_out else None)
        families.append(item)
    return {
        "schema": "tilelang-linear-cost-calibration-v1",
        "unit": "ns",
        "platform_measured": "A2",
        "a3_validated": False,
        "measurement_scope": "resident-data incremental helper costs, not bare ISA latency",
        "inputs": inputs,
        "families": families
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = fit_observations(args.observations)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for family in result["families"]:
        print(family["operation"], family["geometry"], family["measurement_mode"], family["status"],
              family.get("cost"), "holdout_max_error=",
              family.get("held_out_max_abs_relative_error"))


if __name__ == "__main__":
    main()
