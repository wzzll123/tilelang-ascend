"""Audit all three original profiler rounds, without replacing their statistic."""

import argparse
import json
import math
from pathlib import Path
from statistics import median

CASE_IDS = (0, 2, 6, 7, 10, 12, 13, 14, 15, 16, 36, 37, 41, 42)


def summarize(root: Path, prefix="fa128-perf-607-v06e471"):
    rows = []
    for case_id in CASE_IDS:
        durations = {}
        for implementation in ("v06", "auto128"):
            values = []
            for round_id in (1, 2, 3):
                directory = root / f"{prefix}-round{round_id}-{implementation}"
                expected = {f"case_{value}.json" for value in CASE_IDS}
                actual = {path.name for path in directory.glob("case_*.json")}
                if actual != expected:
                    raise ValueError(f"Incomplete/unexpected cases in {directory}: {actual}")
                report = json.loads((directory / f"case_{case_id}.json").read_text())
                if report["case_index"] != case_id:
                    raise ValueError(f"Case identity mismatch: {report}")
                if report["warmup"] != 3 or report["repeat"] != 20:
                    raise ValueError(f"Measurement schedule mismatch: {report}")
                duration = float(report["elapsed_us"])
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError(f"Invalid duration: {report}")
                values.append(duration)
            durations[implementation] = values
        baseline = median(durations["v06"])
        automatic = median(durations["auto128"])
        rows.append((case_id, baseline, automatic, baseline / automatic, durations))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).parent / "results")
    parser.add_argument(
        "--prefix",
        default="fa128-perf-607-v06e471",
        help="Frozen campaign directory prefix; measurement checks remain unchanged")
    args = parser.parse_args()
    rows = summarize(args.root, prefix=args.prefix)
    print("| case | v06 median (us) | auto median (us) | v06 / auto | v06 rounds | auto rounds |")
    print("|---:|---:|---:|---:|---|---|")
    for case_id, baseline, automatic, ratio, durations in rows:
        print(f"| {case_id} | {baseline:.1f} | {automatic:.1f} | {ratio:.4f} | "
              f"{durations['v06']} | {durations['auto128']} |")
    geomean = math.exp(sum(math.log(row[3]) for row in rows) / len(rows))
    print(f"Geometric mean (ratios of three-round medians): {geomean:.6f}")
    print(f"Performance gate >= 0.90: {'PASS' if geomean >= 0.90 else 'FAIL'}")


if __name__ == "__main__":
    main()
