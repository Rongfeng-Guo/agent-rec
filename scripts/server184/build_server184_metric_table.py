#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

SUMMARY_METRICS = [
    "CumulativeUtility",
    "AverageSlateUtility",
    "ClickRate",
    "InstructionUplift@1",
    "InstructionUplift@H",
    "OverCorrectionRegret@1",
    "OverCorrectionRegret@H",
    "MemoryContaminationRate",
    "ScopeClassificationAccuracy",
    "PromotionPrecision",
    "PromotionRecall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--outputs-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean_or_none(values: list[float]):
    if not values:
        return None
    return sum(values) / len(values)


def smoke_status(status_counts: dict[str, int]) -> str:
    if not status_counts:
        return "NO_STATUS"
    keys = set(status_counts)
    if keys.issubset({"SMOKE_TEST_ONLY"}):
        return "SMOKE_ONLY"
    if "SMOKE_TEST_ONLY" in keys:
        return "MIXED"
    return "NON_SMOKE"


def summarize_run(run_dir: Path) -> dict:
    rows = read_json(run_dir / "summary.json") or []
    if not isinstance(rows, list):
        rows = []
    by_method: dict[str, list[dict]] = defaultdict(list)
    status_counts = Counter(str(row.get("status")) for row in rows)
    parser_modes = sorted({str(row.get("parser_mode")) for row in rows if row.get("parser_mode") is not None})
    scenarios = sorted({str(row.get("scenario")) for row in rows if row.get("scenario") is not None})
    for row in rows:
        method = row.get("method")
        if method is not None:
            by_method[str(method)].append(row)

    method_rows = []
    for method, method_entries in sorted(by_method.items()):
        metric_means = {}
        for metric in SUMMARY_METRICS:
            values = [float(entry[metric]) for entry in method_entries if isinstance(entry.get(metric), (int, float))]
            metric_means[metric] = mean_or_none(values)
        method_rows.append(
            {
                "run": run_dir.name,
                "path": str(run_dir),
                "method": method,
                "row_count": len(method_entries),
                "scenario_count": len({entry.get("scenario") for entry in method_entries}),
                "parser_modes": parser_modes,
                "run_status": smoke_status(dict(status_counts)),
                "status_counts": dict(sorted(status_counts.items())),
                **metric_means,
            }
        )

    ranked = sorted(
        method_rows,
        key=lambda row: (
            -(row.get("CumulativeUtility") if isinstance(row.get("CumulativeUtility"), (int, float)) else float("-inf")),
            row["method"],
        ),
    )
    best = ranked[0] if ranked else None
    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "row_count": len(rows),
        "scenario_count": len(scenarios),
        "parser_modes": parser_modes,
        "status_counts": dict(sorted(status_counts.items())),
        "run_status": smoke_status(dict(status_counts)),
        "best_method": best["method"] if best else None,
        "best_cumulative_utility": best.get("CumulativeUtility") if best else None,
        "method_rows": method_rows,
    }


def collect_runs(outputs_root: Path) -> list[dict]:
    rows = []
    if not outputs_root.exists():
        return rows
    for child in sorted(outputs_root.iterdir()):
        if child.is_dir() and child.name.startswith("closed_loop_") and (child / "summary.json").exists():
            rows.append(summarize_run(child))
    return rows


def flatten_method_rows(run_rows: list[dict]) -> list[dict]:
    flat = []
    for run in run_rows:
        flat.extend(run["method_rows"])
    return flat


def build_markdown(run_rows: list[dict], method_rows: list[dict]) -> str:
    lines = [
        "# Server184 Closed Loop Metric Table",
        "",
        "## Run Summary",
        "| Run | Run Status | Rows | Scenarios | Best Method | Best Avg CU |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    if not run_rows:
        lines.append("| none | none | 0 | 0 | none | 0 |")
    else:
        for row in run_rows:
            lines.append(
                f"| {row['run']} | {row['run_status']} | {row['row_count']} | {row['scenario_count']} | {row['best_method']} | {row['best_cumulative_utility']} |"
            )
    lines.extend([
        "",
        "## Method Summary",
        "| Run | Method | Run Status | Avg CU | Avg Uplift@H | Avg Regret@H | Scope Acc | Memory Contam | Promotion Recall |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    if not method_rows:
        lines.append("| none | none | none | 0 | 0 | 0 | 0 | 0 | 0 |")
    else:
        for row in method_rows:
            lines.append(
                f"| {row['run']} | {row['method']} | {row['run_status']} | {row['CumulativeUtility']} | {row['InstructionUplift@H']} | {row['OverCorrectionRegret@H']} | {row['ScopeClassificationAccuracy']} | {row['MemoryContaminationRate']} | {row['PromotionRecall']} |"
            )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "method",
        "run_status",
        "row_count",
        "scenario_count",
        "CumulativeUtility",
        "AverageSlateUtility",
        "ClickRate",
        "InstructionUplift@1",
        "InstructionUplift@H",
        "OverCorrectionRegret@1",
        "OverCorrectionRegret@H",
        "MemoryContaminationRate",
        "ScopeClassificationAccuracy",
        "PromotionPrecision",
        "PromotionRecall",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    outputs_root = Path(args.outputs_root).resolve() if args.outputs_root else root_dir / "outputs/server184_gimo"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else outputs_root / "metric_table"

    run_rows = collect_runs(outputs_root)
    method_rows = flatten_method_rows(run_rows)
    payload = {
        "outputs_root": str(outputs_root),
        "run_count": len(run_rows),
        "method_row_count": len(method_rows),
        "run_rows": run_rows,
        "method_rows": method_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metric_table.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metric_table.md").write_text(build_markdown(run_rows, method_rows), encoding="utf-8")
    write_csv(output_dir / "metric_table.csv", method_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
