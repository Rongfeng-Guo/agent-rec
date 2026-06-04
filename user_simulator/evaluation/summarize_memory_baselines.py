"""Aggregate memory baseline CSV results into JSON, CSV, and LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List


DEFAULT_METRICS = [
    "instruction_satisfaction",
    "memory_contamination_rate",
    "over_correction_rate",
    "over_correction_regret",
    "promotion_precision",
    "promotion_recall",
    "rollback_accuracy",
    "instruction_uplift",
    "over_application_regret",
    "token_cost",
]


def read_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def aggregate(rows: Iterable[dict], group_keys: List[str], metrics: List[str]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)

    results = []
    for key_values, group_rows in sorted(groups.items()):
        result = dict(zip(group_keys, key_values))
        result["n"] = len(group_rows)
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows if row.get(metric) not in {"", None}]
            result[f"{metric}_mean"] = mean(values) if values else None
        results.append(result)
    return results


def write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_number(value):
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def write_latex(path: Path, rows: List[dict], metrics: List[str]):
    columns = ["method", "n", *[f"{metric}_mean" for metric in metrics]]
    headers = ["Method", "N", *[metric.replace("_", " ").title() for metric in metrics]]
    lines = [
        "\\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [row["method"], str(row["n"])]
        cells.extend(format_number(row.get(column)) for column in columns[2:])
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to summary.csv from run_memory_baselines.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    output_dir = Path(args.output_dir)
    by_method = aggregate(rows, ["method"], args.metrics)
    by_method_scenario = aggregate(rows, ["method", "scenario"], args.metrics)

    write_csv(output_dir / "method_summary.csv", by_method)
    write_json(output_dir / "method_summary.json", by_method)
    write_csv(output_dir / "method_scenario_summary.csv", by_method_scenario)
    write_json(output_dir / "method_scenario_summary.json", by_method_scenario)
    write_latex(output_dir / "method_summary.tex", by_method, args.metrics)

    print(json.dumps({"status": "ok", "methods": len(by_method), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
