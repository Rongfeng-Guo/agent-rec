#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

COMPARE_METRICS = [
    "CumulativeUtility",
    "InstructionUplift@H",
    "OverCorrectionRegret@H",
    "MemoryContaminationRate",
    "ScopeClassificationAccuracy",
    "PromotionRecall",
]

LOWER_IS_BETTER = {"OverCorrectionRegret@H", "MemoryContaminationRate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--outputs-root")
    parser.add_argument("--metric-table-json")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_metric_table(outputs_root: Path, metric_table_json: Path | None) -> dict:
    path = metric_table_json or (outputs_root / "metric_table" / "metric_table.json")
    return read_json(path) or {}


def select_best_non_smoke_run(run_rows: list[dict]) -> dict | None:
    eligible = [row for row in run_rows if row.get("run_status") == "NON_SMOKE"]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -(row.get("best_cumulative_utility") if isinstance(row.get("best_cumulative_utility"), (int, float)) else float("-inf")),
            row.get("run", ""),
        ),
    )
    return ranked[0] if ranked else None


def build_comparison_payload(metric_payload: dict) -> dict:
    run_rows = metric_payload.get("run_rows") or []
    method_rows = metric_payload.get("method_rows") or []
    best_run = select_best_non_smoke_run(run_rows)
    if best_run is None:
        return {
            "overall_status": "NO_NON_SMOKE_RUN",
            "summary": "No non-smoke closed-loop run is available for comparison.",
            "best_run": None,
            "winner": None,
            "comparisons": [],
        }

    run_method_rows = [row for row in method_rows if row.get("run") == best_run.get("run")]
    ranked_methods = sorted(
        run_method_rows,
        key=lambda row: (
            -(row.get("CumulativeUtility") if isinstance(row.get("CumulativeUtility"), (int, float)) else float("-inf")),
            row.get("method", ""),
        ),
    )
    winner = ranked_methods[0] if ranked_methods else None
    comparisons = []
    if winner is not None:
        for row in ranked_methods[1:]:
            metric_deltas = {}
            for metric in COMPARE_METRICS:
                winner_value = winner.get(metric)
                other_value = row.get(metric)
                if isinstance(winner_value, (int, float)) and isinstance(other_value, (int, float)):
                    raw_delta = winner_value - other_value
                    metric_deltas[metric] = {
                        "winner": winner_value,
                        "baseline": other_value,
                        "delta": raw_delta,
                        "winner_is_better": raw_delta <= 0 if metric in LOWER_IS_BETTER else raw_delta >= 0,
                    }
            comparisons.append(
                {
                    "baseline_method": row.get("method"),
                    "metric_deltas": metric_deltas,
                }
            )

    return {
        "overall_status": "READY",
        "summary": f"Best non-smoke run is {best_run.get('run')} with winner {winner.get('method') if winner else None}.",
        "best_run": {
            "run": best_run.get("run"),
            "best_method": best_run.get("best_method"),
            "best_cumulative_utility": best_run.get("best_cumulative_utility"),
        },
        "winner": winner,
        "comparisons": comparisons,
    }


def build_report(payload: dict) -> str:
    lines = [
        "# Server184 Best Run Comparison",
        "",
        f"- overall_status: `{payload['overall_status']}`",
        f"- summary: {payload['summary']}",
        "",
    ]
    if payload["overall_status"] != "READY":
        return "\n".join(lines)
    best_run = payload["best_run"]
    winner = payload["winner"]
    lines.extend([
        "## Best Run",
        f"- run: `{best_run['run']}`",
        f"- best_method: `{best_run['best_method']}`",
        f"- best_cumulative_utility: `{best_run['best_cumulative_utility']}`",
        "",
        "## Winner Metrics",
    ])
    for metric in COMPARE_METRICS:
        lines.append(f"- {metric}: `{winner.get(metric)}`")
    lines.extend(["", "## Winner vs Baselines"])
    for item in payload["comparisons"]:
        lines.append(f"- baseline: `{item['baseline_method']}`")
        for metric in COMPARE_METRICS:
            data = item["metric_deltas"].get(metric)
            if not data:
                continue
            lines.append(
                f"  - {metric}: winner=`{data['winner']}` baseline=`{data['baseline']}` delta=`{data['delta']}` winner_is_better=`{data['winner_is_better']}`"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    outputs_root = Path(args.outputs_root).resolve() if args.outputs_root else root_dir / "outputs/server184_gimo"
    metric_table_json = Path(args.metric_table_json).resolve() if args.metric_table_json else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else outputs_root / "best_run_comparison"

    metric_payload = load_metric_table(outputs_root, metric_table_json)
    payload = build_comparison_payload(metric_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "comparison.md").write_text(build_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
