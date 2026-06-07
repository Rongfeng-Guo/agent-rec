#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--outputs-root")
    parser.add_argument("--comparison-json")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_comparison(outputs_root: Path, comparison_json: Path | None) -> dict:
    path = comparison_json or (outputs_root / "best_run_comparison" / "comparison.json")
    return read_json(path) or {}


def build_facet_payload(comparison_payload: dict) -> dict:
    if comparison_payload.get("overall_status") != "READY":
        return {
            "overall_status": comparison_payload.get("overall_status") or "MISSING_COMPARISON",
            "summary": "Best-run comparison is not ready, so no facet report can be built.",
            "winner": comparison_payload.get("winner"),
            "metric_summary": {},
            "baseline_summaries": [],
        }

    winner = comparison_payload.get("winner") or {}
    comparisons = comparison_payload.get("comparisons") or []
    metric_summary: dict[str, dict] = defaultdict(lambda: {
        "win_count": 0,
        "loss_count": 0,
        "tie_count": 0,
        "wins_against": [],
        "losses_against": [],
        "ties_against": [],
    })
    baseline_summaries = []

    for row in comparisons:
        baseline = row.get("baseline_method")
        metric_deltas = row.get("metric_deltas") or {}
        win_count = 0
        loss_count = 0
        tie_count = 0
        better_metrics = []
        worse_metrics = []
        tie_metrics = []
        for metric, data in metric_deltas.items():
            delta = data.get("delta")
            better = data.get("winner_is_better")
            if isinstance(delta, (int, float)) and abs(delta) < 1e-12:
                tie_count += 1
                tie_metrics.append(metric)
                metric_summary[metric]["tie_count"] += 1
                metric_summary[metric]["ties_against"].append(baseline)
            elif better:
                win_count += 1
                better_metrics.append(metric)
                metric_summary[metric]["win_count"] += 1
                metric_summary[metric]["wins_against"].append(baseline)
            else:
                loss_count += 1
                worse_metrics.append(metric)
                metric_summary[metric]["loss_count"] += 1
                metric_summary[metric]["losses_against"].append(baseline)
        baseline_summaries.append({
            "baseline_method": baseline,
            "win_count": win_count,
            "loss_count": loss_count,
            "tie_count": tie_count,
            "better_metrics": better_metrics,
            "worse_metrics": worse_metrics,
            "tie_metrics": tie_metrics,
        })

    scorecard = Counter()
    for row in baseline_summaries:
        if row["win_count"] > row["loss_count"]:
            scorecard["baseline_majority_win"] += 1
        elif row["win_count"] < row["loss_count"]:
            scorecard["baseline_majority_loss"] += 1
        else:
            scorecard["baseline_draw"] += 1

    return {
        "overall_status": "READY",
        "summary": f"Winner {winner.get('method')} has metric-wise tradeoffs against {len(baseline_summaries)} baselines.",
        "winner": {
            "run": winner.get("run"),
            "method": winner.get("method"),
            "CumulativeUtility": winner.get("CumulativeUtility"),
        },
        "scorecard": dict(scorecard),
        "metric_summary": dict(metric_summary),
        "baseline_summaries": baseline_summaries,
    }


def build_report(payload: dict) -> str:
    lines = [
        "# Server184 Metric Facet Report",
        "",
        f"- overall_status: `{payload['overall_status']}`",
        f"- summary: {payload['summary']}",
        "",
    ]
    if payload["overall_status"] != "READY":
        return "\n".join(lines) + "\n"
    winner = payload["winner"]
    lines.extend([
        "## Winner",
        f"- run: `{winner.get('run')}`",
        f"- method: `{winner.get('method')}`",
        f"- CumulativeUtility: `{winner.get('CumulativeUtility')}`",
        "",
        "## Baseline Scorecard",
    ])
    scorecard = payload.get("scorecard") or {}
    if not scorecard:
        lines.append("- none")
    else:
        for key, value in sorted(scorecard.items()):
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Metric Summary"])
    for metric, row in sorted((payload.get("metric_summary") or {}).items()):
        lines.append(
            f"- {metric}: wins=`{row['win_count']}` losses=`{row['loss_count']}` ties=`{row['tie_count']}`"
        )
    lines.extend(["", "## Baseline Facets"])
    for row in payload.get("baseline_summaries") or []:
        lines.append(
            f"- `{row['baseline_method']}` wins=`{row['win_count']}` losses=`{row['loss_count']}` ties=`{row['tie_count']}` better={row['better_metrics']} worse={row['worse_metrics']} ties={row['tie_metrics']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    outputs_root = Path(args.outputs_root).resolve() if args.outputs_root else root_dir / "outputs/server184_gimo"
    comparison_json = Path(args.comparison_json).resolve() if args.comparison_json else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else outputs_root / "metric_facets"

    comparison_payload = load_comparison(outputs_root, comparison_json)
    payload = build_facet_payload(comparison_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "facet_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "facet_report.md").write_text(build_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
