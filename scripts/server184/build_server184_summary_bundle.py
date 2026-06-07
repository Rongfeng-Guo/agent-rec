#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIMARY_METRICS = [
    "CumulativeUtility",
    "PromotionRecall",
    "InstructionUplift@H",
    "OverCorrectionRegret@H",
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


def load_artifacts(outputs_root: Path) -> dict:
    return {
        "decision": read_json(outputs_root / "decision" / "decision.json") or {},
        "metric_table": read_json(outputs_root / "metric_table" / "metric_table.json") or {},
        "best_run_comparison": read_json(outputs_root / "best_run_comparison" / "comparison.json") or {},
        "metric_facets": read_json(outputs_root / "metric_facets" / "facet_report.json") or {},
    }


def rank_top_run(metric_table: dict) -> dict | None:
    run_rows = metric_table.get("run_rows") or []
    ranked = sorted(
        run_rows,
        key=lambda row: (
            -(row.get("best_cumulative_utility") if isinstance(row.get("best_cumulative_utility"), (int, float)) else float("-inf")),
            row.get("run", ""),
        ),
    )
    return ranked[0] if ranked else None


def build_primary_metric_deltas(comparison: dict) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for metric in PRIMARY_METRICS:
        vs_baselines = []
        better_count = 0
        worse_count = 0
        tie_count = 0
        for row in comparison.get("comparisons") or []:
            data = (row.get("metric_deltas") or {}).get(metric)
            if not data:
                continue
            delta = data.get("delta")
            better = data.get("winner_is_better")
            if isinstance(delta, (int, float)) and abs(delta) < 1e-12:
                tie_count += 1
            elif better:
                better_count += 1
            else:
                worse_count += 1
            vs_baselines.append({
                "baseline_method": row.get("baseline_method"),
                "delta": delta,
                "winner_is_better": better,
            })
        summary[metric] = {
            "better_count": better_count,
            "worse_count": worse_count,
            "tie_count": tie_count,
            "vs_baselines": vs_baselines,
        }
    return summary


def build_actionable_conclusion(
    ready_for_review: bool,
    best_run: dict,
    winner: dict,
    primary_metric_deltas: dict[str, dict],
) -> str:
    if not ready_for_review or not best_run or not winner:
        return "Prerequisite artifacts are incomplete, so no metric-facing recommendation should be made yet."
    cu = primary_metric_deltas.get("CumulativeUtility") or {}
    recall = primary_metric_deltas.get("PromotionRecall") or {}
    uplift = primary_metric_deltas.get("InstructionUplift@H") or {}
    regret = primary_metric_deltas.get("OverCorrectionRegret@H") or {}
    return (
        f"Use {best_run.get('run')} as the review anchor: {winner.get('method')} leads on utility "
        f"({cu.get('better_count', 0)} wins) and recall ({recall.get('better_count', 0)} wins), "
        f"but loses on instruction uplift ({uplift.get('worse_count', 0)} losses) and shows mixed regret "
        f"({regret.get('better_count', 0)} wins, {regret.get('worse_count', 0)} losses)."
    )


def build_summary_payload(artifacts: dict, outputs_root: Path) -> dict:
    decision = artifacts.get("decision") or {}
    metric_table = artifacts.get("metric_table") or {}
    comparison = artifacts.get("best_run_comparison") or {}
    facets = artifacts.get("metric_facets") or {}

    ready_for_review = decision.get("overall_status") == "READY_FOR_METRIC_REVIEW"
    best_run = comparison.get("best_run") or {}
    winner = comparison.get("winner") or {}
    scorecard = facets.get("scorecard") or {}
    metric_summary = facets.get("metric_summary") or {}
    top_run = rank_top_run(metric_table)
    primary_metric_deltas = build_primary_metric_deltas(comparison)

    strongest_metrics = sorted(
        (
            (metric, row.get("win_count", 0), row.get("loss_count", 0), row.get("tie_count", 0))
            for metric, row in metric_summary.items()
        ),
        key=lambda item: (-item[1], item[2], item[0]),
    )
    weakest_metrics = sorted(
        (
            (metric, row.get("loss_count", 0), row.get("win_count", 0), row.get("tie_count", 0))
            for metric, row in metric_summary.items()
        ),
        key=lambda item: (-item[1], item[2], item[0]),
    )

    headline = (
        f"Server184 is ready for metric review; best non-smoke run is {best_run.get('run')} with winner {winner.get('method')}."
        if ready_for_review and best_run and winner
        else "Server184 summary bundle is incomplete because prerequisite artifacts are not all ready."
    )

    return {
        "overall_status": "READY"
        if ready_for_review and comparison.get("overall_status") == "READY" and facets.get("overall_status") == "READY"
        else "INCOMPLETE",
        "headline": headline,
        "actionable_conclusion": build_actionable_conclusion(ready_for_review, best_run, winner, primary_metric_deltas),
        "outputs_root": str(outputs_root),
        "readiness": {
            "decision_status": decision.get("overall_status"),
            "comparison_status": comparison.get("overall_status"),
            "facet_status": facets.get("overall_status"),
        },
        "best_run": best_run,
        "winner": {
            "method": winner.get("method"),
            "run": winner.get("run"),
            "CumulativeUtility": winner.get("CumulativeUtility"),
            "InstructionUplift@H": winner.get("InstructionUplift@H"),
            "OverCorrectionRegret@H": winner.get("OverCorrectionRegret@H"),
            "PromotionRecall": winner.get("PromotionRecall"),
        },
        "metric_table_summary": {
            "run_count": metric_table.get("run_count"),
            "method_row_count": metric_table.get("method_row_count"),
            "top_run": top_run.get("run") if top_run else None,
            "top_run_best_method": top_run.get("best_method") if top_run else None,
            "top_run_best_cumulative_utility": top_run.get("best_cumulative_utility") if top_run else None,
        },
        "scorecard": scorecard,
        "primary_metric_deltas": primary_metric_deltas,
        "strongest_metrics": [
            {"metric": metric, "win_count": wins, "loss_count": losses, "tie_count": ties}
            for metric, wins, losses, ties in strongest_metrics[:3]
        ],
        "weakest_metrics": [
            {"metric": metric, "loss_count": losses, "win_count": wins, "tie_count": ties}
            for metric, losses, wins, ties in weakest_metrics[:3]
        ],
        "next_reading_order": [
            "outputs/server184_gimo/decision/decision.md",
            "outputs/server184_gimo/metric_table/metric_table.md",
            "outputs/server184_gimo/best_run_comparison/comparison.md",
            "outputs/server184_gimo/metric_facets/facet_report.md",
        ],
    }


def build_report(payload: dict) -> str:
    lines = [
        "# Server184 Summary Bundle",
        "",
        f"- overall_status: `{payload['overall_status']}`",
        f"- headline: {payload['headline']}",
        f"- actionable_conclusion: {payload['actionable_conclusion']}",
        "",
        "## Readiness",
        f"- decision_status: `{payload['readiness'].get('decision_status')}`",
        f"- comparison_status: `{payload['readiness'].get('comparison_status')}`",
        f"- facet_status: `{payload['readiness'].get('facet_status')}`",
        "",
        "## Best Run",
        f"- run: `{payload['best_run'].get('run')}`",
        f"- best_method: `{payload['best_run'].get('best_method')}`",
        f"- best_cumulative_utility: `{payload['best_run'].get('best_cumulative_utility')}`",
        "",
        "## Winner Snapshot",
        f"- method: `{payload['winner'].get('method')}`",
        f"- CumulativeUtility: `{payload['winner'].get('CumulativeUtility')}`",
        f"- InstructionUplift@H: `{payload['winner'].get('InstructionUplift@H')}`",
        f"- OverCorrectionRegret@H: `{payload['winner'].get('OverCorrectionRegret@H')}`",
        f"- PromotionRecall: `{payload['winner'].get('PromotionRecall')}`",
        "",
        "## Metric Table Summary",
        f"- run_count: `{payload['metric_table_summary'].get('run_count')}`",
        f"- method_row_count: `{payload['metric_table_summary'].get('method_row_count')}`",
        f"- top_run: `{payload['metric_table_summary'].get('top_run')}`",
        f"- top_run_best_method: `{payload['metric_table_summary'].get('top_run_best_method')}`",
        f"- top_run_best_cumulative_utility: `{payload['metric_table_summary'].get('top_run_best_cumulative_utility')}`",
        "",
        "## Primary Metric Deltas",
    ]
    for metric in PRIMARY_METRICS:
        row = (payload.get("primary_metric_deltas") or {}).get(metric) or {}
        lines.append(
            f"- {metric}: better=`{row.get('better_count')}` worse=`{row.get('worse_count')}` ties=`{row.get('tie_count')}` deltas={row.get('vs_baselines')}"
        )
    lines.extend(["", "## Scorecard"])
    scorecard = payload.get("scorecard") or {}
    if not scorecard:
        lines.append("- none")
    else:
        for key, value in sorted(scorecard.items()):
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Strongest Metrics"])
    for row in payload.get("strongest_metrics") or []:
        lines.append(f"- {row['metric']}: wins=`{row['win_count']}` losses=`{row['loss_count']}` ties=`{row['tie_count']}`")
    lines.extend(["", "## Weakest Metrics"])
    for row in payload.get("weakest_metrics") or []:
        lines.append(f"- {row['metric']}: losses=`{row['loss_count']}` wins=`{row['win_count']}` ties=`{row['tie_count']}`")
    lines.extend(["", "## Reading Order"])
    for item in payload.get("next_reading_order") or []:
        lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    outputs_root = Path(args.outputs_root).resolve() if args.outputs_root else root_dir / "outputs/server184_gimo"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else outputs_root / "summary_bundle"

    artifacts = load_artifacts(outputs_root)
    payload = build_summary_payload(artifacts, outputs_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary_bundle.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary_bundle.md").write_text(build_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
