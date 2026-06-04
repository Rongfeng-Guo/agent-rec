"""Run mechanism-level invariant audits for CritiqueWorld."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from user_simulator.evaluation.closed_loop_metrics import aggregate
from user_simulator.evaluation.run_closed_loop_benchmark import (
    memory_snapshot,
    rollout,
    run_branch_rollouts,
    write_jsonl,
    write_latex,
)
from user_simulator.evaluation.validity_invariants import evaluate_invariants
from user_simulator.scenarios.closed_loop_scenarios import list_scenarios
from user_simulator.worlds.critique_world import CritiqueWorldConfig


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def lifecycle_trace(rows: list[dict]) -> list[dict]:
    traced = []
    for row in rows:
        memory_before = row.get("memory_state_before", {})
        memory_after = row.get("memory_state_after", {})
        events_after = memory_after.get("events", [])
        traced.append(
            {
                "scenario": row.get("scenario"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "turn": row.get("turn"),
                "critique": row.get("generated_critique") or row.get("memory_update", {}).get("applied", []),
                "memory_before": memory_before,
                "memory_after": memory_after,
                "active_fast": [item for item in memory_after.get("fast", []) if item.get("active", True)],
                "active_slow": [item for item in memory_after.get("slow", []) if item.get("active", True)],
                "expired": [event for event in events_after if str(event.get("event", "")).startswith("expire")],
                "rolled_back": [event for event in events_after if str(event.get("event", "")).startswith("rollback")],
            }
        )
    return traced


def score_delta_trace(rows: list[dict]) -> list[dict]:
    traced = []
    for row in rows:
        for item_id, breakdown in row.get("score_breakdowns", {}).items():
            traced.append(
                {
                    "scenario": row.get("scenario"),
                    "seed": row.get("seed"),
                    "method": row.get("method"),
                    "turn": row.get("turn"),
                    "item_id": item_id,
                    "base_score": breakdown.get("base_score"),
                    "stable_match": breakdown.get("stable_match"),
                    "context_match": breakdown.get("context_match"),
                    "drift_match": breakdown.get("drift_match"),
                    "fatigue_penalty": breakdown.get("fatigue_penalty"),
                    "novelty_bonus": breakdown.get("novelty_bonus"),
                    "diversity_bonus": breakdown.get("diversity_bonus", 0.0),
                    "recent_exposure_penalty": breakdown.get("recent_exposure_penalty", 0.0),
                    "intervention_score_delta": breakdown.get("intervention_score_delta", 0.0),
                    "final_score": breakdown.get("final_score", breakdown.get("total")),
                    "rank_before": breakdown.get("rank_before"),
                    "rank_after": breakdown.get("rank_after"),
                }
            )
    return traced


def build_report(results: list[dict], failures: list[dict]) -> str:
    passed = sum(1 for row in results if row["passed"])
    total = len(results)
    critical_total = sum(1 for row in results if row.get("critical"))
    critical_failed = [row for row in failures if row.get("critical")]
    by_scenario = {}
    for row in results:
        by_scenario.setdefault(row["scenario"], []).append(row)

    lines = [
        "# CritiqueWorld Validity Gate",
        "",
        f"- Invariants passed: `{passed}/{total}`",
        f"- Critical failures: `{len(critical_failed)}/{critical_total}`",
        "",
        "## Scenario Summary",
    ]
    for scenario, rows in sorted(by_scenario.items()):
        local_passed = sum(1 for row in rows if row["passed"])
        lines.append(f"- `{scenario}`: {local_passed}/{len(rows)} passed")
    lines.extend(["", "## Critical Failures"])
    if not critical_failed:
        lines.append("- none")
    else:
        for failure in critical_failed:
            lines.append(
                f"- `{failure['scenario']}` / `{failure['method']}` / seed `{failure['seed']}`: "
                f"`{failure['invariant']}`"
            )
    return "\n".join(lines) + "\n"


def run_validity_gate(args: argparse.Namespace) -> dict:
    config = CritiqueWorldConfig(default_branch_horizon=args.branch_horizon)
    scenarios = list_scenarios(args.scenarios)
    rows = []
    branch_rows = []

    for seed in args.seeds:
        for scenario in scenarios:
            scenario.max_turns = args.max_turns
            scenario.top_k = args.top_k
            for mode in args.modes:
                trajectory_rows, _, _, critique_points = rollout(
                    scenario,
                    mode,
                    seed,
                    args.parser_mode,
                    args.max_turns,
                    args.top_k,
                    config,
                )
                branches, _ = run_branch_rollouts(
                    scenario,
                    mode,
                    seed,
                    args.parser_mode,
                    critique_points,
                    config,
                    args.top_k,
                    args.branch_horizon,
                )
                rows.extend(trajectory_rows)
                branch_rows.extend(branches)

    invariant_results = evaluate_invariants(rows, branch_rows, scenarios, args.modes, args.seeds)
    failures = [row for row in invariant_results if not row["passed"]]
    critical_failures = [row for row in failures if row.get("critical")]
    lifecycle_rows = lifecycle_trace(rows)
    score_rows = score_delta_trace(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    invariant_csv_rows = []
    for row in invariant_results:
        invariant_csv_rows.append(
            {
                "scenario": row["scenario"],
                "seed": row["seed"],
                "method": row["method"],
                "invariant": row["invariant"],
                "passed": row["passed"],
                "critical": row["critical"],
                "observed": json.dumps(row["observed"], ensure_ascii=False),
                "expected": json.dumps(row["expected"], ensure_ascii=False),
                "trace_ref": row["trace_ref"],
            }
        )
    write_csv(
        output_dir / "invariant_results.csv",
        invariant_csv_rows,
        ["scenario", "seed", "method", "invariant", "passed", "critical", "observed", "expected", "trace_ref"],
    )
    write_jsonl(output_dir / "invariant_failures.jsonl", failures)
    write_jsonl(output_dir / "lifecycle_trace.jsonl", lifecycle_rows)
    write_jsonl(output_dir / "score_delta_trace.jsonl", score_rows)

    method_scenario = aggregate(
        [
            {
                "method": row["method"],
                "scenario": row["scenario"],
                "seed": row["seed"],
                "CumulativeUtility": float(row["passed"]),
                "AverageSlateUtility": float(row["passed"]),
                "ClickRate": float(row["passed"]),
                "LeaveRate": 0.0 if row["passed"] else 1.0,
                "AverageSessionLength": 1.0,
                "SlateDiversity": float(row["passed"]),
                "CategoryCoverage": float(row["passed"]),
                "InstructionUplift@1": float(row["passed"]),
                "InstructionUplift@H": float(row["passed"]),
                "OverCorrectionRegret@1": 0.0 if row["passed"] else 1.0,
                "OverCorrectionRegret@H": 0.0 if row["passed"] else 1.0,
                "DuringHorizonUtility": float(row["passed"]),
                "PostExpiryRecoveryUtility": float(row["passed"]),
                "PostExpirySuppressionRegret": 0.0 if row["passed"] else 1.0,
                "ExpiredConstraintViolationRate": 0.0 if row["passed"] else 1.0,
                "DriftRecoveryTurns": 0.0,
                "RollbackAccuracy": float(row["passed"]),
                "MemoryContaminationRate": 0.0 if row["passed"] else 1.0,
                "PromotionPrecision": float(row["passed"]),
                "PromotionRecall": float(row["passed"]),
                "ScopeClassificationAccuracy": float(row["passed"]),
                "parser_scope_error": 0.0,
                "memory_update_error": 0.0,
                "policy_application_error": 0.0,
                "candidate_coverage_error": 0.0,
            }
            for row in invariant_results
        ],
        ["method", "scenario"],
    )
    write_csv(output_dir / "method_scenario_invariants.csv", method_scenario, list(method_scenario[0].keys()) if method_scenario else [])
    write_latex(output_dir / "tables.tex", method_scenario)
    report = build_report(invariant_results, failures)
    (output_dir / "scenario_report.md").write_text(report, encoding="utf-8")
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "modes": args.modes,
        "scenarios": [scenario.name for scenario in scenarios],
        "seeds": args.seeds,
        "parser_mode": args.parser_mode,
        "max_turns": args.max_turns,
        "top_k": args.top_k,
        "branch_horizon": args.branch_horizon,
        "fail_on_critical_invariant": args.fail_on_critical_invariant,
        "total_invariants": len(invariant_results),
        "failed_invariants": len(failures),
        "critical_failed_invariants": len(critical_failures),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "status": "PASS" if not critical_failures else "FAIL",
        "output_dir": str(output_dir),
        "total_invariants": len(invariant_results),
        "failed_invariants": len(failures),
        "critical_failed_invariants": len(critical_failures),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["none", "flat", "structured", "time_decay", "critiquescope"])
    parser.add_argument("--scenarios", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--parser-mode", choices=["oracle", "deterministic"], default="oracle")
    parser.add_argument("--branch-horizon", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fail-on-critical-invariant", action="store_true")
    args = parser.parse_args()

    summary = run_validity_gate(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.fail_on_critical_invariant and summary["critical_failed_invariants"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
