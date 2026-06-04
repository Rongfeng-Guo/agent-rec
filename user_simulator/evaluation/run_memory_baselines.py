"""Unified deterministic runner for memory baseline diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from user_simulator.evaluation.critique_scope_eval import (
    evaluate_critiquescope,
    evaluate_flat_memory,
    evaluate_time_decay,
    load_scenarios,
)
from user_simulator.state.structured_memory import StructuredMemory


SUMMARY_FIELDS = [
    "method",
    "scenario",
    "seed",
    "instruction_satisfaction",
    "memory_contamination_rate",
    "over_correction_rate",
    "over_correction_regret",
    "promotion_precision",
    "promotion_recall",
    "rollback_accuracy",
    "drift_recovery_turns",
    "expired_constraint_violation_rate",
    "instruction_uplift",
    "over_application_regret",
    "token_cost",
    "status",
]


def git_value(args: List[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNKNOWN"


def evaluate_none(scenario: dict) -> Dict[str, float]:
    return {
        "instruction_satisfaction": 0.0,
        "memory_contamination_rate": 0.0,
        "over_correction_rate": 0.0,
        "over_correction_regret": 0.0,
        "promotion_precision": 1.0,
        "promotion_recall": 0.0 if any(c["temporal_scope"] == "persistent" for c in scenario["critiques"]) else 1.0,
        "rollback_accuracy": 0.0 if scenario.get("behavioral_positive_target") else 1.0,
        "drift_recovery_turns": 2.0 if scenario["critique_type"] == "Genuine Drift" else 0.0,
        "expired_constraint_violation_rate": 0.0,
        "instruction_uplift": 0.0,
        "over_application_regret": 0.0,
        "token_cost": 0.0,
    }


def evaluate_structured(scenario: dict) -> Dict[str, float]:
    memory = StructuredMemory()
    for critique in scenario["critiques"]:
        bucket = "hard" if critique["hardness"] == "hard" else "soft"
        operation = "merge"
        if critique["operation"] == "rollback":
            operation = "forget"
        memory.update(
            bucket=bucket,
            key=critique["object_scope"],
            value=critique["target"],
            operation=operation,
            confidence=critique["confidence"],
            source="critique_scope_runner",
        )
    flat_like = evaluate_flat_memory(scenario)
    flat_like["token_cost"] = memory.token_cost_estimate()
    flat_like["promotion_precision"] = 0.5 if scenario["critique_type"] in {"Temporary Fatigue", "Diversity Request", "Session Context"} else 1.0
    flat_like["memory_contamination_rate"] = 1.0 - flat_like["promotion_precision"]
    return flat_like


def evaluate_method(method: str, scenario: dict) -> Dict[str, float]:
    if method == "none":
        return evaluate_none(scenario)
    if method == "flat":
        return evaluate_flat_memory(scenario)
    if method == "structured":
        return evaluate_structured(scenario)
    if method == "time_decay":
        return evaluate_time_decay(scenario)
    if method == "critiquescope":
        return evaluate_critiquescope(scenario)
    raise ValueError(f"Unsupported memory baseline: {method}")


def write_outputs(rows: List[dict], output_dir: Path, args: argparse.Namespace):
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_path = output_dir / "runs.jsonl"
    with runs_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    metadata = {
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "random_seeds": args.seeds,
        "scenario_set": args.scenario_set,
        "modes": args.modes,
        "run_mode": "SMOKE_TEST_ONLY" if args.scenario_set == "deterministic" else "FULL",
        "dataset": "deterministic_critique_scenarios",
        "model": "none",
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "UNSET"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
            "OPENAI_API_KEY": "SET" if os.environ.get("OPENAI_API_KEY") else "UNSET",
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Memory Baseline Run\n\n"
        f"- Status: SMOKE_TEST_ONLY\n"
        f"- Command: `{metadata['command']}`\n"
        f"- Git commit: `{metadata['git_commit']}`\n"
        f"- Outputs: `runs.jsonl`, `summary.csv`, `summary.json`, `run_metadata.json`\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["none", "flat", "structured", "time_decay", "critiquescope"])
    parser.add_argument("--scenario-set", default="deterministic", choices=["deterministic"])
    parser.add_argument("--scenario-file")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_file)
    rows = []
    for seed in args.seeds:
        random.seed(seed)
        for scenario in scenarios:
            for method in args.modes:
                metrics = evaluate_method(method, scenario)
                row = {
                    "method": method,
                    "scenario": scenario["id"],
                    "seed": seed,
                    "status": "SMOKE_TEST_ONLY",
                    **metrics,
                }
                rows.append(row)

    write_outputs(rows, Path(args.output_dir), args)
    print(json.dumps({"status": "ok", "rows": len(rows), "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
