"""Adapter for turning CritiqueScope rollouts into benchmark scenarios.

The adapter accepts JSONL rows with utterance, critiques, and branch utilities.
When no input is provided it emits the deterministic built-in scenarios. This is
the bridge for later real GIMO rollouts: keep the schema stable, replace the
utility source.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import List

from user_simulator.evaluation.critique_parser import parse_deterministic
from user_simulator.evaluation.critique_scope_eval import DEFAULT_SCENARIOS
from user_simulator.evaluation.critique_uplift_pairs import build_pairs
from user_simulator.evaluation.validate_critique_scenarios import validate_scenario
from user_simulator.state.critique_scope import TARGET_ALIASES


REQUIRED_BRANCHES = ["follow_value", "ignore_value", "over_apply_value"]


def load_rollouts(path: str | None) -> List[dict]:
    if not path:
        return DEFAULT_SCENARIOS
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            row = json.loads(line)
            validate_rollout(row, line_no)
            rows.append(row)
    return rows


def validate_rollout(row: dict, line_no: int):
    errors = validate_scenario(row, index=line_no)
    if errors:
        raise ValueError(f"line {line_no}: " + "; ".join(errors))
    for branch in REQUIRED_BRANCHES:
        if not isinstance(row[branch], list) or not all(isinstance(value, (int, float)) for value in row[branch]):
            raise ValueError(f"line {line_no}: {branch} must be a list of numbers")


def trajectory_value(values: list[float]) -> float:
    return float(sum(values))


def _normalize_target(target: str) -> str:
    lowered = str(target).strip().lower()
    if lowered in TARGET_ALIASES:
        return TARGET_ALIASES[lowered].lower()
    if "family" in lowered:
        return "family"
    if "politic" in lowered:
        return "politics"
    if "mac" in lowered:
        return "mac"
    if "window" in lowered:
        return "windows"
    return lowered


def _signature(critique: dict) -> tuple[str, str, str]:
    return (
        _normalize_target(str(critique.get("target", ""))),
        str(critique.get("operation", "")).lower(),
        str(critique.get("temporal_scope", "")).replace("-", "_").lower(),
    )


def audit_rollouts(rows: List[dict]) -> List[dict]:
    findings: List[dict] = []
    for row in rows:
        row_id = row.get("id", "UNKNOWN")
        branch_lengths = {branch: len(row.get(branch, [])) for branch in REQUIRED_BRANCHES}
        same_length = len(set(branch_lengths.values())) == 1
        findings.append(
            {
                "scenario_id": row_id,
                "check": "branch_length_consistency",
                "passed": same_length,
                "observed": branch_lengths,
                "expected": {"all_equal": True},
            }
        )

        follow_sum = trajectory_value(row["follow_value"])
        ignore_sum = trajectory_value(row["ignore_value"])
        over_apply_sum = trajectory_value(row["over_apply_value"])
        findings.append(
            {
                "scenario_id": row_id,
                "check": "follow_outperforms_at_least_one_counterfactual",
                "passed": bool(follow_sum > ignore_sum or follow_sum > over_apply_sum),
                "observed": {
                    "follow_sum": follow_sum,
                    "ignore_sum": ignore_sum,
                    "over_apply_sum": over_apply_sum,
                },
                "expected": {"follow_gt_ignore_or_over_apply": True},
            }
        )

        parsed = parse_deterministic(row.get("utterance", ""))
        provided = row.get("critiques", [])
        parsed_signatures = {_signature(item) for item in parsed}
        provided_signatures = {_signature(item) for item in provided}
        findings.append(
            {
                "scenario_id": row_id,
                "check": "deterministic_parser_alignment",
                "passed": parsed_signatures == provided_signatures,
                "observed": {
                    "parsed": sorted(parsed_signatures),
                    "provided": sorted(provided_signatures),
                },
                "expected": {"parsed_equals_provided": True},
            }
        )
    return findings


def summarize_audit(findings: List[dict]) -> dict:
    by_check = Counter(finding["check"] for finding in findings)
    failed_by_check = Counter(finding["check"] for finding in findings if not finding["passed"])
    failed_scenarios = sorted({finding["scenario_id"] for finding in findings if not finding["passed"]})
    return {
        "total_checks": len(findings),
        "failed_checks": sum(1 for finding in findings if not finding["passed"]),
        "checks_by_type": dict(sorted(by_check.items())),
        "failed_by_type": dict(sorted(failed_by_check.items())),
        "failed_scenarios": failed_scenarios,
    }


def render_report(summary: dict) -> str:
    lines = [
        "# Rollout Adapter Audit",
        "",
        f"- Total checks: `{summary['total_checks']}`",
        f"- Failed checks: `{summary['failed_checks']}`",
        f"- Failed scenarios: `{', '.join(summary['failed_scenarios']) if summary['failed_scenarios'] else 'none'}`",
        "",
        "## Check Summary",
        "| Check | Count | Failed |",
        "| --- | ---: | ---: |",
    ]
    for check, count in summary["checks_by_type"].items():
        lines.append(f"| {check} | {count} | {summary['failed_by_type'].get(check, 0)} |")
    return "\n".join(lines) + "\n"


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario_id", "check", "passed"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Optional real rollout JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fail-on-audit-error", action="store_true")
    args = parser.parse_args()

    scenarios = load_rollouts(args.input)
    output_dir = Path(args.output_dir)
    audit = audit_rollouts(scenarios)
    failures = [finding for finding in audit if not finding["passed"]]
    summary = summarize_audit(audit)
    write_jsonl(output_dir / "normalized_scenarios.jsonl", scenarios)
    write_jsonl(output_dir / "critique_pairs.jsonl", build_pairs(scenarios))
    write_jsonl(output_dir / "adapter_audit.jsonl", audit)
    write_jsonl(output_dir / "adapter_failures.jsonl", failures)
    write_csv(output_dir / "adapter_audit_summary.csv", audit)
    (output_dir / "adapter_report.md").write_text(render_report(summary), encoding="utf-8")
    metadata = {
        "status": "PASS" if not failures else "FAIL",
        "input_status": "SMOKE_TEST_ONLY" if args.input is None else "REAL_ROLLOUT_INPUT",
        "input": args.input or "DEFAULT_SCENARIOS",
        "scenario_count": len(scenarios),
        "pair_count": len(build_pairs(scenarios)),
        "audit_checks": summary["total_checks"],
        "audit_failures": summary["failed_checks"],
        "audit_failed_scenarios": summary["failed_scenarios"],
        "audit_failed_by_type": summary["failed_by_type"],
    }
    (output_dir / "adapter_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", **metadata, "output_dir": str(output_dir)}, indent=2))
    if failures and args.fail_on_audit_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
