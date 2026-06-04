"""Generate a readable report for CritiqueWorld closed-loop output folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


KEY_METRICS = [
    "CumulativeUtility_mean",
    "ClickRate_mean",
    "InstructionUplift@H_mean",
    "OverCorrectionRegret@H_mean",
    "ScopeClassificationAccuracy_mean",
    "parser_scope_error_mean",
    "memory_update_error_mean",
    "candidate_coverage_error_mean",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def fmt(value: Any) -> str:
    if value in {None, ""}:
        return "NA"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def audit_output_dir(output_dir: Path) -> dict:
    required = [
        "run_metadata.json",
        "summary.csv",
        "method_summary.csv",
        "method_scenario_summary.csv",
        "trajectories.jsonl",
        "branch_rollouts.jsonl",
        "dpo_pairs.jsonl",
        "cdpo_pairs.jsonl",
        "cdpo_validation.json",
        "cdpo_dataset_manifest.json",
        "llamafactory_dataset_info_snippet.json",
        "cdpo_train.jsonl",
        "cdpo_dev.jsonl",
    ]
    errors = []
    for name in required:
        if not (output_dir / name).exists():
            errors.append(f"missing {name}")

    if errors:
        return {"status": "FAIL", "errors": errors}

    metadata = read_json(output_dir / "run_metadata.json")
    validation = read_json(output_dir / "cdpo_validation.json")
    manifest = read_json(output_dir / "cdpo_dataset_manifest.json")
    summary_rows = read_csv(output_dir / "summary.csv")
    method_rows = read_csv(output_dir / "method_summary.csv")
    counts = {
        "summary_rows": len(summary_rows),
        "trajectory_rows": count_jsonl(output_dir / "trajectories.jsonl"),
        "branch_rows": count_jsonl(output_dir / "branch_rollouts.jsonl"),
        "dpo_pairs": count_jsonl(output_dir / "dpo_pairs.jsonl"),
        "cdpo_pairs": count_jsonl(output_dir / "cdpo_pairs.jsonl"),
        "cdpo_train": count_jsonl(output_dir / "cdpo_train.jsonl"),
        "cdpo_dev": count_jsonl(output_dir / "cdpo_dev.jsonl"),
    }

    if validation.get("status") != "PASS":
        errors.append("cdpo_validation status is not PASS")
    if manifest.get("validation_status") != "PASS":
        errors.append("manifest validation status is not PASS")
    if counts["cdpo_pairs"] != validation.get("rows"):
        errors.append("cdpo pair count does not match validation rows")
    if counts["cdpo_pairs"] != manifest.get("row_count"):
        errors.append("cdpo pair count does not match manifest row_count")
    if counts["dpo_pairs"] != counts["cdpo_pairs"]:
        errors.append("raw dpo pair count does not match cdpo bridge pair count")
    if metadata.get("dpo_pair_count") != counts["dpo_pairs"]:
        errors.append("metadata dpo_pair_count does not match dpo_pairs.jsonl")
    if metadata.get("cdpo_pair_count") != counts["cdpo_pairs"]:
        errors.append("metadata cdpo_pair_count does not match cdpo_pairs.jsonl")
    splits = manifest.get("splits", {})
    if splits.get("train_count") != counts["cdpo_train"]:
        errors.append("manifest train_count does not match cdpo_train.jsonl")
    if splits.get("dev_count") != counts["cdpo_dev"]:
        errors.append("manifest dev_count does not match cdpo_dev.jsonl")
    if counts["cdpo_train"] + counts["cdpo_dev"] != counts["cdpo_pairs"]:
        errors.append("train/dev row counts do not sum to cdpo pair count")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "metadata": metadata,
        "validation": validation,
        "manifest": manifest,
        "counts": counts,
        "method_rows": method_rows,
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(output_dir: Path, audit: dict) -> str:
    metadata = audit["metadata"]
    validation = audit["validation"]
    manifest = audit["manifest"]
    counts = audit["counts"]
    method_rows = audit["method_rows"]

    metric_rows = []
    for row in method_rows:
        metric_rows.append([row.get("method", ""), *[fmt(row.get(metric)) for metric in KEY_METRICS]])

    dataset_rows = [
        ["dataset", manifest.get("dataset_name", "NA")],
        ["status", manifest.get("status", "NA")],
        ["validation", validation.get("status", "NA")],
        ["rows", str(manifest.get("row_count", "NA"))],
        ["train/dev", f"{manifest.get('splits', {}).get('train_count', 'NA')}/{manifest.get('splits', {}).get('dev_count', 'NA')}"],
        ["score_delta", f"{fmt(validation.get('score_delta_min'))} / {fmt(validation.get('score_delta_mean'))} / {fmt(validation.get('score_delta_max'))}"],
        ["sha256", manifest.get("input_sha256", "NA")],
    ]

    count_rows = [[key, str(value)] for key, value in counts.items()]
    rejected_rows = [[key, str(value)] for key, value in sorted(validation.get("by_rejected_branch", {}).items())]
    scenario_rows = [[key, str(value)] for key, value in sorted(validation.get("by_scenario", {}).items())]

    return "\n\n".join(
        [
            "# CritiqueWorld Closed-Loop Report",
            "This is a controlled counterfactual rollout proxy report. It is not human evaluation and not complete causal inference.",
            f"- Output directory: `{output_dir}`\n- Parser mode: `{metadata.get('parser_mode')}`\n- Git commit recorded by run: `{metadata.get('git_commit')}`\n- Audit status: `{audit['status']}`",
            "## Dataset Gate\n" + markdown_table(["Field", "Value"], dataset_rows),
            "## Output Counts\n" + markdown_table(["Artifact", "Rows"], count_rows),
            "## Method Summary\n" + markdown_table(["Method", *KEY_METRICS], metric_rows),
            "## Pair Distribution By Rejected Branch\n" + markdown_table(["Rejected Branch", "Rows"], rejected_rows),
            "## Pair Distribution By Scenario\n" + markdown_table(["Scenario", "Rows"], scenario_rows),
            "## Audit Errors\n" + ("\n".join(f"- {error}" for error in audit["errors"]) if audit["errors"] else "- none"),
        ]
    ) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audit = audit_output_dir(output_dir)
    if audit["status"] == "FAIL" and "metadata" not in audit:
        text = "# CritiqueWorld Closed-Loop Report\n\nAudit failed before report generation:\n"
        text += "\n".join(f"- {error}" for error in audit["errors"]) + "\n"
    else:
        text = build_report(output_dir, audit)

    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    print(json.dumps({"status": audit["status"], "report": str(report_path), "errors": audit["errors"]}, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
