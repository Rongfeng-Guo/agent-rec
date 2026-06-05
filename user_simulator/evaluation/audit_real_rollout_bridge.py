"""Audit real or fixture rollout bridge outputs.

The auditor inspects exporter outputs, bridge pair files, and the optional
source JSONL emitted by the exporter. It is intentionally conservative: any
missing artifact is reported, but only a handful of conditions are considered
critical.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from user_simulator.evaluation.critique_rollout_adapter import normalize_task_type
from user_simulator.evaluation.validate_cdpo_pairs import validate_file as validate_cdpo_file


@dataclass
class ArtifactLoad:
    rows: list[dict]
    parse_errors: int
    path: Path | None


def read_jsonl(path: Path) -> ArtifactLoad:
    rows: list[dict] = []
    parse_errors = 0
    if not path.exists():
        return ArtifactLoad(rows=[], parse_errors=0, path=None)

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors += 1
                rows.append({"__error__": str(exc), "__line_no__": line_no})
                continue
            row["__line_no__"] = line_no
            rows.append(row)
    return ArtifactLoad(rows=rows, parse_errors=parse_errors, path=path)


def read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def source_ref(row: dict) -> str:
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ["source_ref", "branch_id"]:
            value = metadata.get(key)
            if value:
                return str(value)
    for key in ["source_ref", "branch_id", "id"]:
        value = row.get(key)
        if value:
            return str(value)
    return "UNKNOWN"


def row_signature(row: dict) -> str:
    return json.dumps(
        {
            "chosen": row.get("chosen", {}),
            "rejected": row.get("rejected", {}),
            "task_type": row.get("task_type"),
            "source_ref": row.get("source_ref"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def trajectory_len(text: Any) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len([line for line in text.splitlines() if line.strip()])


def build_row_error(level: str, artifact: str, message: str, row: dict | None = None) -> dict:
    payload = {
        "level": level,
        "artifact": artifact,
        "message": message,
    }
    if row is not None:
        payload["line_no"] = row.get("__line_no__")
        if row.get("id") is not None:
            payload["id"] = row.get("id")
        if row.get("source_ref") is not None:
            payload["source_ref"] = row.get("source_ref")
        if row.get("task_type") is not None:
            payload["task_type"] = row.get("task_type")
    return payload


def summarize_task_types(rows: Iterable[dict]) -> Counter:
    counts = Counter()
    for row in rows:
        task_type = normalize_task_type(row.get("task_type"))
        counts[task_type] += 1
    return counts


def summarize_pair_quality(rows: list[dict]) -> dict:
    chosen_lengths = []
    rejected_lengths = []
    uplift_values = []
    positives = zeros = negatives = 0
    same_pair = 0
    policy_empty = 0
    search_query_empty = 0
    missing_uplift = 0

    for row in rows:
        chosen = row.get("chosen", {})
        rejected = row.get("rejected", {})
        chosen_policy = str(chosen.get("policy", "")).strip()
        rejected_policy = str(rejected.get("policy", "")).strip()
        chosen_trajectory = str(chosen.get("trajectory", "")).strip()
        rejected_trajectory = str(rejected.get("trajectory", "")).strip()
        chosen_lengths.append(trajectory_len(chosen_trajectory))
        rejected_lengths.append(trajectory_len(rejected_trajectory))

        try:
            uplift = float(row.get("score_delta"))
        except (TypeError, ValueError):
            missing_uplift += 1
        else:
            uplift_values.append(uplift)
            if uplift > 0:
                positives += 1
            elif uplift < 0:
                negatives += 1
            else:
                zeros += 1

        if not chosen_policy or not rejected_policy:
            policy_empty += 1
        if chosen_policy == rejected_policy and chosen_trajectory == rejected_trajectory:
            same_pair += 1

        if normalize_task_type(row.get("task_type")) == "search":
            if not chosen_policy or "query" not in chosen_policy.lower() or not chosen_trajectory:
                search_query_empty += 1

    total = len(rows)
    uplift_total = len(uplift_values)
    return {
        "pair_count": total,
        "chosen_trajectory_length_min": min(chosen_lengths) if chosen_lengths else 0,
        "chosen_trajectory_length_mean": mean(chosen_lengths) if chosen_lengths else 0.0,
        "chosen_trajectory_length_max": max(chosen_lengths) if chosen_lengths else 0,
        "rejected_trajectory_length_min": min(rejected_lengths) if rejected_lengths else 0,
        "rejected_trajectory_length_mean": mean(rejected_lengths) if rejected_lengths else 0.0,
        "rejected_trajectory_length_max": max(rejected_lengths) if rejected_lengths else 0,
        "uplift_min": min(uplift_values) if uplift_values else None,
        "uplift_mean": mean(uplift_values) if uplift_values else None,
        "uplift_max": max(uplift_values) if uplift_values else None,
        "uplift_missing_count": missing_uplift,
        "uplift_zero_count": zeros,
        "uplift_positive_count": positives,
        "uplift_negative_count": negatives,
        "uplift_zero_ratio": (zeros / uplift_total) if uplift_total else None,
        "uplift_positive_ratio": (positives / uplift_total) if uplift_total else None,
        "uplift_negative_ratio": (negatives / uplift_total) if uplift_total else None,
        "chosen_rejected_same_count": same_pair,
        "policy_empty_count": policy_empty,
        "search_query_rewrite_empty_count": search_query_empty,
    }


def write_json(path: Path, payload: dict):
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def render_markdown(summary: dict) -> str:
    lines = [
        "# Real Rollout Bridge Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Critical errors: `{summary['critical_error_count']}`",
        f"- Errors: `{summary['error_count']}`",
        f"- Warnings: `{summary['warning_count']}`",
        "",
        "## Counts",
    ]
    for key in [
        "source_row_count",
        "converted_row_count",
        "skipped_row_count",
        "parse_error_count",
        "branch_rollout_count",
        "dpo_pair_count",
        "cdpo_pair_count",
        "duplicate_row_count",
        "combined_log_missing_count",
        "missing_state_snapshot_count",
        "missing_critique_count",
        "missing_metadata_count",
        "follow_branch_missing_count",
        "ignore_branch_missing_count",
        "over_apply_branch_missing_count",
        "policy_empty_count",
        "search_query_rewrite_empty_count",
        "chosen_rejected_same_count",
        "uplift_missing_count",
    ]:
        lines.append(f"- {key}: `{summary.get(key)}`")
    lines.append("")
    lines.append("## Task Types")
    for task_type, count in summary.get("task_type_counts", {}).items():
        lines.append(f"- {task_type}: `{count}`")
    return "\n".join(lines) + "\n"


def audit_bridge(input_dir: Path, output_dir: Path) -> dict:
    export_metadata = read_json(input_dir / "export_metadata.json") or {}
    source = read_jsonl(input_dir / "adapter_input.jsonl")
    if not source.path:
        source = read_jsonl(input_dir / "normalized_scenarios.jsonl")
    branch = read_jsonl(input_dir / "branch_rollouts.jsonl")
    dpo = read_jsonl(input_dir / "dpo_pairs.jsonl")
    cdpo = read_jsonl(input_dir / "cdpo_pairs.jsonl")
    adapter_audit = read_jsonl(input_dir / "adapter_audit.jsonl")
    adapter_failures = read_jsonl(input_dir / "adapter_failures.jsonl")

    if source.path is None:
        raise FileNotFoundError("No source rows found in adapter_input.jsonl or normalized_scenarios.jsonl")

    row_errors: list[dict] = []
    warnings = 0
    errors = 0
    critical_errors = 0

    if source.parse_errors:
        for row in source.rows:
            if "__error__" in row:
                row_errors.append(build_row_error("CRITICAL", source.path.name, f"JSON decode error: {row['__error__']}", row))
                critical_errors += 1

    source_total_count = len(source.rows)
    source_rows = [row for row in source.rows if "__error__" not in row]
    branch_rows = [row for row in branch.rows if "__error__" not in row]
    dpo_rows = [row for row in dpo.rows if "__error__" not in row]
    cdpo_rows = [row for row in cdpo.rows if "__error__" not in row]

    source_row_count = len(source_rows)
    converted_row_count = len(branch_rows)
    skipped_row_count = max(0, source_total_count - source_row_count)
    parse_error_count = (
        source.parse_errors + branch.parse_errors + dpo.parse_errors + cdpo.parse_errors + adapter_audit.parse_errors + adapter_failures.parse_errors
    )

    if source_row_count == 0:
        critical_errors += 1
        row_errors.append(build_row_error("CRITICAL", source.path.name, "No source rows discovered"))

    task_type_counts = summarize_task_types(source_rows)
    unknown_task_type_count = task_type_counts.get("generic", 0)
    combined_log_missing_count = sum(1 for row in source_rows if not row.get("combined_log"))
    missing_state_snapshot_count = sum(1 for row in source_rows if not row.get("state_snapshot"))
    missing_critique_count = sum(1 for row in source_rows if not row.get("critiques"))
    missing_metadata_count = sum(1 for row in cdpo_rows if not isinstance(row.get("metadata"), dict))

    source_ref_counts = Counter(source_ref(row) for row in source_rows)
    duplicate_row_count = sum(count - 1 for count in source_ref_counts.values() if count > 1)
    if duplicate_row_count:
        warnings += 1
        row_errors.append(build_row_error("WARNING", source.path.name, f"Duplicate source refs detected: {duplicate_row_count}"))

    branch_by_source = defaultdict(set)
    for row in branch_rows:
        branch_by_source[source_ref(row)].add(str(row.get("branch", "")))
    follow_branch_missing_count = sum(1 for branches in branch_by_source.values() if "follow" not in branches)
    ignore_branch_missing_count = sum(1 for branches in branch_by_source.values() if "ignore" not in branches)
    over_apply_branch_missing_count = sum(1 for branches in branch_by_source.values() if "over_apply" not in branches)

    pair_quality = summarize_pair_quality(cdpo_rows)
    uplift_missing_count = pair_quality["uplift_missing_count"]
    policy_empty_count = pair_quality["policy_empty_count"]
    search_query_rewrite_empty_count = pair_quality["search_query_rewrite_empty_count"]
    chosen_rejected_same_count = pair_quality["chosen_rejected_same_count"]

    for row in cdpo_rows:
        if not isinstance(row.get("metadata"), dict):
            row_errors.append(build_row_error("ERROR", cdpo.path.name, "metadata missing or invalid", row))
            errors += 1
        if not row.get("score_delta"):
            if row.get("score_delta") in [0, 0.0]:
                pass
            else:
                row_errors.append(build_row_error("WARNING", cdpo.path.name, "uplift missing", row))
                warnings += 1
        if not str(row.get("chosen", {}).get("policy", "")).strip():
            row_errors.append(build_row_error("ERROR", cdpo.path.name, "chosen policy empty", row))
            errors += 1
        if not str(row.get("rejected", {}).get("policy", "")).strip():
            row_errors.append(build_row_error("ERROR", cdpo.path.name, "rejected policy empty", row))
            errors += 1

    if cdpo_rows:
        validation = validate_cdpo_file(cdpo.path or (input_dir / "cdpo_pairs.jsonl"))
        if validation["status"] != "PASS":
            critical_errors += 1
            row_errors.append(
                {
                    "level": "CRITICAL",
                    "artifact": "cdpo_pairs.jsonl",
                    "message": "CDPO validation failed",
                    "details": validation,
                }
            )
        if validation["rows"] != len(cdpo_rows):
            warnings += 1
    else:
        critical_errors += 1
        row_errors.append(build_row_error("CRITICAL", "cdpo_pairs.jsonl", "No CDPO rows found"))

    if len(dpo_rows) != len(cdpo_rows):
        critical_errors += 1
        row_errors.append(
            build_row_error(
                "CRITICAL",
                "dpo_pairs.jsonl",
                f"DPO/CDPO mismatch: dpo={len(dpo_rows)} cdpo={len(cdpo_rows)}",
            )
        )

    unique_pair_sigs = {row_signature(row) for row in cdpo_rows}
    if cdpo_rows and len(unique_pair_sigs) == 1:
        critical_errors += 1
        row_errors.append(build_row_error("CRITICAL", "cdpo_pairs.jsonl", "All pairs are identical"))

    if cdpo_rows and policy_empty_count >= len(cdpo_rows):
        critical_errors += 1
        row_errors.append(build_row_error("CRITICAL", "cdpo_pairs.jsonl", "All policy text is empty"))

    row_errors_path = output_dir / "row_errors.jsonl"
    write_jsonl(row_errors_path, row_errors)

    task_type_rows = [
        {
            "task_type": task_type,
            "source_rows": count,
            "branch_rows": sum(1 for row in branch_rows if normalize_task_type(row.get("task_type")) == task_type),
            "dpo_pairs": sum(1 for row in dpo_rows if normalize_task_type(row.get("task_type")) == task_type),
            "cdpo_pairs": sum(1 for row in cdpo_rows if normalize_task_type(row.get("task_type")) == task_type),
        }
        for task_type, count in sorted(task_type_counts.items())
    ]
    write_csv(output_dir / "task_type_summary.csv", task_type_rows, ["task_type", "source_rows", "branch_rows", "dpo_pairs", "cdpo_pairs"])

    pair_quality_rows = [{"metric": key, "value": value} for key, value in pair_quality.items()]
    write_csv(output_dir / "pair_quality_summary.csv", pair_quality_rows, ["metric", "value"])

    status = "PASS"
    if critical_errors:
        status = "FAIL"
    elif errors:
        status = "WARN"

    summary = {
        "status": status,
        "critical_error_count": critical_errors,
        "error_count": errors,
        "warning_count": warnings,
        "source_row_count": source_row_count,
        "converted_row_count": converted_row_count,
        "skipped_row_count": skipped_row_count,
        "parse_error_count": parse_error_count,
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "unknown_task_type_count": unknown_task_type_count,
        "branch_rollout_count": len(branch_rows),
        "dpo_pair_count": len(dpo_rows),
        "cdpo_pair_count": len(cdpo_rows),
        "duplicate_row_count": duplicate_row_count,
        "combined_log_missing_count": combined_log_missing_count,
        "missing_state_snapshot_count": missing_state_snapshot_count,
        "missing_critique_count": missing_critique_count,
        "missing_metadata_count": missing_metadata_count,
        "follow_branch_missing_count": follow_branch_missing_count,
        "ignore_branch_missing_count": ignore_branch_missing_count,
        "over_apply_branch_missing_count": over_apply_branch_missing_count,
        "chosen_rejected_same_count": chosen_rejected_same_count,
        "uplift_missing_count": uplift_missing_count,
        "uplift_zero_ratio": pair_quality["uplift_zero_ratio"],
        "uplift_positive_ratio": pair_quality["uplift_positive_ratio"],
        "uplift_negative_ratio": pair_quality["uplift_negative_ratio"],
        "policy_empty_count": policy_empty_count,
        "search_query_rewrite_empty_count": search_query_rewrite_empty_count,
        "export_metadata": export_metadata,
        "adapter_audit_rows": len(adapter_audit.rows),
        "adapter_failure_rows": len(adapter_failures.rows),
    }
    summary["critical_error_count"] = critical_errors
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fail-on-critical-error", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    summary = audit_bridge(input_dir, output_dir)
    audit_path = output_dir / "audit.json"
    markdown_path = output_dir / "audit.md"
    audit_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")

    print(json.dumps({"status": summary["status"], "output_dir": str(output_dir), "audit": str(audit_path)}, indent=2, ensure_ascii=False))
    if summary["status"] == "FAIL" and args.fail_on_critical_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
