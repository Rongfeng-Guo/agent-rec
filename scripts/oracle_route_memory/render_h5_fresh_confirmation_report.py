#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.oracle_route_memory.train_candidate_level_source_ranker import summarize_eval_rows

try:
    from scripts.oracle_route_memory.handoff_io import (
    ensure_empty_output_dir,
    repo_relative_or_absolute as repo_relative,
    resolve_path_under_repo_root as resolve_path,
)
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from handoff_io import (
    ensure_empty_output_dir,
    repo_relative_or_absolute as repo_relative,
    resolve_path_under_repo_root as resolve_path,
)


CLAIM_BOUNDARY = (
    "Fresh confirmation metrics must remain separate from locked validation metrics. "
    "This report is valid only for a clearly fresh/unconsumed split that passed registration "
    "and readiness gates before locked-model scoring."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_json_input(repo_root: Path, value: str | Path, default_name: str) -> Path:
    path = resolve_path(repo_root, value)
    if path.is_dir():
        path = path / default_name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_output_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of scored output rows: {path}")
    return [dict(row) for row in payload]


def hit_count(rows: Sequence[Mapping[str, Any]], topk: int) -> int:
    count = 0
    for row in rows:
        rank = row.get("match_rank")
        if rank is not None and int(rank) <= topk:
            count += 1
    return count


def metric_with_hits(rows: Sequence[Mapping[str, Any]], topk: int) -> dict[str, Any]:
    metric = dict(summarize_eval_rows(rows, topk=topk))
    metric[f"hits_at_{topk}"] = hit_count(rows, topk=topk)
    return metric


def domain_metrics(rows: Sequence[Mapping[str, Any]], topk: int) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("domain", "")), []).append(row)
    return {domain: metric_with_hits(grouped[domain], topk=topk) for domain in sorted(grouped)}


def numeric_delta(fresh: Mapping[str, Any], validation: Mapping[str, Any], key: str) -> float | None:
    if fresh.get(key) is None or validation.get(key) is None:
        return None
    try:
        return float(fresh[key]) - float(validation[key])
    except (TypeError, ValueError):
        return None


def metric_deltas(fresh: Mapping[str, Any], validation: Mapping[str, Any], topk: int) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in (f"Recall@{topk}", "CandidatePoolHitRate", "ConditionalRecall@50GivenPoolHit"):
        delta = numeric_delta(fresh, validation, key)
        if delta is not None:
            deltas[key] = delta
    return deltas


def registration_gate_errors(registration: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if registration.get("status") != "ok":
        errors.append("fresh split registration status is not ok")
    if not str(registration.get("fresh_split_id", "")).strip():
        errors.append("fresh split registration is missing fresh_split_id")
    split_manifest = registration.get("split_manifest")
    if not isinstance(split_manifest, Mapping):
        errors.append("fresh split registration is missing split_manifest metadata")
    else:
        if not str(split_manifest.get("path", "")).strip():
            errors.append("fresh split registration split_manifest.path is missing")
        if not str(split_manifest.get("sha256", "")).strip():
            errors.append("fresh split registration split_manifest.sha256 is missing")
    required_fields = registration.get("required_manifest_fields")
    if not isinstance(required_fields, list) or not required_fields:
        errors.append("fresh split registration has no required_manifest_fields evidence")
    else:
        failed_fields = [
            field
            for field in required_fields
            if not isinstance(field, Mapping) or field.get("status") != "ok"
        ]
        if failed_fields:
            errors.append(f"fresh split registration has {len(failed_fields)} non-ok required_manifest_fields")
    return errors


def require_zero_readiness_count(readiness: Mapping[str, Any], key: str, errors: list[str]) -> None:
    if key not in readiness:
        errors.append(f"fresh readiness {key} is missing")
        return
    try:
        count = int(readiness.get(key))
    except (TypeError, ValueError):
        errors.append(f"fresh readiness {key} is not an integer: {readiness.get(key)!r}")
        return
    if count != 0:
        errors.append(f"fresh readiness {key} is not zero: {count}")


def readiness_gate_errors(readiness: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if readiness.get("status") != "ok":
        errors.append("fresh readiness status is not ok")
    if readiness.get("errors"):
        errors.append("fresh readiness contains errors")
    require_zero_readiness_count(readiness, "bundle_audit_source_drift_count", errors)
    require_zero_readiness_count(readiness, "loaded_model_replay_mismatch_count", errors)
    return errors


def gate_errors(registration: Mapping[str, Any], readiness: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    errors.extend(registration_gate_errors(registration))
    errors.extend(readiness_gate_errors(readiness))
    if not rows:
        errors.append("fresh domain-routed outputs contain zero rows")
    return errors


def build_fresh_confirmation_report(
    *,
    locked_policy_manifest: Path,
    fresh_split_registration: Path,
    fresh_readiness: Path,
    fresh_domain_routed_outputs: Path,
    repo_root: Path,
    topk: int = 50,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    manifest_path = resolve_path(repo_root, locked_policy_manifest)
    registration_path = resolve_json_input(repo_root, fresh_split_registration, "fresh_split_registration.json")
    readiness_path = resolve_json_input(repo_root, fresh_readiness, "fresh_readiness.json")
    outputs_path = resolve_path(repo_root, fresh_domain_routed_outputs)
    manifest = read_json(manifest_path)
    registration = read_json(registration_path)
    readiness = read_json(readiness_path)
    rows = read_output_rows(outputs_path)
    validation_metric = dict(manifest.get("validation_metrics", {}))
    fresh_metric = metric_with_hits(rows, topk=topk)
    errors = gate_errors(registration, readiness, rows)
    report = {
        "name": "H5FreshConfirmationReport",
        "created_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "status": "ok" if not errors else "failed",
        "claim_boundary": CLAIM_BOUNDARY,
        "locked_policy_name": manifest.get("name"),
        "locked_policy_manifest": repo_relative(manifest_path, repo_root),
        "fresh_split_registration": {
            "path": repo_relative(registration_path, repo_root),
            "status": registration.get("status"),
            "fresh_split_id": registration.get("fresh_split_id"),
            "split_manifest": registration.get("split_manifest"),
        },
        "fresh_readiness": {
            "path": repo_relative(readiness_path, repo_root),
            "status": readiness.get("status"),
        },
        "fresh_domain_routed_outputs": {
            "path": repo_relative(outputs_path, repo_root),
            "sha256": sha256_file(outputs_path),
            "row_count": len(rows),
        },
        "locked_validation_metric": validation_metric,
        "fresh_confirmation_metric": fresh_metric,
        "fresh_confirmation_domain_metrics": domain_metrics(rows, topk=topk),
        "fresh_minus_validation": metric_deltas(fresh_metric, validation_metric, topk=topk),
        "errors": errors,
        "next_target": (
            "If status is ok, archive this report with the registration/readiness artifacts and update "
            "paper-facing summaries without merging validation and fresh metrics; if status is failed, stop "
            "and fix the failed gate before reporting fresh confirmation results."
        ),
    }
    return report


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def metric_lines(metric: Mapping[str, Any], topk: int) -> list[str]:
    return [
        f"- sample_count: `{format_value(metric.get('sample_count'))}`",
        f"- hits_at_{topk}: `{format_value(metric.get(f'hits_at_{topk}'))}`",
        f"- Recall@{topk}: `{format_value(metric.get(f'Recall@{topk}'))}`",
        f"- CandidatePoolHitRate: `{format_value(metric.get('CandidatePoolHitRate'))}`",
        f"- ConditionalRecall@50GivenPoolHit: `{format_value(metric.get('ConditionalRecall@50GivenPoolHit'))}`",
    ]


def render_report(report: Mapping[str, Any], topk: int) -> str:
    registration = report["fresh_split_registration"]
    split_manifest = registration.get("split_manifest")
    if not isinstance(split_manifest, Mapping):
        split_manifest = {}
    lines = [
        "# H5-D Fresh Confirmation Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Claim Boundary",
        "",
        str(report["claim_boundary"]),
        "",
        "## Inputs",
        "",
        f"- locked policy: `{report['locked_policy_name']}`",
        f"- locked manifest: `{report['locked_policy_manifest']}`",
        f"- fresh registration: `{registration['path']}` status=`{registration['status']}`",
        f"- fresh split id: `{registration['fresh_split_id']}`",
        f"- fresh split manifest: `{format_value(split_manifest.get('path'))}`",
        f"- fresh split manifest sha256: `{format_value(split_manifest.get('sha256'))}`",
        f"- readiness: `{report['fresh_readiness']['path']}` status=`{report['fresh_readiness']['status']}`",
        f"- fresh outputs: `{report['fresh_domain_routed_outputs']['path']}`",
        f"- fresh outputs sha256: `{report['fresh_domain_routed_outputs']['sha256']}`",
        "",
        "## Locked Validation Metric",
        "",
        *metric_lines(report["locked_validation_metric"], topk=topk),
        "",
        "## Fresh Confirmation Metric",
        "",
        *metric_lines(report["fresh_confirmation_metric"], topk=topk),
        "",
        "## Fresh Domain Metrics",
        "",
        "| domain | sample_count | hits | Recall@50 | CandidatePoolHitRate |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain, metric in report["fresh_confirmation_domain_metrics"].items():
        lines.append(
            f"| {domain} | {format_value(metric.get('sample_count'))} | "
            f"{format_value(metric.get(f'hits_at_{topk}'))} | "
            f"{format_value(metric.get(f'Recall@{topk}'))} | "
            f"{format_value(metric.get('CandidatePoolHitRate'))} |"
        )
    lines.extend(["", "## Fresh Minus Validation", ""])
    for key, value in report["fresh_minus_validation"].items():
        lines.append(f"- {key}: `{value:.6f}`")
    if report.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    lines.extend(["", "## Next Target", "", str(report["next_target"]), ""])
    return "\n".join(lines)


def write_report(output_dir: Path, report: Mapping[str, Any], topk: int) -> None:
    ensure_empty_output_dir(output_dir)
    write_json(output_dir / "fresh_confirmation_report.json", report)
    (output_dir / "fresh_confirmation_report.md").write_text(render_report(report, topk=topk), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a gated H5-D fresh confirmation report.")
    parser.add_argument("--locked-policy-manifest", required=True)
    parser.add_argument("--fresh-split-registration", required=True)
    parser.add_argument("--fresh-readiness", required=True)
    parser.add_argument("--fresh-domain-routed-outputs", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    report = build_fresh_confirmation_report(
        locked_policy_manifest=Path(args.locked_policy_manifest),
        fresh_split_registration=Path(args.fresh_split_registration),
        fresh_readiness=Path(args.fresh_readiness),
        fresh_domain_routed_outputs=Path(args.fresh_domain_routed_outputs),
        repo_root=repo_root,
        topk=int(args.topk),
    )
    output_dir = resolve_path(repo_root, args.output_dir)
    write_report(output_dir, report, topk=int(args.topk))
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output_dir),
                f"fresh_Recall@{args.topk}": report["fresh_confirmation_metric"][f"Recall@{args.topk}"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
