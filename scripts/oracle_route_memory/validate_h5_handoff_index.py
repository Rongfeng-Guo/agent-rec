#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.oracle_route_memory.validate_locked_policy_manifest import validate_manifest

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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def path_check(name: str, path: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "name": name,
        "path": repo_relative(path, repo_root),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }


def audit_errors(audit: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if audit.get("status") != "ok":
        errors.append("bundle audit status is not ok")
    if audit.get("errors"):
        errors.append("bundle audit contains errors")
    if audit.get("source_drift"):
        errors.append("bundle audit contains source_drift")
    rerun = audit.get("rerun_validator")
    if not isinstance(rerun, Mapping) or rerun.get("status") != "ok":
        errors.append("bundle audit rerun_validator is missing or not ok")
    return errors


def readiness_errors(readiness: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if readiness.get("status") != "ok":
        errors.append("fresh readiness status is not ok")
    if readiness.get("errors"):
        errors.append("fresh readiness contains errors")
    if int(readiness.get("bundle_audit_source_drift_count", 0)) != 0:
        errors.append("fresh readiness bundle_audit_source_drift_count is not zero")
    if int(readiness.get("loaded_model_replay_mismatch_count", 0)) != 0:
        errors.append("fresh readiness loaded_model_replay_mismatch_count is not zero")
    return errors


def bundle_artifact_checks(bundle_manifest: Mapping[str, Any], required_artifacts: list[str]) -> list[dict[str, Any]]:
    bundled_sources = {
        str(artifact.get("source_path"))
        for artifact in bundle_manifest.get("artifacts", [])
        if isinstance(artifact, Mapping)
    }
    return [
        {
            "source_path": artifact,
            "included": artifact in bundled_sources,
        }
        for artifact in required_artifacts
    ]


def doc_check(repo_root: Path, item: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_path(repo_root, str(item["path"]))
    required_mentions = [str(value) for value in item.get("required_mentions", [])]
    if not path.exists():
        return {
            "path": repo_relative(path, repo_root),
            "exists": False,
            "missing_mentions": required_mentions,
            "status": "failed",
        }
    text = path.read_text(encoding="utf-8")
    missing = [mention for mention in required_mentions if mention not in text]
    return {
        "path": repo_relative(path, repo_root),
        "exists": True,
        "missing_mentions": missing,
        "status": "ok" if not missing else "failed",
    }


def require_current_key(current: Mapping[str, Any], key: str, errors: list[str]) -> str:
    value = current.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"current_handoff.{key} is required")
        return ""
    return value


def validate_handoff_index(
    *,
    handoff_index: Path,
    repo_root: Path,
    topk: int = 50,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    handoff_index = resolve_path(repo_root, handoff_index)
    index = read_json(handoff_index)
    current = index.get("current_handoff")
    if not isinstance(current, Mapping):
        raise ValueError("handoff index is missing current_handoff")

    errors: list[str] = []
    manifest_path = resolve_path(repo_root, str(index.get("locked_policy_manifest", "")))
    prep_bundle_dir = resolve_path(repo_root, require_current_key(current, "prep_bundle_dir", errors))
    prep_bundle_manifest = resolve_path(repo_root, require_current_key(current, "prep_bundle_manifest", errors))
    audit_dir = resolve_path(repo_root, require_current_key(current, "prep_bundle_audit_dir", errors))
    audit_json = resolve_path(repo_root, require_current_key(current, "prep_bundle_audit_json", errors))
    readiness_dir = resolve_path(repo_root, require_current_key(current, "fresh_readiness_dir", errors))
    readiness_json = resolve_path(repo_root, require_current_key(current, "fresh_readiness_json", errors))
    report_renderer = resolve_path(repo_root, require_current_key(current, "report_renderer", errors))

    checks = [
        path_check("handoff_index", handoff_index, repo_root),
        path_check("locked_policy_manifest", manifest_path, repo_root),
        path_check("prep_bundle_dir", prep_bundle_dir, repo_root),
        path_check("prep_bundle_manifest", prep_bundle_manifest, repo_root),
        path_check("prep_bundle_audit_dir", audit_dir, repo_root),
        path_check("prep_bundle_audit_json", audit_json, repo_root),
        path_check("fresh_readiness_dir", readiness_dir, repo_root),
        path_check("fresh_readiness_json", readiness_json, repo_root),
        path_check("report_renderer", report_renderer, repo_root),
    ]
    for check in checks:
        if not check["exists"]:
            errors.append(f"{check['name']} path is missing: {check['path']}")

    manifest_validation: dict[str, Any] | None = None
    if manifest_path.exists():
        manifest_validation = validate_manifest(manifest_path, repo_root=repo_root, topk=topk)
        if manifest_validation.get("status") != "ok":
            errors.append("locked manifest validation status is not ok")

    audit: Mapping[str, Any] = {}
    if audit_json.exists():
        audit = read_json(audit_json)
        errors.extend(audit_errors(audit))

    readiness: Mapping[str, Any] = {}
    if readiness_json.exists():
        readiness = read_json(readiness_json)
        errors.extend(readiness_errors(readiness))

    bundle_manifest: Mapping[str, Any] = {}
    artifact_checks: list[dict[str, Any]] = []
    required_artifacts = [str(value) for value in index.get("bundle_required_artifacts", [])]
    if not required_artifacts:
        errors.append("handoff index has no bundle_required_artifacts")
    if prep_bundle_manifest.exists():
        bundle_manifest = read_json(prep_bundle_manifest)
        artifact_checks = bundle_artifact_checks(bundle_manifest, required_artifacts)
        for check in artifact_checks:
            if not check["included"]:
                errors.append(f"bundle manifest is missing required artifact {check['source_path']!r}")

    doc_checks = [doc_check(repo_root, item) for item in index.get("doc_checks", [])]
    if not doc_checks:
        errors.append("handoff index has no doc_checks")
    for check in doc_checks:
        if check["status"] != "ok":
            errors.append(f"doc check failed for {check['path']}")

    bundle_artifact_included_count = sum(1 for check in artifact_checks if check["included"])
    doc_check_ok_count = sum(1 for check in doc_checks if check["status"] == "ok")

    return {
        "name": "H5HandoffIndexValidation",
        "created_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "status": "ok" if not errors else "failed",
        "handoff_index": repo_relative(handoff_index, repo_root),
        "current_handoff": dict(current),
        "path_checks": checks,
        "manifest_validation": manifest_validation,
        "bundle_audit_status": audit.get("status"),
        "bundle_audit_source_drift_count": len(audit.get("source_drift", [])),
        "bundle_artifact_count": len(artifact_checks),
        "bundle_artifact_included_count": bundle_artifact_included_count,
        "bundle_artifact_checks": artifact_checks,
        "fresh_readiness_status": readiness.get("status"),
        "fresh_readiness_errors": list(readiness.get("errors", [])),
        "doc_check_count": len(doc_checks),
        "doc_check_ok_count": doc_check_ok_count,
        "doc_checks": doc_checks,
        "errors": errors,
        "next_target": index.get(
            "next_target",
            "Keep the handoff index validator at status=ok after handoff doc or bundle version changes.",
        ),
    }


def render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# H5-D Handoff Index Validation",
        "",
        f"Status: `{result['status']}`",
        "",
        "## Current Handoff",
        "",
    ]
    for key, value in result["current_handoff"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- bundle_audit_status: `{result['bundle_audit_status']}`",
            f"- bundle_audit_source_drift_count: `{result['bundle_audit_source_drift_count']}`",
            f"- fresh_readiness_status: `{result['fresh_readiness_status']}`",
            (
                "- bundle_artifacts: "
                f"`{result.get('bundle_artifact_included_count', 0)}`/"
                f"`{result.get('bundle_artifact_count', 0)}` included"
            ),
            (
                "- document_checks: "
                f"`{result.get('doc_check_ok_count', 0)}`/"
                f"`{result.get('doc_check_count', 0)}` ok"
            ),
            "",
            "## Bundle Artifact Checks",
            "",
        ]
    )
    for check in result["bundle_artifact_checks"]:
        lines.append(f"- `{check['source_path']}` included=`{check['included']}`")
    lines.extend(
        [
            "",
            "## Document Checks",
            "",
        ]
    )
    for check in result["doc_checks"]:
        lines.append(f"- `{check['path']}`: `{check['status']}` missing={check['missing_mentions']}")
    if result.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in result["errors"])
    lines.extend(["", "## Next Target", "", str(result["next_target"]), ""])
    return "\n".join(lines)


def write_validation(output_dir: Path, result: Mapping[str, Any]) -> None:
    ensure_empty_output_dir(output_dir)
    write_json(output_dir / "handoff_index_validation.json", result)
    (output_dir / "handoff_index_validation.md").write_text(render_report(result), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate H5-D handoff index paths and document mentions.")
    parser.add_argument(
        "--handoff-index",
        default="experiments/h5-candidate-level-source-reranker/handoff_index.json",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    result = validate_handoff_index(
        handoff_index=Path(args.handoff_index),
        repo_root=repo_root,
        topk=int(args.topk),
    )
    output_dir = resolve_path(repo_root, args.output_dir)
    write_validation(output_dir, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
