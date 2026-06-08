#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.oracle_route_memory.prepare_h5_fresh_confirmation_bundle import sha256_file, write_json
from scripts.oracle_route_memory.validate_locked_policy_manifest import validate_manifest

try:
    from scripts.oracle_route_memory.handoff_io import ensure_empty_output_dir, resolve_repo_path as resolve_optional_repo_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from handoff_io import ensure_empty_output_dir, resolve_repo_path as resolve_optional_repo_path

NEXT_TARGET = (
    "Keep this audit with the immutable prep bundle. Before any future confirmation run, record the fresh "
    "split manifest path and hash, then rerun this audit with --rerun-validator and fail on any source drift."
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_close(name: str, actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(f"{name} mismatch: actual={actual!r} expected={expected!r}")


def compare_metric_dict(actual: Mapping[str, Any], expected: Mapping[str, Any], *, prefix: str = "metric") -> list[str]:
    errors: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            errors.append(f"{prefix}.{key} missing from actual metric")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, float):
            try:
                assert_close(f"{prefix}.{key}", float(actual_value), expected_value)
            except ValueError as exc:
                errors.append(str(exc))
        elif actual_value != expected_value:
            errors.append(f"{prefix}.{key} mismatch: actual={actual_value!r} expected={expected_value!r}")
    return errors


def audit_artifacts(bundle_dir: Path, bundle_manifest: Mapping[str, Any], repo_root: Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    for artifact in bundle_manifest.get("artifacts", []):
        bundle_path = bundle_dir / str(artifact["bundle_path"])
        check: dict[str, Any] = {
            "source_path": str(artifact["source_path"]),
            "bundle_path": str(artifact["bundle_path"]),
            "expected_sha256": str(artifact["sha256"]),
            "bundle_exists": bundle_path.exists(),
        }
        if not bundle_path.exists():
            errors.append(f"Missing bundled artifact: {bundle_path}")
            check["status"] = "missing"
            checks.append(check)
            continue
        actual_hash = sha256_file(bundle_path)
        check["actual_sha256"] = actual_hash
        if actual_hash != str(artifact["sha256"]):
            errors.append(f"Bundled artifact hash mismatch: {artifact['bundle_path']}")
            check["status"] = "hash_mismatch"
        else:
            check["status"] = "ok"
        if repo_root is not None:
            source_path = repo_root / str(artifact["source_path"])
            check["source_exists"] = source_path.exists()
            if source_path.exists():
                source_hash = sha256_file(source_path)
                check["source_sha256"] = source_hash
                check["source_matches_bundle"] = source_hash == str(artifact["sha256"])
            else:
                check["source_matches_bundle"] = None
        checks.append(check)
    return checks, errors


def audit_bundle(
    bundle_dir: Path,
    *,
    repo_root: Path | None = None,
    topk: int = 50,
    rerun_validator: bool = False,
    fail_on_source_drift: bool = False,
) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "bundle_manifest.json"
    validator_path = bundle_dir / "validator_output.json"
    errors: list[str] = []
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not validator_path.exists():
        raise FileNotFoundError(validator_path)

    bundle_manifest = read_json(manifest_path)
    validator_output = read_json(validator_path)
    artifact_checks, artifact_errors = audit_artifacts(bundle_dir, bundle_manifest, repo_root.resolve() if repo_root else None)
    errors.extend(artifact_errors)

    manifest_validation = bundle_manifest.get("validation", {})
    if validator_output.get("status") != "ok":
        errors.append("validator_output.json does not report status=ok")
    errors.extend(compare_metric_dict(validator_output.get("metric", {}), manifest_validation.get("metric", {}), prefix="validator.metric"))

    rerun_result: dict[str, Any] | None = None
    if rerun_validator:
        if repo_root is None:
            errors.append("rerun_validator requires repo_root")
        else:
            locked_manifest = repo_root.resolve() / str(bundle_manifest["locked_manifest"])
            rerun_result = validate_manifest(locked_manifest, repo_root=repo_root.resolve(), topk=topk)
            errors.extend(compare_metric_dict(rerun_result.get("metric", {}), manifest_validation.get("metric", {}), prefix="rerun.metric"))

    source_drift = [
        check
        for check in artifact_checks
        if check.get("source_matches_bundle") is False
    ]
    if fail_on_source_drift and source_drift:
        errors.append(f"Source drift detected for {len(source_drift)} bundled artifact(s)")
    return {
        "status": "ok" if not errors else "failed",
        "bundle_dir": str(bundle_dir),
        "locked_policy_name": bundle_manifest.get("locked_policy_name"),
        "claim_boundary": bundle_manifest.get("claim_boundary"),
        "guardrails": bundle_manifest.get("guardrails", []),
        "metric": manifest_validation.get("metric", {}),
        "domain_results": manifest_validation.get("domain_results", {}),
        "artifact_check_count": len(artifact_checks),
        "artifact_checks": artifact_checks,
        "source_drift_count": len(source_drift),
        "source_drift": source_drift,
        "rerun_validator": rerun_result,
        "errors": errors,
        "next_target": NEXT_TARGET,
    }


def render_report(audit: Mapping[str, Any]) -> str:
    metric = audit.get("metric", {})
    lines = [
        "# H5-D Prep Bundle Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "## Locked Validation Metric",
        "",
        f"- sample_count: `{metric.get('sample_count')}`",
        f"- hits_at_50: `{metric.get('hits_at_50')}`",
        f"- Recall@50: `{float(metric.get('Recall@50', 0.0)):.6f}`",
        f"- CandidatePoolHitRate: `{float(metric.get('CandidatePoolHitRate', 0.0)):.6f}`",
        "",
        "## Gate Summary",
        "",
        f"- artifact_check_count: `{audit.get('artifact_check_count', len(audit.get('artifact_checks', [])))}`",
        f"- source_drift_count: `{audit.get('source_drift_count', len(audit.get('source_drift', [])))}`",
        f"- error_count: `{len(audit.get('errors', []))}`",
        "",
        "## Artifact Checks",
        "",
    ]
    for check in audit.get("artifact_checks", []):
        drift = " source-drift" if check.get("source_matches_bundle") is False else ""
        lines.append(f"- `{check['bundle_path']}`: `{check['status']}`{drift}")
    if audit.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in audit["errors"])
    if audit.get("source_drift"):
        lines.extend(["", "## Source Drift", ""])
        lines.extend(f"- `{check['source_path']}`" for check in audit["source_drift"])
    lines.extend(
        [
            "",
            "## Next Target",
            "",
            str(audit.get("next_target", NEXT_TARGET)),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an H5-D fresh-confirmation prep bundle.")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--rerun-validator", action="store_true")
    parser.add_argument("--fail-on-source-drift", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    bundle_dir = resolve_optional_repo_path(args.bundle_dir, repo_root)
    audit = audit_bundle(
        bundle_dir,
        repo_root=repo_root,
        topk=int(args.topk),
        rerun_validator=bool(args.rerun_validator),
        fail_on_source_drift=bool(args.fail_on_source_drift),
    )
    if args.output_dir:
        output_dir = resolve_optional_repo_path(args.output_dir, repo_root)
        ensure_empty_output_dir(output_dir)
        write_json(output_dir / "bundle_audit.json", audit)
        (output_dir / "bundle_audit.md").write_text(render_report(audit), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
