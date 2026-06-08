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

from scripts.oracle_route_memory.validate_locked_policy_manifest import validate_manifest

try:
    from scripts.oracle_route_memory.handoff_io import ensure_empty_output_dir, resolve_path_under_repo_root as resolve_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from handoff_io import ensure_empty_output_dir, resolve_path_under_repo_root as resolve_path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def replay_errors(replay: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if replay.get("status") != "ok":
        errors.append("loaded-model replay validation status is not ok")
    if int(replay.get("mismatch_count", 0)) != 0:
        errors.append(f"loaded-model replay mismatch_count is {replay.get('mismatch_count')!r}")
    if replay.get("metric_errors"):
        errors.append("loaded-model replay validation contains metric_errors")
    return errors


def component_model_checks(manifest: Mapping[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name, component in sorted(manifest.get("component_rankers", {}).items()):
        output_dir = resolve_path(repo_root, component["output_dir"])
        model_path = output_dir / "model.pkl"
        checks.append(
            {
                "name": name,
                "output_dir": str(output_dir),
                "model_path": str(model_path),
                "exists": model_path.exists(),
            }
        )
    return checks


def check_readiness(
    *,
    manifest_path: Path,
    bundle_audit_path: Path,
    replay_validation_path: Path,
    repo_root: Path,
    topk: int = 50,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    manifest_path = resolve_path(repo_root, manifest_path)
    bundle_audit_path = resolve_path(repo_root, bundle_audit_path)
    replay_validation_path = resolve_path(repo_root, replay_validation_path)
    manifest = read_json(manifest_path)
    manifest_validation = validate_manifest(manifest_path, repo_root=repo_root, topk=topk)
    audit = read_json(bundle_audit_path)
    replay = read_json(replay_validation_path)
    model_checks = component_model_checks(manifest, repo_root)
    errors: list[str] = []
    errors.extend(audit_errors(audit))
    errors.extend(replay_errors(replay))
    missing_models = [check for check in model_checks if not check["exists"]]
    if missing_models:
        errors.append(f"missing component model files: {[check['name'] for check in missing_models]}")
    component_model_count = len(model_checks)
    missing_component_model_count = len(missing_models)
    return {
        "status": "ok" if not errors else "failed",
        "manifest": str(manifest_path),
        "bundle_audit": str(bundle_audit_path),
        "loaded_model_replay_validation": str(replay_validation_path),
        "manifest_validation": manifest_validation,
        "bundle_audit_status": audit.get("status"),
        "bundle_audit_source_drift_count": len(audit.get("source_drift", [])),
        "loaded_model_replay_status": replay.get("status"),
        "loaded_model_replay_mismatch_count": int(replay.get("mismatch_count", 0)),
        "component_model_count": component_model_count,
        "missing_component_model_count": missing_component_model_count,
        "component_model_checks": model_checks,
        "errors": errors,
        "next_target": (
            "If status is ok, wait for a clearly fresh/unconsumed split, register its manifest path/SHA-256, "
            "then score with locked model.pkl files without retraining."
        ),
    }


def render_report(result: Mapping[str, Any]) -> str:
    metric = result["manifest_validation"]["metric"]
    lines = [
        "# H5-D Fresh Confirmation Readiness",
        "",
        f"Status: `{result['status']}`",
        "",
        "## Locked Validation",
        "",
        f"- sample_count: `{metric['sample_count']}`",
        f"- hits_at_50: `{metric['hits_at_50']}`",
        f"- Recall@50: `{metric['Recall@50']:.6f}`",
        f"- CandidatePoolHitRate: `{metric['CandidatePoolHitRate']:.6f}`",
        "",
        "## Gates",
        "",
        f"- bundle_audit_status: `{result['bundle_audit_status']}`",
        f"- bundle_audit_source_drift_count: `{result['bundle_audit_source_drift_count']}`",
        f"- loaded_model_replay_status: `{result['loaded_model_replay_status']}`",
        f"- loaded_model_replay_mismatch_count: `{result['loaded_model_replay_mismatch_count']}`",
        f"- component_model_count: `{result.get('component_model_count', len(result['component_model_checks']))}`",
        f"- missing_component_model_count: `{result.get('missing_component_model_count', 0)}`",
        "",
        "## Component Models",
        "",
    ]
    for check in result["component_model_checks"]:
        lines.append(f"- `{check['name']}`: `{check['model_path']}` exists=`{check['exists']}`")
    if result.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in result["errors"])
    lines.extend(["", "## Next Target", "", str(result["next_target"]), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check H5-D readiness before a future fresh confirmation split.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bundle-audit", required=True)
    parser.add_argument("--loaded-model-replay-validation", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    result = check_readiness(
        manifest_path=Path(args.manifest),
        bundle_audit_path=Path(args.bundle_audit),
        replay_validation_path=Path(args.loaded_model_replay_validation),
        repo_root=repo_root,
        topk=int(args.topk),
    )
    output_dir = resolve_path(repo_root, args.output_dir)
    ensure_empty_output_dir(output_dir)
    write_json(output_dir / "fresh_readiness.json", result)
    (output_dir / "fresh_readiness.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
