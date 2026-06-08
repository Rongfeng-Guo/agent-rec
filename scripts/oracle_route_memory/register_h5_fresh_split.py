#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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


GUARDRAILS = (
    "Fresh split registration is metadata-only; do not score or inspect labels during registration.",
    "Reject a split manifest that is the same file or SHA-256 as the consumed protocol-v3 confirmation split.",
    "Require the H5-D prep bundle audit to pass before recording a fresh split for scoring.",
    "Require explicit fresh/unconsumed metadata fields in the split manifest before scoring.",
    "After registration, keep H5-D model, route, seed, hard-negative, and domain-routing parameters unchanged.",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_audit_json(path: Path) -> Path:
    return path / "bundle_audit.json" if path.is_dir() else path


def audit_gate_errors(audit: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if audit.get("status") != "ok":
        errors.append("bundle audit status is not ok")
    if audit.get("errors"):
        errors.append("bundle audit contains errors")
    if audit.get("source_drift"):
        errors.append("bundle audit contains source_drift")
    rerun = audit.get("rerun_validator")
    if not isinstance(rerun, Mapping) or rerun.get("status") != "ok":
        errors.append("bundle audit is missing a successful rerun_validator result")
    return errors


def parse_expected_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_required_field(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise ValueError(f"Expected KEY=VALUE for required manifest field, got {value!r}")
    key, expected = value.split("=", 1)
    if not key:
        raise ValueError(f"Missing key in required manifest field {value!r}")
    return key, parse_expected_value(expected)


def lookup_json_path(payload: Any, key_path: str) -> tuple[bool, Any]:
    current = payload
    for part in key_path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return False, None
    return True, current


def required_manifest_field_results(split_payload: Any, required_fields: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    results: list[dict[str, Any]] = []
    if not required_fields:
        errors.append("at least one --require-manifest-field KEY=VALUE is required to prove explicit fresh/unconsumed metadata")
        return results, errors
    for raw_field in required_fields:
        key_path, expected = parse_required_field(raw_field)
        exists, actual = lookup_json_path(split_payload, key_path)
        ok = exists and actual == expected
        results.append(
            {
                "field": key_path,
                "expected": expected,
                "actual": actual if exists else None,
                "exists": exists,
                "status": "ok" if ok else "mismatch",
            }
        )
        if not exists:
            errors.append(f"required split manifest field {key_path!r} is missing")
        elif actual != expected:
            errors.append(f"required split manifest field {key_path!r} mismatch: actual={actual!r} expected={expected!r}")
    return results, errors


def build_registration(
    *,
    split_manifest: Path,
    locked_policy_manifest: Path,
    bundle_audit: Path,
    repo_root: Path,
    fresh_split_id: str,
    operator_note: str,
    required_manifest_fields: list[str] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    split_manifest = resolve_path(repo_root, split_manifest)
    locked_policy_manifest = resolve_path(repo_root, locked_policy_manifest)
    bundle_audit = resolve_audit_json(resolve_path(repo_root, bundle_audit))
    errors: list[str] = []
    for path in (split_manifest, locked_policy_manifest, bundle_audit):
        if not path.exists():
            raise FileNotFoundError(path)

    locked_manifest = read_json(locked_policy_manifest)
    split_payload = read_json(split_manifest)
    required_field_results, required_field_errors = required_manifest_field_results(
        split_payload, list(required_manifest_fields or [])
    )
    errors.extend(required_field_errors)
    consumed_protocol_manifest = resolve_path(repo_root, locked_manifest["protocol"]["manifest"])
    split_hash = sha256_file(split_manifest)
    consumed_hash = sha256_file(consumed_protocol_manifest) if consumed_protocol_manifest.exists() else None
    consumed_manifest_path_match = split_manifest == consumed_protocol_manifest
    consumed_manifest_sha256_match = consumed_hash is not None and split_hash == consumed_hash
    required_manifest_field_count = len(required_field_results)
    required_manifest_field_ok_count = sum(1 for item in required_field_results if item.get("status") == "ok")
    audit = read_json(bundle_audit)
    errors.extend(audit_gate_errors(audit))

    if consumed_manifest_path_match:
        errors.append("fresh split manifest path matches the consumed protocol-v3 confirmation split")
    if consumed_manifest_sha256_match:
        errors.append("fresh split manifest SHA-256 matches the consumed protocol-v3 confirmation split")
    if not fresh_split_id.strip():
        errors.append("fresh_split_id is required")
    if not operator_note.strip():
        errors.append("operator_note is required")

    return {
        "status": "ok" if not errors else "rejected",
        "created_at_utc": utc_now(),
        "fresh_split_id": fresh_split_id,
        "operator_note": operator_note,
        "repo_root": str(repo_root),
        "split_manifest": {
            "path": repo_relative(split_manifest, repo_root),
            "sha256": split_hash,
        },
        "consumed_protocol_v3_split": {
            "path": repo_relative(consumed_protocol_manifest, repo_root),
            "sha256": consumed_hash,
        },
        "locked_policy_manifest": repo_relative(locked_policy_manifest, repo_root),
        "bundle_audit": repo_relative(bundle_audit, repo_root),
        "required_manifest_field_count": required_manifest_field_count,
        "required_manifest_field_ok_count": required_manifest_field_ok_count,
        "required_manifest_fields": required_field_results,
        "consumed_manifest_path_match": consumed_manifest_path_match,
        "consumed_manifest_sha256_match": consumed_manifest_sha256_match,
        "audit_status": audit.get("status"),
        "audit_source_drift_count": len(audit.get("source_drift", [])),
        "audit_error_count": len(audit.get("errors", [])),
        "guardrails": list(GUARDRAILS),
        "errors": errors,
        "next_target": (
            "If status is ok, score the registered fresh split with the locked H5-D pipeline only; "
            "if status is rejected, stop and do not score the split."
        ),
    }


def render_report(registration: Mapping[str, Any]) -> str:
    lines = [
        "# H5-D Fresh Split Registration",
        "",
        f"Status: `{registration['status']}`",
        "",
        "## Split Manifest",
        "",
        f"- id: `{registration['fresh_split_id']}`",
        f"- path: `{registration['split_manifest']['path']}`",
        f"- sha256: `{registration['split_manifest']['sha256']}`",
        "",
        "## Gate Summary",
        "",
        f"- required_manifest_field_count: `{registration.get('required_manifest_field_count', len(registration.get('required_manifest_fields', [])))}`",
        f"- required_manifest_field_ok_count: `{registration.get('required_manifest_field_ok_count', 0)}`",
        f"- consumed_manifest_path_match: `{registration.get('consumed_manifest_path_match')}`",
        f"- consumed_manifest_sha256_match: `{registration.get('consumed_manifest_sha256_match')}`",
        "",
        "## Required Manifest Fields",
        "",
    ]
    for field in registration.get("required_manifest_fields", []):
        lines.append(
            f"- `{field['field']}` expected `{field['expected']}` actual `{field['actual']}`: `{field['status']}`"
        )
    lines.extend(
        [
            "",
            "## Consumed Protocol-V3 Guard",
            "",
            f"- consumed path: `{registration['consumed_protocol_v3_split']['path']}`",
            f"- consumed sha256: `{registration['consumed_protocol_v3_split']['sha256']}`",
            "",
            "## Audit Gate",
            "",
            f"- bundle audit: `{registration['bundle_audit']}`",
            f"- audit_status: `{registration['audit_status']}`",
            f"- audit_error_count: `{registration['audit_error_count']}`",
            f"- audit_source_drift_count: `{registration['audit_source_drift_count']}`",
            "",
        ]
    )
    if registration.get("errors"):
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in registration["errors"])
        lines.append("")
    lines.extend(["## Next Target", "", str(registration["next_target"]), ""])
    return "\n".join(lines)


def write_registration(output_dir: Path, registration: Mapping[str, Any]) -> None:
    ensure_empty_output_dir(output_dir)
    write_json(output_dir / "fresh_split_registration.json", registration)
    (output_dir / "fresh_split_registration.md").write_text(render_report(registration), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a fresh H5-D confirmation split before scoring.")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--locked-policy-manifest", required=True)
    parser.add_argument("--bundle-audit", required=True, help="Path to bundle_audit.json or its containing directory.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--fresh-split-id", required=True)
    parser.add_argument("--operator-note", required=True)
    parser.add_argument("--require-manifest-field", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    registration = build_registration(
        split_manifest=Path(args.split_manifest),
        locked_policy_manifest=Path(args.locked_policy_manifest),
        bundle_audit=Path(args.bundle_audit),
        repo_root=repo_root,
        fresh_split_id=str(args.fresh_split_id),
        operator_note=str(args.operator_note),
        required_manifest_fields=[str(value) for value in args.require_manifest_field],
    )
    write_registration(resolve_path(repo_root, args.output_dir), registration)
    print(json.dumps(registration, indent=2, ensure_ascii=False))
    if registration["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
