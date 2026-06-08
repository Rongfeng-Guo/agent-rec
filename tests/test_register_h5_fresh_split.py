from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "register_h5_fresh_split.py"
    spec = importlib.util.spec_from_file_location("register_h5_fresh_split", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_registration_fixture(tmp_path: Path) -> dict[str, Path]:
    consumed = tmp_path / "outputs" / "consumed" / "split_manifest.json"
    fresh = tmp_path / "outputs" / "fresh" / "split_manifest.json"
    locked = tmp_path / "experiments" / "h5" / "locked_policy_manifest.json"
    audit = tmp_path / "outputs" / "bundle_audit" / "bundle_audit.json"
    write_json(consumed, {"name": "consumed-v3", "sample_ids": ["a", "b"], "fresh_status": "consumed", "consumed": True})
    write_json(fresh, {"name": "fresh-v4", "sample_ids": ["c", "d"], "fresh_status": "fresh", "consumed": False})
    write_json(locked, {"protocol": {"manifest": "outputs/consumed/split_manifest.json"}})
    write_json(
        audit,
        {
            "status": "ok",
            "errors": [],
            "source_drift": [],
            "rerun_validator": {"status": "ok"},
        },
    )
    return {"consumed": consumed, "fresh": fresh, "locked": locked, "audit": audit}


def test_registers_fresh_split_when_audit_passes(tmp_path) -> None:
    module = load_module()
    paths = make_registration_fixture(tmp_path)

    registration = module.build_registration(
        split_manifest=paths["fresh"],
        locked_policy_manifest=paths["locked"],
        bundle_audit=paths["audit"].parent,
        repo_root=tmp_path,
        fresh_split_id="fresh-20260608-a",
        operator_note="synthetic fresh split for test",
        required_manifest_fields=["fresh_status=fresh", "consumed=false"],
    )

    assert registration["status"] == "ok"
    assert registration["split_manifest"]["sha256"] != registration["consumed_protocol_v3_split"]["sha256"]
    assert registration["required_manifest_field_count"] == 2
    assert registration["required_manifest_field_ok_count"] == 2
    assert registration["consumed_manifest_path_match"] is False
    assert registration["consumed_manifest_sha256_match"] is False
    assert registration["audit_error_count"] == 0

    rendered = module.render_report(registration)
    assert "## Gate Summary" in rendered
    assert "required_manifest_field_ok_count" in rendered


def test_rejects_consumed_split_by_path_and_hash(tmp_path) -> None:
    module = load_module()
    paths = make_registration_fixture(tmp_path)

    registration = module.build_registration(
        split_manifest=paths["consumed"],
        locked_policy_manifest=paths["locked"],
        bundle_audit=paths["audit"],
        repo_root=tmp_path,
        fresh_split_id="fresh-20260608-a",
        operator_note="should reject consumed split",
        required_manifest_fields=["fresh_status=fresh", "consumed=false"],
    )

    assert registration["status"] == "rejected"
    assert registration["consumed_manifest_path_match"] is True
    assert registration["consumed_manifest_sha256_match"] is True
    assert any("path matches" in error for error in registration["errors"])
    assert any("SHA-256 matches" in error for error in registration["errors"])


def test_rejects_split_when_bundle_audit_has_source_drift(tmp_path) -> None:
    module = load_module()
    paths = make_registration_fixture(tmp_path)
    write_json(
        paths["audit"],
        {
            "status": "failed",
            "errors": ["Source drift detected"],
            "source_drift": [{"source_path": "research-log.md"}],
            "rerun_validator": {"status": "ok"},
        },
    )

    registration = module.build_registration(
        split_manifest=paths["fresh"],
        locked_policy_manifest=paths["locked"],
        bundle_audit=paths["audit"],
        repo_root=tmp_path,
        fresh_split_id="fresh-20260608-a",
        operator_note="audit should reject",
        required_manifest_fields=["fresh_status=fresh", "consumed=false"],
    )

    assert registration["status"] == "rejected"
    assert any("audit status" in error for error in registration["errors"])
    assert any("source_drift" in error for error in registration["errors"])


def test_rejects_split_without_required_manifest_fields(tmp_path) -> None:
    module = load_module()
    paths = make_registration_fixture(tmp_path)

    registration = module.build_registration(
        split_manifest=paths["fresh"],
        locked_policy_manifest=paths["locked"],
        bundle_audit=paths["audit"],
        repo_root=tmp_path,
        fresh_split_id="fresh-20260608-a",
        operator_note="missing field requirements should reject",
    )

    assert registration["status"] == "rejected"
    assert any("require-manifest-field" in error for error in registration["errors"])


def test_rejects_split_when_required_manifest_field_mismatches(tmp_path) -> None:
    module = load_module()
    paths = make_registration_fixture(tmp_path)

    registration = module.build_registration(
        split_manifest=paths["fresh"],
        locked_policy_manifest=paths["locked"],
        bundle_audit=paths["audit"],
        repo_root=tmp_path,
        fresh_split_id="fresh-20260608-a",
        operator_note="field mismatch should reject",
        required_manifest_fields=["fresh_status=fresh", "consumed=true"],
    )

    assert registration["status"] == "rejected"
    assert registration["required_manifest_field_count"] == 2
    assert registration["required_manifest_field_ok_count"] == 1
    assert any("mismatch" in error for error in registration["errors"])
