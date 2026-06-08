from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_module(name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(sample_id: str, domain: str, rank: int | None) -> dict:
    return {
        "sample_id": sample_id,
        "domain": domain,
        "target_item_id": f"target-{sample_id}",
        "match_rank": rank,
        "route_hit": True,
        "candidate_pool_hit": rank is not None,
        "candidate_pool_match_rank": rank,
    }


def write_minimal_locked_policy(repo_root: Path) -> Path:
    output_dir = repo_root / "out"
    output_dir.mkdir()
    rows = [row("s1", "Book", 4), row("s2", "Game", 80)]
    (output_dir / "cold_like_outputs.json").write_text(json.dumps(rows), encoding="utf-8")
    (output_dir / "summary.json").write_text("{}", encoding="utf-8")
    (output_dir / "analyzer").mkdir()

    (repo_root / "candidate_export").mkdir()
    (repo_root / "component").mkdir()
    (repo_root / "component" / "cold_like_outputs.json").write_text(json.dumps(rows), encoding="utf-8")
    doc_path = repo_root / "doc.md"
    doc_path.write_text("# Bundle doc\n", encoding="utf-8")

    manifest = {
        "name": "h5_pairwise_domain_routed_test",
        "claim_boundary": "Validation-only locked test policy.",
        "candidate_export": {"output_dir": "candidate_export"},
        "component_rankers": {
            "component": {
                "output_dir": "component",
                "cold_like_outputs": "component/cold_like_outputs.json",
            }
        },
        "selected_outputs": {
            "output_dir": "out",
            "cold_like_outputs": "out/cold_like_outputs.json",
            "summary": "out/summary.json",
            "analyzer": "out/analyzer",
        },
        "validation_metrics": {
            "sample_count": 2,
            "hits_at_50": 1,
            "Recall@50": 0.5,
            "CandidatePoolHitRate": 1.0,
            "ConditionalRecall@50GivenPoolHit": 0.5,
            "domain_metrics": {
                "Book": {"sample_count": 1, "hits_at_50": 1, "Recall@50": 1.0},
                "Game": {"sample_count": 1, "hits_at_50": 0, "Recall@50": 0.0},
            },
        },
    }
    manifest_path = repo_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def make_bundle(tmp_path: Path) -> Path:
    prepare = load_module("prepare_h5_fresh_confirmation_bundle", "scripts/oracle_route_memory/prepare_h5_fresh_confirmation_bundle.py")
    manifest_path = write_minimal_locked_policy(tmp_path)
    bundle_dir = tmp_path / "bundle"
    prepare.build_bundle(manifest_path, repo_root=tmp_path, output_dir=bundle_dir, artifacts=[tmp_path / "doc.md"], topk=50)
    return bundle_dir


def test_audit_bundle_validates_hashes_and_reruns_manifest(tmp_path) -> None:
    module = load_module("audit_h5_fresh_confirmation_bundle", "scripts/oracle_route_memory/audit_h5_fresh_confirmation_bundle.py")
    bundle_dir = make_bundle(tmp_path)

    audit = module.audit_bundle(bundle_dir, repo_root=tmp_path, topk=50, rerun_validator=True, fail_on_source_drift=True)

    assert audit["status"] == "ok"
    assert audit["metric"]["hits_at_50"] == 1
    assert audit["rerun_validator"]["metric"]["Recall@50"] == 0.5
    assert audit["artifact_check_count"] == len(audit["artifact_checks"])
    assert audit["source_drift_count"] == 0
    assert "rerun this audit" in audit["next_target"]
    assert not audit["source_drift"]

    rendered = module.render_report(audit)
    assert "## Gate Summary" in rendered
    assert "artifact_check_count" in rendered


def test_audit_bundle_reports_tampered_artifact(tmp_path) -> None:
    module = load_module("audit_h5_fresh_confirmation_bundle", "scripts/oracle_route_memory/audit_h5_fresh_confirmation_bundle.py")
    bundle_dir = make_bundle(tmp_path)
    copied_doc = bundle_dir / "artifacts" / "doc.md"
    copied_doc.write_text("tampered\n", encoding="utf-8")

    audit = module.audit_bundle(bundle_dir, repo_root=tmp_path, topk=50, rerun_validator=False)

    assert audit["status"] == "failed"
    assert any("hash mismatch" in error for error in audit["errors"])



def test_audit_bundle_can_fail_on_source_drift(tmp_path) -> None:
    module = load_module("audit_h5_fresh_confirmation_bundle", "scripts/oracle_route_memory/audit_h5_fresh_confirmation_bundle.py")
    bundle_dir = make_bundle(tmp_path)
    (tmp_path / "doc.md").write_text("source changed\n", encoding="utf-8")

    audit = module.audit_bundle(bundle_dir, repo_root=tmp_path, topk=50, rerun_validator=False, fail_on_source_drift=True)

    assert audit["status"] == "failed"
    assert audit["source_drift_count"] == 1
    assert audit["source_drift"]
    assert any("Source drift" in error for error in audit["errors"])


def test_main_resolves_relative_bundle_and_output_dirs_under_repo_root(tmp_path, monkeypatch) -> None:
    module = load_module("audit_h5_fresh_confirmation_bundle", "scripts/oracle_route_memory/audit_h5_fresh_confirmation_bundle.py")
    make_bundle(tmp_path)
    outside = tmp_path / "outside_cwd"
    outside.mkdir()
    monkeypatch.chdir(outside)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_h5_fresh_confirmation_bundle.py",
            "--bundle-dir",
            "bundle",
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            "outputs/audit",
            "--rerun-validator",
            "--fail-on-source-drift",
            "--topk",
            "50",
        ],
    )

    module.main()

    assert (tmp_path / "outputs" / "audit" / "bundle_audit.json").exists()
    assert not (outside / "outputs" / "audit" / "bundle_audit.json").exists()
