from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "validate_h5_handoff_index.py"
    spec = importlib.util.spec_from_file_location("validate_h5_handoff_index", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def make_fixture(tmp_path: Path) -> dict[str, Path]:
    rows = [row("s1", "Book", 3), row("s2", "Game", 80)]
    write_json(tmp_path / "selected" / "cold_like_outputs.json", rows)
    write_json(tmp_path / "selected" / "summary.json", {})
    (tmp_path / "selected" / "analyzer").mkdir(parents=True)
    (tmp_path / "candidate_export").mkdir()
    write_json(tmp_path / "component" / "cold_like_outputs.json", rows)
    manifest = {
        "name": "h5_test_policy",
        "candidate_export": {"output_dir": "candidate_export"},
        "component_rankers": {
            "h100": {
                "output_dir": "component",
                "cold_like_outputs": "component/cold_like_outputs.json",
            }
        },
        "selected_outputs": {
            "output_dir": "selected",
            "cold_like_outputs": "selected/cold_like_outputs.json",
            "summary": "selected/summary.json",
            "analyzer": "selected/analyzer",
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
    write_json(tmp_path / "manifest.json", manifest)
    bundled_artifacts = [
        {"source_path": "manifest.json"},
        {"source_path": "doc.md"},
        {"source_path": "handoff_index.json"},
    ]
    write_json(tmp_path / "bundle" / "bundle_manifest.json", {"artifacts": bundled_artifacts})
    write_json(
        tmp_path / "audit" / "bundle_audit.json",
        {"status": "ok", "errors": [], "source_drift": [], "rerun_validator": {"status": "ok"}},
    )
    write_json(
        tmp_path / "readiness" / "fresh_readiness.json",
        {
            "status": "ok",
            "errors": [],
            "bundle_audit_source_drift_count": 0,
            "loaded_model_replay_mismatch_count": 0,
        },
    )
    script_path = tmp_path / "scripts" / "oracle_route_memory" / "render_h5_fresh_confirmation_report.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# renderer\n", encoding="utf-8")
    mentions = ["bundle", "audit", "readiness", "render_h5_fresh_confirmation_report.py"]
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("\n".join(mentions), encoding="utf-8")
    index = {
        "locked_policy_manifest": "manifest.json",
        "current_handoff": {
            "version": "test",
            "prep_bundle_dir": "bundle",
            "prep_bundle_manifest": "bundle/bundle_manifest.json",
            "prep_bundle_audit_dir": "audit",
            "prep_bundle_audit_json": "audit/bundle_audit.json",
            "fresh_readiness_dir": "readiness",
            "fresh_readiness_json": "readiness/fresh_readiness.json",
            "report_renderer": "scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py",
        },
        "doc_checks": [{"path": "doc.md", "required_mentions": mentions}],
        "bundle_required_artifacts": ["manifest.json", "doc.md", "handoff_index.json"],
        "next_target": "keep validating",
    }
    write_json(tmp_path / "handoff_index.json", index)
    return {"index": tmp_path / "handoff_index.json", "doc": doc_path, "audit": tmp_path / "audit" / "bundle_audit.json"}


def test_validate_handoff_index_accepts_consistent_fixture(tmp_path) -> None:
    module = load_module()
    paths = make_fixture(tmp_path)

    result = module.validate_handoff_index(handoff_index=paths["index"], repo_root=tmp_path, topk=50)

    assert result["status"] == "ok"
    assert result["manifest_validation"]["metric"]["Recall@50"] == 0.5
    assert result["bundle_artifact_count"] == 3
    assert result["bundle_artifact_included_count"] == 3
    assert result["doc_check_count"] == 1
    assert result["doc_check_ok_count"] == 1
    assert result["doc_checks"][0]["status"] == "ok"
    assert all(check["included"] for check in result["bundle_artifact_checks"])

    report = module.render_report(result)
    assert "- bundle_artifacts: `3`/`3` included" in report
    assert "- document_checks: `1`/`1` ok" in report


def test_validate_handoff_index_rejects_missing_doc_mention(tmp_path) -> None:
    module = load_module()
    paths = make_fixture(tmp_path)
    paths["doc"].write_text("bundle\naudit\n", encoding="utf-8")

    result = module.validate_handoff_index(handoff_index=paths["index"], repo_root=tmp_path, topk=50)

    assert result["status"] == "failed"
    assert any("doc check failed" in error for error in result["errors"])


def test_validate_handoff_index_rejects_audit_drift(tmp_path) -> None:
    module = load_module()
    paths = make_fixture(tmp_path)
    write_json(
        paths["audit"],
        {"status": "ok", "errors": [], "source_drift": [{"source_path": "doc.md"}], "rerun_validator": {"status": "ok"}},
    )

    result = module.validate_handoff_index(handoff_index=paths["index"], repo_root=tmp_path, topk=50)

    assert result["status"] == "failed"
    assert "bundle audit contains source_drift" in result["errors"]


def test_validate_handoff_index_rejects_missing_bundle_artifact(tmp_path) -> None:
    module = load_module()
    paths = make_fixture(tmp_path)
    index = json.loads(paths["index"].read_text(encoding="utf-8"))
    index["bundle_required_artifacts"].append("missing.md")
    write_json(paths["index"], index)

    result = module.validate_handoff_index(handoff_index=paths["index"], repo_root=tmp_path, topk=50)

    assert result["status"] == "failed"
    assert result["bundle_artifact_count"] == 4
    assert result["bundle_artifact_included_count"] == 3
    assert "bundle manifest is missing required artifact 'missing.md'" in result["errors"]
