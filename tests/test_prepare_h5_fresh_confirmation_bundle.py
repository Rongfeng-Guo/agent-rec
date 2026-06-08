from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "prepare_h5_fresh_confirmation_bundle.py"
    spec = importlib.util.spec_from_file_location("prepare_h5_fresh_confirmation_bundle", module_path)
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


def test_build_bundle_validates_and_copies_artifacts(tmp_path) -> None:
    module = load_module()
    manifest_path = write_minimal_locked_policy(tmp_path)
    doc_path = tmp_path / "doc.md"
    doc_path.write_text("# Locked policy doc\n", encoding="utf-8")

    bundle = module.build_bundle(
        manifest_path,
        repo_root=tmp_path,
        output_dir=Path("bundle"),
        artifacts=[doc_path],
        topk=50,
    )

    assert bundle["validation"]["metric"]["hits_at_50"] == 1
    assert bundle["artifact_count"] == len(bundle["artifacts"])
    assert (tmp_path / "bundle" / "validator_output.json").exists()
    assert (tmp_path / "bundle" / "artifacts" / "manifest.json").exists()
    assert (tmp_path / "bundle" / "artifacts" / "doc.md").read_text(encoding="utf-8") == "# Locked policy doc\n"
    readme = (tmp_path / "bundle" / "README.md").read_text(encoding="utf-8")
    assert "Validation-only locked test policy" in readme
    assert "consumed protocol-v3 blind-confirmation labels" in readme
    assert "artifact_count" in readme


def test_build_bundle_rejects_non_empty_output_dir(tmp_path) -> None:
    module = load_module()
    manifest_path = write_minimal_locked_policy(tmp_path)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        module.build_bundle(manifest_path, repo_root=tmp_path, output_dir=output_dir, artifacts=[], topk=50)
