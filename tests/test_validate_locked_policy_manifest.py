from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "validate_locked_policy_manifest.py"
    spec = importlib.util.spec_from_file_location("validate_locked_policy_manifest", module_path)
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


def test_validate_manifest_recomputes_metrics(tmp_path) -> None:
    module = load_module()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    rows = [
        row("s1", "Book", 4),
        row("s2", "Game", 80),
    ]
    (output_dir / "cold_like_outputs.json").write_text(json.dumps(rows), encoding="utf-8")
    (output_dir / "summary.json").write_text("{}", encoding="utf-8")
    (output_dir / "analyzer").mkdir()
    (tmp_path / "candidate_export").mkdir()
    (tmp_path / "component").mkdir()
    (tmp_path / "component" / "cold_like_outputs.json").write_text(json.dumps(rows), encoding="utf-8")
    manifest = {
        "candidate_export": {
            "output_dir": "candidate_export"
        },
        "component_rankers": {
            "component": {
                "output_dir": "component",
                "cold_like_outputs": "component/cold_like_outputs.json"
            }
        },
        "selected_outputs": {
            "output_dir": "out",
            "cold_like_outputs": "out/cold_like_outputs.json",
            "summary": "out/summary.json",
            "analyzer": "out/analyzer"
        },
        "validation_metrics": {
            "sample_count": 2,
            "hits_at_50": 1,
            "Recall@50": 0.5,
            "CandidatePoolHitRate": 1.0,
            "ConditionalRecall@50GivenPoolHit": 0.5,
            "domain_metrics": {
                "Book": {
                    "sample_count": 1,
                    "hits_at_50": 1,
                    "Recall@50": 1.0
                },
                "Game": {
                    "sample_count": 1,
                    "hits_at_50": 0,
                    "Recall@50": 0.0
                }
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = module.validate_manifest(manifest_path, repo_root=tmp_path, topk=50)

    assert result["status"] == "ok"
    assert "Validation-only" in result["claim_boundary"]
    assert result["metric"]["hits_at_50"] == 1
    assert result["domain_results"]["Book"]["Recall@50"] == 1.0
    assert result["component_model_checks"] == [
        {
            "name": "component",
            "output_dir": str(tmp_path / "component"),
            "model_path": str(tmp_path / "component" / "model.pkl"),
            "exists": False,
        }
    ]
    assert "fresh metrics separate" in result["next_target"]
