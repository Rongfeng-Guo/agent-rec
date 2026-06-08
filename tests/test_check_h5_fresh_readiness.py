from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "check_h5_fresh_readiness.py"
    spec = importlib.util.spec_from_file_location("check_h5_fresh_readiness", module_path)
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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_fixture(tmp_path: Path) -> dict[str, Path]:
    rows = [row("s1", "Book", 4), row("s2", "Game", 80)]
    out = tmp_path / "out"
    out.mkdir()
    write_json(out / "cold_like_outputs.json", rows)
    write_json(out / "summary.json", {})
    (out / "analyzer").mkdir()
    (tmp_path / "candidate_export").mkdir()
    component = tmp_path / "component"
    component.mkdir()
    write_json(component / "cold_like_outputs.json", rows)
    (component / "model.pkl").write_bytes(b"model")
    manifest = {
        "candidate_export": {"output_dir": "candidate_export"},
        "component_rankers": {
            "component": {"output_dir": "component", "cold_like_outputs": "component/cold_like_outputs.json"}
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
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    audit_path = tmp_path / "audit.json"
    write_json(audit_path, {"status": "ok", "errors": [], "source_drift": [], "rerun_validator": {"status": "ok"}})
    replay_path = tmp_path / "replay.json"
    write_json(replay_path, {"status": "ok", "mismatch_count": 0, "metric_errors": []})
    return {"manifest": manifest_path, "audit": audit_path, "replay": replay_path, "component_model": component / "model.pkl"}


def test_check_readiness_accepts_complete_gates(tmp_path) -> None:
    module = load_module()
    paths = make_fixture(tmp_path)

    result = module.check_readiness(
        manifest_path=paths["manifest"],
        bundle_audit_path=paths["audit"],
        replay_validation_path=paths["replay"],
        repo_root=tmp_path,
        topk=50,
    )

    assert result["status"] == "ok"
    assert result["manifest_validation"]["metric"]["Recall@50"] == 0.5
    assert result["component_model_count"] == 1
    assert result["missing_component_model_count"] == 0
    assert result["component_model_checks"][0]["exists"] is True

    rendered = module.render_report(result)
    assert "component_model_count" in rendered
    assert "missing_component_model_count" in rendered


def test_check_readiness_rejects_missing_model(tmp_path) -> None:
    module = load_module()
    paths = make_fixture(tmp_path)
    paths["component_model"].unlink()

    result = module.check_readiness(
        manifest_path=paths["manifest"],
        bundle_audit_path=paths["audit"],
        replay_validation_path=paths["replay"],
        repo_root=tmp_path,
        topk=50,
    )

    assert result["status"] == "failed"
    assert result["component_model_count"] == 1
    assert result["missing_component_model_count"] == 1
    assert any("missing component model" in error for error in result["errors"])


def test_main_resolves_relative_output_dir_under_repo_root(tmp_path, monkeypatch) -> None:
    module = load_module()
    paths = make_fixture(tmp_path)
    outside = tmp_path / "outside_cwd"
    outside.mkdir()
    monkeypatch.chdir(outside)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_h5_fresh_readiness.py",
            "--manifest",
            str(paths["manifest"]),
            "--bundle-audit",
            str(paths["audit"]),
            "--loaded-model-replay-validation",
            str(paths["replay"]),
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            "outputs/readiness",
            "--topk",
            "50",
        ],
    )

    module.main()

    assert (tmp_path / "outputs" / "readiness" / "fresh_readiness.json").exists()
    assert not (outside / "outputs" / "readiness" / "fresh_readiness.json").exists()
