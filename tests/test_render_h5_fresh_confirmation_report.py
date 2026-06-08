from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "render_h5_fresh_confirmation_report.py"
    spec = importlib.util.spec_from_file_location("render_h5_fresh_confirmation_report", module_path)
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
        "oracle_source_match_rank": rank,
    }


def make_fixture(tmp_path: Path) -> dict[str, Path]:
    manifest = {
        "name": "h5_pairwise_domain_routed_book_h100_game_h300",
        "validation_metrics": {
            "sample_count": 2,
            "hits_at_50": 1,
            "Recall@50": 0.5,
            "CandidatePoolHitRate": 1.0,
            "ConditionalRecall@50GivenPoolHit": 0.5,
        },
    }
    registration = {
        "status": "ok",
        "fresh_split_id": "fresh-20260608",
        "split_manifest": {"path": "fresh/split_manifest.json", "sha256": "abc123"},
        "required_manifest_fields": [
            {
                "field": "fresh_status",
                "expected": "fresh",
                "actual": "fresh",
                "exists": True,
                "status": "ok",
            },
            {
                "field": "consumed",
                "expected": False,
                "actual": False,
                "exists": True,
                "status": "ok",
            },
        ],
    }
    readiness = {
        "status": "ok",
        "errors": [],
        "bundle_audit_source_drift_count": 0,
        "loaded_model_replay_mismatch_count": 0,
    }
    rows = [row("s1", "Book", 2), row("s2", "Game", 60), row("s3", "Game", None)]
    paths = {
        "manifest": tmp_path / "locked_manifest.json",
        "registration": tmp_path / "registration" / "fresh_split_registration.json",
        "readiness": tmp_path / "readiness" / "fresh_readiness.json",
        "outputs": tmp_path / "fresh_outputs" / "cold_like_outputs.json",
    }
    write_json(paths["manifest"], manifest)
    write_json(paths["registration"], registration)
    write_json(paths["readiness"], readiness)
    write_json(paths["outputs"], rows)
    return paths


def test_build_report_separates_validation_and_fresh_metrics(tmp_path) -> None:
    module = load_module()
    paths = make_fixture(tmp_path)

    report = module.build_fresh_confirmation_report(
        locked_policy_manifest=paths["manifest"],
        fresh_split_registration=paths["registration"].parent,
        fresh_readiness=paths["readiness"].parent,
        fresh_domain_routed_outputs=paths["outputs"],
        repo_root=tmp_path,
        topk=50,
    )

    assert report["status"] == "ok"
    assert report["locked_validation_metric"]["Recall@50"] == 0.5
    assert report["fresh_confirmation_metric"]["sample_count"] == 3
    assert report["fresh_confirmation_metric"]["hits_at_50"] == 1
    assert report["fresh_confirmation_metric"]["Recall@50"] == 1 / 3
    assert report["fresh_confirmation_domain_metrics"]["Book"]["hits_at_50"] == 1
    assert report["fresh_minus_validation"]["Recall@50"] == (1 / 3) - 0.5

    rendered = module.render_report(report, topk=50)
    assert "Locked Validation Metric" in rendered
    assert "Fresh Confirmation Metric" in rendered
    assert "fresh split manifest sha256" in rendered


def test_build_report_fails_when_gates_are_not_ok(tmp_path) -> None:
    module = load_module()
    paths = make_fixture(tmp_path)
    write_json(paths["registration"], {"status": "rejected"})
    write_json(paths["readiness"], {"status": "failed"})

    report = module.build_fresh_confirmation_report(
        locked_policy_manifest=paths["manifest"],
        fresh_split_registration=paths["registration"],
        fresh_readiness=paths["readiness"],
        fresh_domain_routed_outputs=paths["outputs"],
        repo_root=tmp_path,
        topk=50,
    )

    assert report["status"] == "failed"
    assert "fresh split registration status is not ok" in report["errors"]
    assert "fresh readiness status is not ok" in report["errors"]


def test_build_report_rejects_incomplete_registration_evidence(tmp_path) -> None:
    module = load_module()
    paths = make_fixture(tmp_path)
    write_json(paths["registration"], {"status": "ok"})

    report = module.build_fresh_confirmation_report(
        locked_policy_manifest=paths["manifest"],
        fresh_split_registration=paths["registration"],
        fresh_readiness=paths["readiness"],
        fresh_domain_routed_outputs=paths["outputs"],
        repo_root=tmp_path,
        topk=50,
    )

    assert report["status"] == "failed"
    assert "fresh split registration is missing fresh_split_id" in report["errors"]
    assert "fresh split registration is missing split_manifest metadata" in report["errors"]
    assert "fresh split registration has no required_manifest_fields evidence" in report["errors"]


def test_build_report_rejects_incomplete_readiness_evidence(tmp_path) -> None:
    module = load_module()
    paths = make_fixture(tmp_path)
    write_json(
        paths["readiness"],
        {
            "status": "ok",
            "errors": [],
            "bundle_audit_source_drift_count": 1,
        },
    )

    report = module.build_fresh_confirmation_report(
        locked_policy_manifest=paths["manifest"],
        fresh_split_registration=paths["registration"],
        fresh_readiness=paths["readiness"],
        fresh_domain_routed_outputs=paths["outputs"],
        repo_root=tmp_path,
        topk=50,
    )

    assert report["status"] == "failed"
    assert "fresh readiness bundle_audit_source_drift_count is not zero: 1" in report["errors"]
    assert "fresh readiness loaded_model_replay_mismatch_count is missing" in report["errors"]


def test_main_writes_report_files(tmp_path, monkeypatch) -> None:
    module = load_module()
    paths = make_fixture(tmp_path)
    output_dir = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_h5_fresh_confirmation_report.py",
            "--locked-policy-manifest",
            str(paths["manifest"]),
            "--fresh-split-registration",
            str(paths["registration"].parent),
            "--fresh-readiness",
            str(paths["readiness"].parent),
            "--fresh-domain-routed-outputs",
            str(paths["outputs"]),
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--topk",
            "50",
        ],
    )

    module.main()

    payload = json.loads((output_dir / "fresh_confirmation_report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert (output_dir / "fresh_confirmation_report.md").exists()
