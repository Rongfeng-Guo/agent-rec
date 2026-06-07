from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_bridge_check_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "server184" / "bridge_check.py"
    spec = importlib.util.spec_from_file_location("server184_bridge_check", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_bridge_metadata_accepts_latest_real_branch_run(tmp_path):
    module = load_bridge_check_module()
    root_dir = tmp_path
    real_branch_root = root_dir / "outputs" / "server184_gimo" / "real_branch_replay"
    run_dir = real_branch_root / "20260607_120000"

    write_json(run_dir / "cdpo_validation.json", {"status": "PASS", "rows": 1, "error_count": 0})
    write_json(
        run_dir / "cdpo_dataset_manifest.json",
        {
            "status": "SMOKE_TEST_ONLY",
            "source": "RealBranchReplay",
            "proxy": "controlled real user simulator replay proxy",
            "validation_status": "PASS",
            "row_count": 1,
            "by_provenance": {"REAL_USER_SIM_REPLAY": 1},
        },
    )
    write_json(
        run_dir / "audit" / "audit.json",
        {"status": "PASS", "snapshot_count": 2, "branch_count": 3, "positive_uplift_count": 1},
    )
    write_json(
        run_dir / "adapter" / "adapter_metadata.json",
        {"status": "SMOKE_TEST_ONLY", "pair_count": 2, "positive_pair_count": 1},
    )
    write_jsonl(run_dir / "adapter" / "cdpo_pairs.jsonl", [{"id": "pair-1"}])
    write_jsonl(run_dir / "train.jsonl", [])
    write_jsonl(run_dir / "dev.jsonl", [{"id": "pair-1"}])

    payload = module.build_bridge_metadata(root_dir, real_branch_root, None)

    assert payload["status"] == "OK"
    assert payload["latest_run_dir"] == str(run_dir)
    assert payload["cdpo_pair_count"] == 1
    assert payload["train_count"] == 0
    assert payload["dev_count"] == 1
    assert payload["issues"] == []


def test_build_bridge_metadata_flags_missing_artifacts(tmp_path):
    module = load_bridge_check_module()
    root_dir = tmp_path
    real_branch_root = root_dir / "outputs" / "server184_gimo" / "real_branch_replay"
    run_dir = real_branch_root / "20260607_120000"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = module.build_bridge_metadata(root_dir, real_branch_root, None)

    assert payload["status"] == "BLOCKED_MISSING_BRIDGE_ARTIFACTS"
    assert payload["latest_run_dir"] == str(run_dir)
    assert "cdpo_validation.json" in payload["missing_files"]


def test_build_bridge_metadata_rejects_non_real_manifest(tmp_path):
    module = load_bridge_check_module()
    root_dir = tmp_path
    real_branch_root = root_dir / "outputs" / "server184_gimo" / "real_branch_replay"
    run_dir = real_branch_root / "20260607_120000"

    write_json(run_dir / "cdpo_validation.json", {"status": "PASS", "rows": 1, "error_count": 0})
    write_json(
        run_dir / "cdpo_dataset_manifest.json",
        {
            "status": "SMOKE_TEST_ONLY",
            "source": "CritiqueWorld",
            "proxy": "controlled counterfactual rollout proxy",
            "validation_status": "PASS",
            "row_count": 1,
            "by_provenance": {"SYNTHETIC_CRITIQUEWORLD": 1},
        },
    )
    write_json(
        run_dir / "audit" / "audit.json",
        {"status": "PASS", "snapshot_count": 2, "branch_count": 3, "positive_uplift_count": 1},
    )
    write_json(
        run_dir / "adapter" / "adapter_metadata.json",
        {"status": "SMOKE_TEST_ONLY", "pair_count": 2, "positive_pair_count": 1},
    )
    write_jsonl(run_dir / "adapter" / "cdpo_pairs.jsonl", [{"id": "pair-1"}])
    write_jsonl(run_dir / "train.jsonl", [])
    write_jsonl(run_dir / "dev.jsonl", [{"id": "pair-1"}])

    payload = module.build_bridge_metadata(root_dir, real_branch_root, None)

    assert payload["status"] == "BLOCKED_BRIDGE_VALIDATION_FAILED"
    assert any("Manifest source" in issue for issue in payload["issues"])
