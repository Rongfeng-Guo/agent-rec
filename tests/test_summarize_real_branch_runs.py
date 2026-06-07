from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "server184" / "summarize_real_branch_runs.py"
    spec = importlib.util.spec_from_file_location("summarize_real_branch_runs", module_path)
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


def test_collect_runs_picks_latest_ok_and_flags_failures(tmp_path):
    module = load_module()
    runs_root = tmp_path / "outputs" / "server184_gimo" / "real_branch_replay"

    older = runs_root / "20260606_180627"
    write_json(older / "cdpo_validation.json", {"status": "PASS", "rows": 1})
    write_json(older / "cdpo_dataset_manifest.json", {"source": "RealBranchReplay", "proxy": "controlled real user simulator replay proxy", "row_count": 1})
    write_json(older / "audit" / "audit.json", {"status": "FAIL", "positive_uplift_count": 1})
    write_json(older / "adapter" / "adapter_metadata.json", {"status": "SMOKE_TEST_ONLY", "positive_pair_count": 1})
    write_jsonl(older / "adapter" / "cdpo_pairs.jsonl", [{"id": "pair-1"}])
    write_jsonl(older / "train.jsonl", [])
    write_jsonl(older / "dev.jsonl", [{"id": "pair-1"}])

    newer = runs_root / "20260606_181246"
    write_json(newer / "cdpo_validation.json", {"status": "PASS", "rows": 1})
    write_json(newer / "cdpo_dataset_manifest.json", {"source": "RealBranchReplay", "proxy": "controlled real user simulator replay proxy", "row_count": 1})
    write_json(newer / "audit" / "audit.json", {"status": "PASS", "positive_uplift_count": 1})
    write_json(newer / "adapter" / "adapter_metadata.json", {"status": "SMOKE_TEST_ONLY", "positive_pair_count": 1})
    write_jsonl(newer / "adapter" / "cdpo_pairs.jsonl", [{"id": "pair-2"}])
    write_jsonl(newer / "train.jsonl", [])
    write_jsonl(newer / "dev.jsonl", [{"id": "pair-2"}])

    rows = module.collect_runs(runs_root)

    assert [row["run"] for row in rows] == ["20260606_180627", "20260606_181246"]
    assert rows[0]["bridge_status"] == "AUDIT_NOT_PASS"
    assert rows[1]["bridge_status"] == "OK"
    assert "audit status='FAIL'" in rows[0]["bridge_issues"]
    report = module.build_report(rows)
    assert "latest_run: `20260606_181246`" in report
    assert "latest_ok_run: `20260606_181246`" in report
    assert "`AUDIT_NOT_PASS`: `1`" in report
    assert "`OK`: `1`" in report


def test_collect_runs_flags_split_mismatch(tmp_path):
    module = load_module()
    run_dir = tmp_path / "outputs" / "server184_gimo" / "real_branch_replay" / "20260606_181500"
    write_json(run_dir / "cdpo_validation.json", {"status": "PASS", "rows": 2})
    write_json(run_dir / "cdpo_dataset_manifest.json", {"source": "RealBranchReplay", "proxy": "controlled real user simulator replay proxy", "row_count": 2})
    write_json(run_dir / "audit" / "audit.json", {"status": "PASS", "positive_uplift_count": 2})
    write_json(run_dir / "adapter" / "adapter_metadata.json", {"status": "SMOKE_TEST_ONLY", "positive_pair_count": 2})
    write_jsonl(run_dir / "adapter" / "cdpo_pairs.jsonl", [{"id": "pair-1"}, {"id": "pair-2"}])
    write_jsonl(run_dir / "train.jsonl", [])
    write_jsonl(run_dir / "dev.jsonl", [{"id": "pair-2"}])

    rows = module.collect_runs(run_dir.parent)

    assert rows[0]["bridge_status"] == "SPLIT_COUNT_MISMATCH"
    assert "train/dev counts do not sum to cdpo_pairs count" in rows[0]["bridge_issues"]


def test_collect_runs_collapses_missing_artifacts_into_single_status(tmp_path):
    module = load_module()
    run_dir = tmp_path / "outputs" / "server184_gimo" / "real_branch_replay" / "20260606_180421"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = module.collect_runs(run_dir.parent)

    assert rows[0]["bridge_status"] == "MISSING_REQUIRED_ARTIFACTS"
    assert rows[0]["bridge_issues"] == ["missing required artifacts"]
    assert "cdpo_validation.json" in rows[0]["missing_artifacts"]
    assert "adapter/adapter_metadata.json" in rows[0]["missing_artifacts"]
