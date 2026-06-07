from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "server184" / "build_server184_decision_report.py"
    spec = importlib.util.spec_from_file_location("build_server184_decision_report", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_ready_for_metric_review_when_non_smoke_closed_loop_exists(tmp_path):
    module = load_module()
    index_payload = {
        "outputs_root": str(tmp_path / "outputs" / "server184_gimo"),
        "env": {
            "exists": True,
            "model_exists": True,
            "vllm_python_exists": True,
        },
        "bridge_latest_real": {
            "exists": True,
            "status": "OK",
            "validation_status": "PASS",
            "audit_status": "PASS",
            "latest_run_dir": "/tmp/replay/20260606_181246",
        },
        "real_branch_replay": {
            "exists": True,
            "latest_ok_run": "20260606_181246",
        },
        "closed_loop_runs": [
            {
                "name": "closed_loop_fullvalid9",
                "status_counts": {"PASS": 35},
                "best_method": "critiquescope",
                "best_avg_cumulative_utility": 15.95,
            }
        ],
    }

    payload = module.build_decision_payload(index_payload)
    assert payload["overall_status"] == "READY_FOR_METRIC_REVIEW"
    assert payload["blockers"] == []
    assert payload["closed_loop_readiness"]["non_smoke_run_count"] == 1
    report = module.build_report(payload)
    assert "READY_FOR_METRIC_REVIEW" in report
    assert "best_method: `critiquescope`" in report


def test_smoke_only_closed_loop_blocks_metric_review(tmp_path):
    module = load_module()
    outputs_root = tmp_path / "outputs" / "server184_gimo"
    index_path = outputs_root / "index" / "index.json"
    write_json(index_path, {
        "outputs_root": str(outputs_root),
        "env": {
            "exists": True,
            "model_exists": True,
            "vllm_python_exists": True,
        },
        "bridge_latest_real": {
            "exists": True,
            "status": "OK",
            "validation_status": "PASS",
            "audit_status": "PASS",
            "latest_run_dir": "/tmp/replay/20260606_181246",
        },
        "real_branch_replay": {
            "exists": True,
            "latest_ok_run": "20260606_181246",
        },
        "closed_loop_runs": [
            {
                "name": "closed_loop_smoke",
                "status_counts": {"SMOKE_TEST_ONLY": 35},
                "best_method": "critiquescope",
                "best_avg_cumulative_utility": 6.86,
            }
        ],
    })

    index_payload = module.read_json(index_path)
    index_payload["index_path"] = str(index_path)
    payload = module.build_decision_payload(index_payload)
    assert payload["overall_status"] == "BLOCKED"
    assert "ONLY_SMOKE_CLOSED_LOOP_RUNS" in payload["blockers"]
    assert payload["closed_loop_readiness"]["status"] == "ONLY_SMOKE_CLOSED_LOOP_RUNS"
    assert payload["closed_loop_readiness"]["smoke_only_run_count"] == 1
    assert any("non-smoke metric run" in step for step in payload["next_steps"])
