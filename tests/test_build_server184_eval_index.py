from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "server184" / "build_server184_eval_index.py"
    spec = importlib.util.spec_from_file_location("build_server184_eval_index", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_collect_closed_loop_and_build_report(tmp_path):
    module = load_module()
    outputs_root = tmp_path / "outputs" / "server184_gimo"
    write_json(outputs_root / "env" / "env_report.json", {
        "model_exists": True,
        "vllm_python_exists": True,
        "vllm_base_url": "http://127.0.0.1:8000/v1",
        "vllm_model_alias": "qwen2.5-3b-instruct",
    })
    write_json(outputs_root / "bridge" / "latest_real" / "bridge_metadata.json", {
        "status": "OK",
        "latest_run_dir": "/tmp/run",
        "validation_status": "PASS",
        "audit_status": "PASS",
        "cdpo_pair_count": 1,
    })
    write_json(outputs_root / "real_branch_replay_summary" / "summary.json", {
        "run_count": 3,
        "latest_run": "20260606_181246",
        "latest_run_status": "OK",
        "latest_ok_run": "20260606_181246",
        "latest_ok_path": "/tmp/run",
        "status_counts": {"OK": 1, "AUDIT_NOT_PASS": 2},
    })
    write_json(outputs_root / "closed_loop_smoke" / "summary.json", [
        {"method": "flat", "scenario": "s1", "parser_mode": "oracle", "CumulativeUtility": 1.0, "status": "SMOKE_TEST_ONLY"},
        {"method": "time_decay", "scenario": "s1", "parser_mode": "oracle", "CumulativeUtility": 2.0, "status": "SMOKE_TEST_ONLY"},
    ])
    write_json(outputs_root / "closed_loop_fullvalid9" / "summary.json", [
        {"method": "flat", "scenario": "s2", "parser_mode": "oracle", "CumulativeUtility": 3.0, "status": "PASS"},
        {"method": "flat", "scenario": "s3", "parser_mode": "oracle", "CumulativeUtility": 5.0, "status": "PASS"},
    ])

    closed_loop = module.collect_closed_loop(outputs_root)
    assert [row["name"] for row in closed_loop] == ["closed_loop_fullvalid9", "closed_loop_smoke"]
    assert closed_loop[0]["best_method"] == "flat"
    assert closed_loop[0]["best_avg_cumulative_utility"] == 4.0
    assert closed_loop[1]["status_counts"] == {"SMOKE_TEST_ONLY": 2}

    payload = {
        "outputs_root": str(outputs_root),
        "env": module.summarize_env(outputs_root),
        "bridge_latest_real": module.summarize_bridge(outputs_root),
        "real_branch_replay": module.summarize_replay(outputs_root),
        "closed_loop_runs": closed_loop,
    }
    decision = module.build_decision_artifacts(payload, outputs_root)
    payload["decision_report"] = decision
    report = module.build_report(payload)
    assert "# Server184 Eval Index" in report
    assert "latest_run: `20260606_181246`" in report
    assert "`closed_loop_fullvalid9`" in report
    assert "best_method=`flat`" in report
    assert decision["overall_status"] == "READY_FOR_METRIC_REVIEW"
    assert (outputs_root / "decision" / "decision.json").exists()


def test_missing_inputs_degrade_gracefully(tmp_path):
    module = load_module()
    outputs_root = tmp_path / "outputs" / "server184_gimo"

    payload = {
        "outputs_root": str(outputs_root),
        "env": module.summarize_env(outputs_root),
        "bridge_latest_real": module.summarize_bridge(outputs_root),
        "real_branch_replay": module.summarize_replay(outputs_root),
        "closed_loop_runs": module.collect_closed_loop(outputs_root),
    }

    assert payload["env"]["exists"] is False
    assert payload["bridge_latest_real"]["exists"] is False
    assert payload["real_branch_replay"]["exists"] is False
    assert payload["closed_loop_runs"] == []
    decision = module.build_decision_artifacts(payload, outputs_root)
    payload["decision_report"] = decision
    report = module.build_report(payload)
    assert "env_report: `missing`" in report
    assert "## Closed Loop Runs" in report
    assert decision["overall_status"] == "BLOCKED"
