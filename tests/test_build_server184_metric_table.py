from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "server184" / "build_server184_metric_table.py"
    spec = importlib.util.spec_from_file_location("build_server184_metric_table", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_collect_runs_and_flatten_method_rows(tmp_path):
    module = load_module()
    outputs_root = tmp_path / "outputs" / "server184_gimo"
    write_json(outputs_root / "closed_loop_alpha" / "summary.json", [
        {"method": "none", "scenario": "s1", "parser_mode": "oracle", "CumulativeUtility": 1.0, "InstructionUplift@H": 0.0, "OverCorrectionRegret@H": 0.0, "MemoryContaminationRate": 0.0, "ScopeClassificationAccuracy": 1.0, "PromotionRecall": 1.0, "PromotionPrecision": 1.0, "status": "SMOKE_TEST_ONLY"},
        {"method": "flat", "scenario": "s1", "parser_mode": "oracle", "CumulativeUtility": 2.0, "InstructionUplift@H": 0.5, "OverCorrectionRegret@H": 0.1, "MemoryContaminationRate": 0.0, "ScopeClassificationAccuracy": 1.0, "PromotionRecall": 1.0, "PromotionPrecision": 1.0, "status": "SMOKE_TEST_ONLY"},
    ])
    write_json(outputs_root / "closed_loop_beta" / "summary.json", [
        {"method": "none", "scenario": "s1", "parser_mode": "oracle", "CumulativeUtility": 3.0, "InstructionUplift@H": 0.0, "OverCorrectionRegret@H": 0.0, "MemoryContaminationRate": 0.0, "ScopeClassificationAccuracy": 1.0, "PromotionRecall": 1.0, "PromotionPrecision": 1.0, "status": "PASS"},
        {"method": "flat", "scenario": "s1", "parser_mode": "oracle", "CumulativeUtility": 5.0, "InstructionUplift@H": 0.8, "OverCorrectionRegret@H": 0.2, "MemoryContaminationRate": 0.0, "ScopeClassificationAccuracy": 1.0, "PromotionRecall": 1.0, "PromotionPrecision": 1.0, "status": "PASS"},
        {"method": "flat", "scenario": "s2", "parser_mode": "oracle", "CumulativeUtility": 7.0, "InstructionUplift@H": 0.6, "OverCorrectionRegret@H": 0.4, "MemoryContaminationRate": 0.0, "ScopeClassificationAccuracy": 1.0, "PromotionRecall": 1.0, "PromotionPrecision": 1.0, "status": "PASS"},
    ])

    run_rows = module.collect_runs(outputs_root)
    assert [row["run"] for row in run_rows] == ["closed_loop_alpha", "closed_loop_beta"]
    assert run_rows[0]["run_status"] == "SMOKE_ONLY"
    assert run_rows[1]["run_status"] == "NON_SMOKE"
    assert run_rows[1]["best_method"] == "flat"
    assert run_rows[1]["best_cumulative_utility"] == 6.0

    method_rows = module.flatten_method_rows(run_rows)
    assert len(method_rows) == 4
    beta_flat = next(row for row in method_rows if row["run"] == "closed_loop_beta" and row["method"] == "flat")
    assert beta_flat["CumulativeUtility"] == 6.0
    assert beta_flat["InstructionUplift@H"] == 0.7
    assert beta_flat["OverCorrectionRegret@H"] == 0.30000000000000004

    report = module.build_markdown(run_rows, method_rows)
    assert "# Server184 Closed Loop Metric Table" in report
    assert "| closed_loop_beta | NON_SMOKE" in report


def test_empty_outputs_degrade_gracefully(tmp_path):
    module = load_module()
    outputs_root = tmp_path / "outputs" / "server184_gimo"
    run_rows = module.collect_runs(outputs_root)
    method_rows = module.flatten_method_rows(run_rows)
    assert run_rows == []
    assert method_rows == []
    report = module.build_markdown(run_rows, method_rows)
    assert "Method Summary" in report
