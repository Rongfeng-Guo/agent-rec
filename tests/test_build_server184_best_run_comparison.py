from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "server184" / "build_server184_best_run_comparison.py"
    spec = importlib.util.spec_from_file_location("build_server184_best_run_comparison", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_select_best_non_smoke_run_and_compare(tmp_path):
    module = load_module()
    metric_json = tmp_path / "outputs" / "server184_gimo" / "metric_table" / "metric_table.json"
    write_json(metric_json, {
        "run_rows": [
            {"run": "closed_loop_smoke", "run_status": "SMOKE_ONLY", "best_cumulative_utility": 3.0, "best_method": "none"},
            {"run": "closed_loop_fullvalid9", "run_status": "NON_SMOKE", "best_cumulative_utility": 10.0, "best_method": "critiquescope"},
        ],
        "method_rows": [
            {"run": "closed_loop_fullvalid9", "method": "critiquescope", "CumulativeUtility": 10.0, "InstructionUplift@H": 0.3, "OverCorrectionRegret@H": 0.1, "MemoryContaminationRate": 0.0, "ScopeClassificationAccuracy": 1.0, "PromotionRecall": 1.0},
            {"run": "closed_loop_fullvalid9", "method": "none", "CumulativeUtility": 7.0, "InstructionUplift@H": 0.0, "OverCorrectionRegret@H": 0.0, "MemoryContaminationRate": 0.0, "ScopeClassificationAccuracy": 1.0, "PromotionRecall": 0.5},
            {"run": "closed_loop_fullvalid9", "method": "flat", "CumulativeUtility": 8.0, "InstructionUplift@H": 0.1, "OverCorrectionRegret@H": 0.2, "MemoryContaminationRate": 0.0, "ScopeClassificationAccuracy": 1.0, "PromotionRecall": 0.75},
        ],
    })

    payload = module.build_comparison_payload(module.read_json(metric_json))
    assert payload["overall_status"] == "READY"
    assert payload["best_run"]["run"] == "closed_loop_fullvalid9"
    assert payload["winner"]["method"] == "critiquescope"
    baseline_none = next(row for row in payload["comparisons"] if row["baseline_method"] == "none")
    assert baseline_none["metric_deltas"]["CumulativeUtility"]["delta"] == 3.0
    assert baseline_none["metric_deltas"]["PromotionRecall"]["winner_is_better"] is True
    baseline_flat = next(row for row in payload["comparisons"] if row["baseline_method"] == "flat")
    assert baseline_flat["metric_deltas"]["OverCorrectionRegret@H"]["winner_is_better"] is True
    report = module.build_report(payload)
    assert "Best Run" in report
    assert "baseline: `none`" in report


def test_no_non_smoke_run_returns_blocked_payload(tmp_path):
    module = load_module()
    metric_json = tmp_path / "outputs" / "server184_gimo" / "metric_table" / "metric_table.json"
    write_json(metric_json, {
        "run_rows": [
            {"run": "closed_loop_smoke", "run_status": "SMOKE_ONLY", "best_cumulative_utility": 3.0, "best_method": "none"},
        ],
        "method_rows": [],
    })

    payload = module.build_comparison_payload(module.read_json(metric_json))
    assert payload["overall_status"] == "NO_NON_SMOKE_RUN"
    assert payload["winner"] is None
    report = module.build_report(payload)
    assert "No non-smoke closed-loop run" in report
