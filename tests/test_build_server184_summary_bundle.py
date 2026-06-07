from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "server184" / "build_server184_summary_bundle.py"
    spec = importlib.util.spec_from_file_location("build_server184_summary_bundle", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_ready_summary_bundle(tmp_path):
    module = load_module()
    outputs_root = tmp_path / "outputs" / "server184_gimo"
    write_json(outputs_root / "decision" / "decision.json", {
        "overall_status": "READY_FOR_METRIC_REVIEW",
    })
    write_json(outputs_root / "metric_table" / "metric_table.json", {
        "run_count": 4,
        "method_row_count": 16,
        "run_rows": [
            {"run": "closed_loop_fullsmoke", "best_method": "critiquescope", "best_cumulative_utility": 6.86},
            {"run": "closed_loop_fullvalid9", "best_method": "critiquescope", "best_cumulative_utility": 15.95},
            {"run": "closed_loop_smoke", "best_method": "critiquescope", "best_cumulative_utility": 6.98},
        ],
    })
    write_json(outputs_root / "best_run_comparison" / "comparison.json", {
        "overall_status": "READY",
        "best_run": {
            "run": "closed_loop_fullvalid9",
            "best_method": "critiquescope",
            "best_cumulative_utility": 15.95,
        },
        "winner": {
            "method": "critiquescope",
            "run": "closed_loop_fullvalid9",
            "CumulativeUtility": 15.95,
            "InstructionUplift@H": -0.37,
            "OverCorrectionRegret@H": 0.11,
            "PromotionRecall": 1.0,
        },
        "comparisons": [
            {
                "baseline_method": "time_decay",
                "metric_deltas": {
                    "CumulativeUtility": {"delta": 3.49, "winner_is_better": True},
                    "PromotionRecall": {"delta": 0.20, "winner_is_better": True},
                    "InstructionUplift@H": {"delta": -0.10, "winner_is_better": False},
                    "OverCorrectionRegret@H": {"delta": -0.02, "winner_is_better": True},
                },
            },
            {
                "baseline_method": "flat",
                "metric_deltas": {
                    "CumulativeUtility": {"delta": 4.29, "winner_is_better": True},
                    "PromotionRecall": {"delta": 0.30, "winner_is_better": True},
                    "InstructionUplift@H": {"delta": -0.11, "winner_is_better": False},
                    "OverCorrectionRegret@H": {"delta": 0.04, "winner_is_better": False},
                },
            },
        ],
    })
    write_json(outputs_root / "metric_facets" / "facet_report.json", {
        "overall_status": "READY",
        "scorecard": {
            "baseline_majority_win": 1,
            "baseline_draw": 1,
        },
        "metric_summary": {
            "CumulativeUtility": {"win_count": 4, "loss_count": 0, "tie_count": 0},
            "PromotionRecall": {"win_count": 4, "loss_count": 0, "tie_count": 0},
            "InstructionUplift@H": {"win_count": 0, "loss_count": 4, "tie_count": 0},
            "OverCorrectionRegret@H": {"win_count": 1, "loss_count": 3, "tie_count": 0},
        },
    })

    artifacts = module.load_artifacts(outputs_root)
    payload = module.build_summary_payload(artifacts, outputs_root)
    assert payload["overall_status"] == "READY"
    assert payload["winner"]["method"] == "critiquescope"
    assert payload["best_run"]["run"] == "closed_loop_fullvalid9"
    assert payload["metric_table_summary"]["top_run"] == "closed_loop_fullvalid9"
    assert payload["primary_metric_deltas"]["CumulativeUtility"]["better_count"] == 2
    assert payload["primary_metric_deltas"]["InstructionUplift@H"]["worse_count"] == 2
    assert "review anchor" in payload["actionable_conclusion"]
    assert payload["strongest_metrics"][0]["metric"] == "CumulativeUtility"
    assert payload["weakest_metrics"][0]["metric"] == "InstructionUplift@H"
    report = module.build_report(payload)
    assert "Server184 Summary Bundle" in report
    assert "Primary Metric Deltas" in report


def test_incomplete_summary_bundle_when_prereqs_missing(tmp_path):
    module = load_module()
    outputs_root = tmp_path / "outputs" / "server184_gimo"
    write_json(outputs_root / "decision" / "decision.json", {
        "overall_status": "BLOCKED",
    })
    write_json(outputs_root / "best_run_comparison" / "comparison.json", {
        "overall_status": "NO_NON_SMOKE_RUN",
        "best_run": {},
        "winner": {},
    })
    write_json(outputs_root / "metric_facets" / "facet_report.json", {
        "overall_status": "NO_NON_SMOKE_RUN",
        "metric_summary": {},
    })

    payload = module.build_summary_payload(module.load_artifacts(outputs_root), outputs_root)
    assert payload["overall_status"] == "INCOMPLETE"
    assert "incomplete" in payload["headline"].lower()
    assert "no metric-facing recommendation" in payload["actionable_conclusion"].lower()
    report = module.build_report(payload)
    assert "decision_status: `BLOCKED`" in report
