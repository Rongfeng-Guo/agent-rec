from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "server184" / "build_server184_metric_facet_report.py"
    spec = importlib.util.spec_from_file_location("build_server184_metric_facet_report", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_facet_payload_for_ready_comparison(tmp_path):
    module = load_module()
    comparison_json = tmp_path / "outputs" / "server184_gimo" / "best_run_comparison" / "comparison.json"
    write_json(comparison_json, {
        "overall_status": "READY",
        "winner": {"run": "closed_loop_fullvalid9", "method": "critiquescope", "CumulativeUtility": 10.0},
        "comparisons": [
            {
                "baseline_method": "none",
                "metric_deltas": {
                    "CumulativeUtility": {"delta": 3.0, "winner_is_better": True},
                    "InstructionUplift@H": {"delta": -0.5, "winner_is_better": False},
                    "MemoryContaminationRate": {"delta": 0.0, "winner_is_better": True},
                },
            },
            {
                "baseline_method": "flat",
                "metric_deltas": {
                    "CumulativeUtility": {"delta": 2.0, "winner_is_better": True},
                    "InstructionUplift@H": {"delta": -0.1, "winner_is_better": False},
                    "MemoryContaminationRate": {"delta": 0.0, "winner_is_better": True},
                },
            },
        ],
    })

    payload = module.build_facet_payload(module.read_json(comparison_json))
    assert payload["overall_status"] == "READY"
    assert payload["winner"]["method"] == "critiquescope"
    assert payload["metric_summary"]["CumulativeUtility"]["win_count"] == 2
    assert payload["metric_summary"]["InstructionUplift@H"]["loss_count"] == 2
    assert payload["metric_summary"]["MemoryContaminationRate"]["tie_count"] == 2
    assert payload["scorecard"]["baseline_draw"] == 2
    report = module.build_report(payload)
    assert "Metric Summary" in report
    assert "baseline_draw" in report


def test_non_ready_comparison_degrades_gracefully(tmp_path):
    module = load_module()
    comparison_json = tmp_path / "outputs" / "server184_gimo" / "best_run_comparison" / "comparison.json"
    write_json(comparison_json, {
        "overall_status": "NO_NON_SMOKE_RUN",
        "summary": "not ready",
        "winner": None,
        "comparisons": [],
    })

    payload = module.build_facet_payload(module.read_json(comparison_json))
    assert payload["overall_status"] == "NO_NON_SMOKE_RUN"
    assert payload["baseline_summaries"] == []
    report = module.build_report(payload)
    assert "no facet report" in report.lower()
    assert "overall_status: `NO_NON_SMOKE_RUN`" in report
