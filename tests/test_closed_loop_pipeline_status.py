from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


module = importlib.import_module("user_simulator.evaluation.run_closed_loop_pipeline")


def make_args(**overrides) -> argparse.Namespace:
    payload = {
        "modes": ["none", "flat", "structured", "time_decay", "critiquescope"],
        "scenarios": ["all"],
        "seeds": [0],
        "max_turns": 9,
        "top_k": 5,
        "parser_mode": "oracle",
        "branch_horizon": 5,
        "dev_fraction": 0.2,
        "run_validity_gate": True,
        "fail_on_critical_invariant": True,
        "output_dir": "unused",
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_classify_run_status_requires_full_coverage_and_clean_validity_gate():
    args = make_args()
    audit = {"status": "PASS"}
    validity_gate = {"failed_invariants": 0, "critical_failed_invariants": 0}

    assert module.classify_run_status(args, audit, validity_gate) == "PASS"

    partial_args = make_args(scenarios=["temporary_fatigue"])
    assert module.classify_run_status(partial_args, audit, validity_gate) == "SMOKE_TEST_ONLY"
    assert module.classify_run_status(args, audit, None) == "SMOKE_TEST_ONLY"


def test_normalize_closed_loop_artifacts_rewrites_status_fields(tmp_path):
    output_dir = tmp_path / "closed_loop_fullvalid9"
    write_json(output_dir / "summary.json", [{"method": "flat", "status": "SMOKE_TEST_ONLY"}])
    (output_dir / "summary.csv").write_text("method,status\nflat,SMOKE_TEST_ONLY\n", encoding="utf-8")
    write_json(output_dir / "run_metadata.json", {"status": "SMOKE_TEST_ONLY"})
    write_json(output_dir / "cdpo_dataset_manifest.json", {"status": "SMOKE_TEST_ONLY"})
    (output_dir / "README.md").write_text("# Closed-loop CritiqueWorld Run\n\n- Status: SMOKE_TEST_ONLY\n", encoding="utf-8")

    module.normalize_closed_loop_artifacts(output_dir, "PASS")

    summary_rows = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_rows[0]["status"] == "PASS"
    assert "flat,PASS" in (output_dir / "summary.csv").read_text(encoding="utf-8")
    assert json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))["status"] == "PASS"
    assert json.loads((output_dir / "cdpo_dataset_manifest.json").read_text(encoding="utf-8"))["status"] == "PASS"
    assert "- Status: PASS" in (output_dir / "README.md").read_text(encoding="utf-8")
