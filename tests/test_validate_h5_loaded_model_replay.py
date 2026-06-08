from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "validate_h5_loaded_model_replay.py"
    spec = importlib.util.spec_from_file_location("validate_h5_loaded_model_replay", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(sample_id: str, rank: int | None, selected_source: str = "h100") -> dict:
    return {
        "sample_id": sample_id,
        "domain": "Book",
        "target_item_id": f"target-{sample_id}",
        "match_rank": rank,
        "candidate_pool_hit": rank is not None,
        "candidate_pool_match_rank": rank,
        "route_hit": True,
        "candidate_pool_size": 10,
        "selected_source": selected_source,
        "oracle_source_match_rank": rank,
    }


def test_validate_replay_accepts_matching_rows(tmp_path) -> None:
    module = load_module()
    locked = [row("s1", 4), row("s2", None, "h300")]
    replay = [dict(item) for item in locked]
    locked_path = tmp_path / "locked.json"
    replay_path = tmp_path / "replay.json"
    locked_path.write_text(json.dumps(locked), encoding="utf-8")
    replay_path.write_text(json.dumps(replay), encoding="utf-8")

    result = module.validate_replay(locked_outputs=locked_path, replay_outputs=replay_path, topk=50)

    assert result["status"] == "ok"
    assert result["mismatch_count"] == 0
    assert result["replay_metric"]["Recall@50"] == 0.5
    assert "readiness input" in result["next_target"]

    rendered = module.render_report(result, topk=50)
    assert "## Next Target" in rendered
    assert "fresh-confirmation gate" in rendered


def test_validate_replay_reports_field_mismatch(tmp_path) -> None:
    module = load_module()
    locked = [row("s1", 4)]
    replay = [row("s1", 80)]
    locked_path = tmp_path / "locked.json"
    replay_path = tmp_path / "replay.json"
    locked_path.write_text(json.dumps(locked), encoding="utf-8")
    replay_path.write_text(json.dumps(replay), encoding="utf-8")

    result = module.validate_replay(locked_outputs=locked_path, replay_outputs=replay_path, topk=50)

    assert result["status"] == "failed"
    assert result["mismatches"][0]["field"] == "match_rank"
    assert result["metric_errors"]
    assert "resolve row or metric mismatches" in result["next_target"]
