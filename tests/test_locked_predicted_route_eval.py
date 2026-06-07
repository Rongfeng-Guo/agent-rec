from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "eval_predicted_route.py"
    spec = importlib.util.spec_from_file_location("eval_predicted_route", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_fusion_config_derives_required_members_and_runtime_settings(tmp_path: Path) -> None:
    module = load_module()
    payload = {
        "config_hash": "abc123",
        "domain_query_sources": {"Book": "mean", "Game": "prefix1_head"},
        "default_domain_query_source": "learned",
        "extra_prefix1_route_sources": ["domain_prior"],
        "default_query_policy": {"query_source": "fusion", "mode": "fusion_best_rrf"},
        "domain_query_policies": {},
        "fusion_specs": [
            {
                "name": "best_rrf",
                "members": [
                    ["learned", "domain_prior_p1"],
                    ["domain_adaptive", "predicted_route_p1_top2_zscore"],
                    ["domain_adaptive", "domain_prior_p1_top2_quota"],
                ],
            }
        ],
        "route_score_weight": 0.0,
        "per_route_topk": None,
        "prefix1_beam_sizes": [2],
        "fusion_method": "rrf",
        "metadata": {
            "policy_name": "fusion_best_rrf",
            "merge_strategy": "score",
            "route_beam": 2,
        },
    }
    path = tmp_path / "fusion_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = module.load_fusion_config(path)
    assert config["config_hash"] == "abc123"
    assert config["route_score_weight"] == 0.0
    assert config["selected_policy_name"] == "fusion_best_rrf"
    assert config["required_prefix1_beam_sizes"] == [1, 2]
    assert set(config["required_merge_strategies"]) == {"score", "quota", "zscore"}
    assert config["required_query_sources"] == ["learned", "domain_adaptive"]
    assert ("domain_adaptive", "predicted_route_p1_top2_zscore") in config["required_pairs"]


def test_load_fusion_config_requires_default_query_policy(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"domain_query_sources": {}, "fusion_specs": []}), encoding="utf-8")
    try:
        module.load_fusion_config(path)
    except ValueError as exc:
        assert "default_query_policy" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing default_query_policy")
