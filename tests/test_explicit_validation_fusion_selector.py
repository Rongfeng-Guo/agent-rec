from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "select_validation_fusion_policy_explicit.py"
    spec = importlib.util.spec_from_file_location("select_validation_fusion_policy_explicit", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builtin_preset_contains_domain_prior_and_heterogeneous_fusion() -> None:
    module = load_module()
    config = module.resolve_explicit_policy_config(None, "explicit_script_oldv0")

    assert config["extra_prefix1_route_sources"] == ["domain_prior"]
    assert config["default_domain_query_source"] == "learned"
    assert config["domain_query_sources"] == {"Book": "mean", "Game": "prefix1_head"}
    assert config["config_hash"]

    fusion = next(policy for policy in config["policies"] if policy["query_source"] == "fusion")
    assert fusion["policy_name"] == "fusion_best_rrf"
    assert fusion["fusion_method"] == "rrf"
    assert fusion["route_beam"] == 2
    members = {tuple(member) for member in fusion["fusion_specs"][0]["members"]}
    assert ("learned", "domain_prior_p1") in members
    assert ("domain_adaptive", "predicted_route_p1_top2_zscore") in members
    assert ("domain_adaptive", "domain_prior_p1_top2_quota") in members


def test_v3_validation_rrf_candidate_preset_is_preblind_only() -> None:
    module = load_module()
    config = module.resolve_explicit_policy_config(None, "v3_validation_rrf_candidate")

    assert config["extra_prefix1_route_sources"] == ["domain_prior"]
    assert config["default_domain_query_source"] == "learned"
    assert config["domain_query_sources"] == {"Book": "mean", "Game": "prefix1_head"}
    assert "fresh blind confirmation" in config["metadata"]["claim_boundary"]

    policy_names = {policy["policy_name"] for policy in config["policies"]}
    assert "domain_adaptive_domain_prior_p1_top4" in policy_names
    assert "domain_adaptive_predicted_route_p1_top4" in policy_names
    assert "mean_predicted_route_p1_top4" in policy_names
    assert "fusion_comparison_rrf" in policy_names

    fusion = next(policy for policy in config["policies"] if policy["policy_name"] == "fusion_comparison_rrf")
    assert fusion["fusion_method"] == "rrf"
    assert fusion["route_beam"] == 4
    members = {tuple(member) for member in fusion["fusion_specs"][0]["members"]}
    assert members == {
        ("domain_adaptive", "predicted_route_p1_top4"),
        ("mean", "predicted_route_p1_top4"),
    }


def test_normalize_explicit_policy_payload_hash_changes_with_candidates() -> None:
    module = load_module()
    base_payload = {
        "domain_query_sources": {"Book": "mean"},
        "default_domain_query_source": "learned",
        "extra_prefix1_route_sources": ["domain_prior"],
        "policy_candidates": [
            {"policy_name": "a", "query_source": "domain_adaptive", "mode": "domain_prior_p1"}
        ],
    }
    first = module._normalize_explicit_policy_payload(base_payload, "memory", "memory")
    changed = dict(base_payload)
    changed["policy_candidates"] = [
        {"policy_name": "a", "query_source": "domain_adaptive", "mode": "domain_prior_p1_top2_quota"}
    ]
    second = module._normalize_explicit_policy_payload(changed, "memory", "memory")

    assert first["config_hash"] != second["config_hash"]


def test_required_retrieval_keys_keep_policy_controls_distinct() -> None:
    module = load_module()
    policies = [
        {
            "policy_name": "p1_k50",
            "query_source": "learned",
            "mode": "predicted_route_p1_top4_zscore",
            "route_score_weight": 0.0,
            "per_route_topk": 50,
        },
        {
            "policy_name": "p1_k200",
            "query_source": "learned",
            "mode": "predicted_route_p1_top4_zscore",
            "route_score_weight": 0.0,
            "per_route_topk": 200,
        },
        {
            "policy_name": "fusion_w1",
            "query_source": "fusion",
            "mode": "fusion_lr",
            "route_score_weight": 1.0,
            "per_route_topk": 50,
            "fusion_specs": [
                {
                    "name": "lr",
                    "members": [
                        ("learned", "predicted_route_p1_top4_zscore"),
                        ("residual", "predicted_route_p1_top4_zscore"),
                    ],
                }
            ],
        },
    ]

    keys = module.required_retrieval_keys(policies)

    assert ("learned", "predicted_route_p1_top4_zscore", 0.0, 50) in keys
    assert ("learned", "predicted_route_p1_top4_zscore", 0.0, 200) in keys
    assert ("learned", "predicted_route_p1_top4_zscore", 1.0, 50) in keys
    assert ("residual", "predicted_route_p1_top4_zscore", 1.0, 50) in keys
    assert len(keys) == 4


def test_output_dir_guard_refuses_non_empty_directory(tmp_path) -> None:
    module = load_module()
    output_dir = tmp_path / "selector"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        module.ensure_empty_output_dir(output_dir)

    assert (output_dir / "old.txt").read_text(encoding="utf-8") == "keep"
    assert "validation-only" in module.NEXT_TARGET


def test_resolve_output_dir_uses_repo_root_for_relative_paths(tmp_path) -> None:
    module = load_module()
    repo_root = tmp_path / "repo"
    absolute = tmp_path / "absolute"

    assert module.resolve_output_dir("outputs/selector", repo_root) == repo_root / "outputs" / "selector"
    assert module.resolve_output_dir(absolute, repo_root) == absolute
    assert module.resolve_output_dir("outputs/selector") == Path("outputs/selector")


def test_resolve_repo_path_uses_repo_root_for_relative_inputs(tmp_path) -> None:
    module = load_module()
    repo_root = tmp_path / "repo"
    absolute = tmp_path / "policy.json"

    assert module.resolve_repo_path("configs/policy.json", repo_root) == repo_root / "configs" / "policy.json"
    assert module.resolve_repo_path(absolute, repo_root) == absolute
    assert module.resolve_repo_path("configs/policy.json") == Path("configs/policy.json")
    assert module.resolve_repo_path(None, repo_root) is None
