from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from genrec.models.late_bound_fusion_router import LateBoundFusionRouter
from genrec.models.late_bound_fusion_router import LateBoundFusionRouter
from genrec.models.late_bound_router import LateBoundRouter
from genrec.memory.catalog_memory import CatalogMemory


def test_late_bound_router_shapes() -> None:
    model = LateBoundRouter(embedding_dim=8, hidden_dim=16, num_prefix1=4, num_prefix2=3)
    history_embs = torch.randn(2, 5, 8)
    history_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    outputs = model(history_embs, history_mask)
    assert outputs["route1_logits"].shape == (2, 4)
    assert outputs["route2_logits"].shape == (2, 4, 3)
    assert outputs["query_embedding"].shape == (2, 8)


def test_late_bound_router_joint_log_probs() -> None:
    model = LateBoundRouter(embedding_dim=4, hidden_dim=8, num_prefix1=2, num_prefix2=2)
    history_embs = torch.randn(1, 3, 4)
    history_mask = torch.ones(1, 3, dtype=torch.bool)
    outputs = model(history_embs, history_mask)
    _, _, joint = model.route_log_probs(outputs)
    probs = joint.exp().sum().item()
    assert abs(probs - 1.0) < 1e-5


def test_late_bound_fusion_router_outputs_candidate_logits() -> None:
    model = LateBoundFusionRouter(num_features=5, num_sources=3, hidden_dim=8)
    sample_features = torch.randn(2, 5)
    source_scores = torch.randn(2, 7, 3)
    route_scores = torch.randn(2, 7)

    logits, weights = model(sample_features, source_scores, route_scores)

    assert logits.shape == (2, 7)
    assert weights.shape == (2, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)


from genrec.training.router_dataset import RouteVocab
from scripts.oracle_route_memory.eval_predicted_route import (
    add_fusion_query_sources,
    build_fusion_retrieval_row,
    enumerate_prefix1_candidates,
    fuse_ranked_lists,
    load_domain_query_source_config,
    merge_candidate_rows,
    parse_domain_query_sources,
    parse_fusion_specs,
    resolve_query_source,
    rerank_with_routes,
)
from scripts.oracle_route_memory.select_validation_fusion_policy import make_policy_grid


def test_prefix1_candidates_use_prefix1_logits_only() -> None:
    route_vocab = RouteVocab(
        prefix1_values=[10, 20, 30],
        prefix2_values={10: [0, 1], 20: [0, 1], 30: [0, 1]},
    )
    route1_log_probs = torch.log_softmax(torch.tensor([[0.1, 3.0, 2.0]]), dim=-1)

    candidates = enumerate_prefix1_candidates(route1_log_probs, route_vocab, beam_size=2)

    assert [route for route, _ in candidates[0]] == [(20,), (30,)]
    assert all(len(route) == 1 for route, _ in candidates[0])


def test_merge_candidate_rows_supports_bucket_normalized_rerank() -> None:
    per_route_candidates = [
        [("a", 0.90), ("b", 0.89), ("d", 0.88)],
        [("target", 0.30), ("c", 0.10), ("e", 0.10)],
    ]
    route_log_probs = [0.0, 0.0]

    assert merge_candidate_rows(per_route_candidates, route_log_probs, 2, "score") == ["a", "b"]
    assert merge_candidate_rows(per_route_candidates, route_log_probs, 2, "zscore") == ["target", "a"]
    assert merge_candidate_rows(per_route_candidates, route_log_probs, 3, "rrf") == ["a", "target", "b"]


def test_domain_adaptive_query_source_resolution() -> None:
    mapping = parse_domain_query_sources(["Book=residual", "Game=learned"])

    assert resolve_query_source("domain_adaptive", "Book", mapping, "pooled") == "residual"
    assert resolve_query_source("domain_adaptive", "Game", mapping, "pooled") == "learned"
    assert resolve_query_source("domain_adaptive", "Yelp", mapping, "pooled") == "pooled"
    assert resolve_query_source("learned", "Book", mapping, "pooled") == "learned"


def test_domain_query_source_config_loads_mapping(tmp_path) -> None:
    config_path = tmp_path / "selector.json"
    config_path.write_text(
        json.dumps(
            {
                "domain_query_sources": {"Book": "residual", "Game": "learned"},
                "default_domain_query_source": "pooled",
                "metadata": {"selector_type": "validation"},
            }
        ),
        encoding="utf-8",
    )

    config = load_domain_query_source_config(config_path)

    assert config["domain_query_sources"] == {"Book": "residual", "Game": "learned"}
    assert config["default_domain_query_source"] == "pooled"
    assert config["metadata"] == {"selector_type": "validation"}


def test_fusion_specs_parse_and_add_query_sources() -> None:
    specs = parse_fusion_specs([
        "best=learned:domain_prior_p1+domain_adaptive:predicted_route_p1_top2_zscore"
    ])

    assert specs == [
        {
            "name": "best",
            "members": [
                ("learned", "domain_prior_p1"),
                ("domain_adaptive", "predicted_route_p1_top2_zscore"),
            ],
        }
    ]
    assert add_fusion_query_sources(["domain_adaptive"], specs) == ["domain_adaptive", "learned"]


def test_fuse_ranked_lists_rrf_and_round_robin() -> None:
    ranked_lists = [
        ["a", "target", "c"],
        ["target", "b", "a"],
    ]

    assert fuse_ranked_lists(ranked_lists, max_k=3, method="rrf", rrf_k=60.0)[0] == "target"
    assert fuse_ranked_lists(ranked_lists, max_k=3, method="round_robin") == ["a", "target", "b"]


def test_build_fusion_retrieval_row_combines_members() -> None:
    fusion_spec = {
        "name": "lr_p1t4",
        "members": [
            ("learned", "predicted_route_p1_top4_zscore"),
            ("residual", "predicted_route_p1_top4_zscore"),
        ],
    }
    sample_rows = {
        ("learned", "predicted_route_p1_top4_zscore"): {
            "subset": "cold",
            "domain": "Book",
            "ranked_ids": ["a", "target", "c"],
            "latency_ms": 1.25,
            "fallback_used": False,
            "route_hit": False,
            "candidate_pool_hit": True,
            "num_route_candidates": 2,
        },
        ("residual", "predicted_route_p1_top4_zscore"): {
            "subset": "cold",
            "domain": "Book",
            "ranked_ids": ["target", "b", "a"],
            "latency_ms": 2.5,
            "fallback_used": True,
            "route_hit": True,
            "candidate_pool_hit": True,
            "num_route_candidates": 3,
        },
    }

    row = build_fusion_retrieval_row(
        fusion_spec=fusion_spec,
        sample_retrieval_rows=sample_rows,
        sample_id="sample-1",
        target_item_id="target",
        true_route=(1, 2),
        topks=[10, 20, 50],
        fusion_method="round_robin",
        fusion_rrf_k=60.0,
        per_route_topk=50,
    )

    assert row["query_source"] == "fusion"
    assert row["mode"] == "fusion_lr_p1t4"
    assert row["ranked_ids"][:4] == ["a", "target", "b", "c"]
    assert row["match_rank"] == 2
    assert row["route_hit"] is True
    assert row["member_route_hit_count"] == 1
    assert row["member_candidate_pool_hit_count"] == 2
    assert row["candidate_pool_hit"] is True
    assert row["candidate_pool_size"] == 4
    assert row["fallback_used"] is True
    assert row["num_route_candidates"] == 5
    assert row["true_route"] == "1|2"


def test_build_fusion_retrieval_row_reports_missing_member() -> None:
    fusion_spec = {"name": "bad", "members": [("learned", "present"), ("residual", "missing")]}
    sample_rows = {("learned", "present"): {"subset": "cold", "domain": "Book", "ranked_ids": ["a"]}}

    with pytest.raises(ValueError, match="Fusion member residual:missing"):
        build_fusion_retrieval_row(fusion_spec, sample_rows, "sample-1", "target", (1,), [50], "rrf", 60.0, None)


from genrec.training.router_trainer import RouterTrainer, TrainerConfig, evaluate_route_predictions
from torch.utils.data import DataLoader


def _tiny_batch() -> dict:
    return {
        "history_embs": torch.randn(4, 3, 8),
        "history_mask": torch.ones(4, 3, dtype=torch.bool),
        "target_embedding": torch.randn(4, 8),
        "route1_idx": torch.tensor([0, 1, 2, 3]),
        "route2_idx": torch.tensor([0, 1, 0, 1]),
    }


def test_prefix1_training_objective_uses_route1_loss_only() -> None:
    model = LateBoundRouter(embedding_dim=8, hidden_dim=16, num_prefix1=4, num_prefix2=2)
    trainer = RouterTrainer(model, TrainerConfig(training_objective="prefix1"))

    losses = trainer._compute_losses(_tiny_batch())

    assert torch.allclose(losses["loss"], losses["route1_loss"])
    assert losses["route2_loss"].item() == 0.0
    assert losses["contrastive_loss"].item() == 0.0


def test_route_prediction_eval_reports_prefix1_topk() -> None:
    model = LateBoundRouter(embedding_dim=8, hidden_dim=16, num_prefix1=4, num_prefix2=2)
    batch = _tiny_batch()
    loader = DataLoader([batch], batch_size=None)

    metrics = evaluate_route_predictions(model, loader, device="cpu")

    assert "prefix1_top2_accuracy" in metrics
    assert "prefix1_top4_accuracy" in metrics


def test_selector_grid_can_include_domain_adaptive_and_multiple_merge_strategies() -> None:
    policies = make_policy_grid(
        route_beams=[4],
        route_score_weights=[0.0],
        per_route_topks=[50],
        merge_strategies=["zscore", "rrf"],
        fusion_methods=["round_robin"],
        include_single_sources=True,
        candidate_query_sources=["learned", "domain_adaptive"],
    )

    names = {policy.name for policy in policies}

    assert "domain_adaptive_single_p1b4_w0p0_k50_zscore" in names
    assert "domain_adaptive_single_p1b4_w0p0_k50_rrf" in names


def test_selector_grid_can_include_mixed_optional_fusion_groups() -> None:
    policies = make_policy_grid(
        route_beams=[4],
        route_score_weights=[0.0],
        per_route_topks=[50],
        merge_strategies=["zscore"],
        fusion_methods=["round_robin"],
        include_single_sources=False,
        candidate_query_sources=["learned", "pooled", "domain_adaptive", "prefix1_head"],
    )

    names = {policy.name for policy in policies}

    assert "learned_domain_adaptive_round_robin_p1b4_w0p0_k50_zscore" in names
    assert "pooled_prefix1_head_round_robin_p1b4_w0p0_k50_zscore" in names
    assert "domain_adaptive_prefix1_head_round_robin_p1b4_w0p0_k50_zscore" in names


def test_late_bound_fusion_router_outputs_candidate_logits_and_gate_weights() -> None:
    model = LateBoundFusionRouter(num_features=5, num_sources=3, hidden_dim=8)
    sample_features = torch.randn(2, 5)
    source_scores = torch.randn(2, 7, 3)
    route_scores = torch.randn(2, 7)

    logits, weights = model(sample_features, source_scores, route_scores)

    assert logits.shape == (2, 7)
    assert weights.shape == (2, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_rerank_with_routes_reports_candidate_pool_match_rank() -> None:
    memory = CatalogMemory(normalize=True, prefer_faiss=False)
    memory.add_items(
        ["a", "b", "target"],
        np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32),
        routes=[(1,), (1,), (1,)],
    )

    ranked_ids, _, _, diagnostics = rerank_with_routes(
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        route_candidates=[((1,), 0.0)],
        prefix_len=1,
        memory=memory,
        topks=[2],
        route_score_weight=0.0,
        merge_strategy="score",
        per_route_topk=3,
        target_item_id="target",
    )

    assert ranked_ids == ["a", "b"]
    assert diagnostics["candidate_pool_hit"] is True
    assert diagnostics["candidate_pool_size"] == 3
    assert diagnostics["candidate_pool_match_rank"] == 3
    assert diagnostics["candidate_pool_rank_cutoff"] == 3
