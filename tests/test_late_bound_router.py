from __future__ import annotations

import torch
import json

from genrec.models.late_bound_router import LateBoundRouter


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


from genrec.training.router_dataset import RouteVocab
from scripts.oracle_route_memory.eval_predicted_route import (
    add_fusion_query_sources,
    enumerate_prefix1_candidates,
    fuse_ranked_lists,
    load_domain_query_source_config,
    merge_candidate_rows,
    parse_domain_query_sources,
    parse_fusion_specs,
    resolve_query_source,
)


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
