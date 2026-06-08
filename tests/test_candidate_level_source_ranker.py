from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "train_candidate_level_source_ranker.py"
    spec = importlib.util.spec_from_file_location("train_candidate_level_source_ranker", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def base_row(sample_id: str, candidate_id: str, label: int, best_rank: int | None, score: float) -> dict:
    return {
        "sample_id": sample_id,
        "domain": "Book",
        "target_item_id": "target",
        "candidate_id": candidate_id,
        "label": label,
        "route_hit": True,
        "candidate_pool_hit": True,
        "candidate_pool_size": 3,
        "num_route_candidates": 2,
        "member_route_hit_count": 2,
        "member_candidate_pool_hit_count": 2,
        "route_score": score,
        "history_len_norm": 0.4,
        "route_top1_confidence": 0.8,
        "route_entropy": 0.2,
        "bucket_size_log": 0.1,
        "query_agreement": 0.6,
        "source_presence_count": 2,
        "source_score_max": score,
        "source_score_mean": score,
        "source_score_std": 0.0,
        "best_source_score": score,
        "best_source_rank": best_rank,
        "mean_source_rank": best_rank,
        "min_source_rank": best_rank,
        "learned_score": score,
        "learned_present": best_rank is not None,
        "learned_rank": best_rank,
        "residual_score": score / 2,
        "residual_present": best_rank is not None,
        "residual_rank": best_rank,
        "mean_score": 0.0,
        "mean_present": False,
        "mean_rank": None,
        "prefix1_head_score": 0.0,
        "prefix1_head_present": False,
        "prefix1_head_rank": None,
    }


def test_feature_vector_has_declared_length_and_handles_missing_ranks() -> None:
    module = load_module()
    row = base_row("s1", "a", 0, None, 0.1)

    features = module.feature_vector(row)

    assert len(features) == len(module.FEATURE_NAMES)
    assert all(np.isfinite(features))
    assert features[module.FEATURE_NAMES.index("best_source_rank_inv")] == 0.0


def test_iter_sample_groups_preserves_contiguous_groups(tmp_path) -> None:
    module = load_module()
    path = tmp_path / "rows.jsonl"
    rows = [
        base_row("s1", "a", 0, 2, 0.1),
        base_row("s1", "target", 1, 1, 0.9),
        base_row("s2", "b", 0, None, 0.2),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    groups = list(module.iter_sample_groups(path))

    assert [[row["candidate_id"] for row in group] for group in groups] == [["a", "target"], ["b"]]


def test_evaluate_model_outputs_analyzer_compatible_rows(tmp_path) -> None:
    module = load_module()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    rows = [
        base_row("s1", "a", 0, 3, 0.1),
        base_row("s1", "target", 1, 1, 0.9),
        base_row("s1", "b", 0, 2, 0.2),
        base_row("s2", "c", 0, 2, 0.2),
        base_row("s2", "target", 1, 1, 0.8),
        base_row("s2", "d", 0, 3, 0.1),
    ]
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    train_path.write_text(payload, encoding="utf-8")
    eval_path.write_text(payload, encoding="utf-8")
    features, labels, sampling = module.sample_training_rows(
        train_path,
        negatives_per_positive=2,
        hard_negatives_per_positive=1,
        seed=7,
    )
    model = module.train_model(features, labels, seed=7, max_iter=200)

    metrics, outputs = module.evaluate_model(model, eval_path, topk=2)

    assert sampling["hard_negative_count"] == 2
    assert metrics["sample_count"] == 2
    assert metrics["Recall@2"] == 1.0
    assert outputs[0]["candidate_pool_match_rank"] == 1
    assert outputs[0]["ranked_ids"][0] == "target"


def test_ensure_empty_output_dir_rejects_non_empty_dir(tmp_path) -> None:
    module = load_module()
    output_dir = tmp_path / "ranker"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("do not overwrite\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists and is not empty"):
        module.ensure_empty_output_dir(output_dir)


def test_render_report_includes_next_target_and_handles_missing_rank_metrics() -> None:
    module = load_module()
    summary = {
        "model": "PairwiseLinearRanker(softplus(pos-neg))",
        "objective": "pairwise",
        "train_rows_path": "train.jsonl",
        "eval_rows_path": "eval.jsonl",
        "negatives_per_positive": 500,
        "hard_negatives_per_positive": 100,
        "train_sampling": {
            "positive_count": 1,
            "sampled_negative_count": 1,
            "hard_negative_count": 0,
            "random_negative_count": 1,
            "pair_count": 1,
            "skipped_groups_without_positive": 0,
        },
        "eval_metric": {
            "Recall@50": 0.0,
            "CandidatePoolHitRate": 0.0,
            "ConditionalRecall@50GivenPoolHit": 0.0,
            "AvgCandidatePoolMatchRank": None,
            "AvgPoolHitRankMissMatchRank": None,
            "OracleSourceHit@50Rate": 0.0,
            "AvgOracleSourceMatchRank": None,
        },
        "next_target": module.NEXT_TARGET,
    }

    rendered = module.render_report(summary, topk=50)

    assert "## Next Target" in rendered
    assert "score_candidate_level_source_ranker.py" in rendered
    assert "- AvgCandidatePoolMatchRank: `n/a`" in rendered


def test_pairwise_model_outputs_analyzer_compatible_rows(tmp_path) -> None:
    module = load_module()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    rows = [
        base_row("s1", "a", 0, 3, 0.1),
        base_row("s1", "target", 1, 1, 0.9),
        base_row("s1", "b", 0, 2, 0.2),
        base_row("s2", "c", 0, 2, 0.2),
        base_row("s2", "target", 1, 1, 0.8),
        base_row("s2", "d", 0, 3, 0.1),
    ]
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    train_path.write_text(payload, encoding="utf-8")
    eval_path.write_text(payload, encoding="utf-8")
    positive_features, negative_features, sampling = module.sample_pairwise_training_pairs(
        train_path,
        negatives_per_positive=2,
        hard_negatives_per_positive=1,
        seed=7,
    )
    model = module.train_pairwise_model(
        positive_features,
        negative_features,
        seed=7,
        epochs=60,
        batch_size=2,
        learning_rate=0.05,
        weight_decay=0.0,
    )

    metrics, outputs = module.evaluate_model(model, eval_path, topk=2)

    assert sampling["pair_count"] == 4
    assert metrics["sample_count"] == 2
    assert metrics["Recall@2"] == 1.0
    assert outputs[0]["candidate_pool_match_rank"] == 1
    assert outputs[0]["ranked_ids"][0] == "target"
