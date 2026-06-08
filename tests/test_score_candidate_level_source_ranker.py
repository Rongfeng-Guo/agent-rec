from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def load_module(name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def base_row(sample_id: str, candidate_id: str, label: int, rank: int | None, score: float) -> dict:
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
        "best_source_rank": rank,
        "mean_source_rank": rank,
        "min_source_rank": rank,
        "learned_score": score,
        "learned_present": rank is not None,
        "learned_rank": rank,
        "residual_score": score / 2,
        "residual_present": rank is not None,
        "residual_rank": rank,
        "mean_score": 0.0,
        "mean_present": False,
        "mean_rank": None,
        "prefix1_head_score": 0.0,
        "prefix1_head_present": False,
        "prefix1_head_rank": None,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_score_loaded_ranker_matches_direct_evaluation(tmp_path) -> None:
    train = load_module("train_candidate_level_source_ranker", "scripts/oracle_route_memory/train_candidate_level_source_ranker.py")
    scorer = load_module("score_candidate_level_source_ranker", "scripts/oracle_route_memory/score_candidate_level_source_ranker.py")
    rows = [
        base_row("s1", "a", 0, 3, 0.1),
        base_row("s1", "target", 1, 1, 0.9),
        base_row("s1", "b", 0, 2, 0.2),
        base_row("s2", "c", 0, 2, 0.2),
        base_row("s2", "target", 1, 1, 0.8),
        base_row("s2", "d", 0, 3, 0.1),
    ]
    train_rows = tmp_path / "train.jsonl"
    eval_rows = tmp_path / "eval.jsonl"
    write_rows(train_rows, rows)
    write_rows(eval_rows, rows)
    positives, negatives, _ = train.sample_pairwise_training_pairs(
        train_rows,
        negatives_per_positive=2,
        hard_negatives_per_positive=1,
        seed=7,
    )
    model = train.train_pairwise_model(
        positives,
        negatives,
        seed=7,
        epochs=60,
        batch_size=2,
        learning_rate=0.05,
        weight_decay=0.0,
    )
    direct_metric, _ = train.evaluate_model(model, eval_rows, topk=2)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"model": model, "feature_names": train.FEATURE_NAMES, "query_sources": train.QUERY_SOURCES}, handle)

    summary = scorer.score_loaded_ranker(
        model_path=model_path,
        eval_rows_path=eval_rows,
        output_dir=tmp_path / "scored",
        topk=2,
    )

    assert summary["eval_metric"]["Recall@2"] == direct_metric["Recall@2"]
    assert "fresh confirmation report" in summary["next_target"]
    assert (tmp_path / "scored" / "scored_outputs.json").exists()

    rendered = (tmp_path / "scored" / "report.md").read_text(encoding="utf-8")
    assert "## Next Target" in rendered
    assert "do not retrain" in rendered


def test_load_model_bundle_rejects_feature_mismatch(tmp_path) -> None:
    scorer = load_module("score_candidate_level_source_ranker", "scripts/oracle_route_memory/score_candidate_level_source_ranker.py")
    model_path = tmp_path / "bad_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"model": object(), "feature_names": ["wrong"], "query_sources": []}, handle)

    try:
        scorer.load_model_bundle(model_path)
    except ValueError as exc:
        assert "feature_names" in str(exc)
    else:
        raise AssertionError("expected feature mismatch")



def test_load_model_bundle_accepts_legacy_main_pairwise_pickle(tmp_path) -> None:
    train = load_module("train_candidate_level_source_ranker", "scripts/oracle_route_memory/train_candidate_level_source_ranker.py")
    scorer = load_module("score_candidate_level_source_ranker", "scripts/oracle_route_memory/score_candidate_level_source_ranker.py")
    model = train.PairwiseLinearRanker(
        mean=np.zeros(len(train.FEATURE_NAMES), dtype=np.float32),
        scale=np.ones(len(train.FEATURE_NAMES), dtype=np.float32),
        weight=np.ones(len(train.FEATURE_NAMES), dtype=np.float32),
        bias=0.0,
    )
    main_module = sys.modules["__main__"]
    had_attr = hasattr(main_module, "PairwiseLinearRanker")
    old_attr = getattr(main_module, "PairwiseLinearRanker", None)
    old_module = train.PairwiseLinearRanker.__module__
    try:
        setattr(main_module, "PairwiseLinearRanker", train.PairwiseLinearRanker)
        train.PairwiseLinearRanker.__module__ = "__main__"
        model_path = tmp_path / "legacy_model.pkl"
        with model_path.open("wb") as handle:
            pickle.dump({"model": model, "feature_names": train.FEATURE_NAMES, "query_sources": train.QUERY_SOURCES}, handle)
    finally:
        train.PairwiseLinearRanker.__module__ = old_module
        if had_attr:
            setattr(main_module, "PairwiseLinearRanker", old_attr)
        else:
            delattr(main_module, "PairwiseLinearRanker")

    bundle = scorer.load_model_bundle(model_path)

    assert bundle["model"].__class__.__name__ == "PairwiseLinearRanker"
    assert hasattr(bundle["model"], "predict_scores")
