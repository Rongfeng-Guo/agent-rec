#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

QUERY_SOURCES = ("learned", "residual", "mean", "prefix1_head")
try:
    from scripts.oracle_route_memory.handoff_io import ensure_empty_output_dir
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from handoff_io import ensure_empty_output_dir

FEATURE_NAMES = (
    "route_score",
    "history_len_norm",
    "route_top1_confidence",
    "route_entropy",
    "bucket_size_log",
    "query_agreement",
    "candidate_pool_size_log",
    "num_route_candidates",
    "member_route_hit_count",
    "member_candidate_pool_hit_count",
    "source_presence_count",
    "source_score_max",
    "source_score_mean",
    "source_score_std",
    "best_source_score",
    "best_source_rank_inv",
    "best_source_rank_log_inv",
    "mean_source_rank_inv",
    "min_source_rank_inv",
    "learned_score",
    "learned_present",
    "learned_rank_inv",
    "learned_rank_log_inv",
    "residual_score",
    "residual_present",
    "residual_rank_inv",
    "residual_rank_log_inv",
    "mean_score",
    "mean_present",
    "mean_rank_inv",
    "mean_rank_log_inv",
    "prefix1_head_score",
    "prefix1_head_present",
    "prefix1_head_rank_inv",
    "prefix1_head_rank_log_inv",
)

NEXT_TARGET = (
    "If this is a validation training run, compare and lock the selected policy before any fresh labels are "
    "available. If this model is already locked, use score_candidate_level_source_ranker.py for loaded-model "
    "validation replay or fresh scoring without retraining, retuning, or changing the locked feature schema."
)


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def rank_inv(value: Any) -> float:
    rank = as_float(value, default=0.0)
    return 0.0 if rank <= 0 else 1.0 / rank


def rank_log_inv(value: Any) -> float:
    rank = as_float(value, default=0.0)
    return 0.0 if rank <= 0 else 1.0 / np.log1p(rank)


def feature_vector(row: Mapping[str, Any]) -> list[float]:
    values = [
        as_float(row.get("route_score")),
        as_float(row.get("history_len_norm")),
        as_float(row.get("route_top1_confidence")),
        as_float(row.get("route_entropy")),
        as_float(row.get("bucket_size_log")),
        as_float(row.get("query_agreement")),
        np.log1p(as_float(row.get("candidate_pool_size"))),
        as_float(row.get("num_route_candidates")),
        as_float(row.get("member_route_hit_count")),
        as_float(row.get("member_candidate_pool_hit_count")),
        as_float(row.get("source_presence_count")),
        as_float(row.get("source_score_max")),
        as_float(row.get("source_score_mean")),
        as_float(row.get("source_score_std")),
        as_float(row.get("best_source_score")),
        rank_inv(row.get("best_source_rank")),
        rank_log_inv(row.get("best_source_rank")),
        rank_inv(row.get("mean_source_rank")),
        rank_inv(row.get("min_source_rank")),
    ]
    for source in QUERY_SOURCES:
        values.extend(
            [
                as_float(row.get(f"{source}_score")),
                float(bool(row.get(f"{source}_present"))),
                rank_inv(row.get(f"{source}_rank")),
                rank_log_inv(row.get(f"{source}_rank")),
            ]
        )
    return [float(value) for value in values]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_sample_groups(path: Path) -> Iterator[list[dict[str, Any]]]:
    current_sample_id = None
    current_rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        sample_id = str(row["sample_id"])
        if current_sample_id is not None and sample_id != current_sample_id:
            yield current_rows
            current_rows = []
        current_sample_id = sample_id
        current_rows.append(row)
    if current_rows:
        yield current_rows


def hard_negative_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    rank = as_float(row.get("min_source_rank"), default=1e9)
    if rank <= 0:
        rank = 1e9
    return (
        rank,
        -as_float(row.get("source_score_max")),
        -as_float(row.get("route_score")),
    )


def select_negative_indices(
    negatives: Sequence[Mapping[str, Any]],
    *,
    positive_count: int,
    negatives_per_positive: int,
    hard_negatives_per_positive: int,
    rng: random.Random,
) -> tuple[list[int], int, int]:
    total_limit = min(len(negatives), positive_count * negatives_per_positive)
    hard_limit = min(total_limit, positive_count * hard_negatives_per_positive)
    hard_indices = set(sorted(range(len(negatives)), key=lambda idx: hard_negative_key(negatives[idx]))[:hard_limit])
    remaining_indices = [idx for idx in range(len(negatives)) if idx not in hard_indices]
    random_limit = max(total_limit - len(hard_indices), 0)
    random_indices = set(rng.sample(remaining_indices, random_limit)) if random_limit < len(remaining_indices) else set(remaining_indices)
    return sorted(hard_indices | random_indices), len(hard_indices), len(random_indices)


def sample_training_rows(
    train_rows_path: Path,
    *,
    negatives_per_positive: int,
    hard_negatives_per_positive: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = random.Random(seed)
    features: list[list[float]] = []
    labels: list[int] = []
    skipped_groups = 0
    sampled_negative_count = 0
    hard_negative_count = 0
    random_negative_count = 0
    positive_count = 0
    for group in iter_sample_groups(train_rows_path):
        positives = [row for row in group if int(row.get("label", 0)) == 1]
        if not positives:
            skipped_groups += 1
            continue
        negatives = [row for row in group if int(row.get("label", 0)) == 0]
        sampled_indices, hard_count, random_count = select_negative_indices(
            negatives,
            positive_count=len(positives),
            negatives_per_positive=negatives_per_positive,
            hard_negatives_per_positive=hard_negatives_per_positive,
            rng=rng,
        )
        sampled_negatives = [negatives[idx] for idx in sampled_indices]
        for row in positives + sampled_negatives:
            features.append(feature_vector(row))
            labels.append(int(row.get("label", 0)))
        positive_count += len(positives)
        sampled_negative_count += len(sampled_negatives)
        hard_negative_count += hard_count
        random_negative_count += random_count
    if not features:
        raise ValueError(f"No supervised rows found in {train_rows_path}")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        {
            "positive_count": positive_count,
            "sampled_negative_count": sampled_negative_count,
            "hard_negative_count": hard_negative_count,
            "random_negative_count": random_negative_count,
            "skipped_groups_without_positive": skipped_groups,
        },
    )


def train_model(features: np.ndarray, labels: np.ndarray, *, seed: int, max_iter: int) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    alpha=1e-5,
                    l1_ratio=0.05,
                    class_weight="balanced",
                    max_iter=max_iter,
                    tol=1e-4,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(features, labels)
    return model


@dataclass
class PairwiseLinearRanker:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean) / self.scale
        return normalized @ self.weight + self.bias


def sample_pairwise_training_pairs(
    train_rows_path: Path,
    *,
    negatives_per_positive: int,
    hard_negatives_per_positive: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = random.Random(seed)
    positive_features: list[list[float]] = []
    negative_features: list[list[float]] = []
    skipped_groups = 0
    positive_count = 0
    pair_count = 0
    sampled_negative_count = 0
    hard_negative_count = 0
    random_negative_count = 0
    for group in iter_sample_groups(train_rows_path):
        positives = [row for row in group if int(row.get("label", 0)) == 1]
        if not positives:
            skipped_groups += 1
            continue
        negatives = [row for row in group if int(row.get("label", 0)) == 0]
        sampled_indices, hard_count, random_count = select_negative_indices(
            negatives,
            positive_count=len(positives),
            negatives_per_positive=negatives_per_positive,
            hard_negatives_per_positive=hard_negatives_per_positive,
            rng=rng,
        )
        sampled_negatives = [negatives[idx] for idx in sampled_indices]
        for positive in positives:
            positive_vector = feature_vector(positive)
            for negative in sampled_negatives:
                positive_features.append(positive_vector)
                negative_features.append(feature_vector(negative))
        positive_count += len(positives)
        sampled_negative_count += len(sampled_negatives)
        hard_negative_count += hard_count
        random_negative_count += random_count
        pair_count += len(positives) * len(sampled_negatives)
    if not positive_features:
        raise ValueError(f"No pairwise training pairs found in {train_rows_path}")
    return (
        np.asarray(positive_features, dtype=np.float32),
        np.asarray(negative_features, dtype=np.float32),
        {
            "positive_count": positive_count,
            "sampled_negative_count": sampled_negative_count,
            "hard_negative_count": hard_negative_count,
            "random_negative_count": random_negative_count,
            "pair_count": pair_count,
            "skipped_groups_without_positive": skipped_groups,
        },
    )


def train_pairwise_model(
    positive_features: np.ndarray,
    negative_features: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> PairwiseLinearRanker:
    torch.manual_seed(seed)
    all_features = np.concatenate([positive_features, negative_features], axis=0)
    mean = all_features.mean(axis=0).astype(np.float32)
    scale = all_features.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    positive_tensor = torch.from_numpy((positive_features - mean) / scale).float()
    negative_tensor = torch.from_numpy((negative_features - mean) / scale).float()
    model = torch.nn.Linear(positive_tensor.shape[1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_pairs = int(positive_tensor.shape[0])
    for _ in range(epochs):
        permutation = torch.randperm(num_pairs)
        for start in range(0, num_pairs, batch_size):
            indices = permutation[start : start + batch_size]
            pos_scores = model(positive_tensor[indices]).squeeze(-1)
            neg_scores = model(negative_tensor[indices]).squeeze(-1)
            loss = F.softplus(-(pos_scores - neg_scores)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    weight = model.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
    bias = float(model.bias.detach().cpu().item())
    return PairwiseLinearRanker(mean=mean, scale=scale, weight=weight, bias=bias)


def score_features(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_scores"):
        return np.asarray(model.predict_scores(features), dtype=np.float32)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float32)
    return np.asarray(model.decision_function(features), dtype=np.float32)


def summarize_eval_rows(rows: Sequence[Mapping[str, Any]], topk: int) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            f"Recall@{topk}": 0.0,
            "CandidatePoolHitRate": 0.0,
            "RouteHitRate": 0.0,
            "CandidatePoolLossRate": 0.0,
            "ConditionalRecall@50GivenPoolHit": 0.0,
            "AvgCandidatePoolMatchRank": None,
            "AvgPoolHitRankMissMatchRank": None,
            f"OracleSourceHit@{topk}Rate": 0.0,
            "AvgOracleSourceMatchRank": None,
        }
    hits = []
    pool_hits = []
    route_hits = []
    pool_losses = []
    candidate_pool_match_ranks = []
    pool_hit_rank_miss_ranks = []
    oracle_source_hits = []
    oracle_source_ranks = []
    for row in rows:
        rank = row.get("match_rank")
        hit = rank is not None and int(rank) <= topk
        pool_hit = bool(row.get("candidate_pool_hit"))
        hits.append(float(hit))
        pool_hits.append(float(pool_hit))
        route_hits.append(float(row.get("route_hit")))
        pool_losses.append(float(pool_hit and not hit))
        if row.get("candidate_pool_match_rank") is not None:
            candidate_pool_match_ranks.append(float(row["candidate_pool_match_rank"]))
            if pool_hit and not hit:
                pool_hit_rank_miss_ranks.append(float(row["candidate_pool_match_rank"]))
        oracle_rank = row.get("oracle_source_match_rank")
        oracle_source_hits.append(float(oracle_rank is not None and float(oracle_rank) <= topk))
        if oracle_rank is not None:
            oracle_source_ranks.append(float(oracle_rank))
    recall = float(np.mean(hits))
    pool_hit_rate = float(np.mean(pool_hits))
    return {
        "sample_count": len(rows),
        f"Recall@{topk}": recall,
        "CandidatePoolHitRate": pool_hit_rate,
        "RouteHitRate": float(np.mean(route_hits)),
        "CandidatePoolLossRate": float(np.mean(pool_losses)),
        "ConditionalRecall@50GivenPoolHit": float(recall / pool_hit_rate) if pool_hit_rate > 0 else 0.0,
        "AvgCandidatePoolMatchRank": float(np.mean(candidate_pool_match_ranks)) if candidate_pool_match_ranks else None,
        "AvgPoolHitRankMissMatchRank": float(np.mean(pool_hit_rank_miss_ranks)) if pool_hit_rank_miss_ranks else None,
        f"OracleSourceHit@{topk}Rate": float(np.mean(oracle_source_hits)),
        "AvgOracleSourceMatchRank": float(np.mean(oracle_source_ranks)) if oracle_source_ranks else None,
    }


def evaluate_model(model: Any, eval_rows_path: Path, *, topk: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outputs: list[dict[str, Any]] = []
    for group in iter_sample_groups(eval_rows_path):
        features = np.asarray([feature_vector(row) for row in group], dtype=np.float32)
        scores = score_features(model, features)
        ranked_indices = np.argsort(scores)[::-1].tolist()
        positive_indices = [idx for idx, row in enumerate(group) if int(row.get("label", 0)) == 1]
        positive_index = positive_indices[0] if positive_indices else None
        match_rank = None
        if positive_index is not None:
            match_rank = next(rank for rank, candidate_idx in enumerate(ranked_indices, start=1) if candidate_idx == positive_index)
        first = group[0]
        positive_row = group[positive_index] if positive_index is not None else None
        source_target_ranks = {}
        source_candidate_hits = {}
        if positive_row is not None:
            for source in QUERY_SOURCES:
                rank_value = positive_row.get(f"{source}_rank")
                source_target_ranks[source] = rank_value
                source_candidate_hits[source] = rank_value is not None
        ranked_ids = [str(group[candidate_idx]["candidate_id"]) for candidate_idx in ranked_indices[:topk]]
        outputs.append(
            {
                "sample_id": str(first["sample_id"]),
                "domain": str(first["domain"]),
                "target_item_id": str(first["target_item_id"]),
                "match_rank": match_rank,
                "route_hit": bool(first.get("route_hit")),
                "candidate_pool_hit": bool(first.get("candidate_pool_hit")),
                "candidate_pool_size": int(first.get("candidate_pool_size", len(group))),
                "candidate_pool_match_rank": match_rank if positive_index is not None else None,
                "candidate_pool_rank_cutoff": len(group),
                "num_route_candidates": int(first.get("num_route_candidates", 0)),
                "member_route_hit_count": int(first.get("member_route_hit_count", 0)),
                "member_candidate_pool_hit_count": int(first.get("member_candidate_pool_hit_count", 0)),
                "source_target_ranks": source_target_ranks,
                "source_candidate_hits": source_candidate_hits,
                "oracle_source_match_rank": positive_row.get("min_source_rank") if positive_row is not None else None,
                "oracle_source_hit_at_topk": bool(
                    positive_row is not None
                    and positive_row.get("min_source_rank") is not None
                    and float(positive_row["min_source_rank"]) <= topk
                ),
                "ranked_ids": ranked_ids,
                "ranker_score_top1": float(scores[ranked_indices[0]]) if ranked_indices else None,
            }
        )
    return summarize_eval_rows(outputs, topk=topk), outputs


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_optional_float(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def render_report(summary: Mapping[str, Any], topk: int) -> str:
    train = summary["train_sampling"]
    metric = summary["eval_metric"]
    lines = [
        "# H5 Candidate-Level Source Ranker",
        "",
        f"- Model: `{summary['model']}`",
        f"- Objective: `{summary['objective']}`",
        f"- Train rows: `{summary['train_rows_path']}`",
        f"- Eval rows: `{summary['eval_rows_path']}`",
        f"- Negatives per positive: `{summary['negatives_per_positive']}`",
        f"- Hard negatives per positive: `{summary['hard_negatives_per_positive']}`",
        "",
        "## Training Sample",
        "",
        f"- positives: `{train['positive_count']}`",
        f"- sampled negatives: `{train['sampled_negative_count']}`",
        f"- hard negatives: `{train['hard_negative_count']}`",
        f"- random negatives: `{train['random_negative_count']}`",
        f"- pairs: `{train.get('pair_count', '')}`",
        f"- skipped groups without positives: `{train['skipped_groups_without_positive']}`",
        "",
        "## Evaluation",
        "",
        f"- Recall@{topk}: `{metric[f'Recall@{topk}']:.6f}`",
        f"- CandidatePoolHitRate: `{metric['CandidatePoolHitRate']:.6f}`",
        f"- ConditionalRecall@50GivenPoolHit: `{metric['ConditionalRecall@50GivenPoolHit']:.6f}`",
        f"- AvgCandidatePoolMatchRank: `{format_optional_float(metric['AvgCandidatePoolMatchRank'])}`",
        f"- AvgPoolHitRankMissMatchRank: `{format_optional_float(metric['AvgPoolHitRankMissMatchRank'])}`",
        f"- OracleSourceHit@{topk}: `{metric[f'OracleSourceHit@{topk}Rate']:.6f}`",
        f"- AvgOracleSourceMatchRank: `{format_optional_float(metric['AvgOracleSourceMatchRank'])}`",
        "",
        "## Next Target",
        "",
        str(summary.get("next_target", NEXT_TARGET)),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an H5 candidate-level source/rank ranker.")
    parser.add_argument("--train-rows", required=True)
    parser.add_argument("--eval-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", choices=["pointwise", "pairwise"], default="pointwise")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--negatives-per-positive", type=int, default=200)
    parser.add_argument("--hard-negatives-per-positive", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--pairwise-epochs", type=int, default=20)
    parser.add_argument("--pairwise-batch-size", type=int, default=4096)
    parser.add_argument("--pairwise-learning-rate", type=float, default=5e-3)
    parser.add_argument("--pairwise-weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_empty_output_dir(output_dir)
    train_rows_path = Path(args.train_rows)
    eval_rows_path = Path(args.eval_rows)
    if args.objective == "pointwise":
        features, labels, train_sampling = sample_training_rows(
            train_rows_path,
            negatives_per_positive=int(args.negatives_per_positive),
            hard_negatives_per_positive=int(args.hard_negatives_per_positive),
            seed=int(args.seed),
        )
        model = train_model(features, labels, seed=int(args.seed), max_iter=int(args.max_iter))
        model_name = "sklearn.Pipeline(StandardScaler, SGDClassifier(log_loss, elasticnet))"
    else:
        positive_features, negative_features, train_sampling = sample_pairwise_training_pairs(
            train_rows_path,
            negatives_per_positive=int(args.negatives_per_positive),
            hard_negatives_per_positive=int(args.hard_negatives_per_positive),
            seed=int(args.seed),
        )
        model = train_pairwise_model(
            positive_features,
            negative_features,
            seed=int(args.seed),
            epochs=int(args.pairwise_epochs),
            batch_size=int(args.pairwise_batch_size),
            learning_rate=float(args.pairwise_learning_rate),
            weight_decay=float(args.pairwise_weight_decay),
        )
        model_name = "PairwiseLinearRanker(softplus(pos-neg))"
    eval_metric, eval_outputs = evaluate_model(model, eval_rows_path, topk=int(args.topk))

    model_path = output_dir / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"model": model, "feature_names": FEATURE_NAMES, "query_sources": QUERY_SOURCES}, handle)
    outputs_path = output_dir / "cold_like_outputs.json"
    write_json(outputs_path, eval_outputs)
    summary = {
        "name": "H5CandidateLevelSourceRanker",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "model": model_name,
        "objective": str(args.objective),
        "feature_names": list(FEATURE_NAMES),
        "query_sources": list(QUERY_SOURCES),
        "train_rows_path": str(train_rows_path),
        "eval_rows_path": str(eval_rows_path),
        "negatives_per_positive": int(args.negatives_per_positive),
        "hard_negatives_per_positive": int(args.hard_negatives_per_positive),
        "max_iter": int(args.max_iter),
        "pairwise_epochs": int(args.pairwise_epochs),
        "pairwise_batch_size": int(args.pairwise_batch_size),
        "pairwise_learning_rate": float(args.pairwise_learning_rate),
        "pairwise_weight_decay": float(args.pairwise_weight_decay),
        "seed": int(args.seed),
        "train_sampling": train_sampling,
        "eval_metric": eval_metric,
        "files": {
            "model": str(model_path),
            "cold_like_outputs": str(outputs_path),
        },
        "next_target": NEXT_TARGET,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(render_report(summary, int(args.topk)), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir.resolve()),
                f"Recall@{args.topk}": eval_metric[f"Recall@{args.topk}"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
