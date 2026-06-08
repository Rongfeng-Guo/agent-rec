#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from genrec.models import LateBoundFusionRouter
from genrec.training import build_training_examples, load_protocol_manifest, load_route_mapping, protocol_split_examples
from genrec.training.router_dataset import RouterDataset
from scripts.oracle_route_memory.eval_predicted_route import (
    build_memory,
    enumerate_prefix1_candidates,
    load_model,
    load_prefix1_query_head,
    mean_history_embedding_from_ids,
)
from genrec.memory.data_adapter import load_item_embeddings

try:
    from scripts.oracle_route_memory.handoff_io import ensure_empty_output_dir, resolve_output_dir, resolve_repo_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from handoff_io import ensure_empty_output_dir, resolve_output_dir, resolve_repo_path


NEXT_TARGET = (
    "Use this H4 late-bound fusion run only as validation evidence, then compare "
    "its bottleneck rows against the locked H5-D candidate-level source ranker "
    "before any fresh-confirmation claim."
)


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


@dataclass
class FusionConfig:
    query_sources: Tuple[str, ...]
    prefix1_beam: int
    per_route_topk: int
    topk: int
    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    device: str
    seed: int


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "t"}


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _normalize_scores(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-6:
        std = 1.0
    return [float((score - mean) / std) for score in arr]


def _route_entropy(route1_log_probs: torch.Tensor) -> float:
    probs = route1_log_probs.exp()
    entropy = float(-(probs * route1_log_probs).sum().item())
    denom = math.log(max(int(route1_log_probs.shape[-1]), 2))
    return entropy / denom


def _query_agreement(top_lists: Sequence[Sequence[str]], k: int = 10) -> float:
    usable = [set(ranked[:k]) for ranked in top_lists if ranked]
    if len(usable) < 2:
        return 0.0
    overlaps = []
    for left, right in combinations(usable, 2):
        denom = len(left | right)
        overlaps.append(0.0 if denom == 0 else len(left & right) / denom)
    return float(sum(overlaps) / len(overlaps)) if overlaps else 0.0


def collect_source_candidates(
    memory: Any,
    query_embedding: np.ndarray,
    route_candidates: Sequence[Tuple[Tuple[int, ...], float]],
    prefix_len: int,
    per_route_topk: int,
    rank_topk: int,
) -> Dict[str, Any]:
    candidate_rows: Dict[str, float] = {}
    route_scores: Dict[str, float] = {}
    per_route_ranked: List[List[str]] = []
    fallback_used = False
    bucket_sizes = []

    for route_tuple, route_log_prob in route_candidates:
        route_key = route_tuple[:prefix_len]
        bucket_sizes.append(int(memory.candidate_count(route_key)))
        if not memory.has_route(route_key):
            fallback_used = True
        results = memory.search(query_embedding, route=route_key, topk=per_route_topk)
        if not results:
            continue
        item_ids = [str(result["item_id"]) for result in results]
        normalized = _normalize_scores([float(result["score"]) for result in results])
        per_route_ranked.append(item_ids[:rank_topk])
        for item_id, score in zip(item_ids, normalized):
            if score > candidate_rows.get(item_id, -1e9):
                candidate_rows[item_id] = float(score)
            route_scores[item_id] = max(route_scores.get(item_id, -1e9), float(route_log_prob))

    if not candidate_rows:
        fallback_used = True
        results = memory.search(query_embedding, route=None, topk=per_route_topk)
        item_ids = [str(result["item_id"]) for result in results]
        normalized = _normalize_scores([float(result["score"]) for result in results])
        per_route_ranked.append(item_ids[:rank_topk])
        for item_id, score in zip(item_ids, normalized):
            candidate_rows[item_id] = float(score)
            route_scores[item_id] = 0.0
        if not bucket_sizes:
            bucket_sizes.append(int(memory.candidate_count(None)))

    ranked_ids = [
        item_id
        for item_id, _ in sorted(
            candidate_rows.items(),
            key=lambda item: (item[1] + 0.1 * route_scores.get(item[0], 0.0)),
            reverse=True,
        )[:rank_topk]
    ]
    return {
        "ranked_ids": ranked_ids,
        "score_map": candidate_rows,
        "route_score_map": route_scores,
        "fallback_used": fallback_used,
        "bucket_size": float(sum(bucket_sizes) / max(len(bucket_sizes), 1)),
        "candidate_pool_size": len(candidate_rows),
        "top_lists": per_route_ranked,
    }


def candidate_union_from_source_outputs(
    source_outputs: Mapping[str, Mapping[str, Any]],
    query_sources: Sequence[str],
) -> List[str]:
    return sorted(
        {
            str(item_id)
            for source_name in query_sources
            for item_id in source_outputs[source_name].get("score_map", {}).keys()
        }
    )


def summarize_rows(rows: Sequence[Mapping[str, Any]], topk: int) -> Dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "Recall@10": 0.0,
            "Recall@20": 0.0,
            "Recall@50": 0.0,
            "RouteHitRate": 0.0,
            "CandidatePoolHitRate": 0.0,
            "CandidatePoolLossRate": 0.0,
            "ConditionalRecall@50GivenPoolHit": 0.0,
            "AvgCandidatePoolMatchRank": None,
            "AvgPoolHitRankMissMatchRank": None,
            f"OracleSourceHit@{topk}Rate": 0.0,
            "AvgOracleSourceMatchRank": None,
        }
    recalls = {10: [], 20: [], 50: []}
    route_hits = []
    pool_hits = []
    pool_losses = []
    for row in rows:
        rank = row["match_rank"]
        is_hit_at_topk = rank is not None and rank <= topk
        for k in recalls:
            recalls[k].append(float(rank is not None and rank <= k))
        route_hits.append(float(row["route_hit"]))
        pool_hits.append(float(row["candidate_pool_hit"]))
        pool_losses.append(float(row["candidate_pool_hit"] and not is_hit_at_topk))
    candidate_pool_match_ranks = [
        float(row["candidate_pool_match_rank"]) for row in rows if row.get("candidate_pool_match_rank") is not None
    ]
    pool_hit_rank_miss_ranks = [
        float(row["candidate_pool_match_rank"])
        for row in rows
        if row.get("candidate_pool_match_rank") is not None and row["candidate_pool_hit"] and row["match_rank"] > topk
    ]
    oracle_source_hits = [float(row.get("oracle_source_hit_at_topk", False)) for row in rows]
    oracle_source_match_ranks = [
        float(row["oracle_source_match_rank"]) for row in rows if row.get("oracle_source_match_rank") is not None
    ]
    recall50 = float(np.mean(recalls[50]))
    pool_hit_rate = float(np.mean(pool_hits))
    return {
        "sample_count": len(rows),
        "Recall@10": float(np.mean(recalls[10])),
        "Recall@20": float(np.mean(recalls[20])),
        "Recall@50": recall50,
        "RouteHitRate": float(np.mean(route_hits)),
        "CandidatePoolHitRate": pool_hit_rate,
        "CandidatePoolLossRate": float(np.mean(pool_losses)),
        "ConditionalRecall@50GivenPoolHit": float(recall50 / pool_hit_rate) if pool_hit_rate > 0 else 0.0,
        "AvgCandidatePoolMatchRank": float(np.mean(candidate_pool_match_ranks)) if candidate_pool_match_ranks else None,
        "AvgPoolHitRankMissMatchRank": float(np.mean(pool_hit_rank_miss_ranks)) if pool_hit_rank_miss_ranks else None,
        f"OracleSourceHit@{topk}Rate": float(np.mean(oracle_source_hits)),
        "AvgOracleSourceMatchRank": float(np.mean(oracle_source_match_ranks)) if oracle_source_match_ranks else None,
    }


def build_gate_examples(
    examples: Sequence[Any],
    item_embeddings: Mapping[str, np.ndarray],
    route_mapping: Mapping[str, Tuple[int, int]],
    memory: Any,
    router_model: Any,
    route_vocab: Any,
    prefix1_query_head: Any,
    config: FusionConfig,
) -> List[Dict[str, Any]]:
    dataset = RouterDataset(examples, item_embeddings, route_vocab)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=dataset.collate_fn)
    rows: List[Dict[str, Any]] = []
    device = config.device

    with torch.no_grad():
        for batch in loader:
            outputs = router_model(batch["history_embs"].to(device), batch["history_mask"].to(device))
            route1_log_probs, _, _ = router_model.route_log_probs(outputs)
            prefix1_candidates = enumerate_prefix1_candidates(route1_log_probs.cpu(), route_vocab, config.prefix1_beam)
            pooled_embeddings = torch.nn.functional.normalize(outputs["pooled_history"], dim=-1).cpu().numpy()
            learned_embeddings = outputs["query_embedding"].cpu().numpy()
            residual_embeddings = torch.nn.functional.normalize(outputs["query_embedding"] + outputs["pooled_history"], dim=-1).cpu().numpy()
            mean_embeddings = np.stack(
                [mean_history_embedding_from_ids(history, item_embeddings) for history in batch["history_item_ids"]],
                axis=0,
            )
            query_embeddings = {
                "learned": learned_embeddings,
                "residual": residual_embeddings,
                "pooled": pooled_embeddings,
                "mean": mean_embeddings,
            }
            if prefix1_query_head is not None:
                query_embeddings["prefix1_head"] = prefix1_query_head(
                    batch["history_embs"].to(device),
                    batch["history_mask"].to(device),
                ).cpu().numpy()

            for idx, target_item_id in enumerate(batch["target_item_id"]):
                route_candidates = prefix1_candidates[idx]
                target_item_id_str = str(target_item_id)
                true_prefix1 = (int(route_mapping[str(target_item_id)][0]),)
                route_hit = any(tuple(route) == true_prefix1 for route, _ in route_candidates)
                source_outputs = {}
                route_score_union: Dict[str, float] = {}
                ranked_lists = []
                bucket_sizes = []

                for source_name in config.query_sources:
                    source_outputs[source_name] = collect_source_candidates(
                        memory=memory,
                        query_embedding=query_embeddings[source_name][idx],
                        route_candidates=route_candidates,
                        prefix_len=1,
                        per_route_topk=config.per_route_topk,
                        rank_topk=config.topk,
                    )
                    ranked_lists.append(source_outputs[source_name]["ranked_ids"])
                    bucket_sizes.append(float(source_outputs[source_name]["bucket_size"]))
                    for item_id, route_score in source_outputs[source_name]["route_score_map"].items():
                        route_score_union[item_id] = max(route_score_union.get(item_id, -1e9), float(route_score))

                candidate_union = candidate_union_from_source_outputs(source_outputs, config.query_sources)
                if not candidate_union:
                    continue

                source_score_matrix = np.zeros((len(candidate_union), len(config.query_sources)), dtype=np.float32)
                source_presence_matrix = np.zeros((len(candidate_union), len(config.query_sources)), dtype=np.bool_)
                route_score_vector = np.zeros((len(candidate_union),), dtype=np.float32)
                for item_idx, item_id in enumerate(candidate_union):
                    route_score_vector[item_idx] = float(route_score_union.get(item_id, 0.0))
                    for source_idx, source_name in enumerate(config.query_sources):
                        if item_id in source_outputs[source_name]["score_map"]:
                            source_presence_matrix[item_idx, source_idx] = True
                        source_score_matrix[item_idx, source_idx] = float(
                            source_outputs[source_name]["score_map"].get(item_id, 0.0)
                        )

                sample_features = np.asarray(
                    [
                        min(len(batch["history_item_ids"][idx]), 10) / 10.0,
                        float(route1_log_probs[idx].exp().max().item()),
                        _route_entropy(route1_log_probs[idx].cpu()),
                        math.log1p(sum(bucket_sizes) / max(len(bucket_sizes), 1)) / 8.0,
                        _query_agreement(ranked_lists, k=min(10, config.topk)),
                    ],
                    dtype=np.float32,
                )
                member_candidate_pool_hit_count = sum(
                    int(target_item_id_str in source_outputs[source_name]["score_map"]) for source_name in config.query_sources
                )
                rows.append(
                    {
                        "sample_id": str(batch["sample_id"][idx]),
                        "domain": str(batch["domain"][idx]),
                        "target_item_id": str(target_item_id),
                        "candidate_ids": candidate_union,
                        "query_sources": list(config.query_sources),
                        "source_scores": source_score_matrix,
                        "source_presence": source_presence_matrix,
                        "route_scores": route_score_vector,
                        "sample_features": sample_features,
                        "target_index": candidate_union.index(target_item_id_str) if target_item_id_str in candidate_union else -1,
                        "route_hit": bool(route_hit),
                        "candidate_pool_hit": target_item_id_str in candidate_union,
                        "candidate_pool_size": len(candidate_union),
                        "num_route_candidates": len(route_candidates),
                        "member_route_hit_count": len(config.query_sources) if route_hit else 0,
                        "member_candidate_pool_hit_count": member_candidate_pool_hit_count,
                    }
                )
    return rows


def evaluate_gate(model: LateBoundFusionRouter, rows: Sequence[Mapping[str, Any]], device: str, topk: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    outputs: List[Dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            sample_features = torch.from_numpy(np.asarray(row["sample_features"], dtype=np.float32)).unsqueeze(0).to(device)
            source_score_array = np.asarray(row["source_scores"], dtype=np.float32)
            source_scores = torch.from_numpy(source_score_array).unsqueeze(0).to(device)
            route_scores = torch.from_numpy(np.asarray(row["route_scores"], dtype=np.float32)).unsqueeze(0).to(device)
            logits, weights = model(sample_features, source_scores, route_scores)
            candidate_ids = list(row["candidate_ids"])
            ranked = torch.argsort(logits[0], descending=True).cpu().tolist()
            ranked_ids = [candidate_ids[candidate_idx] for candidate_idx in ranked[:topk]]
            target_index = int(row["target_index"])
            match_rank = None
            if target_index >= 0:
                for rank, candidate_idx in enumerate(ranked, start=1):
                    if int(candidate_idx) == target_index:
                        match_rank = rank
                        break
            candidate_pool_hit = bool(row["candidate_pool_hit"])
            weights_list = [float(value) for value in weights[0].cpu().tolist()]
            num_sources = int(source_score_array.shape[-1])
            query_sources = [str(value) for value in row.get("query_sources", [f"source_{idx}" for idx in range(num_sources)])]
            source_presence = np.asarray(row.get("source_presence", np.ones_like(source_score_array, dtype=np.bool_)), dtype=np.bool_)
            source_target_ranks: Dict[str, int | None] = {}
            source_candidate_hits: Dict[str, bool] = {}
            for source_idx, source_name in enumerate(query_sources):
                target_present = target_index >= 0 and bool(source_presence[target_index, source_idx])
                source_candidate_hits[source_name] = target_present
                if not target_present:
                    source_target_ranks[source_name] = None
                    continue
                present_indices = np.flatnonzero(source_presence[:, source_idx])
                ranked_source_indices = sorted(
                    (int(candidate_idx) for candidate_idx in present_indices),
                    key=lambda candidate_idx: float(source_score_array[candidate_idx, source_idx]),
                    reverse=True,
                )
                source_target_ranks[source_name] = next(
                    rank for rank, candidate_idx in enumerate(ranked_source_indices, start=1) if candidate_idx == target_index
                )
            observed_source_ranks = [rank for rank in source_target_ranks.values() if rank is not None]
            oracle_source_match_rank = min(observed_source_ranks) if observed_source_ranks else None
            outputs.append(
                {
                    "sample_id": row["sample_id"],
                    "domain": row["domain"],
                    "target_item_id": row["target_item_id"],
                    "match_rank": match_rank,
                    "route_hit": bool(row["route_hit"]),
                    "candidate_pool_hit": candidate_pool_hit,
                    "candidate_pool_size": int(row.get("candidate_pool_size", len(candidate_ids))),
                    "candidate_pool_match_rank": match_rank if candidate_pool_hit else None,
                    "candidate_pool_rank_cutoff": len(candidate_ids),
                    "num_route_candidates": int(row.get("num_route_candidates", 0)),
                    "member_route_hit_count": int(row.get("member_route_hit_count", 0)),
                    "member_candidate_pool_hit_count": int(row.get("member_candidate_pool_hit_count", 0)),
                    "source_target_ranks": source_target_ranks,
                    "source_candidate_hits": source_candidate_hits,
                    "oracle_source_match_rank": oracle_source_match_rank,
                    "oracle_source_hit_at_topk": bool(oracle_source_match_rank is not None and oracle_source_match_rank <= topk),
                    "ranked_ids": ranked_ids,
                    "source_weights": weights_list[:num_sources],
                    "route_weight": weights_list[num_sources] if len(weights_list) > num_sources else None,
                    "gate_weights": weights_list,
                }
            )
    return summarize_rows(outputs, topk=topk), outputs


def train_gate(
    model: LateBoundFusionRouter,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    config: FusionConfig,
) -> Dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    device = config.device
    best_metric = -1.0
    best_state = None
    history = []
    train_supervised_rows = [row for row in train_rows if int(row["target_index"]) >= 0]

    for epoch in range(1, config.epochs + 1):
        model.train()
        random.shuffle(train_supervised_rows)
        losses = []
        for row in train_supervised_rows:
            optimizer.zero_grad(set_to_none=True)
            sample_features = torch.from_numpy(np.asarray(row["sample_features"], dtype=np.float32)).unsqueeze(0).to(device)
            source_scores = torch.from_numpy(np.asarray(row["source_scores"], dtype=np.float32)).unsqueeze(0).to(device)
            route_scores = torch.from_numpy(np.asarray(row["route_scores"], dtype=np.float32)).unsqueeze(0).to(device)
            target = torch.tensor([int(row["target_index"])], dtype=torch.long, device=device)
            logits, _ = model(sample_features, source_scores, route_scores)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_metric, _ = evaluate_gate(model, train_rows, device, config.topk)
        val_metric, _ = evaluate_gate(model, val_rows, device, config.topk)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "train_recall@50": float(train_metric["Recall@50"]),
            "val_recall@50": float(val_metric["Recall@50"]),
            "val_candidate_pool_hit_rate": float(val_metric["CandidatePoolHitRate"]),
            "val_route_hit_rate": float(val_metric["RouteHitRate"]),
        }
        history.append(record)
        if record["val_recall@50"] > best_metric:
            best_metric = record["val_recall@50"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train_metric, train_outputs = evaluate_gate(model, train_rows, device, config.topk)
    final_val_metric, val_outputs = evaluate_gate(model, val_rows, device, config.topk)
    return {
        "history": history,
        "best_val_recall@50": best_metric,
        "train_metric": final_train_metric,
        "val_metric": final_val_metric,
        "train_outputs": train_outputs,
        "val_outputs": val_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight LateBoundFusionRouter on the official protocol.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--item-embedding-path", required=True)
    parser.add_argument("--item-sid-path", required=True)
    parser.add_argument("--router-checkpoint-dir", required=True)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--prefix1-query-head-checkpoint", default=None)
    parser.add_argument("--query-sources", nargs="+", default=["learned", "residual", "pooled"])
    parser.add_argument("--prefix1-beam", type=int, default=4)
    parser.add_argument("--per-route-topk", type=int, default=50)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, args.repo_root)
    data_dir = resolve_repo_path(args.data_dir, args.repo_root)
    item_embedding_path = resolve_repo_path(args.item_embedding_path, args.repo_root)
    item_sid_path = resolve_repo_path(args.item_sid_path, args.repo_root)
    router_checkpoint_dir = resolve_repo_path(args.router_checkpoint_dir, args.repo_root)
    protocol_manifest_path = resolve_repo_path(args.protocol_manifest, args.repo_root)
    prefix1_query_head_checkpoint = resolve_repo_path(args.prefix1_query_head_checkpoint, args.repo_root)
    ensure_empty_output_dir(output_dir)

    config = FusionConfig(
        query_sources=tuple(str(value) for value in args.query_sources),
        prefix1_beam=int(args.prefix1_beam),
        per_route_topk=int(args.per_route_topk),
        topk=int(args.topk),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        device=device,
        seed=int(args.seed),
    )

    protocol_manifest = load_protocol_manifest(protocol_manifest_path)
    item_embeddings = load_item_embeddings(data_dir, item_embedding_path)
    route_mapping = load_route_mapping(item_sid_path)
    router_model, route_vocab, router_meta = load_model(router_checkpoint_dir, device)
    prefix1_query_head = None
    prefix1_query_head_meta = None
    if "prefix1_head" in config.query_sources:
        if prefix1_query_head_checkpoint is None:
            raise ValueError("--prefix1-query-head-checkpoint is required when prefix1_head is used.")
        prefix1_query_head, prefix1_query_head_meta = load_prefix1_query_head(prefix1_query_head_checkpoint, device)

    all_examples = build_training_examples(data_dir, item_embeddings, route_mapping, max_history=args.max_history)
    train_examples = protocol_split_examples(all_examples, protocol_manifest, "train")
    cold_like_examples = protocol_split_examples(all_examples, protocol_manifest, "cold_like_val")
    heldout_items = set(str(item_id) for item_id in protocol_manifest.get("heldout_item_ids", []))

    train_visible_embeddings = {item_id: embedding for item_id, embedding in item_embeddings.items() if item_id not in heldout_items}
    train_visible_routes = {item_id: route for item_id, route in route_mapping.items() if item_id in train_visible_embeddings}
    train_memory = build_memory(train_visible_embeddings, train_visible_routes, prefix_len=1)
    full_memory = build_memory(item_embeddings, route_mapping, prefix_len=1)

    train_rows = build_gate_examples(
        train_examples,
        item_embeddings=item_embeddings,
        route_mapping=route_mapping,
        memory=train_memory,
        router_model=router_model,
        route_vocab=route_vocab,
        prefix1_query_head=prefix1_query_head,
        config=config,
    )
    cold_like_rows = build_gate_examples(
        cold_like_examples,
        item_embeddings=item_embeddings,
        route_mapping=route_mapping,
        memory=full_memory,
        router_model=router_model,
        route_vocab=route_vocab,
        prefix1_query_head=prefix1_query_head,
        config=config,
    )

    model = LateBoundFusionRouter(
        num_features=5,
        num_sources=len(config.query_sources),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    result = train_gate(model, train_rows, cold_like_rows, config)

    torch.save(model.state_dict(), output_dir / "model.pt")
    payload = {
        "model_config": {
            "num_features": 5,
            "num_sources": len(config.query_sources),
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
        },
        "feature_names": [
            "history_len_norm",
            "route_top1_confidence",
            "route_entropy",
            "bucket_size_log",
            "query_agreement",
        ],
        "query_sources": list(config.query_sources),
        "protocol_manifest": str(protocol_manifest_path.resolve()),
        "protocol_config_hash": protocol_manifest.get("config_hash"),
        "router_checkpoint_dir": str(router_checkpoint_dir.resolve()),
        "prefix1_query_head_checkpoint": str(prefix1_query_head_checkpoint.resolve()) if prefix1_query_head_checkpoint else None,
        "fusion_config": asdict(config),
        "train_result": result,
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "hostname": socket.gethostname(),
            "num_train_examples": len(train_examples),
            "num_cold_like_examples": len(cold_like_examples),
            "num_train_rows": len(train_rows),
            "num_cold_like_rows": len(cold_like_rows),
            "num_supervised_train_rows": sum(int(row["target_index"] >= 0) for row in train_rows),
            "num_supervised_cold_like_rows": sum(int(row["target_index"] >= 0) for row in cold_like_rows),
            "router_best": router_meta.get("train_result", {}).get("best", {}),
            "prefix1_query_head_best": (prefix1_query_head_meta or {}).get("train_result", {}).get("best", {}),
        },
        "method_config": {
            "name": "LateBoundFusionRouter",
            "query_sources": list(config.query_sources),
            "prefix1_beam": config.prefix1_beam,
            "per_route_topk": config.per_route_topk,
            "topk": config.topk,
            "candidate_scoring": "source_zscore + learned_gate(sample_features) + route_weight",
        },
        "next_target": NEXT_TARGET,
    }
    (output_dir / "checkpoint_meta.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "train_outputs.json").write_text(json.dumps(result["train_outputs"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "cold_like_outputs.json").write_text(json.dumps(result["val_outputs"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "method_config.json").write_text(json.dumps(payload["method_config"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_lines = [
        "# LateBoundFusionRouter",
        "",
        f"- Query sources: `{', '.join(config.query_sources)}`",
        f"- Prefix1 beam / per-route topk: `{config.prefix1_beam}` / `{config.per_route_topk}`",
        f"- Protocol hash: `{protocol_manifest.get('config_hash', '')}`",
        "",
        "## Train",
        "",
        f"- Recall@50: `{format_metric(result['train_metric']['Recall@50'])}`",
        f"- CandidatePoolHitRate: `{format_metric(result['train_metric']['CandidatePoolHitRate'])}`",
        "",
        "## Cold-Like Validation",
        "",
        f"- Recall@50: `{format_metric(result['val_metric']['Recall@50'])}`",
        f"- CandidatePoolHitRate: `{format_metric(result['val_metric']['CandidatePoolHitRate'])}`",
        f"- ConditionalRecall@50GivenPoolHit: `{format_metric(result['val_metric']['ConditionalRecall@50GivenPoolHit'])}`",
        f"- AvgCandidatePoolMatchRank: `{format_metric(result['val_metric']['AvgCandidatePoolMatchRank'])}`",
        f"- AvgPoolHitRankMissMatchRank: `{format_metric(result['val_metric']['AvgPoolHitRankMissMatchRank'])}`",
        f"- OracleSourceHit@{config.topk}: `{format_metric(result['val_metric'][f'OracleSourceHit@{config.topk}Rate'])}`",
        f"- AvgOracleSourceMatchRank: `{format_metric(result['val_metric']['AvgOracleSourceMatchRank'])}`",
        "",
        "## Next Target",
        "",
        NEXT_TARGET,
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir.resolve()),
                "train_recall@50": result["train_metric"]["Recall@50"],
                "cold_like_recall@50": result["val_metric"]["Recall@50"],
                "next_target": NEXT_TARGET,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
