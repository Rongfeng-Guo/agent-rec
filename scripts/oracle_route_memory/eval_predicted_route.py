#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from genrec.models import LateBoundRouter, Prefix1QueryHead
from genrec.memory.catalog_memory import CatalogMemory
from genrec.memory.data_adapter import load_item_embeddings, load_train_item_set
from genrec.training import (
    RouteVocab,
    RouterDataset,
    build_eval_router_samples,
    build_training_examples,
    load_route_mapping,
)

RouteCandidate = Tuple[Tuple[int, ...], float]


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "t"}


def dcg(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def reciprocal_rank(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def build_memory(
    item_embeddings: Mapping[str, np.ndarray],
    route_mapping: Mapping[str, Tuple[int, int]],
    prefix_len: int,
    train_item_set: set[str] | None = None,
) -> CatalogMemory:
    memory = CatalogMemory(normalize=True, prefer_faiss=True)
    item_ids = []
    embs = []
    routes = []
    for item_id, emb in item_embeddings.items():
        if train_item_set is not None and item_id not in train_item_set:
            continue
        route = route_mapping.get(item_id)
        if route is None:
            continue
        item_ids.append(item_id)
        embs.append(np.asarray(emb, dtype=np.float32))
        routes.append(route[:prefix_len])
    if not item_ids:
        raise ValueError(f"No items were available to build prefix-{prefix_len} memory.")
    memory.add_items(item_ids=item_ids, item_embs=np.stack(embs, axis=0), routes=routes, labels=item_ids)
    return memory


def load_model(checkpoint_dir: str | Path, device: str) -> tuple[LateBoundRouter, RouteVocab, dict]:
    checkpoint_dir = Path(checkpoint_dir)
    meta = json.loads((checkpoint_dir / "checkpoint_meta.json").read_text(encoding="utf-8"))
    route_vocab = RouteVocab.from_dict(meta["route_vocab"])
    model = LateBoundRouter(**meta["model_config"])
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, route_vocab, meta


def load_prefix1_query_head(checkpoint_dir: str | Path, device: str) -> tuple[Prefix1QueryHead, dict]:
    checkpoint_dir = Path(checkpoint_dir)
    meta = json.loads((checkpoint_dir / "checkpoint_meta.json").read_text(encoding="utf-8"))
    model = Prefix1QueryHead(**meta["model_config"])
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, meta


def enumerate_prefix1_candidates(route1_log_probs: torch.Tensor, route_vocab: RouteVocab, beam_size: int) -> List[List[RouteCandidate]]:
    topk = torch.topk(route1_log_probs, k=min(int(beam_size), route1_log_probs.shape[-1]), dim=-1)
    candidates: List[List[RouteCandidate]] = []
    for row_idx in range(route1_log_probs.shape[0]):
        rows = []
        for prefix1_idx, score in zip(topk.indices[row_idx].tolist(), topk.values[row_idx].tolist()):
            rows.append(((route_vocab.decode_prefix1(prefix1_idx),), float(score)))
        candidates.append(rows)
    return candidates


def enumerate_route_candidates(joint_log_probs: torch.Tensor, route_vocab: RouteVocab, beam_size: int) -> List[List[RouteCandidate]]:
    flat = joint_log_probs.view(joint_log_probs.shape[0], -1)
    topk = torch.topk(flat, k=min(int(beam_size), flat.shape[-1]), dim=-1)
    candidates: List[List[RouteCandidate]] = []
    for row_idx in range(flat.shape[0]):
        rows = []
        for flat_index, score in zip(topk.indices[row_idx].tolist(), topk.values[row_idx].tolist()):
            prefix1_idx = flat_index // route_vocab.num_prefix2
            prefix2_idx = flat_index % route_vocab.num_prefix2
            rows.append((route_vocab.decode_prefix2(prefix1_idx, prefix2_idx), float(score)))
        candidates.append(rows)
    return candidates


def selected_query_sources(query_source: str) -> List[str]:
    if query_source == "all":
        return ["learned", "pooled", "residual", "mean"]
    if query_source == "all_plus_prefix1_head":
        return ["learned", "pooled", "residual", "mean", "prefix1_head"]
    return [query_source]


def parse_domain_query_sources(values: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    allowed_sources = {"learned", "pooled", "residual", "mean", "prefix1_head"}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected DOMAIN=QUERY_SOURCE, got {value!r}.")
        domain, query_source = value.split("=", 1)
        domain = domain.strip()
        query_source = query_source.strip()
        if not domain:
            raise ValueError(f"Domain cannot be empty in {value!r}.")
        if query_source not in allowed_sources:
            raise ValueError(f"Unknown query source {query_source!r} for domain {domain!r}.")
        mapping[domain] = query_source
    return mapping


def resolve_query_source(
    query_source: str,
    domain: str,
    domain_query_sources: Mapping[str, str],
    default_query_source: str,
) -> str:
    if query_source != "domain_adaptive":
        return query_source
    return domain_query_sources.get(domain, default_query_source)


def parse_fusion_specs(values: Sequence[str]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    allowed_sources = {"learned", "pooled", "residual", "mean", "prefix1_head", "domain_adaptive"}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected FUSION_NAME=QUERY_SOURCE:MODE+..., got {value!r}.")
        name, members_text = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Fusion name cannot be empty in {value!r}.")
        members: List[Tuple[str, str]] = []
        for member_text in members_text.split("+"):
            member_text = member_text.strip()
            if not member_text:
                continue
            if ":" not in member_text:
                raise ValueError(f"Expected QUERY_SOURCE:MODE in fusion member {member_text!r}.")
            query_source, mode = member_text.split(":", 1)
            query_source = query_source.strip()
            mode = mode.strip()
            if query_source not in allowed_sources:
                raise ValueError(f"Unknown fusion query source {query_source!r} in {value!r}.")
            if not mode:
                raise ValueError(f"Fusion mode cannot be empty in {value!r}.")
            members.append((query_source, mode))
        if len(members) < 2:
            raise ValueError(f"Fusion spec {value!r} must contain at least two members.")
        specs.append({"name": name, "members": members})
    return specs


def add_fusion_query_sources(query_sources: Sequence[str], fusion_specs: Sequence[Mapping[str, Any]]) -> List[str]:
    rows = list(query_sources)
    for spec in fusion_specs:
        for query_source, _ in spec["members"]:
            if query_source not in rows:
                rows.append(query_source)
    return rows


def fuse_ranked_lists(
    ranked_lists: Sequence[Sequence[str]],
    max_k: int,
    method: str = "rrf",
    rrf_k: float = 60.0,
) -> List[str]:
    if method == "rrf":
        scores: Dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, item_id in enumerate(ranked[:max_k], start=1):
                scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (float(rrf_k) + rank)
        return [item_id for item_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max_k]]
    if method == "round_robin":
        fused: List[str] = []
        seen = set()
        for rank in range(max_k):
            for ranked in ranked_lists:
                if rank >= len(ranked):
                    continue
                item_id = ranked[rank]
                if item_id in seen:
                    continue
                fused.append(item_id)
                seen.add(item_id)
                if len(fused) >= max_k:
                    return fused
        return fused
    raise ValueError(f"Unknown fusion method: {method}")


def load_domain_query_source_config(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Domain query source config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    mapping = payload.get("domain_query_sources", {})
    if not isinstance(mapping, Mapping):
        raise ValueError("domain_query_sources must be a mapping in the selector config.")
    allowed_sources = {"learned", "pooled", "residual", "mean", "prefix1_head"}
    normalized_mapping: Dict[str, str] = {}
    for domain, query_source in mapping.items():
        query_source = str(query_source)
        if query_source not in allowed_sources:
            raise ValueError(f"Unknown query source {query_source!r} for domain {domain!r} in selector config.")
        normalized_mapping[str(domain)] = query_source
    default_query_source = str(payload.get("default_domain_query_source", "learned"))
    if default_query_source not in allowed_sources:
        raise ValueError(f"Unknown default query source {default_query_source!r} in selector config.")
    return {
        "path": str(config_path),
        "domain_query_sources": normalized_mapping,
        "default_domain_query_source": default_query_source,
        "metadata": payload.get("metadata", {}),
        "metrics": payload.get("metrics", {}),
    }


def selected_extra_prefix1_route_sources(route_sources: Sequence[str]) -> List[str]:
    if not route_sources:
        return []
    if "all" in route_sources:
        return ["domain_prior", "history_last", "history_vote", "history_recency"]
    return list(dict.fromkeys(route_sources))


def summarize(rows: Sequence[Mapping[str, Any]], topks: Sequence[int], train_item_set: set[str]) -> Dict[str, Any]:
    recalls = defaultdict(list)
    ndcgs = defaultdict(list)
    mrrs = defaultdict(list)
    candidate_pool_hits = []
    candidate_pool_losses = []
    max_k = max(topks)
    for row in rows:
        ranked = row["ranked_ids"]
        target = row["target_item_id"]
        rank = row["match_rank"]
        if "candidate_pool_hit" in row:
            pool_hit = bool(row["candidate_pool_hit"])
            final_hit = rank is not None and rank <= max_k
            candidate_pool_hits.append(1.0 if pool_hit else 0.0)
            candidate_pool_losses.append(1.0 if pool_hit and not final_hit else 0.0)
        for k in topks:
            recalls[k].append(1.0 if target in ranked[:k] else 0.0)
            ndcgs[k].append(dcg(rank) if rank is not None and rank <= k else 0.0)
            mrrs[k].append(reciprocal_rank(rank, k))
    top50_total = sum(min(50, len(row["ranked_ids"])) for row in rows)
    iid_hits = sum(sum(1 for item_id in row["ranked_ids"][:50] if item_id not in train_item_set) for row in rows)
    return {
        "sample_count": len(rows),
        **{f"Recall@{k}": float(np.mean(recalls[k])) if recalls[k] else 0.0 for k in topks},
        **{f"NDCG@{k}": float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0 for k in topks},
        **{f"MRR@{k}": float(np.mean(mrrs[k])) if mrrs[k] else 0.0 for k in topks},
        "CandidatePoolHitRate": float(np.mean(candidate_pool_hits)) if candidate_pool_hits else 0.0,
        "CandidatePoolLossRate": float(np.mean(candidate_pool_losses)) if candidate_pool_losses else 0.0,
        "IIDRatio@50": (iid_hits / top50_total) if top50_total else 0.0,
        "avg_latency_ms": float(np.mean([row["latency_ms"] for row in rows])) if rows else 0.0,
        "fallback_rate": float(np.mean([1.0 if row["fallback_used"] else 0.0 for row in rows])) if rows else 0.0,
    }


def grouped_summary(
    rows: Sequence[Mapping[str, Any]],
    group_keys: Sequence[str],
    topks: Sequence[int],
    train_item_set: set[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary_rows = []
    for key_values, subset_rows in sorted(grouped.items()):
        metrics = summarize(subset_rows, topks, train_item_set)
        metrics.update({key: value for key, value in zip(group_keys, key_values)})
        summary_rows.append(metrics)
    return summary_rows


def route_to_text(route: Sequence[Any]) -> str:
    return "|".join(str(part) for part in route)


def mean_history_embedding_from_ids(history_item_ids: Sequence[str], item_embeddings: Mapping[str, np.ndarray]) -> np.ndarray:
    vectors = [np.asarray(item_embeddings[item_id], dtype=np.float32) for item_id in history_item_ids if item_id in item_embeddings]
    if not vectors:
        raise ValueError("Cannot build a mean query for an empty embedded history.")
    return np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)


def prefix1_prior_candidates(counter: Counter, fallback: Counter, beam_size: int) -> List[RouteCandidate]:
    source = counter if counter else fallback
    rows = []
    for route_value, _ in source.most_common(max(int(beam_size), 0)):
        rows.append(((int(route_value),), 0.0))
    return rows


def history_prefix1_candidates(
    history_item_ids: Sequence[str],
    route_mapping: Mapping[str, Tuple[int, int]],
    strategy: str,
    beam_size: int,
    fallback: Sequence[RouteCandidate],
) -> List[RouteCandidate]:
    routes = [int(route_mapping[item_id][0]) for item_id in history_item_ids if item_id in route_mapping]
    scored: List[Tuple[int, float]] = []
    if strategy == "history_last":
        seen = set()
        for route_value in reversed(routes):
            if route_value in seen:
                continue
            seen.add(route_value)
            scored.append((route_value, 0.0))
    elif strategy == "history_vote":
        counts = Counter(routes)
        scored = [(route_value, float(count)) for route_value, count in counts.most_common()]
    elif strategy == "history_recency":
        counts = Counter()
        seq_len = len(routes)
        for pos, route_value in enumerate(routes):
            counts[route_value] += math.exp((pos - float(seq_len - 1)) / max(seq_len / 2.0, 1.0))
        scored = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    else:
        raise ValueError(f"Unknown history prefix-1 strategy: {strategy}")

    rows: List[RouteCandidate] = []
    seen_routes = set()
    for route_value, _ in scored:
        if route_value in seen_routes:
            continue
        rows.append(((int(route_value),), 0.0))
        seen_routes.add(route_value)
        if len(rows) >= beam_size:
            return rows
    for route_tuple, score in fallback:
        route_value = int(route_tuple[0])
        if route_value in seen_routes:
            continue
        rows.append(((route_value,), score))
        seen_routes.add(route_value)
        if len(rows) >= beam_size:
            break
    return rows


def summarize_route_source_predictions(route_source_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in route_source_rows:
        grouped[(str(row["subset"]), "ALL", str(row["mode"]))].append(row)
        grouped[(str(row["subset"]), str(row["domain"]), str(row["mode"]))].append(row)

    summary_rows = []
    for (subset, domain, mode), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "subset": subset,
                "domain": domain,
                "mode": mode,
                "sample_count": len(rows),
                "candidate_route_recall": float(np.mean([int(row["candidate_route_hit"]) for row in rows])) if rows else 0.0,
            }
        )
    return summary_rows


def example_route(example: Any, prefix_len: int) -> Tuple[int, ...]:
    route = (int(example.route_prefix1), int(example.route_prefix2))
    return route[:prefix_len]


def route_distribution_outputs(examples_by_subset: Mapping[str, Sequence[Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    distribution_rows: List[Dict[str, Any]] = []
    imbalance_rows: List[Dict[str, Any]] = []
    for subset, examples in examples_by_subset.items():
        domain_groups: Dict[str, List[Any]] = {"ALL": list(examples)}
        for example in examples:
            domain_groups.setdefault(str(example.domain), []).append(example)

        for domain, domain_examples in sorted(domain_groups.items()):
            for prefix_len in (1, 2):
                counts = Counter(example_route(example, prefix_len) for example in domain_examples)
                total = sum(counts.values())
                if total == 0:
                    continue
                probs = np.asarray([count / total for count in counts.values()], dtype=np.float64)
                entropy = float(-(probs * np.log(probs)).sum()) if len(probs) else 0.0
                normalized_entropy = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
                top_counts = sorted(counts.values(), reverse=True)
                imbalance_rows.append(
                    {
                        "subset": subset,
                        "domain": domain,
                        "prefix_len": prefix_len,
                        "sample_count": total,
                        "num_routes": len(counts),
                        "top1_share": float(top_counts[0] / total),
                        "top3_share": float(sum(top_counts[:3]) / total),
                        "entropy": entropy,
                        "normalized_entropy": float(normalized_entropy),
                    }
                )
                for route, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                    distribution_rows.append(
                        {
                            "subset": subset,
                            "domain": domain,
                            "prefix_len": prefix_len,
                            "route": route_to_text(route),
                            "count": count,
                            "ratio": float(count / total),
                        }
                    )
    return distribution_rows, imbalance_rows


def summarize_route_predictions(prediction_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[(str(row["subset"]), "ALL")].append(row)
        grouped[(str(row["subset"]), str(row["domain"]))].append(row)

    summary_rows = []
    metric_keys = [
        "prefix1_top1_hit",
        "prefix1_top2_hit",
        "prefix1_top4_hit",
        "prefix2_top1_hit",
        "prefix2_top4_hit",
        "prefix2_top8_hit",
    ]
    for (subset, domain), rows in sorted(grouped.items()):
        summary = {"subset": subset, "domain": domain, "sample_count": len(rows)}
        for key in metric_keys:
            summary[key.replace("_hit", "_accuracy")] = float(np.mean([int(row[key]) for row in rows])) if rows else 0.0
        summary_rows.append(summary)
    return summary_rows


def find_route_metric(route_summary_rows: Sequence[Mapping[str, Any]], subset: str, domain: str, metric: str) -> float:
    for row in route_summary_rows:
        if row["subset"] == subset and row["domain"] == domain:
            return float(row.get(metric, 0.0))
    return 0.0


def find_recall50(summary_rows: Sequence[Mapping[str, Any]], query_source: str, subset: str, mode: str) -> float:
    for row in summary_rows:
        if row["query_source"] == query_source and row["subset"] == subset and row["mode"] == mode:
            return float(row.get("Recall@50", 0.0))
    return 0.0


def _score_merged(candidates: Sequence[Tuple[str, float]], max_k: int) -> List[str]:
    best_scores: Dict[str, float] = {}
    for item_id, score in candidates:
        best_scores[item_id] = max(best_scores.get(item_id, -1e9), float(score))
    return [item_id for item_id, _ in sorted(best_scores.items(), key=lambda item: item[1], reverse=True)[:max_k]]


def _weighted_flat_candidates(
    per_route_candidates: Sequence[Sequence[Tuple[str, float]]],
    route_log_probs: Sequence[float],
    route_score_weight: float,
) -> List[Tuple[str, float]]:
    candidates: List[Tuple[str, float]] = []
    for route_rows, route_log_prob in zip(per_route_candidates, route_log_probs):
        route_bonus = route_score_weight * float(route_log_prob)
        candidates.extend((item_id, float(score) + route_bonus) for item_id, score in route_rows)
    return candidates


def _score_backfill(ranked_ids: List[str], candidates: Sequence[Tuple[str, float]], max_k: int) -> List[str]:
    if len(ranked_ids) >= max_k:
        return ranked_ids[:max_k]
    seen = set(ranked_ids)
    for item_id in _score_merged(candidates, max_k):
        if item_id in seen:
            continue
        ranked_ids.append(item_id)
        seen.add(item_id)
        if len(ranked_ids) >= max_k:
            break
    return ranked_ids


def merge_candidate_rows(
    per_route_candidates: Sequence[Sequence[Tuple[str, float]]],
    route_log_probs: Sequence[float],
    max_k: int,
    merge_strategy: str,
    route_score_weight: float = 0.0,
    rrf_k: float = 60.0,
) -> List[str]:
    if len(per_route_candidates) != len(route_log_probs):
        raise ValueError("per_route_candidates and route_log_probs must have the same length.")

    candidates = _weighted_flat_candidates(per_route_candidates, route_log_probs, route_score_weight)
    if merge_strategy == "score":
        return _score_merged(candidates, max_k)

    if merge_strategy == "zscore":
        normalized_candidates: List[Tuple[str, float]] = []
        for route_rows, route_log_prob in zip(per_route_candidates, route_log_probs):
            if not route_rows:
                continue
            scores = np.asarray([float(score) for _, score in route_rows], dtype=np.float32)
            mean = float(scores.mean())
            std = float(scores.std())
            if std < 1e-6:
                std = 1.0
            route_bonus = route_score_weight * float(route_log_prob)
            normalized_candidates.extend((item_id, ((float(score) - mean) / std) + route_bonus) for item_id, score in route_rows)
        return _score_merged(normalized_candidates, max_k)

    if merge_strategy == "rrf":
        rrf_candidates: List[Tuple[str, float]] = []
        for route_rows, route_log_prob in zip(per_route_candidates, route_log_probs):
            route_bonus = route_score_weight * float(route_log_prob)
            for rank, (item_id, _) in enumerate(route_rows, start=1):
                rrf_candidates.append((item_id, (1.0 / (rrf_k + rank)) + route_bonus))
        return _score_merged(rrf_candidates, max_k)

    if merge_strategy == "round_robin":
        ranked_ids = []
        seen = set()
        for local_rank in range(max_k):
            for route_rows in per_route_candidates:
                if local_rank >= len(route_rows):
                    continue
                item_id, _ = route_rows[local_rank]
                if item_id in seen:
                    continue
                ranked_ids.append(item_id)
                seen.add(item_id)
                if len(ranked_ids) >= max_k:
                    break
            if len(ranked_ids) >= max_k:
                break
        return _score_backfill(ranked_ids, candidates, max_k)

    if merge_strategy == "quota":
        ranked_ids = []
        seen = set()
        quota = max(1, math.ceil(max_k / max(len(per_route_candidates), 1)))
        for route_rows in per_route_candidates:
            taken = 0
            for item_id, _ in route_rows:
                if item_id in seen:
                    continue
                ranked_ids.append(item_id)
                seen.add(item_id)
                taken += 1
                if taken >= quota or len(ranked_ids) >= max_k:
                    break
            if len(ranked_ids) >= max_k:
                break
        return _score_backfill(ranked_ids, candidates, max_k)

    raise ValueError(f"Unknown merge strategy: {merge_strategy}")


def merge_mode_label(mode_label: str, merge_strategy: str) -> str:
    return mode_label if merge_strategy == "score" else f"{mode_label}_{merge_strategy}"


def rerank_with_routes(
    query_embedding: np.ndarray,
    route_candidates: Sequence[RouteCandidate],
    prefix_len: int,
    memory: CatalogMemory,
    topks: Sequence[int],
    route_score_weight: float,
    merge_strategy: str = "score",
    per_route_topk: int | None = None,
    target_item_id: str | None = None,
) -> Tuple[List[str], float, bool, Dict[str, Any]]:
    start_search = time.perf_counter()
    max_k = max(topks)
    search_topk = int(per_route_topk) if per_route_topk is not None else max_k
    search_topk = max(1, search_topk)
    per_route_candidates: List[List[Tuple[str, float]]] = []
    route_log_probs: List[float] = []
    fallback_used = False
    for route_tuple, route_log_prob in route_candidates:
        route_key = route_tuple[:prefix_len]
        if not memory.has_route(route_key):
            fallback_used = True
        current_route_candidates = []
        for result in memory.search(query_embedding, route=route_key, topk=search_topk):
            current_route_candidates.append((result["item_id"], float(result["score"])))
        if current_route_candidates:
            per_route_candidates.append(current_route_candidates)
            route_log_probs.append(float(route_log_prob))
    if not per_route_candidates:
        fallback_used = True
        current_route_candidates = []
        for result in memory.search(query_embedding, route=None, topk=search_topk):
            current_route_candidates.append((result["item_id"], float(result["score"])))
        if current_route_candidates:
            per_route_candidates.append(current_route_candidates)
            route_log_probs.append(0.0)

    candidate_pool = {item_id for route_rows in per_route_candidates for item_id, _ in route_rows}
    ranked_ids = merge_candidate_rows(
        per_route_candidates=per_route_candidates,
        route_log_probs=route_log_probs,
        max_k=max_k,
        merge_strategy=merge_strategy,
        route_score_weight=route_score_weight,
    )

    latency_ms = (time.perf_counter() - start_search) * 1000.0
    diagnostics = {
        "candidate_pool_size": len(candidate_pool),
        "candidate_pool_hit": bool(target_item_id is not None and target_item_id in candidate_pool),
        "num_route_candidates": len(route_candidates),
        "merge_strategy": merge_strategy,
        "per_route_topk": search_topk,
    }
    return ranked_ids, latency_ms, fallback_used, diagnostics


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--item-embedding-path", required=True)
    parser.add_argument("--item-sid-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--prefix1-query-head-checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cold-only", type=str2bool, default=True)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--beam-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--prefix1-beam-sizes", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--extra-prefix1-route-sources",
        nargs="*",
        default=[],
        choices=["domain_prior", "history_last", "history_vote", "history_recency", "all"],
    )
    parser.add_argument("--topk", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument(
        "--query-source",
        default="learned",
        choices=["learned", "pooled", "residual", "mean", "prefix1_head", "domain_adaptive", "all", "all_plus_prefix1_head"],
    )
    parser.add_argument("--domain-query-source", nargs="*", default=[])
    parser.add_argument("--domain-query-source-config", default=None)
    parser.add_argument(
        "--default-domain-query-source",
        default="learned",
        choices=["learned", "pooled", "residual", "mean", "prefix1_head"],
    )
    parser.add_argument("--route-score-weight", type=float, default=1.0)
    parser.add_argument("--per-route-topk", type=int, default=None)
    parser.add_argument(
        "--merge-strategies",
        nargs="+",
        default=["score"],
        choices=["score", "zscore", "round_robin", "quota", "rrf"],
    )
    parser.add_argument(
        "--fusion-spec",
        nargs="*",
        default=[],
        help="Named list fusion specs: NAME=QUERY_SOURCE:MODE+QUERY_SOURCE:MODE+...",
    )
    parser.add_argument("--fusion-method", default="rrf", choices=["rrf", "round_robin"])
    parser.add_argument("--fusion-rrf-k", type=float, default=60.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    fusion_specs = parse_fusion_specs(args.fusion_spec)
    query_sources = add_fusion_query_sources(selected_query_sources(args.query_source), fusion_specs)
    domain_query_source_config = load_domain_query_source_config(args.domain_query_source_config)
    domain_query_sources = dict(domain_query_source_config.get("domain_query_sources", {}))
    domain_query_sources.update(parse_domain_query_sources(args.domain_query_source))
    default_domain_query_source = str(
        domain_query_source_config.get("default_domain_query_source", args.default_domain_query_source)
        if domain_query_source_config
        else args.default_domain_query_source
    )
    extra_prefix1_route_sources = selected_extra_prefix1_route_sources(args.extra_prefix1_route_sources)
    item_embeddings = load_item_embeddings(args.data_dir, args.item_embedding_path)
    route_mapping = load_route_mapping(args.item_sid_path)
    train_item_set = load_train_item_set(args.data_dir)
    model, route_vocab, meta = load_model(args.checkpoint_dir, device)
    prefix1_query_head = None
    prefix1_query_head_meta = None
    domain_uses_prefix1_head = "prefix1_head" in set(domain_query_sources.values()) or (
        "domain_adaptive" in query_sources and default_domain_query_source == "prefix1_head"
    )
    if "prefix1_head" in query_sources or domain_uses_prefix1_head:
        if not args.prefix1_query_head_checkpoint:
            raise ValueError("--prefix1-query-head-checkpoint is required when --query-source includes prefix1_head.")
        prefix1_query_head, prefix1_query_head_meta = load_prefix1_query_head(args.prefix1_query_head_checkpoint, device)

    train_examples = build_training_examples(args.data_dir, item_embeddings, route_mapping, max_history=args.max_history)
    domain_prefix1_counts: Dict[str, Counter] = defaultdict(Counter)
    global_prefix1_counts: Counter = Counter()
    for example in train_examples:
        domain_prefix1_counts[str(example.domain)][int(example.route_prefix1)] += 1
        global_prefix1_counts[int(example.route_prefix1)] += 1

    cold_examples = build_eval_router_samples(args.data_dir, route_mapping, cold_only=True, item_embeddings=item_embeddings, max_history=args.max_history)
    warm_examples = build_eval_router_samples(args.data_dir, route_mapping, cold_only=False, item_embeddings=item_embeddings, max_history=args.max_history)
    warm_examples = [row for row in warm_examples if not row.cold]

    cold_dataset = RouterDataset(cold_examples, item_embeddings, route_vocab)
    warm_dataset = RouterDataset(warm_examples, item_embeddings, route_vocab)
    cold_loader = DataLoader(cold_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=cold_dataset.collate_fn)
    warm_loader = DataLoader(warm_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=warm_dataset.collate_fn)

    full_memory_p1 = build_memory(item_embeddings, route_mapping, prefix_len=1)
    full_memory_p2 = build_memory(item_embeddings, route_mapping, prefix_len=2)
    train_memory = build_memory(item_embeddings, route_mapping, prefix_len=2, train_item_set=train_item_set)

    unique_cold_targets = sorted({row.target_item_id for row in cold_examples})
    start = time.perf_counter()
    train_memory.add_items(
        item_ids=unique_cold_targets,
        item_embs=np.stack([item_embeddings[item_id] for item_id in unique_cold_targets], axis=0),
        routes=[route_mapping[item_id] for item_id in unique_cold_targets],
        labels=unique_cold_targets,
    )
    cold_insertion_ms = (time.perf_counter() - start) * 1000.0

    retrieval_rows = []
    prediction_rows = []
    route_source_rows = []
    prefix1_beams_for_prediction = sorted(set(args.prefix1_beam_sizes + [1, 2, 4]))
    prefix2_beams_for_prediction = sorted(set(args.beam_sizes + [1, 4, 8]))

    for subset_name, loader in [("cold", cold_loader), ("warm", warm_loader)]:
        with torch.no_grad():
            for batch in loader:
                outputs = model(batch["history_embs"].to(device), batch["history_mask"].to(device))
                route1_log_probs, _, joint_log_probs = model.route_log_probs(outputs)
                prefix1_candidates_by_beam = {
                    beam: enumerate_prefix1_candidates(route1_log_probs.cpu(), route_vocab, beam)
                    for beam in prefix1_beams_for_prediction
                }
                prefix2_candidates_by_beam = {
                    beam: enumerate_route_candidates(joint_log_probs.cpu(), route_vocab, beam)
                    for beam in prefix2_beams_for_prediction
                }
                pooled_embeddings = torch.nn.functional.normalize(outputs["pooled_history"], dim=-1).cpu().numpy()
                learned_embeddings = outputs["query_embedding"].cpu().numpy()
                residual_embeddings = torch.nn.functional.normalize(outputs["query_embedding"] + outputs["pooled_history"], dim=-1).cpu().numpy()
                mean_embeddings = np.stack(
                    [mean_history_embedding_from_ids(history, item_embeddings) for history in batch["history_item_ids"]],
                    axis=0,
                )
                query_embeddings_by_source = {
                    "learned": learned_embeddings,
                    "pooled": pooled_embeddings,
                    "residual": residual_embeddings,
                    "mean": mean_embeddings,
                }
                if prefix1_query_head is not None:
                    query_embeddings_by_source["prefix1_head"] = prefix1_query_head(
                        batch["history_embs"].to(device),
                        batch["history_mask"].to(device),
                    ).cpu().numpy()

                for idx, target_item_id in enumerate(batch["target_item_id"]):
                    true_route = tuple(int(part) for part in route_mapping[target_item_id])
                    true_prefix1 = true_route[:1]
                    domain = str(batch["domain"][idx])
                    sample_id = str(batch["sample_id"][idx])
                    prediction_row = {
                        "subset": subset_name,
                        "domain": domain,
                        "sample_id": sample_id,
                        "target_item_id": target_item_id,
                        "true_prefix1": route_to_text(true_prefix1),
                        "true_prefix2": route_to_text(true_route),
                        "predicted_prefix1_top1": route_to_text(prefix1_candidates_by_beam[1][idx][0][0]),
                        "predicted_prefix2_top1": route_to_text(prefix2_candidates_by_beam[1][idx][0][0]),
                    }
                    for beam in (1, 2, 4):
                        prediction_row[f"prefix1_top{beam}_hit"] = int(
                            any(candidate[0] == true_prefix1 for candidate in prefix1_candidates_by_beam[beam][idx])
                        )
                    for beam in (1, 4, 8):
                        prediction_row[f"prefix2_top{beam}_hit"] = int(
                            any(candidate[0] == true_route for candidate in prefix2_candidates_by_beam[beam][idx])
                        )
                    prediction_rows.append(prediction_row)

                    search_specs = []
                    for beam in args.prefix1_beam_sizes:
                        mode_label = "predicted_route_p1" if beam == 1 else f"predicted_route_p1_top{beam}"
                        search_specs.append((1, beam, mode_label, full_memory_p1, prefix1_candidates_by_beam))

                    heuristic_prefix1_candidates_by_mode: Dict[str, Dict[int, List[List[RouteCandidate]]]] = {}
                    for source_name in extra_prefix1_route_sources:
                        heuristic_prefix1_candidates_by_mode[source_name] = {}
                        for beam in prefix1_beams_for_prediction:
                            prior = prefix1_prior_candidates(domain_prefix1_counts.get(domain, Counter()), global_prefix1_counts, beam)
                            if source_name == "domain_prior":
                                candidates = prior
                            else:
                                candidates = history_prefix1_candidates(batch["history_item_ids"][idx], route_mapping, source_name, beam, prior)
                            heuristic_prefix1_candidates_by_mode[source_name][beam] = [candidates]
                    for source_name, candidates_by_beam in heuristic_prefix1_candidates_by_mode.items():
                        for beam in args.prefix1_beam_sizes:
                            mode_label = f"{source_name}_p1" if beam == 1 else f"{source_name}_p1_top{beam}"
                            search_specs.append((1, beam, mode_label, full_memory_p1, candidates_by_beam))

                    for beam in args.beam_sizes:
                        search_specs.append((2, beam, f"predicted_route_p2_top{beam}", full_memory_p2, prefix2_candidates_by_beam))

                    for prefix_len, beam_size, mode_label, _, candidates_by_beam in search_specs:
                        route_candidates = candidates_by_beam[beam_size][0 if mode_label.startswith(("domain_prior", "history_")) else idx]
                        true_key = true_prefix1 if prefix_len == 1 else true_route
                        route_source_rows.append(
                            {
                                "subset": subset_name,
                                "domain": domain,
                                "mode": mode_label,
                                "sample_id": sample_id,
                                "target_item_id": target_item_id,
                                "true_route": route_to_text(true_key),
                                "candidate_routes": [route_to_text(route) for route, _ in route_candidates],
                                "candidate_route_hit": int(any(route[:prefix_len] == true_key for route, _ in route_candidates)),
                            }
                        )

                    sample_retrieval_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
                    for query_source in query_sources:
                        effective_query_source = resolve_query_source(
                            query_source,
                            domain,
                            domain_query_sources,
                            default_domain_query_source,
                        )
                        query_embedding = query_embeddings_by_source[effective_query_source][idx]
                        for prefix_len, beam_size, mode_label, current_memory, candidates_by_beam in search_specs:
                            route_candidates = candidates_by_beam[beam_size][0 if mode_label.startswith(("domain_prior", "history_")) else idx]
                            for merge_strategy in args.merge_strategies:
                                ranked_ids, latency_ms, fallback_used, rerank_diagnostics = rerank_with_routes(
                                    query_embedding=query_embedding,
                                    route_candidates=route_candidates,
                                    prefix_len=prefix_len,
                                    memory=current_memory,
                                    topks=args.topk,
                                    route_score_weight=args.route_score_weight,
                                    merge_strategy=merge_strategy,
                                    per_route_topk=args.per_route_topk,
                                    target_item_id=target_item_id,
                                )
                                match_rank = None
                                for rank, item_id in enumerate(ranked_ids, start=1):
                                    if item_id == target_item_id:
                                        match_rank = rank
                                        break
                                retrieval_row = {
                                    "query_source": query_source,
                                    "effective_query_source": effective_query_source,
                                    "subset": subset_name,
                                    "domain": domain,
                                    "mode": merge_mode_label(mode_label, merge_strategy),
                                    "sample_id": sample_id,
                                    "target_item_id": target_item_id,
                                    "true_route": route_to_text(true_route),
                                    "route_candidates": [
                                        {"route": route_to_text(route), "score": score}
                                        for route, score in route_candidates
                                    ],
                                    "ranked_ids": ranked_ids,
                                    "match_rank": match_rank,
                                    "latency_ms": latency_ms,
                                    "fallback_used": fallback_used,
                                    **rerank_diagnostics,
                                }
                                retrieval_rows.append(retrieval_row)
                                sample_retrieval_rows[(query_source, retrieval_row["mode"])] = retrieval_row

                    for fusion_spec in fusion_specs:
                        member_rows = []
                        for query_source, mode in fusion_spec["members"]:
                            row = sample_retrieval_rows.get((query_source, mode))
                            if row is None:
                                available = sorted(f"{source}:{mode_name}" for source, mode_name in sample_retrieval_rows)
                                raise ValueError(
                                    f"Fusion member {query_source}:{mode} was not produced for sample {sample_id}. "
                                    f"Available members: {available}"
                                )
                            member_rows.append(row)
                        ranked_ids = fuse_ranked_lists(
                            [row["ranked_ids"] for row in member_rows],
                            max_k=max(args.topk),
                            method=args.fusion_method,
                            rrf_k=args.fusion_rrf_k,
                        )
                        match_rank = None
                        for rank, item_id in enumerate(ranked_ids, start=1):
                            if item_id == target_item_id:
                                match_rank = rank
                                break
                        candidate_pool = {item_id for row in member_rows for item_id in row["ranked_ids"][: max(args.topk)]}
                        fusion_row = {
                            "query_source": "fusion",
                            "effective_query_source": "fusion",
                            "subset": subset_name,
                            "domain": domain,
                            "mode": f"fusion_{fusion_spec['name']}",
                            "sample_id": sample_id,
                            "target_item_id": target_item_id,
                            "true_route": route_to_text(true_route),
                            "route_candidates": [],
                            "ranked_ids": ranked_ids,
                            "match_rank": match_rank,
                            "latency_ms": sum(float(row.get("latency_ms", 0.0)) for row in member_rows),
                            "fallback_used": any(bool(row.get("fallback_used")) for row in member_rows),
                            "candidate_pool_size": len(candidate_pool),
                            "candidate_pool_hit": target_item_id in candidate_pool,
                            "num_route_candidates": sum(int(row.get("num_route_candidates", 0)) for row in member_rows),
                            "merge_strategy": f"fusion_{args.fusion_method}",
                            "per_route_topk": args.per_route_topk,
                            "fusion_members": [f"{source}:{mode}" for source, mode in fusion_spec["members"]],
                        }
                        retrieval_rows.append(fusion_row)

    summary_rows = grouped_summary(retrieval_rows, ["query_source", "subset", "mode"], args.topk, train_item_set)
    summary_by_domain_rows = grouped_summary(retrieval_rows, ["query_source", "subset", "domain", "mode"], args.topk, train_item_set)
    route_summary_rows = summarize_route_predictions(prediction_rows)
    route_source_summary_rows = summarize_route_source_predictions(route_source_rows)
    distribution_rows, imbalance_rows = route_distribution_outputs(
        {"train": train_examples, "cold": cold_examples, "warm": warm_examples}
    )

    summary_query_sources = list(query_sources)
    if fusion_specs:
        summary_query_sources.append("fusion")
    warm_retention_by_source = {
        query_source: find_recall50(summary_rows, query_source, "warm", "predicted_route_p2_top8")
        for query_source in summary_query_sources
    }
    cold_recall50_by_query_source_and_mode = {
        query_source: {
            row["mode"]: float(row["Recall@50"])
            for row in summary_rows
            if row["query_source"] == query_source and row["subset"] == "cold"
        }
        for query_source in summary_query_sources
    }
    diagnostics = {
        "prefix1_route_accuracy": find_route_metric(route_summary_rows, "cold", "ALL", "prefix1_top1_accuracy"),
        "prefix1_route_top2_accuracy": find_route_metric(route_summary_rows, "cold", "ALL", "prefix1_top2_accuracy"),
        "prefix1_route_top4_accuracy": find_route_metric(route_summary_rows, "cold", "ALL", "prefix1_top4_accuracy"),
        "prefix2_route_top1_accuracy": find_route_metric(route_summary_rows, "cold", "ALL", "prefix2_top1_accuracy"),
        "prefix2_route_top4_accuracy": find_route_metric(route_summary_rows, "cold", "ALL", "prefix2_top4_accuracy"),
        "prefix2_route_top8_accuracy": find_route_metric(route_summary_rows, "cold", "ALL", "prefix2_top8_accuracy"),
        "target_prefix1_candidate_recall": {
            "predicted_route_p1": find_route_metric(route_summary_rows, "cold", "ALL", "prefix1_top1_accuracy"),
            "predicted_route_p1_top2": find_route_metric(route_summary_rows, "cold", "ALL", "prefix1_top2_accuracy"),
            "predicted_route_p1_top4": find_route_metric(route_summary_rows, "cold", "ALL", "prefix1_top4_accuracy"),
        },
        "target_route_candidate_recall": {
            "predicted_route_p2_top1": find_route_metric(route_summary_rows, "cold", "ALL", "prefix2_top1_accuracy"),
            "predicted_route_p2_top4": find_route_metric(route_summary_rows, "cold", "ALL", "prefix2_top4_accuracy"),
            "predicted_route_p2_top8": find_route_metric(route_summary_rows, "cold", "ALL", "prefix2_top8_accuracy"),
        },
        "route_prediction_by_subset_domain": route_summary_rows,
        "route_source_prediction_by_subset_domain": route_source_summary_rows,
        "class_imbalance_by_subset_domain": imbalance_rows,
        "cold_recall50_by_query_source_and_mode": cold_recall50_by_query_source_and_mode,
        "warm_retention_recall50_by_query_source": warm_retention_by_source,
        "cold_insertion_time_ms_total": cold_insertion_ms,
        "cold_insertion_time_ms_per_item": cold_insertion_ms / max(len(unique_cold_targets), 1),
        "num_unique_cold_targets": len(unique_cold_targets),
        "checkpoint_best": meta.get("train_result", {}).get("best", {}),
        "prefix1_query_head_checkpoint": args.prefix1_query_head_checkpoint,
        "prefix1_query_head_best": (prefix1_query_head_meta or {}).get("train_result", {}).get("best", {}),
        "query_source": args.query_source,
        "query_sources": query_sources,
        "domain_query_source_config": domain_query_source_config,
        "domain_query_sources": domain_query_sources,
        "default_domain_query_source": default_domain_query_source,
        "route_score_weight": args.route_score_weight,
        "per_route_topk": args.per_route_topk,
        "prefix1_beam_sizes": args.prefix1_beam_sizes,
        "prefix2_beam_sizes": args.beam_sizes,
        "extra_prefix1_route_sources": extra_prefix1_route_sources,
        "merge_strategies": args.merge_strategies,
        "fusion_specs": fusion_specs,
        "fusion_method": args.fusion_method,
        "fusion_rrf_k": args.fusion_rrf_k,
    }

    metric_fieldnames = ["sample_count"] + [f"{metric}@{k}" for metric in ("Recall", "NDCG", "MRR") for k in args.topk] + [
        "CandidatePoolHitRate",
        "CandidatePoolLossRate",
        "IIDRatio@50",
        "fallback_rate",
        "avg_latency_ms",
    ]
    summary_fieldnames = ["query_source", "subset", "mode"] + metric_fieldnames
    summary_by_domain_fieldnames = ["query_source", "subset", "domain", "mode"] + metric_fieldnames
    write_csv(output_dir / "summary.csv", summary_fieldnames, summary_rows)
    write_csv(output_dir / "summary_by_domain.csv", summary_by_domain_fieldnames, summary_by_domain_rows)
    write_csv(
        output_dir / "route_diagnostics.csv",
        [
            "subset",
            "domain",
            "sample_count",
            "prefix1_top1_accuracy",
            "prefix1_top2_accuracy",
            "prefix1_top4_accuracy",
            "prefix2_top1_accuracy",
            "prefix2_top4_accuracy",
            "prefix2_top8_accuracy",
        ],
        route_summary_rows,
    )
    write_csv(
        output_dir / "route_source_diagnostics.csv",
        ["subset", "domain", "mode", "sample_count", "candidate_route_recall"],
        route_source_summary_rows,
    )
    write_csv(
        output_dir / "route_distribution.csv",
        ["subset", "domain", "prefix_len", "route", "count", "ratio"],
        distribution_rows,
    )
    write_csv(
        output_dir / "route_imbalance.csv",
        ["subset", "domain", "prefix_len", "sample_count", "num_routes", "top1_share", "top3_share", "entropy", "normalized_entropy"],
        imbalance_rows,
    )

    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "summary_by_domain.json").write_text(json.dumps(summary_by_domain_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "route_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retrieval_rows),
        encoding="utf-8",
    )

    report_lines = [
        "# Predicted Route Eval",
        "",
        f"- Prefix-1 route top-1/top-2/top-4 accuracy: `{diagnostics['prefix1_route_accuracy']:.4f}` / `{diagnostics['prefix1_route_top2_accuracy']:.4f}` / `{diagnostics['prefix1_route_top4_accuracy']:.4f}`",
        f"- Prefix-2 route top-1/top-4/top-8 accuracy: `{diagnostics['prefix2_route_top1_accuracy']:.4f}` / `{diagnostics['prefix2_route_top4_accuracy']:.4f}` / `{diagnostics['prefix2_route_top8_accuracy']:.4f}`",
        f"- Cold insertion ms per item: `{diagnostics['cold_insertion_time_ms_per_item']:.6f}`",
        "",
        "## Cold Recall@50",
    ]
    for query_source in query_sources:
        mode_scores = cold_recall50_by_query_source_and_mode.get(query_source, {})
        for mode in sorted(mode_scores):
            report_lines.append(f"- `{query_source}` `{mode}`: `{mode_scores[mode]:.4f}`")
    report_lines.extend(
        [
            "",
            "## Diagnostics Files",
            "",
            "- `summary_by_domain.csv` reports retrieval metrics by Book/Game/Yelp.",
            "- `route_diagnostics.csv` reports route prediction accuracy by Book/Game/Yelp.",
            "- `route_distribution.csv` and `route_imbalance.csv` report prefix class imbalance.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(output_dir.resolve()), "diagnostics": diagnostics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
