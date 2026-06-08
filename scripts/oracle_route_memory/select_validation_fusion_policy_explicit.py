#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import socket
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
from torch.utils.data import DataLoader

from genrec.memory.data_adapter import load_item_embeddings
from genrec.training import RouterDataset, build_training_examples, load_route_mapping
from genrec.training import (
    compute_config_hash,
    default_cold_like_validation_split_name,
    default_validation_split_name,
    load_protocol_manifest,
    protocol_split_examples,
)
from eval_predicted_route import (
    build_fusion_retrieval_row,
    build_memory,
    enumerate_prefix1_candidates,
    history_prefix1_candidates,
    load_model,
    load_prefix1_query_head,
    mean_history_embedding_from_ids,
    merge_mode_label,
    prefix1_prior_candidates,
    rerank_with_routes,
    resolve_query_source,
    route_to_text,
)

try:
    from scripts.oracle_route_memory.handoff_io import ensure_empty_output_dir, resolve_output_dir, resolve_repo_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from handoff_io import ensure_empty_output_dir, resolve_output_dir, resolve_repo_path

TOPKS = (10, 20, 50)
MERGE_STRATEGIES = ("score", "zscore", "quota", "round_robin", "rrf")
NEXT_TARGET = (
    "Treat the selected explicit policy as validation-only evidence; keep its "
    "policy hash and ranking outputs separate from any fresh-confirmation "
    "claim until a fresh split is registered and scored."
)


BUILTIN_POLICY_PRESETS: Dict[str, Dict[str, Any]] = {
    "explicit_script_oldv0": {
        "domain_query_sources": {"Book": "mean", "Game": "prefix1_head"},
        "default_domain_query_source": "learned",
        "extra_prefix1_route_sources": ["domain_prior"],
        "policy_candidates": [
            {
                "policy_name": "domain_adaptive_domain_prior_p1",
                "query_source": "domain_adaptive",
                "mode": "domain_prior_p1",
                "route_score_weight": 0.0,
                "per_route_topk": None,
            },
            {
                "policy_name": "domain_adaptive_domain_prior_p1_top2_quota",
                "query_source": "domain_adaptive",
                "mode": "domain_prior_p1_top2_quota",
                "route_score_weight": 0.0,
                "per_route_topk": None,
            },
            {
                "policy_name": "fusion_best_rrf",
                "query_source": "fusion",
                "mode": "fusion_best_rrf",
                "fusion_method": "rrf",
                "route_score_weight": 0.0,
                "per_route_topk": None,
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
            },
        ],
        "metadata": {
            "preset_name": "explicit_script_oldv0",
            "description": "Frozen heterogeneous domain-prior fusion preset matching the 2026-06-07 development result.",
        },
    },
    "v3_validation_rrf_candidate": {
        "domain_query_sources": {"Book": "mean", "Game": "prefix1_head"},
        "default_domain_query_source": "learned",
        "extra_prefix1_route_sources": ["domain_prior"],
        "policy_candidates": [
            {
                "policy_name": "domain_adaptive_domain_prior_p1_top4",
                "query_source": "domain_adaptive",
                "mode": "domain_prior_p1_top4",
                "route_score_weight": 0.0,
                "per_route_topk": None,
            },
            {
                "policy_name": "domain_adaptive_predicted_route_p1_top4",
                "query_source": "domain_adaptive",
                "mode": "predicted_route_p1_top4",
                "route_score_weight": 0.0,
                "per_route_topk": None,
            },
            {
                "policy_name": "mean_predicted_route_p1_top4",
                "query_source": "mean",
                "mode": "predicted_route_p1_top4",
                "route_score_weight": 0.0,
                "per_route_topk": None,
            },
            {
                "policy_name": "fusion_comparison_rrf",
                "query_source": "fusion",
                "mode": "fusion_comparison_rrf",
                "fusion_method": "rrf",
                "route_score_weight": 0.0,
                "per_route_topk": None,
                "fusion_specs": [
                    {
                        "name": "comparison_rrf",
                        "members": [
                            ["domain_adaptive", "predicted_route_p1_top4"],
                            ["mean", "predicted_route_p1_top4"],
                        ],
                    }
                ],
            },
        ],
        "metadata": {
            "preset_name": "v3_validation_rrf_candidate",
            "description": (
                "Validation-only candidate preset for a future blind-confirmation lock. "
                "It mirrors the strongest v3 comparison diagnostics without making them retroactive claims."
            ),
            "claim_boundary": "Select only on validation before a fresh blind confirmation run; do not apply to the consumed 2026-06-07 blind set.",
        },
    },
}


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "t"}


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def selected_extra_prefix1_route_sources(route_sources: Sequence[str]) -> List[str]:
    if not route_sources:
        return []
    if "all" in route_sources:
        return ["domain_prior", "history_last", "history_vote", "history_recency"]
    return list(dict.fromkeys(str(source) for source in route_sources))


def parse_mode_details(mode: str) -> Tuple[str, int, str]:
    merge_strategy = "score"
    base_mode = str(mode)
    for candidate in MERGE_STRATEGIES[1:]:
        suffix = f"_{candidate}"
        if base_mode.endswith(suffix):
            merge_strategy = candidate
            base_mode = base_mode[: -len(suffix)]
            break
    beam = 1
    if "_top" in base_mode:
        stem, beam_text = base_mode.rsplit("_top", 1)
        base_mode = stem + f"_top{beam_text}"
        beam = int(beam_text)
    return base_mode, beam, merge_strategy


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "sample_count": 0,
            "RouteHitRate": 0.0,
            "CandidatePoolHitRate": 0.0,
            "CandidatePoolLossRate": 0.0,
            "ConditionalRecall@50GivenPoolHit": 0.0,
            "RouteMissRate": 0.0,
            "CandidatePoolMissRate": 0.0,
            "RankingMissRate": 0.0,
            **{f"Recall@{k}": 0.0 for k in TOPKS},
        }
    recalls = {}
    for k in TOPKS:
        recalls[f"Recall@{k}"] = float(np.mean([1.0 if row["match_rank"] is not None and row["match_rank"] <= k else 0.0 for row in rows]))
    route_hit_rate = float(np.mean([1.0 if row.get("route_hit") else 0.0 for row in rows]))
    pool_hit_rate = float(np.mean([1.0 if row.get("candidate_pool_hit") else 0.0 for row in rows]))
    recall50 = recalls["Recall@50"]
    pool_loss = float(np.mean([1.0 if row.get("candidate_pool_hit") and not (row["match_rank"] is not None and row["match_rank"] <= 50) else 0.0 for row in rows]))
    return {
        "sample_count": n,
        "RouteHitRate": route_hit_rate,
        "CandidatePoolHitRate": pool_hit_rate,
        "CandidatePoolLossRate": pool_loss,
        "ConditionalRecall@50GivenPoolHit": float(recall50 / pool_hit_rate) if pool_hit_rate > 0 else 0.0,
        "RouteMissRate": float(1.0 - route_hit_rate),
        "CandidatePoolMissRate": float(1.0 - pool_hit_rate),
        "RankingMissRate": pool_loss,
        **recalls,
    }


def grouped_summary(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, subset in sorted(groups.items()):
        metric = summarize_rows(subset)
        metric.update({key: value for key, value in zip(keys, key_values)})
        out.append(metric)
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    extras = sorted({key for row in rows for key in row.keys() if key not in fieldnames})
    fieldnames.extend(extras)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def normalize_candidate(raw: Mapping[str, Any]) -> Dict[str, Any]:
    policy_name = str(raw["policy_name"])
    query_source = str(raw["query_source"])
    mode = str(raw["mode"])
    per_route_topk = raw.get("per_route_topk")
    if per_route_topk is not None:
        per_route_topk = int(per_route_topk)
    fusion_specs = list(raw.get("fusion_specs", []))
    fusion_method = raw.get("fusion_method")
    query_sources = [query_source]
    route_beam = 1
    merge_strategy = "score"
    if query_source == "fusion":
        if not fusion_specs:
            raise ValueError(f"Fusion candidate {policy_name!r} requires fusion_specs.")
        member_modes = []
        query_sources = []
        max_beam = 1
        merge_values = set()
        for query_source_name, member_mode in fusion_specs[0]["members"]:
            member_mode = str(member_mode)
            _, beam, member_merge = parse_mode_details(member_mode)
            max_beam = max(max_beam, beam)
            merge_values.add(member_merge)
            query_sources.append(str(query_source_name))
            member_modes.append(member_mode)
        route_beam = max_beam
        merge_strategy = merge_values.pop() if len(merge_values) == 1 else "score"
    else:
        _, route_beam, merge_strategy = parse_mode_details(mode)
    return {
        "policy_name": policy_name,
        "query_source": query_source,
        "mode": mode,
        "query_sources": list(dict.fromkeys(query_sources)),
        "fusion_method": fusion_method or ("single" if query_source != "fusion" else "rrf"),
        "fusion_specs": fusion_specs,
        "route_beam": int(raw.get("route_beam", route_beam)),
        "prefix1_beam_sizes": [int(raw.get("route_beam", route_beam))],
        "route_score_weight": float(raw.get("route_score_weight", 0.0)),
        "per_route_topk": per_route_topk,
        "merge_strategy": str(raw.get("merge_strategy", merge_strategy)),
    }


def _normalize_explicit_policy_payload(payload: Mapping[str, Any], source_path: str, source_name: str) -> Dict[str, Any]:
    policies = [normalize_candidate(row) for row in payload.get("policy_candidates", [])]
    if not policies:
        raise ValueError(f"No policy_candidates were defined in {source_name}.")
    resolved_payload = {
        "domain_query_sources": {str(k): str(v) for k, v in payload.get("domain_query_sources", {}).items()},
        "default_domain_query_source": str(payload.get("default_domain_query_source", "learned")),
        "extra_prefix1_route_sources": selected_extra_prefix1_route_sources(payload.get("extra_prefix1_route_sources", [])),
        "policy_candidates": payload.get("policy_candidates", []),
        "metadata": payload.get("metadata", {}),
    }
    return {
        "path": source_path,
        "source_name": source_name,
        "config_hash": compute_config_hash(resolved_payload),
        "resolved_payload": resolved_payload,
        "domain_query_sources": resolved_payload["domain_query_sources"],
        "default_domain_query_source": resolved_payload["default_domain_query_source"],
        "extra_prefix1_route_sources": resolved_payload["extra_prefix1_route_sources"],
        "policies": policies,
        "metadata": resolved_payload["metadata"],
    }


def resolve_explicit_policy_config(path: str | Path | None, preset: str | None) -> Dict[str, Any]:
    if path and preset:
        raise ValueError("Use either --explicit-policy-config or --preset, not both.")
    if preset:
        if preset not in BUILTIN_POLICY_PRESETS:
            raise ValueError(f"Unknown explicit policy preset: {preset}")
        payload = BUILTIN_POLICY_PRESETS[preset]
        return _normalize_explicit_policy_payload(payload, f"preset:{preset}", f"preset {preset}")
    if path is None:
        raise ValueError("An explicit policy source is required. Provide --explicit-policy-config or --preset.")
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return _normalize_explicit_policy_payload(payload, str(config_path), str(config_path))


def policy_metadata(policy: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "policy_name": policy["policy_name"],
        "query_source": policy["query_source"],
        "mode": policy["mode"],
        "query_sources": list(policy.get("query_sources", [])),
        "fusion_method": policy.get("fusion_method", "single"),
        "fusion_specs": list(policy.get("fusion_specs", [])),
        "route_beam": int(policy.get("route_beam", 1)),
        "prefix1_beam_sizes": list(policy.get("prefix1_beam_sizes", [int(policy.get("route_beam", 1))])),
        "route_score_weight": float(policy.get("route_score_weight", 0.0)),
        "per_route_topk": policy.get("per_route_topk"),
        "merge_strategy": policy.get("merge_strategy", "score"),
    }


RetrievalKey = Tuple[str, str, float, int | None]


def retrieval_key_for_policy_member(policy: Mapping[str, Any], query_source: str, mode: str) -> RetrievalKey:
    per_route_topk = policy.get("per_route_topk")
    if per_route_topk is not None:
        per_route_topk = int(per_route_topk)
    return (
        str(query_source),
        str(mode),
        float(policy.get("route_score_weight", 0.0)),
        per_route_topk,
    )


def required_retrieval_keys(policies: Sequence[Mapping[str, Any]]) -> set[RetrievalKey]:
    keys: set[RetrievalKey] = set()
    for policy in policies:
        members = [(policy["query_source"], policy["mode"])]
        if policy["query_source"] == "fusion":
            members = list(policy["fusion_specs"][0]["members"])
        for query_source, mode in members:
            keys.add(retrieval_key_for_policy_member(policy, str(query_source), str(mode)))
    return keys


def pick_best_policy(summary_rows: Sequence[Mapping[str, Any]], validation_split: str) -> Mapping[str, Any]:
    candidates = [row for row in summary_rows if row.get("split") == validation_split]
    if not candidates:
        raise ValueError(f"No {validation_split} policy summaries were produced.")
    return max(
        candidates,
        key=lambda row: (
            float(row.get("Recall@50", 0.0)),
            float(row.get("CandidatePoolHitRate", 0.0)),
            float(row.get("Recall@20", 0.0)),
            -int(row.get("route_beam", 999)),
            -int(row.get("per_route_topk", 999999)) if row.get("per_route_topk") is not None else -999999,
        ),
    )


def evaluate_policies(
    splits: Mapping[str, Sequence[Any]],
    policies: Sequence[Mapping[str, Any]],
    item_embeddings: Mapping[str, np.ndarray],
    route_mapping: Mapping[str, Tuple[int, int]],
    model: Any,
    route_vocab: Any,
    device: str,
    batch_size: int,
    prefix1_query_head: Any,
    domain_query_sources: Mapping[str, str],
    default_domain_query_source: str,
    extra_prefix1_route_sources: Sequence[str],
) -> List[Dict[str, Any]]:
    required_keys = required_retrieval_keys(policies)
    required_predicted_beams = {1}
    required_heuristic_beams: Dict[str, set[int]] = defaultdict(set)
    for query_source, mode, _, _ in required_keys:
        base_mode, beam, _ = parse_mode_details(str(mode))
        if base_mode.startswith("predicted_route_p1"):
            required_predicted_beams.add(beam)
        elif base_mode.startswith(("domain_prior_p1", "history_last_p1", "history_vote_p1", "history_recency_p1")):
            source_name = base_mode.split("_p1", 1)[0]
            required_heuristic_beams[source_name].add(beam)
        else:
            raise ValueError(f"Unsupported explicit policy mode: {mode}")

    memory = build_memory(item_embeddings, route_mapping, prefix_len=1)
    rows: List[Dict[str, Any]] = []
    train_examples = build_training_examples("user_simulator", item_embeddings, route_mapping, max_history=10)
    domain_prefix1_counts: Dict[str, Counter] = defaultdict(Counter)
    global_prefix1_counts: Counter = Counter()
    for example in train_examples:
        domain_prefix1_counts[str(example.domain)][int(example.route_prefix1)] += 1
        global_prefix1_counts[int(example.route_prefix1)] += 1

    for split_name, examples in splits.items():
        dataset = RouterDataset(examples, item_embeddings, route_vocab)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        with torch.no_grad():
            for batch in loader:
                outputs = model(batch["history_embs"].to(device), batch["history_mask"].to(device))
                route1_log_probs, _, _ = model.route_log_probs(outputs)
                prefix1_candidates_by_beam = {
                    beam: enumerate_prefix1_candidates(route1_log_probs.cpu(), route_vocab, beam)
                    for beam in sorted(required_predicted_beams)
                }
                pooled_embeddings = torch.nn.functional.normalize(outputs["pooled_history"], dim=-1).cpu().numpy()
                learned_embeddings = outputs["query_embedding"].cpu().numpy()
                residual_embeddings = torch.nn.functional.normalize(outputs["query_embedding"] + outputs["pooled_history"], dim=-1).cpu().numpy()
                mean_embeddings = np.stack([mean_history_embedding_from_ids(history, item_embeddings) for history in batch["history_item_ids"]], axis=0)
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
                    domain = str(batch["domain"][idx])
                    sample_id = str(batch["sample_id"][idx])
                    true_route = tuple(int(part) for part in route_mapping[target_item_id])
                    true_prefix1 = true_route[:1]

                    heuristic_candidates_by_source: Dict[str, Dict[int, Sequence[Any]]] = defaultdict(dict)
                    for source_name, beams in required_heuristic_beams.items():
                        prior_cache: Dict[int, Sequence[Any]] = {}
                        for beam in sorted(beams):
                            prior = prefix1_prior_candidates(domain_prefix1_counts.get(domain, Counter()), global_prefix1_counts, beam)
                            if source_name == "domain_prior":
                                candidates = prior
                            else:
                                candidates = history_prefix1_candidates(batch["history_item_ids"][idx], route_mapping, source_name, beam, prior)
                            prior_cache[beam] = candidates
                        heuristic_candidates_by_source[source_name] = prior_cache

                    sample_rows: Dict[RetrievalKey, Dict[str, Any]] = {}
                    for query_source, mode, route_score_weight, per_route_topk in sorted(required_keys):
                        effective_source = resolve_query_source(query_source, domain, domain_query_sources, default_domain_query_source) if query_source == "domain_adaptive" else query_source
                        if effective_source not in query_embeddings:
                            raise ValueError(f"Query source {query_source!r} resolved to unavailable source {effective_source!r}.")
                        base_mode, beam, merge_strategy = parse_mode_details(mode)
                        if base_mode.startswith("predicted_route_p1"):
                            route_candidates = prefix1_candidates_by_beam[beam][idx]
                        else:
                            source_name = base_mode.split("_p1", 1)[0]
                            route_candidates = heuristic_candidates_by_source[source_name][beam]
                        ranked_ids, latency_ms, fallback_used, diagnostics = rerank_with_routes(
                            query_embedding=query_embeddings[effective_source][idx],
                            route_candidates=route_candidates,
                            prefix_len=1,
                            memory=memory,
                            topks=TOPKS,
                            route_score_weight=route_score_weight,
                            merge_strategy=merge_strategy,
                            per_route_topk=per_route_topk,
                            target_item_id=target_item_id,
                        )
                        match_rank = next((rank for rank, item_id in enumerate(ranked_ids, start=1) if item_id == target_item_id), None)
                        sample_rows[(query_source, mode, route_score_weight, per_route_topk)] = {
                            "query_source": query_source,
                            "effective_query_source": effective_source,
                            "split": split_name,
                            "subset": split_name,
                            "domain": domain,
                            "mode": mode,
                            "sample_id": sample_id,
                            "target_item_id": target_item_id,
                            "true_route": route_to_text(true_route),
                            "ranked_ids": ranked_ids,
                            "match_rank": match_rank,
                            "latency_ms": latency_ms,
                            "fallback_used": fallback_used,
                            "route_hit": any(tuple(route[:1]) == true_prefix1 for route, _ in route_candidates),
                            **diagnostics,
                        }

                    for policy in policies:
                        if policy["query_source"] == "fusion":
                            fusion_sample_rows = {
                                (query_source, mode): sample_rows[retrieval_key_for_policy_member(policy, str(query_source), str(mode))]
                                for query_source, mode in policy["fusion_specs"][0]["members"]
                            }
                            result = build_fusion_retrieval_row(
                                fusion_spec=policy["fusion_specs"][0],
                                sample_retrieval_rows=fusion_sample_rows,
                                sample_id=sample_id,
                                target_item_id=target_item_id,
                                true_route=true_route,
                                topks=TOPKS,
                                fusion_method=str(policy["fusion_method"]),
                                fusion_rrf_k=60.0,
                                per_route_topk=policy["per_route_topk"],
                            )
                            # Keep explicit selector fusion rows aligned with the
                            # single-policy summary schema.
                            result["split"] = split_name
                            result.setdefault("subset", split_name)
                        else:
                            result = dict(sample_rows[retrieval_key_for_policy_member(policy, policy["query_source"], policy["mode"])])
                        result.update(policy_metadata(policy))
                        rows.append(result)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Select an explicit set of prefix-1 policies on a protocol-locked validation split.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--item-embedding-path", required=True)
    parser.add_argument("--item-sid-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--explicit-policy-config", default=None)
    parser.add_argument("--preset", choices=sorted(BUILTIN_POLICY_PRESETS), default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix1-query-head-checkpoint", default=None)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--selection-split", default=None)
    parser.add_argument("--warm-split", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, args.repo_root)
    data_dir = resolve_repo_path(args.data_dir, args.repo_root)
    item_embedding_path = resolve_repo_path(args.item_embedding_path, args.repo_root)
    item_sid_path = resolve_repo_path(args.item_sid_path, args.repo_root)
    checkpoint_dir = resolve_repo_path(args.checkpoint_dir, args.repo_root)
    explicit_policy_config_path = resolve_repo_path(args.explicit_policy_config, args.repo_root)
    protocol_manifest_path = resolve_repo_path(args.protocol_manifest, args.repo_root)
    prefix1_query_head_checkpoint = resolve_repo_path(args.prefix1_query_head_checkpoint, args.repo_root)
    ensure_empty_output_dir(output_dir)

    explicit_config = resolve_explicit_policy_config(explicit_policy_config_path, args.preset)
    item_embeddings = load_item_embeddings(data_dir, item_embedding_path)
    route_mapping = load_route_mapping(item_sid_path)
    model, route_vocab, checkpoint_meta = load_model(checkpoint_dir, device)
    prefix1_query_head = None
    prefix1_query_head_meta = None
    needed_sources = set(explicit_config["domain_query_sources"].values())
    for policy in explicit_config["policies"]:
        needed_sources.update(policy.get("query_sources", []))
    if "prefix1_head" in needed_sources:
        if prefix1_query_head_checkpoint is None:
            raise ValueError("--prefix1-query-head-checkpoint is required when explicit policies use prefix1_head.")
        prefix1_query_head, prefix1_query_head_meta = load_prefix1_query_head(prefix1_query_head_checkpoint, device)

    examples = build_training_examples(data_dir, item_embeddings, route_mapping, max_history=args.max_history)
    protocol_manifest = load_protocol_manifest(protocol_manifest_path)
    selection_split = args.selection_split or default_cold_like_validation_split_name(protocol_manifest)
    warm_split = args.warm_split or default_validation_split_name(protocol_manifest)
    splits = {
        selection_split: protocol_split_examples(examples, protocol_manifest, selection_split),
        warm_split: protocol_split_examples(examples, protocol_manifest, warm_split),
    }

    rows = evaluate_policies(
        splits=splits,
        policies=explicit_config["policies"],
        item_embeddings=item_embeddings,
        route_mapping=route_mapping,
        model=model,
        route_vocab=route_vocab,
        device=device,
        batch_size=args.batch_size,
        prefix1_query_head=prefix1_query_head,
        domain_query_sources=explicit_config["domain_query_sources"],
        default_domain_query_source=explicit_config["default_domain_query_source"],
        extra_prefix1_route_sources=explicit_config["extra_prefix1_route_sources"],
    )
    policy_by_name = {policy["policy_name"]: policy for policy in explicit_config["policies"]}
    enriched_rows = [
        {
            **policy_metadata(policy_by_name[str(row["policy_name"])]),
            **row,
            "policy_config_path": explicit_config["path"],
            "policy_config_hash": explicit_config["config_hash"],
            "policy_origin": explicit_config["source_name"],
        }
        for row in rows
    ]
    summary = grouped_summary(enriched_rows, ["policy_name", "split"])
    summary_by_domain = grouped_summary(enriched_rows, ["policy_name", "split", "domain"])

    enriched_summary = [{**policy_metadata(policy_by_name[str(row["policy_name"])]), **row} for row in summary]
    enriched_by_domain = [{**policy_metadata(policy_by_name[str(row["policy_name"])]), **row} for row in summary_by_domain]

    best = pick_best_policy(enriched_summary, selection_split)
    best_policy = policy_by_name[str(best["policy_name"])]
    best_meta = policy_metadata(best_policy)
    default_query_policy = {"query_source": best_meta["query_source"], "mode": best_meta["mode"]}
    config = {
        "domain_query_sources": explicit_config["domain_query_sources"],
        "default_domain_query_source": explicit_config["default_domain_query_source"],
        "extra_prefix1_route_sources": explicit_config["extra_prefix1_route_sources"],
        "default_query_policy": default_query_policy,
        "domain_query_policies": {},
        "fusion_specs": best_meta["fusion_specs"],
        "route_score_weight": float(best_meta["route_score_weight"]),
        "per_route_topk": best_meta["per_route_topk"],
        "prefix1_beam_sizes": list(best_meta["prefix1_beam_sizes"]),
        "fusion_method": best_meta["fusion_method"],
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "selector": Path(__file__).name,
            "selection_metric": f"{selection_split} Recall@50, tie-broken by CandidatePoolHitRate/Recall@20/smaller beam/topk",
            "leakage_note": "Policy selection uses protocol-locked train-derived validation only and never reads cold test labels.",
            "seed": args.seed,
            "split_sizes": {name: len(values) for name, values in splits.items()},
            "selection_split": selection_split,
            "warm_split": warm_split,
            "args": vars(args),
            "checkpoint_best": checkpoint_meta.get("train_result", {}).get("best", {}),
            "prefix1_query_head_checkpoint": str(prefix1_query_head_checkpoint.resolve()) if prefix1_query_head_checkpoint else None,
            "prefix1_query_head_best": (prefix1_query_head_meta or {}).get("train_result", {}).get("best", {}),
            "domain_query_sources": explicit_config["domain_query_sources"],
            "default_domain_query_source": explicit_config["default_domain_query_source"],
            "extra_prefix1_route_sources": explicit_config["extra_prefix1_route_sources"],
            "protocol_manifest": str(protocol_manifest_path.resolve()),
            "protocol_config_hash": protocol_manifest.get("config_hash"),
            "explicit_policy_config": explicit_config["path"],
            "explicit_policy_config_hash": explicit_config["config_hash"],
            "explicit_policy_source": explicit_config["source_name"],
            "explicit_policy_metadata": explicit_config["metadata"],
            **best_meta,
        },
        "policy_metrics": {
            "selected_validation": dict(best),
            "all_summaries_path": "selector_summary.csv",
            "summary_by_domain_path": "selector_summary_by_domain.csv",
        },
    }
    config["config_hash"] = compute_config_hash(config)

    (output_dir / "resolved_policy_candidates.json").write_text(json.dumps(explicit_config["resolved_payload"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "fusion_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "selector_rows.json").write_text(json.dumps(enriched_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "selector_summary.json").write_text(json.dumps(enriched_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "selector_summary_by_domain.json").write_text(json.dumps(enriched_by_domain, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "selector_rows.csv", enriched_rows)
    write_csv(output_dir / "selector_summary.csv", enriched_summary)
    write_csv(output_dir / "selector_summary_by_domain.csv", enriched_by_domain)
    selector_manifest = {
        "status": "ok",
        "selected_policy": str(best_policy["policy_name"]),
        "selection_split": selection_split,
        "warm_split": warm_split,
        "output_dir": str(output_dir),
        "artifacts": {
            "fusion_config": "fusion_config.json",
            "selector_rows": "selector_rows.json",
            "selector_summary": "selector_summary.json",
            "selector_summary_by_domain": "selector_summary_by_domain.json",
            "report": "report.md",
        },
        "next_target": NEXT_TARGET,
    }
    (output_dir / "selector_manifest.json").write_text(
        json.dumps(selector_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    top = sorted([row for row in enriched_summary if row.get("split") == selection_split], key=lambda row: float(row.get("Recall@50", 0.0)), reverse=True)
    report_lines = [
        "# Explicit Validation Policy Selector",
        "",
        f"- Selected policy: `{best_policy['policy_name']}`",
        f"- Locked eval policy: `{default_query_policy['query_source']}:{default_query_policy['mode']}`",
        f"- Route beam / route score weight / per-route topk: `{best_meta['route_beam']}` / `{best_meta['route_score_weight']}` / `{best_meta['per_route_topk']}`",
        f"- Query sources: `{'+'.join(best_meta['query_sources'])}`",
        f"- Fusion method: `{best_meta['fusion_method']}`",
        f"- Extra prefix1 route sources: `{','.join(explicit_config['extra_prefix1_route_sources']) or 'none'}`",
        f"- Protocol hash: `{config['config_hash']}`",
        "",
        "## Selected Cold-Like Validation Metrics",
        "",
        f"- Recall@10/20/50: `{float(best['Recall@10']):.4f}` / `{float(best['Recall@20']):.4f}` / `{float(best['Recall@50']):.4f}`",
        f"- RouteHitRate / CandidatePoolHitRate / ConditionalRecall@50GivenPoolHit: `{float(best['RouteHitRate']):.4f}` / `{float(best['CandidatePoolHitRate']):.4f}` / `{float(best['ConditionalRecall@50GivenPoolHit']):.4f}`",
        "",
        "## Cold-Like Policy Ranking",
        "",
    ]
    for row in top:
        report_lines.append(f"- `{row['policy_name']}`: R@50 `{float(row['Recall@50']):.4f}`, pool `{float(row['CandidatePoolHitRate']):.4f}`, route `{float(row['RouteHitRate']):.4f}`")
    report_lines.extend(["", "## Next Target", "", NEXT_TARGET])
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir.resolve()),
                "selected": config["metadata"],
                "next_target": NEXT_TARGET,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
