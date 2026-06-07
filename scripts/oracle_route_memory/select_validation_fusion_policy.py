#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import socket
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
from genrec.training import compute_config_hash, load_protocol_manifest, protocol_split_examples
from eval_predicted_route import (
    RouteCandidate,
    build_memory,
    enumerate_prefix1_candidates,
    fuse_ranked_lists,
    load_model,
    load_domain_query_source_config,
    mean_history_embedding_from_ids,
    merge_mode_label,
    load_prefix1_query_head,
    resolve_query_source,
    rerank_with_routes,
    route_to_text,
)


BASE_QUERY_SOURCES = ("learned", "residual", "pooled", "mean")
OPTIONAL_QUERY_SOURCES = ("prefix1_head", "domain_adaptive")
TOPKS = (10, 20, 50)


@dataclass(frozen=True)
class PolicyCandidate:
    name: str
    query_sources: Tuple[str, ...]
    fusion_method: str | None
    route_beam: int
    route_score_weight: float
    per_route_topk: int
    merge_strategy: str

    @property
    def mode(self) -> str:
        suffix = "" if self.route_beam == 1 else f"_top{self.route_beam}"
        return f"predicted_route_p1{suffix}_{self.merge_strategy}"

    @property
    def eval_query_source(self) -> str:
        return "fusion" if len(self.query_sources) > 1 else self.query_sources[0]

    @property
    def eval_mode(self) -> str:
        return f"fusion_{self.name}" if len(self.query_sources) > 1 else self.mode


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "t"}


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def make_policy_grid(
    route_beams: Sequence[int],
    route_score_weights: Sequence[float],
    per_route_topks: Sequence[int],
    merge_strategies: Sequence[str],
    fusion_methods: Sequence[str],
    include_single_sources: bool,
    candidate_query_sources: Sequence[str],
) -> List[PolicyCandidate]:
    allowed_query_sources = tuple(dict.fromkeys(str(source) for source in candidate_query_sources))
    available_base_sources = tuple(source for source in BASE_QUERY_SOURCES if source in allowed_query_sources)
    source_groups: List[Tuple[str, ...]] = []

    if include_single_sources:
        source_groups.extend((source,) for source in allowed_query_sources)

    preferred_groups = (
        ("learned", "residual"),
        ("learned", "pooled"),
        ("learned", "residual", "pooled"),
        ("learned", "residual", "pooled", "mean"),
    )
    for group in preferred_groups:
        if all(source in allowed_query_sources for source in group):
            source_groups.append(group)

    anchor_sources = tuple(source for source in ("learned", "residual", "pooled", "mean") if source in allowed_query_sources)
    optional_sources = tuple(source for source in allowed_query_sources if source in OPTIONAL_QUERY_SOURCES)
    for optional_source in optional_sources:
        for anchor_source in anchor_sources:
            if anchor_source == optional_source:
                continue
            source_groups.append((anchor_source, optional_source))
        if len(available_base_sources) >= 2:
            source_groups.append(tuple(dict.fromkeys((*available_base_sources[:2], optional_source))))

    if len(optional_sources) >= 2:
        source_groups.append(optional_sources)

    deduped_groups: List[Tuple[str, ...]] = []
    seen_groups = set()
    for group in source_groups:
        normalized = tuple(dict.fromkeys(group))
        if len(normalized) < 1:
            continue
        if normalized in seen_groups:
            continue
        seen_groups.add(normalized)
        deduped_groups.append(normalized)
    source_groups = deduped_groups

    policies: List[PolicyCandidate] = []
    for beam in route_beams:
        for weight in route_score_weights:
            for per_route_topk in per_route_topks:
                for merge_strategy in merge_strategies:
                    for sources in source_groups:
                        methods: Iterable[str | None] = fusion_methods if len(sources) > 1 else (None,)
                        for method in methods:
                            source_tag = "_".join(sources)
                            method_tag = method or "single"
                            weight_tag = str(weight).replace(".", "p").replace("-", "m")
                            name = f"{source_tag}_{method_tag}_p1b{beam}_w{weight_tag}_k{per_route_topk}_{merge_strategy}"
                            policies.append(
                                PolicyCandidate(
                                    name=name,
                                    query_sources=tuple(sources),
                                    fusion_method=method,
                                    route_beam=int(beam),
                                    route_score_weight=float(weight),
                                    per_route_topk=int(per_route_topk),
                                    merge_strategy=str(merge_strategy),
                                )
                            )
    return policies


def route_hit(route_candidates: Sequence[RouteCandidate], true_prefix1: Tuple[int, ...]) -> bool:
    return any(tuple(route[:1]) == true_prefix1 for route, _ in route_candidates)


def match_rank(ranked_ids: Sequence[str], target_item_id: str) -> int | None:
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id == target_item_id:
            return rank
    return None


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



def summarize_route_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"sample_count": 0, "RouteHitRate": 0.0, "RouteMissRate": 0.0}
    hit_rate = float(np.mean([1.0 if row.get("route_hit") else 0.0 for row in rows]))
    return {"sample_count": len(rows), "RouteHitRate": hit_rate, "RouteMissRate": float(1.0 - hit_rate)}


def grouped_route_summary(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, subset in sorted(groups.items()):
        metric = summarize_route_rows(subset)
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def evaluate_policies(
    splits: Mapping[str, Sequence[RouterExample]],
    policies: Sequence[PolicyCandidate],
    item_embeddings: Mapping[str, np.ndarray],
    route_mapping: Mapping[str, Tuple[int, int]],
    model: Any,
    route_vocab: Any,
    device: str,
    batch_size: int,
    prefix1_query_head: Any,
    domain_query_sources: Mapping[str, str],
    default_domain_query_source: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    max_beam = max(policy.route_beam for policy in policies)
    memory = build_memory(item_embeddings, route_mapping, prefix_len=1)
    rows: List[Dict[str, Any]] = []
    route_rows: List[Dict[str, Any]] = []
    policies_by_base: Dict[Tuple[int, float, int, str], List[PolicyCandidate]] = defaultdict(list)
    for policy in policies:
        policies_by_base[(policy.route_beam, policy.route_score_weight, policy.per_route_topk, policy.merge_strategy)].append(policy)

    for split_name, examples in splits.items():
        dataset = RouterDataset(examples, item_embeddings, route_vocab)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        with torch.no_grad():
            for batch in loader:
                outputs = model(batch["history_embs"].to(device), batch["history_mask"].to(device))
                route1_log_probs, _, _ = model.route_log_probs(outputs)
                candidates_by_beam = {
                    beam: enumerate_prefix1_candidates(route1_log_probs.cpu(), route_vocab, beam)
                    for beam in sorted({1, 2, 4, max_beam} | {policy.route_beam for policy in policies})
                }
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
                    domain = str(batch["domain"][idx])
                    sample_id = str(batch["sample_id"][idx])
                    true_route = tuple(int(part) for part in route_mapping[target_item_id])
                    true_prefix1 = true_route[:1]
                    for beam in (1, 2, 4):
                        beam_candidates = candidates_by_beam[beam][idx]
                        route_rows.append(
                            {
                                "split": split_name,
                                "domain": domain,
                                "sample_id": sample_id,
                                "beam": beam,
                                "true_prefix1": route_to_text(true_prefix1),
                                "route_hit": int(route_hit(beam_candidates, true_prefix1)),
                            }
                        )

                    base_cache: Dict[Tuple[str, int, float, int], Dict[str, Any]] = {}
                    for (beam, weight, per_route_topk, merge_strategy), base_policies in policies_by_base.items():
                        route_candidates = candidates_by_beam[beam][idx]
                        current_route_hit = route_hit(route_candidates, true_prefix1)
                        eval_sources = sorted({source for policy in base_policies for source in policy.query_sources})
                        for source in eval_sources:
                            effective_source = resolve_query_source(
                                source,
                                domain,
                                domain_query_sources,
                                default_domain_query_source,
                            ) if source == "domain_adaptive" else source
                            if effective_source not in query_embeddings:
                                raise ValueError(f"Query source {source!r} resolved to unavailable source {effective_source!r}.")
                            ranked_ids, latency_ms, fallback_used, diagnostics = rerank_with_routes(
                                query_embedding=query_embeddings[effective_source][idx],
                                route_candidates=route_candidates,
                                prefix_len=1,
                                memory=memory,
                                topks=TOPKS,
                                route_score_weight=weight,
                                merge_strategy=merge_strategy,
                                per_route_topk=per_route_topk,
                                target_item_id=target_item_id,
                            )
                            rank = match_rank(ranked_ids, target_item_id)
                            base_cache[(source, beam, weight, per_route_topk)] = {
                                "query_source": source,
                                "effective_query_source": effective_source,
                                "split": split_name,
                                "domain": domain,
                                "mode": merge_mode_label("predicted_route_p1" if beam == 1 else f"predicted_route_p1_top{beam}", merge_strategy),
                                "sample_id": sample_id,
                                "target_item_id": target_item_id,
                                "true_route": route_to_text(true_route),
                                "route_candidates": [{"route": route_to_text(route), "score": score} for route, score in route_candidates],
                                "ranked_ids": ranked_ids,
                                "match_rank": rank,
                                "latency_ms": latency_ms,
                                "fallback_used": fallback_used,
                                "route_hit": current_route_hit,
                                **diagnostics,
                            }

                        for policy in base_policies:
                            if len(policy.query_sources) == 1:
                                base = dict(base_cache[(policy.query_sources[0], beam, weight, per_route_topk)])
                                base["policy_name"] = policy.name
                                base["policy_query_sources"] = "+".join(policy.query_sources)
                                base["fusion_method"] = "single"
                                base["route_beam"] = beam
                                base["route_score_weight"] = weight
                                base["per_route_topk"] = per_route_topk
                                rows.append(base)
                                continue

                            member_rows = [base_cache[(source, beam, weight, per_route_topk)] for source in policy.query_sources]
                            ranked_ids = fuse_ranked_lists(
                                [member["ranked_ids"] for member in member_rows],
                                max_k=max(TOPKS),
                                method=str(policy.fusion_method),
                            )
                            rank = match_rank(ranked_ids, target_item_id)
                            candidate_pool = {item_id for member in member_rows for item_id in member["ranked_ids"][: max(TOPKS)]}
                            rows.append(
                                {
                                    "query_source": "fusion",
                                    "effective_query_source": "fusion",
                                    "split": split_name,
                                    "domain": domain,
                                    "mode": f"fusion_{policy.name}",
                                    "sample_id": sample_id,
                                    "target_item_id": target_item_id,
                                    "true_route": route_to_text(true_route),
                                    "ranked_ids": ranked_ids,
                                    "match_rank": rank,
                                    "latency_ms": sum(float(member.get("latency_ms", 0.0)) for member in member_rows),
                                    "fallback_used": any(bool(member.get("fallback_used")) for member in member_rows),
                                    "candidate_pool_size": len(candidate_pool),
                                    "candidate_pool_hit": target_item_id in candidate_pool,
                                    "num_route_candidates": sum(int(member.get("num_route_candidates", 0)) for member in member_rows),
                                    "merge_strategy": f"fusion_{policy.fusion_method}",
                                    "per_route_topk": per_route_topk,
                                    "route_hit": current_route_hit,
                                    "policy_name": policy.name,
                                    "policy_query_sources": "+".join(policy.query_sources),
                                    "fusion_method": policy.fusion_method,
                                    "route_beam": beam,
                                    "route_score_weight": weight,
                                    "fusion_members": [f"{source}:{policy.mode}" for source in policy.query_sources],
                                }
                            )
    return rows, route_rows


def policy_metadata(policy: PolicyCandidate) -> Dict[str, Any]:
    fusion_specs = []
    if len(policy.query_sources) > 1:
        fusion_specs.append(
            {
                "name": policy.name,
                "members": [[source, policy.mode] for source in policy.query_sources],
            }
        )
    return {
        "policy_name": policy.name,
        "query_source": policy.eval_query_source,
        "mode": policy.eval_mode,
        "query_sources": list(policy.query_sources),
        "fusion_method": policy.fusion_method or "single",
        "fusion_specs": fusion_specs,
        "route_beam": policy.route_beam,
        "prefix1_beam_sizes": [policy.route_beam],
        "route_score_weight": policy.route_score_weight,
        "per_route_topk": policy.per_route_topk,
        "merge_strategy": policy.merge_strategy,
    }


def pick_best_policy(summary_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    candidates = [row for row in summary_rows if row.get("split") == "cold_like_val"]
    if not candidates:
        raise ValueError("No cold_like_val policy summaries were produced.")
    return max(
        candidates,
        key=lambda row: (
            float(row.get("Recall@50", 0.0)),
            float(row.get("CandidatePoolHitRate", 0.0)),
            float(row.get("Recall@20", 0.0)),
            -int(row.get("route_beam", 999)),
            -int(row.get("per_route_topk", 999999)),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a prefix-1 fusion policy on train-derived validation splits.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--item-embedding-path", required=True)
    parser.add_argument("--item-sid-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix1-query-head-checkpoint", default=None)
    parser.add_argument("--domain-query-source-config", default=None)
    parser.add_argument("--default-domain-query-source", default="learned")
    parser.add_argument("--protocol-manifest", default=None)
    parser.add_argument("--cold-like-item-ratio", type=float, default=0.12)
    parser.add_argument("--warm-val-ratio", type=float, default=0.08)
    parser.add_argument("--max-val-examples", type=int, default=1200)
    parser.add_argument("--route-beams", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--route-score-weights", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--per-route-topks", nargs="+", type=int, default=[25, 50, 100])
    parser.add_argument("--merge-strategies", nargs="+", default=["zscore"], choices=["score", "zscore", "round_robin", "quota", "rrf"])
    parser.add_argument("--fusion-methods", nargs="+", default=["round_robin", "rrf"], choices=["round_robin", "rrf"])
    parser.add_argument("--include-single-sources", type=str2bool, default=True)
    parser.add_argument("--candidate-query-sources", nargs="+", default=list(BASE_QUERY_SOURCES))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    item_embeddings = load_item_embeddings(args.data_dir, args.item_embedding_path)
    route_mapping = load_route_mapping(args.item_sid_path)
    model, route_vocab, checkpoint_meta = load_model(args.checkpoint_dir, device)
    domain_query_source_config = load_domain_query_source_config(args.domain_query_source_config)
    domain_query_sources = dict(domain_query_source_config.get("domain_query_sources", {}))
    default_domain_query_source = str(
        domain_query_source_config.get("default_domain_query_source", args.default_domain_query_source)
        if domain_query_source_config
        else args.default_domain_query_source
    )
    prefix1_query_head = None
    prefix1_query_head_meta = None
    candidate_query_sources = list(dict.fromkeys(args.candidate_query_sources))
    if ("prefix1_head" in candidate_query_sources or default_domain_query_source == "prefix1_head" or "prefix1_head" in set(domain_query_sources.values())):
        if not args.prefix1_query_head_checkpoint:
            raise ValueError("--prefix1-query-head-checkpoint is required when prefix1_head is in selector candidates or domain mapping.")
        prefix1_query_head, prefix1_query_head_meta = load_prefix1_query_head(args.prefix1_query_head_checkpoint, device)
    examples = build_training_examples(args.data_dir, item_embeddings, route_mapping, max_history=args.max_history)
    protocol_manifest = load_protocol_manifest(args.protocol_manifest) if args.protocol_manifest else None
    if protocol_manifest is not None:
        splits = {
            "cold_like_val": protocol_split_examples(examples, protocol_manifest, "cold_like_val"),
            "warm_val": protocol_split_examples(examples, protocol_manifest, "warm_val"),
        }
    else:
        raise ValueError("select_validation_fusion_policy.py now requires --protocol-manifest for leakage-safe selection.")
    policies = make_policy_grid(
        route_beams=args.route_beams,
        route_score_weights=args.route_score_weights,
        per_route_topks=args.per_route_topks,
        merge_strategies=args.merge_strategies,
        fusion_methods=args.fusion_methods,
        include_single_sources=args.include_single_sources,
        candidate_query_sources=candidate_query_sources,
    )
    rows, route_rows = evaluate_policies(
        splits=splits,
        policies=policies,
        item_embeddings=item_embeddings,
        route_mapping=route_mapping,
        model=model,
        route_vocab=route_vocab,
        device=device,
        batch_size=args.batch_size,
        prefix1_query_head=prefix1_query_head,
        domain_query_sources=domain_query_sources,
        default_domain_query_source=default_domain_query_source,
    )
    summary = grouped_summary(rows, ["policy_name", "split"])
    summary_by_domain = grouped_summary(rows, ["policy_name", "split", "domain"])
    route_summary = grouped_route_summary(route_rows, ["split", "beam"])
    route_summary_by_domain = grouped_route_summary(route_rows, ["split", "domain", "beam"])

    policy_by_name = {policy.name: policy for policy in policies}
    enriched_summary = []
    for row in summary:
        policy = policy_by_name[str(row["policy_name"])]
        enriched_summary.append({**policy_metadata(policy), **row})
    enriched_by_domain = []
    for row in summary_by_domain:
        policy = policy_by_name[str(row["policy_name"])]
        enriched_by_domain.append({**policy_metadata(policy), **row})

    best = pick_best_policy(enriched_summary)
    best_policy = policy_by_name[str(best["policy_name"])]
    best_meta = policy_metadata(best_policy)
    default_query_policy = {"query_source": best_meta["query_source"], "mode": best_meta["mode"]}
    config = {
        "domain_query_sources": domain_query_sources,
        "default_domain_query_source": default_domain_query_source,
        "default_query_policy": default_query_policy,
        "domain_query_policies": {},
        "fusion_specs": best_meta["fusion_specs"],
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "selector": "select_validation_fusion_policy.py",
            "selection_metric": "cold_like_val Recall@50, tie-broken by CandidatePoolHitRate/Recall@20/smaller beam/topk",
            "leakage_note": "Policy selection uses protocol-locked train-derived validation only and never reads cold test labels.",
            "seed": args.seed,
            "split_sizes": {name: len(values) for name, values in splits.items()},
            "args": vars(args),
            "checkpoint_best": checkpoint_meta.get("train_result", {}).get("best", {}),
            "prefix1_query_head_checkpoint": str(Path(args.prefix1_query_head_checkpoint).resolve()) if args.prefix1_query_head_checkpoint else None,
            "prefix1_query_head_best": (prefix1_query_head_meta or {}).get("train_result", {}).get("best", {}),
            "domain_query_sources": domain_query_sources,
            "default_domain_query_source": default_domain_query_source,
            "protocol_manifest": str(Path(args.protocol_manifest).resolve()) if args.protocol_manifest else None,
            "protocol_config_hash": protocol_manifest.get("config_hash") if protocol_manifest else None,
            **best_meta,
        },
        "policy_metrics": {
            "selected_cold_like_val": dict(best),
            "all_summaries_path": "selector_summary.csv",
            "summary_by_domain_path": "selector_summary_by_domain.csv",
        },
    }
    config["config_hash"] = compute_config_hash(config)
    (output_dir / "fusion_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "selector_summary.json").write_text(json.dumps(enriched_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "selector_summary_by_domain.json").write_text(json.dumps(enriched_by_domain, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "route_summary.json").write_text(json.dumps(route_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "route_summary_by_domain.json").write_text(json.dumps(route_summary_by_domain, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "selector_summary.csv", enriched_summary)
    write_csv(output_dir / "selector_summary_by_domain.csv", enriched_by_domain)
    write_csv(output_dir / "route_summary.csv", route_summary)
    write_csv(output_dir / "route_summary_by_domain.csv", route_summary_by_domain)

    top = sorted(
        [row for row in enriched_summary if row.get("split") == "cold_like_val"],
        key=lambda row: float(row.get("Recall@50", 0.0)),
        reverse=True,
    )[:10]
    report_lines = [
        "# Validation Fusion Policy Selector",
        "",
        f"- Selected policy: `{best_policy.name}`",
        f"- Locked eval policy: `{default_query_policy['query_source']}:{default_query_policy['mode']}`",
        f"- Route beam / route score weight / per-route topk: `{best_policy.route_beam}` / `{best_policy.route_score_weight}` / `{best_policy.per_route_topk}`",
        f"- Query sources: `{'+'.join(best_policy.query_sources)}`",
        f"- Fusion method: `{best_policy.fusion_method or 'single'}`",
        f"- Protocol hash: `{config['config_hash']}`",
        f"- Validation split sizes: `{ {name: len(values) for name, values in splits.items()} }`",
        "",
        "## Selected Cold-Like Validation Metrics",
        "",
        f"- Recall@10/20/50: `{float(best['Recall@10']):.4f}` / `{float(best['Recall@20']):.4f}` / `{float(best['Recall@50']):.4f}`",
        f"- RouteHitRate / CandidatePoolHitRate / ConditionalRecall@50GivenPoolHit: `{float(best['RouteHitRate']):.4f}` / `{float(best['CandidatePoolHitRate']):.4f}` / `{float(best['ConditionalRecall@50GivenPoolHit']):.4f}`",
        f"- RouteMiss / CandidatePoolMiss / RankingMiss: `{float(best['RouteMissRate']):.4f}` / `{float(best['CandidatePoolMissRate']):.4f}` / `{float(best['RankingMissRate']):.4f}`",
        "",
        "## Top Cold-Like Policies",
        "",
    ]
    for row in top:
        report_lines.append(
            f"- `{row['policy_name']}`: R@50 `{float(row['Recall@50']):.4f}`, pool `{float(row['CandidatePoolHitRate']):.4f}`, route `{float(row['RouteHitRate']):.4f}`"
        )
    report_lines.extend(
        [
            "",
            "## Files",
            "",
            "- `fusion_config.json`: locked policy for cold eval.",
            "- `selector_summary.csv`: all policy metrics on warm/cold-like validation.",
            "- `selector_summary_by_domain.csv`: domain breakdown.",
            "- `route_summary.csv`: prefix-1 route hit rates for validation splits.",
            "",
            "## Leakage Note",
            "",
            "This selector only consumes the locked protocol manifest and does not inspect cold-test labels.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(output_dir.resolve()), "selected": config["metadata"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
