#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import socket
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from genrec.models import Prefix1QueryHead
from genrec.memory.data_adapter import load_item_embeddings
from genrec.training import RouterDataset, build_training_examples, load_route_mapping
from scripts.oracle_route_memory.eval_predicted_route import (
    RouteCandidate,
    build_memory,
    enumerate_prefix1_candidates,
    load_model,
    load_prefix1_query_head,
    mean_history_embedding_from_ids,
    rerank_with_routes,
)

BASE_QUERY_SOURCES = {"learned", "pooled", "residual", "mean"}
ALL_QUERY_SOURCES = BASE_QUERY_SOURCES | {"prefix1_head"}


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "t"}


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def split_validation_dataset(dataset: RouterDataset, val_ratio: float, seed: int):
    val_size = max(1, int(len(dataset) * float(val_ratio)))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(int(seed))
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    return val_dataset, train_size, val_size


def init_metric_row() -> Dict[str, float]:
    return {
        "sample_count": 0.0,
        "hits@50": 0.0,
        "mrr@50": 0.0,
        "candidate_pool_hits": 0.0,
        "candidate_pool_losses": 0.0,
    }


def update_metric(row: Dict[str, float], match_rank: int | None, candidate_pool_hit: bool) -> None:
    row["sample_count"] += 1.0
    hit = match_rank is not None and match_rank <= 50
    row["hits@50"] += 1.0 if hit else 0.0
    row["mrr@50"] += 0.0 if match_rank is None or match_rank > 50 else 1.0 / float(match_rank)
    row["candidate_pool_hits"] += 1.0 if candidate_pool_hit else 0.0
    row["candidate_pool_losses"] += 1.0 if candidate_pool_hit and not hit else 0.0


def finalize_metric(row: Mapping[str, float]) -> Dict[str, float]:
    count = float(row.get("sample_count", 0.0))
    if count <= 0:
        return {
            "sample_count": 0,
            "Recall@50": 0.0,
            "MRR@50": 0.0,
            "CandidatePoolHitRate": 0.0,
            "CandidatePoolLossRate": 0.0,
        }
    return {
        "sample_count": int(count),
        "Recall@50": float(row["hits@50"] / count),
        "MRR@50": float(row["mrr@50"] / count),
        "CandidatePoolHitRate": float(row["candidate_pool_hits"] / count),
        "CandidatePoolLossRate": float(row["candidate_pool_losses"] / count),
    }


def select_best_source(metrics: Mapping[str, Mapping[str, Dict[str, float]]], domain: str, query_sources: Sequence[str]) -> str:
    priority = {query_source: idx for idx, query_source in enumerate(query_sources)}
    candidates = []
    for query_source in query_sources:
        row = metrics.get(domain, {}).get(query_source, {})
        candidates.append(
            (
                float(row.get("Recall@50", 0.0)),
                float(row.get("MRR@50", 0.0)),
                -priority[query_source],
                query_source,
            )
        )
    return max(candidates)[3]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "domain",
        "query_source",
        "selected",
        "sample_count",
        "Recall@50",
        "MRR@50",
        "CandidatePoolHitRate",
        "CandidatePoolLossRate",
    ]
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-sources", nargs="+", default=["learned", "pooled", "residual", "mean"])
    parser.add_argument("--prefix1-beam-size", type=int, default=4)
    parser.add_argument("--merge-strategy", default="zscore", choices=["score", "zscore", "round_robin", "quota", "rrf"])
    parser.add_argument("--route-score-weight", type=float, default=0.0)
    parser.add_argument("--per-route-topk", type=int, default=None)
    parser.add_argument("--include-prefix1-head", type=str2bool, default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    query_sources = list(dict.fromkeys(args.query_sources))
    if args.include_prefix1_head and "prefix1_head" not in query_sources:
        query_sources.append("prefix1_head")
    unknown_sources = [query_source for query_source in query_sources if query_source not in ALL_QUERY_SOURCES]
    if unknown_sources:
        raise ValueError(f"Unknown query sources: {unknown_sources}")
    if "prefix1_head" in query_sources and not args.prefix1_query_head_checkpoint:
        raise ValueError("--prefix1-query-head-checkpoint is required when selecting prefix1_head.")

    item_embeddings = load_item_embeddings(args.data_dir, args.item_embedding_path)
    route_mapping = load_route_mapping(args.item_sid_path)
    model, route_vocab, checkpoint_meta = load_model(args.checkpoint_dir, device)
    examples = build_training_examples(args.data_dir, item_embeddings, route_mapping, max_history=args.max_history)
    dataset = RouterDataset(examples, item_embeddings, route_vocab)
    val_dataset, train_size, val_size = split_validation_dataset(dataset, args.val_ratio, args.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    memory = build_memory(item_embeddings, route_mapping, prefix_len=1)

    prefix1_query_head: Prefix1QueryHead | None = None
    prefix1_query_head_meta: dict[str, Any] | None = None
    if "prefix1_head" in query_sources:
        prefix1_query_head, prefix1_query_head_meta = load_prefix1_query_head(args.prefix1_query_head_checkpoint, device)

    raw_metrics: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(init_metric_row))
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch["history_embs"].to(device), batch["history_mask"].to(device))
            route1_log_probs, _, _ = model.route_log_probs(outputs)
            prefix1_candidates = enumerate_prefix1_candidates(route1_log_probs.cpu(), route_vocab, args.prefix1_beam_size)
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
                domain = str(batch["domain"][idx])
                route_candidates: Sequence[RouteCandidate] = prefix1_candidates[idx]
                for query_source in query_sources:
                    ranked_ids, _, _, rerank_diagnostics = rerank_with_routes(
                        query_embedding=query_embeddings_by_source[query_source][idx],
                        route_candidates=route_candidates,
                        prefix_len=1,
                        memory=memory,
                        topks=[50],
                        route_score_weight=args.route_score_weight,
                        merge_strategy=args.merge_strategy,
                        per_route_topk=args.per_route_topk,
                        target_item_id=str(target_item_id),
                    )
                    match_rank = None
                    for rank, item_id in enumerate(ranked_ids, start=1):
                        if item_id == target_item_id:
                            match_rank = rank
                            break
                    for key in ("ALL", domain):
                        update_metric(raw_metrics[key][query_source], match_rank, bool(rerank_diagnostics.get("candidate_pool_hit", False)))

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for domain, rows in raw_metrics.items():
        metrics[domain] = {query_source: finalize_metric(rows[query_source]) for query_source in query_sources}

    domain_query_sources = {
        domain: select_best_source(metrics, domain, query_sources)
        for domain in sorted(metrics)
        if domain != "ALL"
    }
    default_domain_query_source = select_best_source(metrics, "ALL", query_sources)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for domain in sorted(metrics):
        selected = default_domain_query_source if domain == "ALL" else domain_query_sources[domain]
        for query_source in query_sources:
            row = dict(metrics[domain][query_source])
            row.update({"domain": domain, "query_source": query_source, "selected": int(query_source == selected)})
            csv_rows.append(row)
    write_csv(output_dir / "selector_summary.csv", csv_rows)

    payload = {
        "selector_type": "validation_domain_query_source",
        "domain_query_sources": domain_query_sources,
        "default_domain_query_source": default_domain_query_source,
        "metrics": metrics,
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "hostname": socket.gethostname(),
            "data_dir": str(Path(args.data_dir).resolve()),
            "item_embedding_path": str(Path(args.item_embedding_path).resolve()),
            "item_sid_path": str(Path(args.item_sid_path).resolve()),
            "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
            "prefix1_query_head_checkpoint": str(Path(args.prefix1_query_head_checkpoint).resolve()) if args.prefix1_query_head_checkpoint else None,
            "checkpoint_best": checkpoint_meta.get("train_result", {}).get("best", {}),
            "prefix1_query_head_best": (prefix1_query_head_meta or {}).get("train_result", {}).get("best", {}),
            "num_examples": len(dataset),
            "num_train_examples": train_size,
            "num_val_examples": val_size,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "query_sources": query_sources,
            "prefix1_beam_size": args.prefix1_beam_size,
            "merge_strategy": args.merge_strategy,
            "route_score_weight": args.route_score_weight,
            "per_route_topk": args.per_route_topk,
            "max_history": args.max_history,
            "device": device,
        },
    }
    (output_dir / "domain_query_source_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Validation Query Source Selector",
                "",
                f"- Default query source: `{default_domain_query_source}`",
                f"- Domain query sources: `{domain_query_sources}`",
                f"- Validation examples: `{val_size}`",
                f"- Prefix-1 beam: `{args.prefix1_beam_size}`",
                f"- Merge strategy: `{args.merge_strategy}`",
                "",
                "See `selector_summary.csv` for per-domain metrics.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir.resolve()), "domain_query_sources": domain_query_sources, "default_domain_query_source": default_domain_query_source}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
