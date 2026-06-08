#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.memory.data_adapter import load_item_embeddings
from genrec.training import build_training_examples, load_protocol_manifest, load_route_mapping, protocol_split_examples
from scripts.oracle_route_memory.eval_predicted_route import build_memory, load_model, load_prefix1_query_head
from scripts.oracle_route_memory.train_late_bound_fusion_router import FusionConfig, build_gate_examples, choose_device

try:
    from scripts.oracle_route_memory.handoff_io import ensure_empty_output_dir
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from handoff_io import ensure_empty_output_dir

SAMPLE_FEATURE_NAMES = (
    "history_len_norm",
    "route_top1_confidence",
    "route_entropy",
    "bucket_size_log",
    "query_agreement",
)

NEXT_TARGET = (
    "If this is a validation export, train or replay only the locked validation-selected H5-D rankers and "
    "validate the selected policy manifest. If this is a fresh export, score these candidate rows with the "
    "locked h100/h300 model.pkl files, apply the locked domain route, and render a fresh confirmation report "
    "without retraining or changing query sources, beam, or per-route depth."
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def compute_source_local_ranks(source_scores: np.ndarray, source_presence: np.ndarray) -> list[list[int | None]]:
    if source_scores.shape != source_presence.shape:
        raise ValueError(
            f"source_scores and source_presence must have the same shape, got {source_scores.shape} and {source_presence.shape}"
        )
    num_candidates, num_sources = source_scores.shape
    ranks: list[list[int | None]] = [[None for _ in range(num_sources)] for _ in range(num_candidates)]
    for source_idx in range(num_sources):
        present_indices = np.flatnonzero(source_presence[:, source_idx])
        ranked_indices = sorted(
            (int(candidate_idx) for candidate_idx in present_indices),
            key=lambda candidate_idx: float(source_scores[candidate_idx, source_idx]),
            reverse=True,
        )
        for rank, candidate_idx in enumerate(ranked_indices, start=1):
            ranks[candidate_idx][source_idx] = rank
    return ranks


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _std(values: Sequence[float]) -> float | None:
    return float(np.std(np.asarray(values, dtype=np.float32))) if values else None


def candidate_feature_rows_from_gate_row(gate_row: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    candidate_ids = [str(candidate_id) for candidate_id in gate_row["candidate_ids"]]
    query_sources = [str(source) for source in gate_row["query_sources"]]
    source_scores = np.asarray(gate_row["source_scores"], dtype=np.float32)
    source_presence = np.asarray(gate_row.get("source_presence", np.ones_like(source_scores, dtype=np.bool_)), dtype=np.bool_)
    route_scores = np.asarray(gate_row["route_scores"], dtype=np.float32)
    sample_features = np.asarray(gate_row["sample_features"], dtype=np.float32)
    target_index = int(gate_row["target_index"])
    local_ranks = compute_source_local_ranks(source_scores, source_presence)
    sample_feature_values = {
        name: float(sample_features[idx]) if idx < len(sample_features) else None
        for idx, name in enumerate(SAMPLE_FEATURE_NAMES)
    }

    rows: list[dict[str, Any]] = []
    for candidate_idx, candidate_id in enumerate(candidate_ids):
        present_scores: list[float] = []
        present_ranks: list[int] = []
        best_source = None
        best_source_rank = None
        best_source_score = None
        row: dict[str, Any] = {
            "split": split,
            "sample_id": str(gate_row["sample_id"]),
            "domain": str(gate_row["domain"]),
            "target_item_id": str(gate_row["target_item_id"]),
            "candidate_id": candidate_id,
            "candidate_index": candidate_idx,
            "label": int(candidate_idx == target_index),
            "route_hit": bool(gate_row["route_hit"]),
            "candidate_pool_hit": bool(gate_row["candidate_pool_hit"]),
            "candidate_pool_size": int(gate_row.get("candidate_pool_size", len(candidate_ids))),
            "num_route_candidates": int(gate_row.get("num_route_candidates", 0)),
            "member_route_hit_count": int(gate_row.get("member_route_hit_count", 0)),
            "member_candidate_pool_hit_count": int(gate_row.get("member_candidate_pool_hit_count", 0)),
            "route_score": float(route_scores[candidate_idx]),
            **sample_feature_values,
        }
        for source_idx, source_name in enumerate(query_sources):
            source_score = float(source_scores[candidate_idx, source_idx])
            source_is_present = bool(source_presence[candidate_idx, source_idx])
            source_rank = local_ranks[candidate_idx][source_idx]
            row[f"{source_name}_score"] = source_score
            row[f"{source_name}_present"] = source_is_present
            row[f"{source_name}_rank"] = source_rank
            if source_is_present:
                present_scores.append(source_score)
                if source_rank is not None:
                    present_ranks.append(int(source_rank))
                    if best_source_rank is None or source_rank < best_source_rank:
                        best_source = source_name
                        best_source_rank = int(source_rank)
                        best_source_score = source_score

        row["source_presence_count"] = len(present_scores)
        row["source_score_max"] = max(present_scores) if present_scores else None
        row["source_score_mean"] = _mean(present_scores)
        row["source_score_std"] = _std(present_scores)
        row["best_source"] = best_source
        row["best_source_rank"] = best_source_rank
        row["best_source_score"] = best_source_score
        row["mean_source_rank"] = _mean([float(rank) for rank in present_ranks])
        row["min_source_rank"] = min(present_ranks) if present_ranks else None
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
            count += 1
    return count


def summarize_exported_split(split: str, gate_rows: Sequence[Mapping[str, Any]], output_path: Path) -> dict[str, Any]:
    sample_groups: list[dict[str, Any]] = []
    candidate_row_count = 0
    positive_row_count = 0
    pool_hits = []
    route_hits = []
    candidate_pool_sizes = []
    oracle_source_ranks = []
    oracle_source_hits_at_50 = []

    def iter_rows() -> Iterable[Mapping[str, Any]]:
        nonlocal candidate_row_count, positive_row_count
        for gate_row in gate_rows:
            feature_rows = candidate_feature_rows_from_gate_row(gate_row, split)
            positive_rows = [row for row in feature_rows if int(row["label"]) == 1]
            oracle_source_rank = None
            if positive_rows:
                oracle_source_rank = positive_rows[0].get("min_source_rank")
                if oracle_source_rank is not None:
                    oracle_source_ranks.append(float(oracle_source_rank))
            oracle_source_hits_at_50.append(bool(oracle_source_rank is not None and oracle_source_rank <= 50))
            candidate_row_count += len(feature_rows)
            positive_row_count += len(positive_rows)
            pool_hits.append(bool(gate_row["candidate_pool_hit"]))
            route_hits.append(bool(gate_row["route_hit"]))
            candidate_pool_sizes.append(float(gate_row.get("candidate_pool_size", len(feature_rows))))
            sample_groups.append(
                {
                    "split": split,
                    "sample_id": str(gate_row["sample_id"]),
                    "domain": str(gate_row["domain"]),
                    "target_item_id": str(gate_row["target_item_id"]),
                    "candidate_count": len(feature_rows),
                    "target_index": int(gate_row["target_index"]),
                    "route_hit": bool(gate_row["route_hit"]),
                    "candidate_pool_hit": bool(gate_row["candidate_pool_hit"]),
                    "oracle_source_match_rank": oracle_source_rank,
                    "oracle_source_hit_at_50": bool(oracle_source_rank is not None and oracle_source_rank <= 50),
                }
            )
            yield from feature_rows

    write_jsonl(output_path, iter_rows())
    return {
        "split": split,
        "sample_count": len(gate_rows),
        "candidate_row_count": candidate_row_count,
        "positive_row_count": positive_row_count,
        "route_hit_rate": _mean([float(value) for value in route_hits]) or 0.0,
        "candidate_pool_hit_rate": _mean([float(value) for value in pool_hits]) or 0.0,
        "avg_candidate_pool_size": _mean(candidate_pool_sizes),
        "oracle_source_hit_at_50_rate": _mean([float(value) for value in oracle_source_hits_at_50]) or 0.0,
        "avg_oracle_source_match_rank": _mean(oracle_source_ranks),
        "sample_groups": sample_groups,
    }


def render_report(summary: Mapping[str, Any], protocol_hash: str | None) -> str:
    lines = [
        "# H5 Candidate-Level Source Feature Export",
        "",
        f"- Protocol hash: `{protocol_hash or ''}`",
        f"- Query sources: `{', '.join(summary['query_sources'])}`",
        f"- Prefix1 beam / per-route topk: `{summary['fusion_config']['prefix1_beam']}` / `{summary['fusion_config']['per_route_topk']}`",
        "",
        "## Split Summary",
        "",
        "| split | samples | candidate rows | positives | pool hit | avg pool size | oracle src Hit@50 | avg oracle src rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_summary in summary["splits"]:
        avg_rank = split_summary["avg_oracle_source_match_rank"]
        lines.append(
            f"| {split_summary['split']} | {split_summary['sample_count']} | "
            f"{split_summary['candidate_row_count']} | {split_summary['positive_row_count']} | "
            f"{split_summary['candidate_pool_hit_rate']:.4f} | {split_summary['avg_candidate_pool_size']:.2f} | "
            f"{split_summary['oracle_source_hit_at_50_rate']:.4f} | "
            f"{avg_rank:.2f} |" if avg_rank is not None else
            f"| {split_summary['split']} | {split_summary['sample_count']} | "
            f"{split_summary['candidate_row_count']} | {split_summary['positive_row_count']} | "
            f"{split_summary['candidate_pool_hit_rate']:.4f} | {split_summary['avg_candidate_pool_size']:.2f} | "
            f"{split_summary['oracle_source_hit_at_50_rate']:.4f} |  |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- train rows: `{summary['files']['train_candidate_rows']}`",
            f"- cold-like rows: `{summary['files']['cold_like_candidate_rows']}`",
            f"- sample groups: `{summary['files']['sample_groups']}`",
            "",
            "## Next Target",
            "",
            str(summary.get("next_target", NEXT_TARGET)),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export candidate-level source/rank features for H5 reranking.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--item-embedding-path", required=True)
    parser.add_argument("--item-sid-path", required=True)
    parser.add_argument("--router-checkpoint-dir", required=True)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix1-query-head-checkpoint", default=None)
    parser.add_argument("--query-sources", nargs="+", default=["learned", "residual", "mean", "prefix1_head"])
    parser.add_argument("--prefix1-beam", type=int, default=4)
    parser.add_argument("--per-route-topk", type=int, default=500)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    ensure_empty_output_dir(output_dir)
    config = FusionConfig(
        query_sources=tuple(str(value) for value in args.query_sources),
        prefix1_beam=int(args.prefix1_beam),
        per_route_topk=int(args.per_route_topk),
        topk=int(args.topk),
        hidden_dim=32,
        dropout=0.0,
        learning_rate=0.0,
        weight_decay=0.0,
        epochs=0,
        device=device,
        seed=int(args.seed),
    )

    protocol_manifest = load_protocol_manifest(args.protocol_manifest)
    item_embeddings = load_item_embeddings(args.data_dir, args.item_embedding_path)
    route_mapping = load_route_mapping(args.item_sid_path)
    router_model, route_vocab, router_meta = load_model(args.router_checkpoint_dir, device)
    prefix1_query_head = None
    prefix1_query_head_meta = None
    if "prefix1_head" in config.query_sources:
        if not args.prefix1_query_head_checkpoint:
            raise ValueError("--prefix1-query-head-checkpoint is required when prefix1_head is used.")
        prefix1_query_head, prefix1_query_head_meta = load_prefix1_query_head(args.prefix1_query_head_checkpoint, device)

    all_examples = build_training_examples(args.data_dir, item_embeddings, route_mapping, max_history=args.max_history)
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

    train_path = output_dir / "train_candidate_rows.jsonl"
    cold_like_path = output_dir / "cold_like_candidate_rows.jsonl"
    train_summary = summarize_exported_split("train", train_rows, train_path)
    cold_like_summary = summarize_exported_split("cold_like_val", cold_like_rows, cold_like_path)
    sample_groups = train_summary.pop("sample_groups") + cold_like_summary.pop("sample_groups")
    sample_groups_path = output_dir / "sample_groups.json"
    sample_groups_path.write_text(json.dumps(sample_groups, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")

    summary = {
        "name": "H5CandidateLevelSourceFeatureExport",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "protocol_manifest": str(Path(args.protocol_manifest).resolve()),
        "protocol_config_hash": protocol_manifest.get("config_hash"),
        "router_checkpoint_dir": str(Path(args.router_checkpoint_dir).resolve()),
        "prefix1_query_head_checkpoint": str(Path(args.prefix1_query_head_checkpoint).resolve()) if args.prefix1_query_head_checkpoint else None,
        "router_best": router_meta.get("train_result", {}).get("best", {}),
        "prefix1_query_head_best": (prefix1_query_head_meta or {}).get("train_result", {}).get("best", {}),
        "query_sources": list(config.query_sources),
        "fusion_config": asdict(config),
        "files": {
            "train_candidate_rows": str(train_path),
            "cold_like_candidate_rows": str(cold_like_path),
            "sample_groups": str(sample_groups_path),
        },
        "splits": [train_summary, cold_like_summary],
        "next_target": NEXT_TARGET,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(summary, protocol_manifest.get("config_hash")), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir.resolve()),
                "train_candidate_rows": train_summary["candidate_row_count"],
                "cold_like_candidate_rows": cold_like_summary["candidate_row_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
