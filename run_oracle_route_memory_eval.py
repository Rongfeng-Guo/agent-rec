"""Oracle-route memory evaluation.

This script tests the research pivot from:

    History -> full SID -> item

to:

    History -> route -> dynamic item memory -> item

The key upper-bound question is:

If the target route is correct, can metadata memory retrieve the cold target
item better than global no-route retrieval?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from genrec.memory.catalog_memory import CatalogMemory
from genrec.memory.data_adapter import (
    build_eval_samples,
    inspect_available_artifacts,
    load_item_embeddings,
    load_item_metadata,
    load_item_sids,
    load_train_item_set,
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value: {value!r}")


def normalize_route(route: Any) -> Optional[Tuple[str, ...]]:
    if route is None:
        return None
    if isinstance(route, tuple):
        return tuple(str(part) for part in route)
    if isinstance(route, list):
        return tuple(str(part) for part in route)
    if isinstance(route, str):
        text = route.strip()
        if not text:
            return tuple()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        for sep in (",", "|", "/"):
            if sep in text:
                return tuple(part.strip() for part in text.split(sep) if part.strip())
        if " " in text:
            return tuple(part for part in text.split() if part)
        return tuple(text[:1]) if len(text) == 1 else tuple(text)
    return (str(route),)


def route_from_sid(sid_value: Any, prefix_len: int) -> Optional[Tuple[str, ...]]:
    if sid_value is None:
        return None
    if isinstance(sid_value, (list, tuple)):
        parts = [str(part) for part in sid_value]
    elif isinstance(sid_value, str):
        text = sid_value.strip()
        if not text:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        for sep in (",", "|", "/"):
            if sep in text:
                parts = [part.strip() for part in text.split(sep) if part.strip()]
                break
        else:
            if " " in text:
                parts = [part for part in text.split() if part]
            else:
                parts = list(text)
    else:
        parts = [str(sid_value)]
    if len(parts) < prefix_len:
        return tuple(parts)
    return tuple(parts[:prefix_len])


def mean_history_embedding(history: Sequence[str], embeddings: Mapping[str, np.ndarray], history_len: int) -> Optional[np.ndarray]:
    vectors = [embeddings[item_id] for item_id in history[-history_len:] if item_id in embeddings]
    if not vectors:
        return None
    return np.mean(np.stack(vectors, axis=0).astype(np.float32), axis=0)


def dcg_from_rank(rank: Optional[int]) -> float:
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def reciprocal_rank(rank: Optional[int], k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def compute_recall(ranked_ids: Sequence[str], target: str, k: int) -> float:
    return 1.0 if target in ranked_ids[:k] else 0.0


def compute_ndcg(ranked_ids: Sequence[str], target: str, k: int) -> float:
    try:
        rank = ranked_ids[:k].index(target) + 1
    except ValueError:
        rank = None
    return dcg_from_rank(rank)


def load_predicted_routes(path: Optional[str]) -> Dict[str, Tuple[str, ...]]:
    if not path:
        return {}
    route_map: Dict[str, Tuple[str, ...]] = {}
    data_path = Path(path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"predicted_routes_path does not exist: {data_path}")
    if data_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        with data_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        rows = payload if isinstance(payload, list) else [payload]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        route_value = row.get("route") or row.get("predicted_route") or row.get("sid_prefix")
        route = normalize_route(route_value)
        if not route:
            continue
        for key in (row.get("sample_id"), row.get("target"), row.get("item_id"), row.get("ItemID")):
            if key not in (None, ""):
                route_map[str(key)] = route
    return route_map


def build_memory(
    item_embeddings: Mapping[str, np.ndarray],
    item_sids: Mapping[str, Any],
    item_metadata: Mapping[str, Mapping[str, Any]],
    prefix_len: int,
) -> Tuple[CatalogMemory, Dict[str, Any]]:
    memory = CatalogMemory(normalize=True, prefer_faiss=True)
    item_ids = []
    embs = []
    routes = []
    labels = []
    metadata_rows = []
    skipped_missing_sid = 0
    for item_id, emb in item_embeddings.items():
        route = route_from_sid(item_sids.get(item_id), prefix_len) if item_id in item_sids else None
        if route is None:
            skipped_missing_sid += 1
        meta = item_metadata.get(item_id, {})
        item_ids.append(item_id)
        embs.append(np.asarray(emb, dtype=np.float32))
        routes.append(route)
        labels.append(str(meta.get("label") or item_id))
        metadata_rows.append(meta.get("metadata") if isinstance(meta, Mapping) else None)
    memory.add_items(item_ids=item_ids, item_embs=np.stack(embs, axis=0), routes=routes, labels=labels, metadata=metadata_rows)
    route_sizes = [row["size"] for row in memory.route_stats()]
    stats = {
        "num_items": memory.num_items,
        "embedding_dim": memory.embedding_dim,
        "backend": memory.backend,
        "num_routes": len(route_sizes),
        "avg_route_size": float(sum(route_sizes) / len(route_sizes)) if route_sizes else 0.0,
        "median_route_size": float(statistics.median(route_sizes)) if route_sizes else 0.0,
        "items_without_route": skipped_missing_sid,
    }
    return memory, stats


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def evaluate_mode(
    mode: str,
    prefix_len: int,
    samples: Sequence[Mapping[str, Any]],
    embeddings: Mapping[str, np.ndarray],
    item_sids: Mapping[str, Any],
    train_item_set: set[str],
    memory: CatalogMemory,
    topks: Sequence[int],
    history_len: int,
    predicted_routes: Mapping[str, Tuple[str, ...]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    max_k = max(topks)
    route_miss_before = memory.route_miss_count
    skipped_missing_embedding = 0
    skipped_missing_sid = 0
    skipped_missing_predicted_route = 0
    per_sample_rows: List[Dict[str, Any]] = []
    recalls = {k: [] for k in topks}
    ndcgs = {k: [] for k in topks}
    mrrs = {k: [] for k in topks}
    unique_top50 = set()
    cold_hits = 0
    total_top50 = 0
    num_cold_samples = 0
    latencies_ms: List[float] = []
    candidate_counts: List[int] = []
    route_query_count = 0
    route_hit_count = 0
    route_fallback_count = 0

    for sample in samples:
        target = str(sample["target"])
        if sample.get("cold"):
            num_cold_samples += 1
        query = mean_history_embedding(sample["history"], embeddings, history_len)
        if query is None or target not in embeddings:
            skipped_missing_embedding += 1
            continue

        route = None
        if mode == "oracle_route":
            if target not in item_sids:
                skipped_missing_sid += 1
                continue
            route = route_from_sid(item_sids[target], prefix_len)
            if route is None:
                skipped_missing_sid += 1
                continue
        elif mode == "predicted_route":
            route = predicted_routes.get(sample["sample_id"]) or predicted_routes.get(target)
            if route is None:
                skipped_missing_predicted_route += 1
                continue
            if prefix_len > 0 and len(route) > prefix_len:
                route = tuple(route[:prefix_len])

        route_available = False
        fallback_used = False
        candidate_count = memory.num_items
        if mode != "metadata" and route is not None:
            route_query_count += 1
            route_available = memory.has_route(route)
            fallback_used = not route_available
            if route_available:
                route_hit_count += 1
            else:
                route_fallback_count += 1
            candidate_count = memory.candidate_count(route)

        start_time = time.perf_counter()
        results = memory.search(query, route=route if mode != "metadata" else None, topk=max_k)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        latencies_ms.append(latency_ms)
        candidate_counts.append(candidate_count)

        ranked_ids = [row["item_id"] for row in results]
        for k in topks:
            recalls[k].append(compute_recall(ranked_ids, target, k))
            ndcgs[k].append(compute_ndcg(ranked_ids, target, k))
        top50 = ranked_ids[:50]
        unique_top50.update(top50)
        cold_hits += sum(1 for item_id in top50 if item_id not in train_item_set)
        total_top50 += len(top50)

        matched_rank = None
        for idx, item_id in enumerate(ranked_ids, start=1):
            if item_id == target:
                matched_rank = idx
                break
        for k in topks:
            mrrs[k].append(reciprocal_rank(matched_rank, k))

        per_sample_rows.append(
            {
                "mode": mode,
                "prefix_len": prefix_len if mode != "metadata" else None,
                "sample_id": sample["sample_id"],
                "user_id": sample["user_id"],
                "domain": sample["domain"],
                "target": target,
                "route": list(route) if route is not None else None,
                "route_available": route_available if mode != "metadata" else None,
                "fallback_used": fallback_used if mode != "metadata" else None,
                "candidate_count": candidate_count,
                "latency_ms": latency_ms,
                "match_rank": matched_rank,
                "topk_ids": ranked_ids,
            }
        )

    metrics = {
        "mode": mode,
        "prefix_len": prefix_len if mode != "metadata" else None,
        "cold_only": True,
        "SampleCount": len(per_sample_rows),
        "ColdSampleCount": num_cold_samples,
        "Recall@10": float(np.mean(recalls.get(10, [0.0]))) if 10 in recalls else None,
        "Recall@20": float(np.mean(recalls.get(20, [0.0]))) if 20 in recalls else None,
        "Recall@50": float(np.mean(recalls.get(50, [0.0]))) if 50 in recalls else None,
        "NDCG@10": float(np.mean(ndcgs.get(10, [0.0]))) if 10 in ndcgs else None,
        "NDCG@20": float(np.mean(ndcgs.get(20, [0.0]))) if 20 in ndcgs else None,
        "NDCG@50": float(np.mean(ndcgs.get(50, [0.0]))) if 50 in ndcgs else None,
        "HitRate@10": float(np.mean(recalls.get(10, [0.0]))) if 10 in recalls else None,
        "HitRate@20": float(np.mean(recalls.get(20, [0.0]))) if 20 in recalls else None,
        "HitRate@50": float(np.mean(recalls.get(50, [0.0]))) if 50 in recalls else None,
        "MRR@10": float(np.mean(mrrs.get(10, [0.0]))) if 10 in mrrs else None,
        "MRR@20": float(np.mean(mrrs.get(20, [0.0]))) if 20 in mrrs else None,
        "MRR@50": float(np.mean(mrrs.get(50, [0.0]))) if 50 in mrrs else None,
        "Coverage@50": (len(unique_top50) / total_top50) if total_top50 else 0.0,
        "ColdRatio@50": (cold_hits / total_top50) if total_top50 else 0.0,
        "RouteQueryCount": route_query_count,
        "RouteHitCount": route_hit_count,
        "RouteMissCount": memory.route_miss_count - route_miss_before,
        "FallbackCount": route_fallback_count,
        "RouteHitRate": (route_hit_count / route_query_count) if route_query_count else None,
        "RouteMissRate": ((memory.route_miss_count - route_miss_before) / route_query_count) if route_query_count else None,
        "FallbackRate": (route_fallback_count / route_query_count) if route_query_count else None,
        "AverageCandidateCount": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "AverageRetrievalLatencyMs": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "LatencyP50Ms": percentile(latencies_ms, 50),
        "LatencyP95Ms": percentile(latencies_ms, 95),
        "SkippedMissingEmbedding": skipped_missing_embedding,
        "SkippedMissingSID": skipped_missing_sid,
        "SkippedMissingPredictedRoute": skipped_missing_predicted_route,
    }
    return metrics, per_sample_rows


def markdown_summary(results: Sequence[Mapping[str, Any]]) -> str:
    header = "| mode | prefix_len | samples | Recall@50 | NDCG@50 | MRR@50 | RouteHitRate | FallbackRate | AvgLatencyMs |"
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, divider]
    for row in results:
        rows.append(
            "| {mode} | {prefix_len} | {samples} | {r50:.4f} | {n50:.4f} | {m50:.4f} | {hit_rate} | {fallback_rate} | {latency:.3f} |".format(
                mode=row["mode"],
                prefix_len=row["prefix_len"] if row["prefix_len"] is not None else "-",
                samples=row.get("SampleCount", 0),
                r50=row.get("Recall@50") or 0.0,
                n50=row.get("NDCG@50") or 0.0,
                m50=row.get("MRR@50") or 0.0,
                hit_rate=(f"{row['RouteHitRate']:.4f}" if row.get("RouteHitRate") is not None else "-"),
                fallback_rate=(f"{row['FallbackRate']:.4f}" if row.get("FallbackRate") is not None else "-"),
                latency=row.get("AverageRetrievalLatencyMs") or 0.0,
            )
        )
    metadata_r50 = next((row.get("Recall@50", 0.0) for row in results if row["mode"] == "metadata"), 0.0)
    best_oracle_r50 = max((row.get("Recall@50", 0.0) for row in results if row["mode"] == "oracle_route"), default=0.0)
    if best_oracle_r50 >= metadata_r50 + 0.05:
        judgment = "route-conditioned memory binding has positive signal"
    elif best_oracle_r50 >= max(metadata_r50 - 0.02, 0.0):
        judgment = "route condition provides limited gain"
    else:
        judgment = "item/user embedding may be insufficient; memory binding upper bound is weak"
    return "\n".join(["# Oracle Route Memory Eval", "", *rows, "", f"Judgment: {judgment}"]) + "\n"


def ensure_output_dir(path: Optional[str]) -> Path:
    if path:
        output_dir = Path(path).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / "oracle_route_memory" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "prefix_len",
        "cold_only",
        "SampleCount",
        "ColdSampleCount",
        "Recall@10",
        "Recall@20",
        "Recall@50",
        "NDCG@10",
        "NDCG@20",
        "NDCG@50",
        "HitRate@10",
        "HitRate@20",
        "HitRate@50",
        "MRR@10",
        "MRR@20",
        "MRR@50",
        "RouteQueryCount",
        "RouteHitCount",
        "RouteMissCount",
        "FallbackCount",
        "RouteHitRate",
        "RouteMissRate",
        "FallbackRate",
        "AverageCandidateCount",
        "AverageRetrievalLatencyMs",
        "LatencyP50Ms",
        "LatencyP95Ms",
        "SkippedMissingEmbedding",
        "SkippedMissingSID",
        "SkippedMissingPredictedRoute",
        "Coverage@50",
        "ColdRatio@50",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--item_embedding_path")
    parser.add_argument("--item_sid_path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--cold_only", type=str2bool, default=True)
    parser.add_argument("--prefix_len", nargs="+", type=int, default=[1], choices=[1, 2])
    parser.add_argument("--history_len", type=int, default=5)
    parser.add_argument("--topk", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--mode", default="all", choices=["metadata", "oracle_route", "predicted_route", "all"])
    parser.add_argument("--predicted_routes_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--max_eval_samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    artifacts = inspect_available_artifacts(args.data_dir)
    item_embeddings = load_item_embeddings(args.data_dir, args.item_embedding_path)
    item_sids = load_item_sids(args.data_dir, args.item_sid_path)
    item_metadata = load_item_metadata(args.data_dir)
    train_item_set = load_train_item_set(args.data_dir)
    eval_samples = build_eval_samples(args.data_dir, split=args.split, cold_only=args.cold_only)
    if args.max_eval_samples:
        eval_samples = eval_samples[: args.max_eval_samples]
    predicted_routes = load_predicted_routes(args.predicted_routes_path)

    selected_modes = [args.mode] if args.mode != "all" else ["metadata", "oracle_route", "predicted_route"]
    if "predicted_route" in selected_modes and not predicted_routes:
        selected_modes = [mode for mode in selected_modes if mode != "predicted_route"]

    all_results: List[Dict[str, Any]] = []
    all_per_sample_rows: List[Dict[str, Any]] = []
    memory_stats_by_prefix: Dict[str, Any] = {}
    route_stats_rows: List[Dict[str, Any]] = []

    for prefix_len in sorted(set(args.prefix_len)):
        memory, memory_stats = build_memory(item_embeddings, item_sids, item_metadata, prefix_len)
        memory_stats_by_prefix[str(prefix_len)] = memory_stats
        total_items = max(memory.num_items, 1)
        for route_row in memory.route_stats():
            route_stats_rows.append(
                {
                    "prefix_len": prefix_len,
                    "route": "|".join(route_row["route"]),
                    "size": route_row["size"],
                    "share": route_row["size"] / total_items,
                }
            )
        for mode in selected_modes:
            if mode == "metadata" and any(row["mode"] == "metadata" for row in all_results):
                continue
            metrics, per_sample_rows = evaluate_mode(
                mode=mode,
                prefix_len=prefix_len,
                samples=eval_samples,
                embeddings=item_embeddings,
                item_sids=item_sids,
                train_item_set=train_item_set,
                memory=memory,
                topks=args.topk,
                history_len=args.history_len,
                predicted_routes=predicted_routes,
            )
            all_results.append(metrics)
            all_per_sample_rows.extend(per_sample_rows)

    run_metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
        "command": " ".join(sys.argv),
        "artifacts": artifacts,
        "num_item_embeddings": len(item_embeddings),
        "num_item_sids": len(item_sids),
        "num_item_metadata": len(item_metadata),
        "num_eval_samples_requested": len(eval_samples),
        "selected_modes": selected_modes,
        "memory_stats_by_prefix": memory_stats_by_prefix,
    }

    (output_dir / "command.sh").write_text("#!/usr/bin/env bash\n" + " ".join(sys.argv) + "\n", encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(markdown_summary(all_results), encoding="utf-8")
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_per_sample_rows),
        encoding="utf-8",
    )
    write_summary_csv(output_dir / "summary.csv", all_results)

    with (output_dir / "route_coverage.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["prefix_len", "route", "size", "share"])
        writer.writeheader()
        for row in route_stats_rows:
            writer.writerow(row)

    with (output_dir / "fallback_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["mode", "prefix_len", "RouteQueryCount", "RouteHitCount", "RouteMissCount", "FallbackCount", "RouteHitRate", "RouteMissRate", "FallbackRate"],
        )
        writer.writeheader()
        for row in all_results:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    with (output_dir / "latency_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["mode", "prefix_len", "SampleCount", "AverageCandidateCount", "AverageRetrievalLatencyMs", "LatencyP50Ms", "LatencyP95Ms"],
        )
        writer.writeheader()
        for row in all_results:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    (output_dir / "eval_results.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "eval_results.md").write_text(markdown_summary(all_results), encoding="utf-8")
    (output_dir / "per_sample_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_per_sample_rows),
        encoding="utf-8",
    )
    (output_dir / "memory_stats.json").write_text(json.dumps(memory_stats_by_prefix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "route_stats.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["prefix_len", "route", "size", "share"])
        writer.writeheader()
        for row in route_stats_rows:
            writer.writerow(row)

    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "results": all_results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
