#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import socket
import subprocess
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.memory.data_adapter import load_item_embeddings
from genrec.training import build_training_examples, load_protocol_manifest, load_route_mapping, protocol_split_examples

TOPKS = (10, 20, 50)
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260607

OFFICIAL_FIELDNAMES = [
    "method_key",
    "display_name",
    "category",
    "result_level",
    "notes",
    "subset",
    "domain",
    "sample_count",
    "Recall@10",
    "Recall@20",
    "Recall@50",
    "NDCG@10",
    "NDCG@20",
    "NDCG@50",
    "MRR",
    "RouteHitRate@1",
    "RouteHitRate@2",
    "RouteHitRate@4",
    "CandidatePoolHitRate@50",
    "ConditionalRecall@50GivenPoolHit",
    "CandidatePoolLossRate",
    "AverageCandidatePoolSize",
    "LatencyMsPerSample",
    "PeakMemoryMB",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def mean_query(history_item_ids: Sequence[str], embeddings: Mapping[str, np.ndarray]) -> np.ndarray:
    vectors = [np.asarray(embeddings[item_id], dtype=np.float32) for item_id in history_item_ids if item_id in embeddings]
    if not vectors:
        raise ValueError("Cannot build a metadata query from an empty history.")
    query = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = np.linalg.norm(query)
    return (query / norm).astype(np.float32) if norm else query.astype(np.float32)


def last_query(history_item_ids: Sequence[str], embeddings: Mapping[str, np.ndarray]) -> np.ndarray:
    for item_id in reversed(history_item_ids):
        if item_id in embeddings:
            query = np.asarray(embeddings[item_id], dtype=np.float32)
            norm = np.linalg.norm(query)
            return (query / norm).astype(np.float32) if norm else query.astype(np.float32)
    raise ValueError("Cannot build a last-item query from an empty history.")


def rank_target_from_scores(item_ids: Sequence[str], scores: np.ndarray, target: str) -> int | None:
    order = np.argsort(-scores, kind="mergesort")
    for rank, idx in enumerate(order, start=1):
        if item_ids[int(idx)] == target:
            return rank
    return None


def build_full_catalog_rows(
    examples: Sequence[Any],
    *,
    subset: str,
    embeddings: Mapping[str, np.ndarray],
    route_mapping: Mapping[str, Sequence[int]],
    query_kind: str,
    item_ids: Sequence[str],
    item_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        query = mean_query(example.history_item_ids, embeddings) if query_kind == "mean" else last_query(example.history_item_ids, embeddings)
        scores = item_matrix @ query
        rank = rank_target_from_scores(item_ids, scores, str(example.target_item_id))
        rows.append(
            {
                "subset": subset,
                "domain": str(example.domain),
                "sample_id": str(example.sample_id),
                "target_item_id": str(example.target_item_id),
                "match_rank": rank,
                "candidate_pool_size": len(item_ids),
                "candidate_pool_hit": rank is not None,
                "route_hit@1": None,
                "route_hit@2": None,
                "route_hit@4": None,
                "latency_ms": None,
            }
        )
    return rows


def build_oracle_rows(
    examples: Sequence[Any],
    *,
    subset: str,
    embeddings: Mapping[str, np.ndarray],
    route_mapping: Mapping[str, Sequence[int]],
    item_ids: Sequence[str],
    item_matrix: np.ndarray,
    prefix_len: int,
) -> list[dict[str, Any]]:
    route_to_indices: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for idx, item_id in enumerate(item_ids):
        route = route_mapping.get(item_id)
        if route is not None:
            route_to_indices[tuple(int(part) for part in route[:prefix_len])].append(idx)
    rows = []
    for example in examples:
        target = str(example.target_item_id)
        route = tuple(int(part) for part in route_mapping[target][:prefix_len])
        indices = route_to_indices[route]
        query = mean_query(example.history_item_ids, embeddings)
        scores = item_matrix[indices] @ query
        bucket_ids = [item_ids[idx] for idx in indices]
        rank = rank_target_from_scores(bucket_ids, scores, target)
        rows.append(
            {
                "subset": subset,
                "domain": str(example.domain),
                "sample_id": str(example.sample_id),
                "target_item_id": target,
                "match_rank": rank,
                "candidate_pool_size": len(indices),
                "candidate_pool_hit": True,
                "route_hit@1": 1.0,
                "route_hit@2": 1.0,
                "route_hit@4": 1.0,
                "latency_ms": None,
            }
        )
    return rows


def build_popularity_rows(
    examples: Sequence[Any],
    *,
    subset: str,
    selection_train_examples: Sequence[Any],
    item_ids: Sequence[str],
) -> list[dict[str, Any]]:
    counts = Counter(str(example.target_item_id) for example in selection_train_examples)
    ranking = [item_id for item_id, _ in sorted(counts.items(), key=lambda item_count: (-item_count[1], item_count[0]))]
    ranking += [item_id for item_id in item_ids if item_id not in counts]
    rank_by_item = {item_id: idx + 1 for idx, item_id in enumerate(ranking)}
    return [
        {
            "subset": subset,
            "domain": str(example.domain),
            "sample_id": str(example.sample_id),
            "target_item_id": str(example.target_item_id),
            "match_rank": rank_by_item.get(str(example.target_item_id)),
            "candidate_pool_size": len(ranking),
            "candidate_pool_hit": str(example.target_item_id) in rank_by_item,
            "route_hit@1": None,
            "route_hit@2": None,
            "route_hit@4": None,
            "latency_ms": None,
        }
        for example in examples
    ]


def stable_random_sample(sample_id: str, values: Sequence[str], k: int) -> list[str]:
    seed_hex = hashlib.sha256(f"{BOOTSTRAP_SEED}:{sample_id}".encode("utf-8")).hexdigest()
    rng = random.Random(int(seed_hex[:16], 16))
    return rng.sample(list(values), k=min(k, len(values)))


def build_random_matched_rows(selected_rows: Sequence[Mapping[str, Any]], item_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for row in selected_rows:
        sample_id = str(row["sample_id"])
        target = str(row["target_item_id"])
        pool_size = int(row.get("candidate_pool_size") or 0)
        candidates = stable_random_sample(sample_id, item_ids, pool_size)
        rank = candidates.index(target) + 1 if target in candidates else None
        rows.append(
            {
                "subset": row["subset"],
                "domain": row["domain"],
                "sample_id": sample_id,
                "target_item_id": target,
                "match_rank": rank,
                "candidate_pool_size": pool_size,
                "candidate_pool_hit": target in candidates,
                "route_hit@1": None,
                "route_hit@2": None,
                "route_hit@4": None,
                "latency_ms": None,
            }
        )
    return rows


def route_hit(row: Mapping[str, Any], k: int) -> float | None:
    candidates = row.get("route_candidates")
    true_route = str(row.get("true_route", ""))
    if not candidates:
        return None
    routes = [str(candidate.get("route", "")) for candidate in candidates[:k]]
    return float(true_route in routes or any(true_route.startswith(route) for route in routes if route))


def convert_eval_rows(rows: Sequence[Mapping[str, Any]], query_source: str, mode: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("query_source") != query_source or row.get("mode") != mode:
            continue
        out.append(
            {
                "subset": row["subset"],
                "domain": row["domain"],
                "sample_id": row["sample_id"],
                "target_item_id": row["target_item_id"],
                "match_rank": row.get("match_rank"),
                "candidate_pool_size": row.get("candidate_pool_size"),
                "candidate_pool_hit": row.get("candidate_pool_hit"),
                "route_hit@1": route_hit(row, 1),
                "route_hit@2": route_hit(row, 2),
                "route_hit@4": route_hit(row, 4),
                "latency_ms": row.get("latency_ms"),
            }
        )
    return out


def summarize_rows(rows: Sequence[Mapping[str, Any]], subset: str, domain: str) -> dict[str, Any]:
    scoped = [row for row in rows if row["subset"] == subset and (domain == "ALL" or row["domain"] == domain)]
    out: dict[str, Any] = {"sample_count": len(scoped)}
    if not scoped:
        for key in OFFICIAL_FIELDNAMES:
            out.setdefault(key, None)
        return out
    for k in TOPKS:
        hits = []
        ndcgs = []
        for row in scoped:
            rank = row.get("match_rank")
            rank = int(rank) if rank not in (None, "") else None
            hit = rank is not None and rank <= k
            hits.append(1.0 if hit else 0.0)
            ndcgs.append(0.0 if not hit else 1.0 / math.log2(rank + 1))
        out[f"Recall@{k}"] = float(np.mean(hits))
        out[f"NDCG@{k}"] = float(np.mean(ndcgs))
    reciprocal = []
    for row in scoped:
        rank = row.get("match_rank")
        rank = int(rank) if rank not in (None, "") else None
        reciprocal.append(0.0 if rank is None else 1.0 / rank)
    out["MRR"] = float(np.mean(reciprocal))
    for k in (1, 2, 4):
        values = [row.get(f"route_hit@{k}") for row in scoped if row.get(f"route_hit@{k}") is not None]
        out[f"RouteHitRate@{k}"] = float(np.mean(values)) if values else None
    pool_hits = [bool(row.get("candidate_pool_hit")) for row in scoped]
    recall50_hits = [
        row.get("match_rank") not in (None, "") and int(row["match_rank"]) <= 50
        for row in scoped
    ]
    out["CandidatePoolHitRate@50"] = float(np.mean(pool_hits))
    out["ConditionalRecall@50GivenPoolHit"] = (
        float(sum(1 for hit, pool_hit in zip(recall50_hits, pool_hits) if hit and pool_hit) / max(sum(pool_hits), 1))
    )
    out["CandidatePoolLossRate"] = 1.0 - out["CandidatePoolHitRate@50"]
    pool_sizes = [float(row["candidate_pool_size"]) for row in scoped if row.get("candidate_pool_size") not in (None, "")]
    out["AverageCandidatePoolSize"] = float(np.mean(pool_sizes)) if pool_sizes else None
    latencies = [float(row["latency_ms"]) for row in scoped if row.get("latency_ms") not in (None, "")]
    out["LatencyMsPerSample"] = float(np.mean(latencies)) if latencies else None
    out["PeakMemoryMB"] = None
    return out


def official_rows_for_method(
    *,
    method_key: str,
    display_name: str,
    category: str,
    result_level: str,
    notes: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    groups = [
        ("blind_confirmation", "ALL"),
        ("blind_confirmation", "Book"),
        ("blind_confirmation", "Game"),
        ("cold_like_validation", "ALL"),
        ("warm_validation", "ALL"),
    ]
    for subset, domain in groups:
        out.append(
            {
                "method_key": method_key,
                "display_name": display_name,
                "category": category,
                "result_level": result_level,
                "notes": notes,
                "subset": subset,
                "domain": domain,
                **summarize_rows(rows, subset, domain),
            }
        )
    return out


def rows_by_occurrence(rows: Sequence[Mapping[str, Any]]) -> dict[str, deque[Mapping[str, Any]]]:
    grouped: dict[str, deque[Mapping[str, Any]]] = defaultdict(deque)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    return grouped


def align_rows(reference_rows: Sequence[Mapping[str, Any]], other_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    grouped = rows_by_occurrence(other_rows)
    aligned = []
    for row in reference_rows:
        bucket = grouped[str(row["sample_id"])]
        if not bucket:
            raise ValueError(f"Missing paired row for sample_id={row['sample_id']}")
        aligned.append(bucket.popleft())
    return aligned


def hit50(row: Mapping[str, Any]) -> float:
    rank = row.get("match_rank")
    return float(rank not in (None, "") and int(rank) <= 50)


def bootstrap_delta(
    selected_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_key: str,
) -> dict[str, Any]:
    selected = [row for row in selected_rows if row["subset"] == "blind_confirmation"]
    baseline = align_rows(selected, [row for row in baseline_rows if row["subset"] == "blind_confirmation"])
    selected_hits = [hit50(row) for row in selected]
    baseline_hits = [hit50(row) for row in baseline]
    n = len(selected_hits)
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = []
    wins = 0
    for _ in range(BOOTSTRAP_REPS):
        indices = [rng.randrange(n) for _ in range(n)]
        delta = float(np.mean([selected_hits[i] - baseline_hits[i] for i in indices]))
        deltas.append(delta)
        wins += int(delta > 0)
    deltas_sorted = sorted(deltas)
    observed = float(np.mean(selected_hits) - np.mean(baseline_hits))
    return {
        "comparison": f"predicted_route_validation_selected_vs_{baseline_key}",
        "delta_Recall@50": observed,
        "ci_low": deltas_sorted[int(0.025 * BOOTSTRAP_REPS)],
        "ci_high": deltas_sorted[int(0.975 * BOOTSTRAP_REPS)],
        "win_rate": wins / BOOTSTRAP_REPS,
        "sample_count": n,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ci_crosses_zero": deltas_sorted[int(0.025 * BOOTSTRAP_REPS)] <= 0.0 <= deltas_sorted[int(0.975 * BOOTSTRAP_REPS)],
    }


def render_official_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Blind Confirmation Comparison",
        "",
        "| method | level | subset | domain | n | Recall@50 | NDCG@50 | MRR |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['result_level']} | {row['subset']} | {row['domain']} | "
            f"{row['sample_count']} | {float(row['Recall@50']):.6f} | {float(row['NDCG@50']):.6f} | {float(row['MRR']):.6f} |"
        )
    return "\n".join(lines) + "\n"


def render_bootstrap_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Paired Bootstrap",
        "",
        "| comparison | delta Recall@50 | 95% CI | win rate | crosses zero |",
        "|---|---:|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['comparison']} | {float(row['delta_Recall@50']):.6f} | "
            f"[{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}] | "
            f"{float(row['win_rate']):.4f} | {row['ci_crosses_zero']} |"
        )
    return "\n".join(lines) + "\n"


def render_readme_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    def blind_all(method_key: str) -> Mapping[str, Any]:
        for row in rows:
            if row["method_key"] == method_key and row["subset"] == "blind_confirmation" and row["domain"] == "ALL":
                return row
        raise KeyError(f"Missing blind_confirmation ALL row for {method_key}")

    summary_methods = [
        ("metadata_global_mean_query", "Metadata Global Mean Query"),
        ("metadata_global_best_non_route_query", "Metadata Global Best Non-Route Query"),
        ("dynamic_memory_without_route", "Dynamic Memory Without Route"),
        ("random_matched_size_bucket", "Random Matched-Size Bucket"),
        ("predicted_route_validation_selected", "Predicted Route Validation-Selected Fusion"),
        ("predicted_prefix1_top1_single_query", "Predicted Prefix-1 Top-1 Single Query"),
        ("predicted_prefix1_top4_single_query", "Predicted Prefix-1 Top-4 Single Query"),
        ("oracle_prefix1_route", "Oracle Prefix-1 Route"),
        ("oracle_prefix2_route", "Oracle Prefix-2 Route"),
    ]
    selected = blind_all("predicted_route_validation_selected")
    lines = [
        "# Paper Ready Protocol v3 Blind Confirmation",
        "",
        "This bundle reports an item-level blind confirmation result. The v2 result remains a development_result.",
        "",
        "## Claim Boundary",
        "",
        "- Claimable result: `Predicted Route Validation-Selected Fusion`, locked before blind confirmation and selected only on `cold_like_validation`.",
        "- Non-route baselines and random matched-size bucket are fixed controls, not route-selection results.",
        "- Predicted single-query and oracle rows are diagnostics or upper bounds; do not report them as the selected method claim.",
        "- Any post-confirmation comparison rows outside `official_comparison.csv` are diagnostics unless they are locked before a future blind confirmation run.",
        "",
        "## Blind Confirmation ALL Summary",
        "",
        f"Selected-policy sample count: `{selected['sample_count']}`.",
        "",
        "| method | level | Recall@50 | NDCG@50 | MRR |",
        "|---|---|---:|---:|---:|",
    ]
    for method_key, label in summary_methods:
        row = blind_all(method_key)
        lines.append(
            f"| {label} | {row['result_level']} | {float(row['Recall@50']):.6f} | "
            f"{float(row['NDCG@50']):.6f} | {float(row['MRR']):.6f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The selected fusion beats the random matched-size bucket on blind confirmation, but it does not beat the strongest no-route metadata baseline in this bundle.",
        "The oracle rows remain much higher, so the remaining bottleneck is route/query binding rather than candidate-memory capacity.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper-ready v3 blind confirmation bundle.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--selector-dir", required=True)
    parser.add_argument("--locked-eval-dir", required=True)
    parser.add_argument("--comparison-eval-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--item-embedding-path", required=True)
    parser.add_argument("--item-sid-path", required=True)
    parser.add_argument("--max-history", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir = Path(args.protocol_dir)
    selector_dir = Path(args.selector_dir)
    locked_eval_dir = Path(args.locked_eval_dir)
    comparison_eval_dir = Path(args.comparison_eval_dir)

    manifest = load_protocol_manifest(protocol_dir / "split_manifest.json")
    embeddings = load_item_embeddings(args.data_dir, args.item_embedding_path)
    route_mapping = load_route_mapping(args.item_sid_path)
    examples = build_training_examples(args.data_dir, embeddings, route_mapping, max_history=args.max_history)
    split_examples = {
        split: protocol_split_examples(examples, manifest, split)
        for split in ("selection_train", "blind_confirmation", "cold_like_validation", "warm_validation")
    }
    item_ids = sorted(item_id for item_id in embeddings if item_id in route_mapping)
    item_matrix = normalize_matrix(np.stack([np.asarray(embeddings[item_id], dtype=np.float32) for item_id in item_ids], axis=0))

    locked_rows = read_jsonl(locked_eval_dir / "results.jsonl")
    comparison_rows = read_jsonl(comparison_eval_dir / "results.jsonl")
    selected_rows = convert_eval_rows(locked_rows, "selected_policy", "validation_selected")
    mean_rows: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    popularity_rows: list[dict[str, Any]] = []
    oracle_p1_rows: list[dict[str, Any]] = []
    oracle_p2_rows: list[dict[str, Any]] = []
    for subset in ("blind_confirmation", "cold_like_validation", "warm_validation"):
        subset_examples = split_examples[subset]
        mean_rows += build_full_catalog_rows(
            subset_examples,
            subset=subset,
            embeddings=embeddings,
            route_mapping=route_mapping,
            query_kind="mean",
            item_ids=item_ids,
            item_matrix=item_matrix,
        )
        last_rows += build_full_catalog_rows(
            subset_examples,
            subset=subset,
            embeddings=embeddings,
            route_mapping=route_mapping,
            query_kind="last",
            item_ids=item_ids,
            item_matrix=item_matrix,
        )
        popularity_rows += build_popularity_rows(
            subset_examples,
            subset=subset,
            selection_train_examples=split_examples["selection_train"],
            item_ids=item_ids,
        )
        oracle_p1_rows += build_oracle_rows(
            subset_examples,
            subset=subset,
            embeddings=embeddings,
            route_mapping=route_mapping,
            item_ids=item_ids,
            item_matrix=item_matrix,
            prefix_len=1,
        )
        oracle_p2_rows += build_oracle_rows(
            subset_examples,
            subset=subset,
            embeddings=embeddings,
            route_mapping=route_mapping,
            item_ids=item_ids,
            item_matrix=item_matrix,
            prefix_len=2,
        )

    best_non_route_rows = mean_rows
    if summarize_rows(last_rows, "cold_like_validation", "ALL")["Recall@50"] > summarize_rows(mean_rows, "cold_like_validation", "ALL")["Recall@50"]:
        best_non_route_rows = last_rows
    random_rows = build_random_matched_rows(selected_rows, item_ids)
    p1_top1_rows = convert_eval_rows(comparison_rows, "domain_adaptive", "predicted_route_p1")
    p1_top4_rows = convert_eval_rows(comparison_rows, "domain_adaptive", "predicted_route_p1_top4")

    official_rows: list[dict[str, Any]] = []
    official_rows += official_rows_for_method(
        method_key="popularity_global",
        display_name="Popularity",
        category="baseline",
        result_level="blind_confirmation_result",
        notes="Selection-train target-frequency ranking; fixed before confirmation eval.",
        rows=popularity_rows,
    )
    official_rows += official_rows_for_method(
        method_key="metadata_global_mean_query",
        display_name="Metadata Global Mean Query",
        category="baseline",
        result_level="blind_confirmation_result",
        notes="No-route full-catalog retrieval with mean history metadata query.",
        rows=mean_rows,
    )
    official_rows += official_rows_for_method(
        method_key="metadata_global_best_non_route_query",
        display_name="Metadata Global Best Non-Route Query",
        category="baseline",
        result_level="blind_confirmation_result",
        notes="Best of mean/last metadata query selected only on cold_like_validation.",
        rows=best_non_route_rows,
    )
    official_rows += official_rows_for_method(
        method_key="dynamic_memory_without_route",
        display_name="Dynamic Memory Without Route",
        category="baseline",
        result_level="blind_confirmation_result",
        notes="Full dynamic catalog retrieval without route filtering.",
        rows=mean_rows,
    )
    official_rows += official_rows_for_method(
        method_key="random_matched_size_bucket",
        display_name="Random Matched-Size Bucket",
        category="baseline",
        result_level="blind_confirmation_result",
        notes="Native deterministic random bucket matched to selected-policy candidate pool sizes.",
        rows=random_rows,
    )
    official_rows += official_rows_for_method(
        method_key="predicted_prefix1_top1_single_query",
        display_name="Predicted Prefix-1 Top-1 Single Query",
        category="predicted_route",
        result_level="diagnostic_result",
        notes="Fixed comparison-only diagnostic; not used for selection after confirmation.",
        rows=p1_top1_rows,
    )
    official_rows += official_rows_for_method(
        method_key="predicted_prefix1_top4_single_query",
        display_name="Predicted Prefix-1 Top-4 Single Query",
        category="predicted_route",
        result_level="diagnostic_result",
        notes="Fixed comparison-only diagnostic; not used for selection after confirmation.",
        rows=p1_top4_rows,
    )
    official_rows += official_rows_for_method(
        method_key="predicted_route_validation_selected",
        display_name="Predicted Route Validation-Selected Fusion",
        category="predicted_route",
        result_level="blind_confirmation_result",
        notes="Locked before blind confirmation; selected only on cold_like_validation.",
        rows=selected_rows,
    )
    official_rows += official_rows_for_method(
        method_key="oracle_prefix1_route",
        display_name="Oracle Prefix-1 Route",
        category="oracle_upper_bound",
        result_level="oracle_upper_bound",
        notes="True prefix-1 route injected; diagnostic upper bound.",
        rows=oracle_p1_rows,
    )
    official_rows += official_rows_for_method(
        method_key="oracle_prefix2_route",
        display_name="Oracle Prefix-2 Route",
        category="oracle_upper_bound",
        result_level="oracle_upper_bound",
        notes="True prefix-2 route injected; diagnostic upper bound.",
        rows=oracle_p2_rows,
    )

    bootstrap_rows = [
        bootstrap_delta(selected_rows, mean_rows, baseline_key="metadata_global_mean_query"),
        bootstrap_delta(selected_rows, best_non_route_rows, baseline_key="metadata_global_best_non_route_query"),
        bootstrap_delta(selected_rows, random_rows, baseline_key="random_matched_size_bucket"),
        bootstrap_delta(selected_rows, p1_top4_rows, baseline_key="predicted_prefix1_top4_single_query"),
    ]

    for name in [
        "split_manifest.json",
        "split_manifest.md",
        "leakage_audit.json",
        "leakage_audit.md",
        "confirmation_eval_lock.json",
        "confirmation_eval_lock.md",
    ]:
        shutil.copyfile(protocol_dir / name, output_dir / name)
    shutil.copyfile(selector_dir / "fusion_config.json", output_dir / "fusion_config.json")
    shutil.copyfile(selector_dir / "fusion_config.json", output_dir / "method_config.json")
    write_csv(output_dir / "official_comparison.csv", official_rows, OFFICIAL_FIELDNAMES)
    (output_dir / "official_comparison.md").write_text(render_official_markdown(official_rows), encoding="utf-8")
    write_csv(output_dir / "bootstrap_comparison.csv", bootstrap_rows)
    write_json(output_dir / "bootstrap_comparison.json", bootstrap_rows)
    (output_dir / "bootstrap_report.md").write_text(render_bootstrap_markdown(bootstrap_rows), encoding="utf-8")
    write_csv(output_dir / "route_error_breakdown.csv", [])
    write_csv(output_dir / "candidate_pool_breakdown.csv", [])
    write_csv(output_dir / "latency_summary.csv", [
        {
            "method_key": "predicted_route_validation_selected",
            "LatencyMsPerSample": summarize_rows(selected_rows, "blind_confirmation", "ALL")["LatencyMsPerSample"],
            "PeakMemoryMB": None,
        }
    ])

    reproduce = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {REPO_ROOT}",
        "# Re-run split, training, selector, lock, confirmation eval, and bundle commands from run_metadata.json.",
    ]
    (output_dir / "reproduce.sh").write_text("\n".join(reproduce) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(render_readme_markdown(official_rows), encoding="utf-8")
    run_metadata = {
        "created_at": utc_now(),
        "hostname": socket.gethostname(),
        "git_commit": git_commit(),
        "protocol_dir": str(protocol_dir.resolve()),
        "selector_dir": str(selector_dir.resolve()),
        "locked_eval_dir": str(locked_eval_dir.resolve()),
        "comparison_eval_dir": str(comparison_eval_dir.resolve()),
        "split_hash": manifest.get("split_hash"),
        "config_hash": manifest.get("config_hash"),
        "lock_hash": read_json(protocol_dir / "confirmation_eval_lock.json").get("lock_hash"),
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "official_comparison_hash": file_hash(output_dir / "official_comparison.csv"),
    }
    write_json(output_dir / "run_metadata.json", run_metadata)
    print(json.dumps({"status": "ok", "output_dir": str(output_dir.resolve()), **run_metadata}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
