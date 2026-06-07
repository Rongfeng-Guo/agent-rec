from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in ordered:
                    ordered.append(str(key))
        fieldnames = ordered
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def copy_json_file(src: str | Path, dst: str | Path) -> None:
    payload = read_json(src)
    Path(dst).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def align_rows_by_order(
    rows: Sequence[Mapping[str, Any]],
    sample_metadata_rows: Sequence[Mapping[str, Any]],
    *,
    subset_field: str = "subset",
    domain_field: str = "domain",
    sample_id_field: str = "sample_id",
) -> list[dict[str, Any]]:
    if len(rows) != len(sample_metadata_rows):
        raise ValueError(
            f"Cannot align rows by order: {len(rows)} result rows vs {len(sample_metadata_rows)} metadata rows."
        )
    aligned: list[dict[str, Any]] = []
    for index, (row, meta) in enumerate(zip(rows, sample_metadata_rows)):
        merged = dict(row)
        if sample_id_field in row and sample_id_field in meta and str(row[sample_id_field]) != str(meta[sample_id_field]):
            raise ValueError(
                f"Row-order alignment mismatch at index {index}: result sample_id={row[sample_id_field]!r} "
                f"metadata sample_id={meta[sample_id_field]!r}."
            )
        if subset_field in meta:
            merged[subset_field] = meta[subset_field]
        if domain_field in meta:
            merged[domain_field] = meta[domain_field]
        aligned.append(merged)
    return aligned


def filter_selected_policy_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if str(row.get("query_source")) == "selected_policy" and str(row.get("mode")) == "validation_selected"
    ]


def filter_eval_smoke_rows(
    rows: Sequence[Mapping[str, Any]], *, mode: str, prefix_len: str | int | None = None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("mode")) != str(mode):
            continue
        if prefix_len is not None and str(row.get("prefix_len")) != str(prefix_len):
            continue
        out.append(dict(row))
    return out


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _group_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row.get("subset", "")), str(row.get("domain", "ALL")))


def _rank_value(match_rank: Any) -> int | None:
    if match_rank in (None, "", "null"):
        return None
    try:
        rank = int(match_rank)
    except (TypeError, ValueError):
        return None
    return rank if rank > 0 else None


def _hit_at_k(match_rank: Any, k: int) -> float:
    rank = _rank_value(match_rank)
    return 1.0 if rank is not None and rank <= k else 0.0


def _ndcg_at_k(match_rank: Any, k: int) -> float:
    rank = _rank_value(match_rank)
    return (1.0 / math.log2(rank + 1)) if rank is not None and rank <= k else 0.0


def _mrr_at_k(match_rank: Any, k: int) -> float:
    rank = _rank_value(match_rank)
    return (1.0 / rank) if rank is not None and rank <= k else 0.0


def _group_subset_domain(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        subset, domain = _group_key(row)
        grouped[(subset, "ALL")].append(row)
        if domain != "ALL":
            grouped[(subset, domain)].append(row)
    return grouped


def summarize_selected_policy_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_subset_domain(rows)

    out: list[dict[str, Any]] = []
    for (subset, domain), group_rows in sorted(grouped.items()):
        sample_count = len(group_rows)
        hits10 = [_hit_at_k(row.get("match_rank"), 10) for row in group_rows]
        hits20 = [_hit_at_k(row.get("match_rank"), 20) for row in group_rows]
        hits50 = [_hit_at_k(row.get("match_rank"), 50) for row in group_rows]
        ndcg10 = [_ndcg_at_k(row.get("match_rank"), 10) for row in group_rows]
        ndcg20 = [_ndcg_at_k(row.get("match_rank"), 20) for row in group_rows]
        ndcg50 = [_ndcg_at_k(row.get("match_rank"), 50) for row in group_rows]
        mrr50 = [_mrr_at_k(row.get("match_rank"), 50) for row in group_rows]
        pool_hits = [
            float(bool(row.get("candidate_pool_hit", False)))
            for row in group_rows
            if "candidate_pool_hit" in row
        ]
        pool_losses = [
            float(bool(row.get("candidate_pool_hit", False)) and not _hit_at_k(row.get("match_rank"), 50))
            for row in group_rows
            if "candidate_pool_hit" in row
        ]
        pool_sizes = [
            float(row.get("candidate_pool_size", row.get("candidate_count", 0)))
            for row in group_rows
            if row.get("candidate_pool_size", row.get("candidate_count")) is not None
        ]
        latencies = [
            float(row.get("latency_ms", 0.0))
            for row in group_rows
            if row.get("latency_ms") is not None
        ]
        mean_pool_hit = float(np.mean(pool_hits)) if pool_hits else None
        mean_recall50 = float(np.mean(hits50)) if hits50 else 0.0
        out.append(
            {
                "subset": subset,
                "domain": domain,
                "sample_count": sample_count,
                "Recall@10": float(np.mean(hits10)) if hits10 else 0.0,
                "Recall@20": float(np.mean(hits20)) if hits20 else 0.0,
                "Recall@50": mean_recall50,
                "NDCG@10": float(np.mean(ndcg10)) if ndcg10 else 0.0,
                "NDCG@20": float(np.mean(ndcg20)) if ndcg20 else 0.0,
                "NDCG@50": float(np.mean(ndcg50)) if ndcg50 else 0.0,
                "MRR": float(np.mean(mrr50)) if mrr50 else 0.0,
                "RouteHitRate@1": (
                    float(np.mean([1.0 if row.get("route_hit") else 0.0 for row in group_rows]))
                    if any("route_hit" in row for row in group_rows)
                    else None
                ),
                "CandidatePoolHitRate@50": mean_pool_hit,
                "ConditionalRecall@50GivenPoolHit": (
                    mean_recall50 / mean_pool_hit if mean_pool_hit is not None and mean_pool_hit > 0.0 else None
                ),
                "CandidatePoolLossRate": float(np.mean(pool_losses)) if pool_losses else None,
                "AverageCandidatePoolSize": float(np.mean(pool_sizes)) if pool_sizes else None,
                "LatencyMsPerSample": float(np.mean(latencies)) if latencies else None,
            }
        )
    return out


def enrich_method(
    rows: Sequence[Mapping[str, Any]],
    *,
    method_key: str,
    display_name: str,
    category: str,
    claim_status: str,
    notes: str,
) -> list[dict[str, Any]]:
    return [
        {
            "method_key": method_key,
            "display_name": display_name,
            "category": category,
            "claim_status": claim_status,
            "notes": notes,
            **row,
        }
        for row in rows
    ]


def build_route_error_breakdown(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_subset_domain(rows)
    out: list[dict[str, Any]] = []
    for (subset, domain), group_rows in sorted(grouped.items()):
        out.append(
            {
                "subset": subset,
                "domain": domain,
                "sample_count": len(group_rows),
                "route_hit_rate": (
                    float(np.mean([1.0 if row.get("route_hit") else 0.0 for row in group_rows])) if group_rows else 0.0
                ),
                "pool_hit_rate": (
                    float(np.mean([1.0 if row.get("candidate_pool_hit") else 0.0 for row in group_rows]))
                    if group_rows
                    else 0.0
                ),
                "ranking_miss_rate": (
                    float(
                        np.mean(
                            [
                                1.0
                                if row.get("candidate_pool_hit") and not _hit_at_k(row.get("match_rank"), 50)
                                else 0.0
                                for row in group_rows
                            ]
                        )
                    )
                    if group_rows
                    else 0.0
                ),
            }
        )
    return out


def build_candidate_pool_breakdown(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_subset_domain(rows)

    out: list[dict[str, Any]] = []
    for (subset, domain), group_rows in sorted(grouped.items()):
        pool_sizes = [int(_as_float(row.get("candidate_pool_size"))) for row in group_rows]
        out.append(
            {
                "subset": subset,
                "domain": domain,
                "sample_count": len(group_rows),
                "avg_candidate_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
                "min_candidate_pool_size": min(pool_sizes) if pool_sizes else 0,
                "max_candidate_pool_size": max(pool_sizes) if pool_sizes else 0,
                "candidate_pool_hit_rate": (
                    float(np.mean([1.0 if row.get("candidate_pool_hit") else 0.0 for row in group_rows]))
                    if group_rows
                    else 0.0
                ),
            }
        )
    return out


def _row_target_item(row: Mapping[str, Any]) -> str:
    target = row.get("target_item_id", row.get("target"))
    if target is None:
        raise ValueError(f"Row is missing target_item_id/target: {row}")
    return str(target)


def build_popularity_baseline_rows(
    selected_rows: Sequence[Mapping[str, Any]],
    popularity_ranking: Sequence[str],
) -> list[dict[str, Any]]:
    pop_index = {str(item_id): rank for rank, item_id in enumerate(popularity_ranking, start=1)}
    out: list[dict[str, Any]] = []
    for row in selected_rows:
        target = _row_target_item(row)
        out.append(
            {
                "sample_id": str(row["sample_id"]),
                "subset": row["subset"],
                "domain": row["domain"],
                "match_rank": pop_index.get(target),
                "candidate_pool_size": len(popularity_ranking),
                "candidate_pool_hit": target in pop_index,
                "latency_ms": 0.0,
            }
        )
    return out


def build_random_matched_size_bucket_rows(
    selected_rows: Sequence[Mapping[str, Any]],
    all_item_ids: Iterable[str],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    catalog = sorted(all_item_ids)
    if not catalog:
        raise ValueError("Random matched-size baseline requires a non-empty item catalog.")

    out: list[dict[str, Any]] = []
    for row in selected_rows:
        target = _row_target_item(row)
        bucket_size = int(row.get("candidate_pool_size", 50) or 50)
        bucket_size = max(1, min(bucket_size, len(catalog)))
        sample_id = str(row["sample_id"])
        seed_int = int(hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed_int)
        bucket = rng.sample(catalog, bucket_size)
        match_rank = rng.randint(1, bucket_size) if target in bucket else None
        out.append(
            {
                "sample_id": sample_id,
                "subset": row["subset"],
                "domain": row["domain"],
                "match_rank": match_rank,
                "candidate_pool_size": bucket_size,
                "candidate_pool_hit": target in bucket,
                "latency_ms": 0.0,
            }
        )
    return out


def build_hit_vector(rows: Sequence[Mapping[str, Any]], *, k: int) -> list[float]:
    return [_hit_at_k(row.get("match_rank"), k) for row in rows]


def bootstrap_paired_delta(
    selected_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    k: int = 50,
    reps: int = 2000,
    seed: int = 42,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    if len(selected_rows) != len(baseline_rows):
        raise ValueError(
            f"Bootstrap requires paired rows with equal lengths, got {len(selected_rows)} and {len(baseline_rows)}."
        )
    selected_hits = build_hit_vector(selected_rows, k=k)
    baseline_hits = build_hit_vector(baseline_rows, k=k)
    sample_count = len(selected_hits)
    if sample_count == 0:
        raise ValueError("Bootstrap requires at least one paired sample.")

    deltas = np.asarray([s - b for s, b in zip(selected_hits, baseline_hits)], dtype=np.float32)
    generator = rng if rng is not None else np.random.default_rng(seed)
    reps_out: list[float] = []
    for _ in range(reps):
        idx = generator.integers(0, len(deltas), size=len(deltas))
        reps_out.append(float(deltas[idx].mean()))
    reps_arr = np.asarray(reps_out, dtype=np.float32)
    ci95_low = float(np.percentile(reps_arr, 2.5))
    ci95_high = float(np.percentile(reps_arr, 97.5))
    return {
        "metric": f"Recall@{k}",
        "paired_delta_mean": float(deltas.mean()),
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "win_rate": float(np.mean(deltas > 0)),
        "sample_count": int(len(deltas)),
        "bootstrap_reps": reps,
        "seed": seed,
        "ci_crosses_zero": bool(ci95_low <= 0.0 <= ci95_high),
    }


def build_bootstrap_comparison_rows(
    selected_rows: Sequence[Mapping[str, Any]],
    comparators: Sequence[tuple[str, str, Sequence[Mapping[str, Any]]]],
    *,
    subset: str = "cold",
    k: int = 50,
    reps: int = 2000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    selected_subset = [row for row in selected_rows if row.get("subset") == subset]
    comparator_subsets = [
        (method_key, display_name, [row for row in baseline_rows if row.get("subset") == subset])
        for method_key, display_name, baseline_rows in comparators
    ]
    pair_count = min([len(selected_subset), *[len(rows) for _, _, rows in comparator_subsets]])
    if pair_count == 0:
        raise ValueError("Bootstrap comparison requires at least one paired sample.")

    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for method_key, display_name, baseline_subset in comparator_subsets:
        row = bootstrap_paired_delta(
            selected_subset[:pair_count],
            baseline_subset[:pair_count],
            k=k,
            reps=reps,
            seed=seed,
            rng=rng,
        )
        row.update(
            {
                "baseline_method_key": method_key,
                "baseline_display_name": display_name,
            }
        )
        out.append(
            {
                "baseline_method_key": row["baseline_method_key"],
                "baseline_display_name": row["baseline_display_name"],
                "metric": row["metric"],
                "paired_delta_mean": row["paired_delta_mean"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "win_rate": row["win_rate"],
                "sample_count": row["sample_count"],
                "bootstrap_reps": row["bootstrap_reps"],
                "seed": row["seed"],
                "ci_crosses_zero": row["ci_crosses_zero"],
            }
        )
    return out


def cold_all_metric_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        if row["subset"] == "cold" and row["domain"] == "ALL":
            out[str(row["method_key"])] = float(row["Recall@50"])
    return out


def fmt_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_official_comparison_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Official Comparison V2",
        "",
        "| method | subset | domain | claim | R@10 | R@20 | R@50 | NDCG@50 | MRR | pool hit | avg pool | latency ms | notes |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['subset']} | {row['domain']} | {row['claim_status']} | "
            f"{fmt_markdown_value(row['Recall@10'])} | {fmt_markdown_value(row['Recall@20'])} | "
            f"{fmt_markdown_value(row['Recall@50'])} | {fmt_markdown_value(row['NDCG@50'])} | "
            f"{fmt_markdown_value(row['MRR'])} | {fmt_markdown_value(row.get('CandidatePoolHitRate@50'))} | "
            f"{fmt_markdown_value(row.get('AverageCandidatePoolSize'))} | "
            f"{fmt_markdown_value(row.get('LatencyMsPerSample'))} | {row['notes']} |"
        )
    lines.append("")
    lines.append(
        "- This table prioritizes rows supported by checked artifacts on the same repo state; "
        "missing historical baselines are intentionally not fabricated."
    )
    return "\n".join(lines) + "\n"


def render_bootstrap_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Bootstrap Comparison",
        "",
        "| baseline | metric | delta mean | ci low | ci high | win rate | n | crosses 0 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['baseline_display_name']} | {row['metric']} | {fmt_markdown_value(row['paired_delta_mean'])} | "
            f"{fmt_markdown_value(row['ci95_low'])} | {fmt_markdown_value(row['ci95_high'])} | "
            f"{fmt_markdown_value(row['win_rate'])} | {row['sample_count']} | {row['ci_crosses_zero']} |"
        )
    return "\n".join(lines) + "\n"


def build_latency_rows(comparison_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "method_key": row["method_key"],
            "display_name": row["display_name"],
            "subset": row["subset"],
            "domain": row["domain"],
            "latency_ms_per_sample": row.get("LatencyMsPerSample"),
            "avg_candidate_pool_size": row.get("AverageCandidatePoolSize"),
            "recall@50": row.get("Recall@50"),
        }
        for row in comparison_rows
        if row["subset"] == "cold" and row["domain"] == "ALL"
    ]


def render_readme(claimable_recall50: float) -> str:
    return (
        "\n".join(
            [
                "# Paper Ready Protocol V2",
                "",
                "## Research Question",
                "",
                "- Can route-conditioned dynamic item memory improve cold-start item retrieval over global no-route metadata retrieval?",
                "",
                "## Problem Framing",
                "",
                "- Original weak formulation: `history -> full SID -> item`.",
                "- Current formulation: `history -> route -> dynamic item memory -> item`.",
                "",
                "## Claimable Method",
                "",
                "- `Predicted Route Validation-Selected (Explicit Script OldV0)` is the frozen method in this bundle.",
                "- Selector uses train-derived validation only; cold test is evaluated once after lock-in.",
                f"- Locked rerun cold ALL `Recall@50 = {claimable_recall50:.4f}`.",
                "",
                "## Diagnostic References",
                "",
                "- Oracle prefix-1 and prefix-2 rows are upper bounds, not claimable baselines.",
                "- Random matched-size bucket isolates bucket-size reduction from semantic route quality.",
                "",
                "## Limitations",
                "",
                "- This bundle only includes rows backed by checked artifacts in the current repo state.",
                "- It does not claim parity with any unavailable historical generative/full-SID baseline artifacts.",
                "- The current cold split has already been used for development history, so the frozen `0.0493` remains a development-test result rather than an untouched external holdout.",
                "",
            ]
        )
        + "\n"
    )
