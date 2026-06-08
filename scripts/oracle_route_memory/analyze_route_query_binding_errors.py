#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts.oracle_route_memory.handoff_io import ensure_empty_output_dir, resolve_output_dir, resolve_repo_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from handoff_io import ensure_empty_output_dir, resolve_output_dir, resolve_repo_path

DEFAULT_GROUP_BY = ("split", "policy_name", "domain")
ERROR_CLASSES = (
    "hit_at_k",
    "route_miss_or_unreported",
    "route_hit_pool_miss",
    "pool_hit_rank_miss",
)
NEXT_TARGET = (
    "Use this bottleneck slice to decide whether the next change should improve "
    "route coverage, candidate-pool construction, or candidate-level reranking; "
    "keep validation and fresh-confirmation claims separate."
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [dict(row) for row in payload]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    parsed = as_float(value)
    if parsed is None:
        return None
    return int(parsed)


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def hit_at_k(row: Mapping[str, Any], top_k: int) -> bool:
    rank = as_int(row.get("match_rank"))
    return rank is not None and rank <= top_k


def classify_error(row: Mapping[str, Any], top_k: int) -> str:
    if hit_at_k(row, top_k):
        return "hit_at_k"
    if not as_bool(row.get("route_hit")):
        return "route_miss_or_unreported"
    if not as_bool(row.get("candidate_pool_hit")):
        return "route_hit_pool_miss"
    return "pool_hit_rank_miss"


def group_rows(rows: Sequence[Mapping[str, Any]], group_by: Sequence[str]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in group_by)].append(row)
    return groups


def summarize_rows(rows: Sequence[Mapping[str, Any]], top_k: int) -> dict[str, Any]:
    n = len(rows)
    classes = Counter(classify_error(row, top_k) for row in rows)
    hits = classes["hit_at_k"]
    misses = n - hits
    route_hits = sum(1 for row in rows if as_bool(row.get("route_hit")))
    pool_hits = sum(1 for row in rows if as_bool(row.get("candidate_pool_hit")))
    pool_given_route = pool_hits / route_hits if route_hits else 0.0
    hit_given_pool = hits / pool_hits if pool_hits else 0.0
    miss_classes = {name: count for name, count in classes.items() if name != "hit_at_k"}
    dominant_miss_class = "none"
    dominant_miss_count = 0
    if miss_classes:
        dominant_miss_class, dominant_miss_count = sorted(
            miss_classes.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
    candidate_pool_sizes = [value for row in rows if (value := as_float(row.get("candidate_pool_size"))) is not None]
    route_candidate_counts = [value for row in rows if (value := as_float(row.get("num_route_candidates"))) is not None]
    member_route_hits = [value for row in rows if (value := as_float(row.get("member_route_hit_count"))) is not None]
    member_pool_hits = [value for row in rows if (value := as_float(row.get("member_candidate_pool_hit_count"))) is not None]
    candidate_pool_match_ranks = [
        value for row in rows if (value := as_float(row.get("candidate_pool_match_rank"))) is not None
    ]
    oracle_source_hits = [as_bool(row.get("oracle_source_hit_at_topk")) for row in rows]
    oracle_source_match_ranks = [
        value for row in rows if (value := as_float(row.get("oracle_source_match_rank"))) is not None
    ]
    pool_hit_rank_miss_ranks = [
        value
        for row in rows
        if classify_error(row, top_k) == "pool_hit_rank_miss"
        and (value := as_float(row.get("candidate_pool_match_rank"))) is not None
    ]

    return {
        "sample_count": n,
        f"Hit@{top_k}Count": hits,
        f"Hit@{top_k}Rate": hits / n if n else 0.0,
        "MissCount": misses,
        "MissRate": misses / n if n else 0.0,
        "RouteHitCount": route_hits,
        "RouteHitRate": route_hits / n if n else 0.0,
        "CandidatePoolHitCount": pool_hits,
        "CandidatePoolHitRate": pool_hits / n if n else 0.0,
        "RouteMissOrUnreportedCount": classes["route_miss_or_unreported"],
        "RouteMissOrUnreportedRate": classes["route_miss_or_unreported"] / n if n else 0.0,
        "RouteHitPoolMissCount": classes["route_hit_pool_miss"],
        "RouteHitPoolMissRate": classes["route_hit_pool_miss"] / n if n else 0.0,
        "PoolHitRankMissCount": classes["pool_hit_rank_miss"],
        "PoolHitRankMissRate": classes["pool_hit_rank_miss"] / n if n else 0.0,
        "PoolHitGivenRouteHit": pool_given_route,
        f"Hit@{top_k}GivenPoolHit": hit_given_pool,
        "AvgCandidatePoolSize": mean(candidate_pool_sizes),
        "AvgNumRouteCandidates": mean(route_candidate_counts),
        "AvgMemberRouteHitCount": mean(member_route_hits),
        "AvgMemberCandidatePoolHitCount": mean(member_pool_hits),
        "AvgCandidatePoolMatchRank": mean(candidate_pool_match_ranks),
        "AvgPoolHitRankMissMatchRank": mean(pool_hit_rank_miss_ranks),
        f"OracleSourceHit@{top_k}Count": sum(1 for value in oracle_source_hits if value),
        f"OracleSourceHit@{top_k}Rate": mean(float(value) for value in oracle_source_hits),
        "AvgOracleSourceMatchRank": mean(oracle_source_match_ranks),
        "DominantMissClass": dominant_miss_class,
        "DominantMissCount": dominant_miss_count,
    }


def build_summary(rows: Sequence[Mapping[str, Any]], group_by: Sequence[str], top_k: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key_values, subset in sorted(group_rows(rows, group_by).items()):
        summary = summarize_rows(subset, top_k)
        for key, value in zip(group_by, key_values):
            summary[key] = value
        out.append(summary)
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def select_bottlenecks(rows: Sequence[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    candidates = [row for row in rows if int(row.get("sample_count", 0)) > 0 and float(row.get("MissRate", 0.0)) > 0.0]
    return sorted(
        candidates,
        key=lambda row: (
            -float(row.get("MissRate", 0.0)),
            -float(row.get("RouteHitRate", 0.0)),
            -int(row.get("sample_count", 0)),
            str(row.get("policy_name", "")),
            str(row.get("domain", "")),
        ),
    )[:limit]


def render_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_path: Path,
    group_by: Sequence[str],
    top_k: int,
    limit: int,
) -> str:
    key_header = " | ".join(group_by)
    key_sep = "|".join("---" for _ in group_by)
    lines = [
        "# Route/Query Binding Error Analysis",
        "",
        f"- source_rows: `{source_path}`",
        f"- group_by: `{', '.join(group_by)}`",
        f"- top_k: `{top_k}`",
        "",
        "## Top Bottlenecks",
        "",
        f"| {key_header} | n | Hit@{top_k} | RouteHit | PoolHit | route miss | route-hit pool miss | pool-hit rank miss | avg pool rank | avg rank-miss rank | oracle src Hit@{top_k} | avg oracle src rank | dominant miss |",
        f"| {key_sep} |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in select_bottlenecks(rows, limit):
        keys = " | ".join(fmt(row.get(key)) for key in group_by)
        lines.append(
            f"| {keys} | {fmt(row.get('sample_count'))} | {fmt(row.get(f'Hit@{top_k}Rate'))} | "
            f"{fmt(row.get('RouteHitRate'))} | {fmt(row.get('CandidatePoolHitRate'))} | "
            f"{fmt(row.get('RouteMissOrUnreportedRate'))} | {fmt(row.get('RouteHitPoolMissRate'))} | "
            f"{fmt(row.get('PoolHitRankMissRate'))} | {fmt(row.get('AvgCandidatePoolMatchRank'))} | "
            f"{fmt(row.get('AvgPoolHitRankMissMatchRank'))} | {fmt(row.get(f'OracleSourceHit@{top_k}Rate'))} | "
            f"{fmt(row.get('AvgOracleSourceMatchRank'))} | {fmt(row.get('DominantMissClass'))} |"
        )
    lines.extend([
        "",
        "## Interpretation Guide",
        "",
        "- `route_miss_or_unreported`: the row did not report a true prefix-1 route hit before top-k ranking.",
        "- `route_hit_pool_miss`: the true prefix-1 route was covered, but the target item never entered the candidate pool.",
        "- `pool_hit_rank_miss`: the target entered the candidate pool, but ranked outside the evaluated top-k cutoff.",
        "- High `RouteHit` with low `PoolHit` points to query/candidate retrieval, not route classification alone.",
        "",
        "## Next Target",
        "",
        NEXT_TARGET,
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice selector rows into route miss, candidate-pool miss, and ranking miss buckets.")
    parser.add_argument("--selector-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--group-by", nargs="+", default=list(DEFAULT_GROUP_BY))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    selector_rows = resolve_repo_path(args.selector_rows, args.repo_root)
    if selector_rows is None:
        raise ValueError("--selector-rows is required.")
    output_dir = resolve_output_dir(args.output_dir, args.repo_root)
    rows = read_rows(selector_rows)
    summary = build_summary(rows, args.group_by, args.top_k)
    ensure_empty_output_dir(output_dir)
    (output_dir / "error_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "error_summary.csv", summary)
    (output_dir / "error_summary.md").write_text(
        render_markdown(summary, source_path=selector_rows, group_by=args.group_by, top_k=args.top_k, limit=args.limit),
        encoding="utf-8",
    )
    manifest = {
        "row_count": len(rows),
        "summary_count": len(summary),
        "source_rows": str(selector_rows),
        "group_by": list(args.group_by),
        "top_k": args.top_k,
        "output_dir": str(output_dir),
        "artifacts": {
            "json": "error_summary.json",
            "csv": "error_summary.csv",
            "markdown": "error_summary.md",
        },
        "next_target": NEXT_TARGET,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
