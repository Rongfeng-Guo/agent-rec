#!/usr/bin/env python3
"""Build deterministic PROXY_HIERARCHICAL_ROUTE codes from metadata embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stable_reindex(assignments: np.ndarray) -> np.ndarray:
    mapping = {}
    next_id = 0
    normalized = np.empty_like(assignments)
    for idx, value in enumerate(assignments.tolist()):
        if value not in mapping:
            mapping[value] = next_id
            next_id += 1
        normalized[idx] = mapping[value]
    return normalized
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from sklearn.cluster import MiniBatchKMeans


def run_kmeans(matrix: np.ndarray, k: int, seed: int, backend: str):
    if backend in {"auto", "faiss"} and faiss is not None:
        km = faiss.Kmeans(d=matrix.shape[1], k=k, niter=25, nredo=1, seed=seed, gpu=False)
        km.train(matrix.astype(np.float32))
        _, assign = km.index.search(matrix.astype(np.float32), 1)
        return _stable_reindex(assign.reshape(-1)), km.centroids.astype(np.float32), "faiss"
    if backend in {"auto", "sklearn"}:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=min(2048, max(256, len(matrix))), n_init=10)
        assign = km.fit_predict(matrix)
        return _stable_reindex(assign.astype(int)), km.cluster_centers_.astype(np.float32), "sklearn"
    raise RuntimeError("No supported clustering backend available")


def entropy_from_counts(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return float(-sum(p * np.log2(p) for p in probs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-embedding-path", required=True)
    parser.add_argument("--item-id-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--branching-factor", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="auto", choices=["auto", "faiss", "sklearn"])
    args = parser.parse_args()

    if args.depth != 2:
        raise ValueError("This implementation currently supports depth=2 only.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    item_ids = json.loads(Path(args.item_id_path).read_text(encoding="utf-8"))
    embeddings = np.load(args.item_embedding_path).astype(np.float32)
    if len(item_ids) != len(embeddings):
        raise ValueError("item_ids and embeddings length mismatch")

    level1_assign, level1_centroids, backend_used = run_kmeans(embeddings, args.branching_factor, args.seed, args.backend)
    level2_assign = np.full(len(item_ids), -1, dtype=int)
    level2_centroids: Dict[int, np.ndarray] = {}
    cluster_sizes_lvl1 = Counter(level1_assign.tolist())

    for cluster_id in sorted(cluster_sizes_lvl1):
        member_idx = np.where(level1_assign == cluster_id)[0]
        local_k = min(args.branching_factor, len(member_idx))
        if local_k <= 1:
            level2_assign[member_idx] = 0
            level2_centroids[cluster_id] = embeddings[member_idx[:1]].copy()
            continue
        local_assign, local_centroids, _ = run_kmeans(embeddings[member_idx], local_k, args.seed + int(cluster_id) + 1, backend_used)
        level2_assign[member_idx] = local_assign
        level2_centroids[cluster_id] = local_centroids

    mapping_rows = []
    prefix1 = Counter()
    prefix2 = Counter()
    for idx, item_id in enumerate(item_ids):
        sid = [int(level1_assign[idx]), int(level2_assign[idx])]
        prefix1[(sid[0],)] += 1
        prefix2[tuple(sid)] += 1
        mapping_rows.append({"item_id": item_id, "sid": sid, "route_type": "PROXY_HIERARCHICAL_ROUTE"})

    (output_dir / "item_sid_mapping.json").write_text(json.dumps({row["item_id"]: row["sid"] for row in mapping_rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "item_sid_mapping.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mapping_rows), encoding="utf-8")
    np.savez(output_dir / "route_centroids.npz", level1=level1_centroids, **{f"level2_{k}": v for k, v in level2_centroids.items()})

    cluster_sizes = list(prefix2.values())
    audit = {
        "item_count": len(item_ids),
        "route_depth": args.depth,
        "prefix1_route_count": len(prefix1),
        "prefix2_route_count": len(prefix2),
        "cluster_size_min": int(min(cluster_sizes)) if cluster_sizes else 0,
        "cluster_size_max": int(max(cluster_sizes)) if cluster_sizes else 0,
        "cluster_size_mean": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        "cluster_size_p50": float(np.percentile(cluster_sizes, 50)) if cluster_sizes else 0.0,
        "cluster_size_p95": float(np.percentile(cluster_sizes, 95)) if cluster_sizes else 0.0,
        "empty_cluster_count": max(args.branching_factor - len(prefix1), 0),
        "singleton_cluster_count": int(sum(1 for c in cluster_sizes if c == 1)),
        "route_entropy": entropy_from_counts(cluster_sizes),
        "coverage": 1.0,
        "seed": args.seed,
        "backend": backend_used,
        "route_type": "PROXY_HIERARCHICAL_ROUTE",
    }
    manifest = {
        "item_embedding_path": str(Path(args.item_embedding_path).resolve()),
        "item_id_path": str(Path(args.item_id_path).resolve()),
        "branching_factor": args.branching_factor,
        "depth": args.depth,
        "seed": args.seed,
        "backend_requested": args.backend,
        "backend_used": backend_used,
        "route_type": "PROXY_HIERARCHICAL_ROUTE",
    }

    with (output_dir / "route_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix_len", "route", "count"])
        writer.writeheader()
        for route, count in sorted(prefix1.items()):
            writer.writerow({"prefix_len": 1, "route": "|".join(map(str, route)), "count": count})
        for route, count in sorted(prefix2.items()):
            writer.writerow({"prefix_len": 2, "route": "|".join(map(str, route)), "count": count})

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "audit.md").write_text(
        "\n".join([
            "# Proxy Route Audit",
            "",
            f"- Route type: `{audit['route_type']}`",
            f"- Item count: `{audit['item_count']}`",
            f"- Route depth: `{audit['route_depth']}`",
            f"- Prefix-1 routes: `{audit['prefix1_route_count']}`",
            f"- Prefix-2 routes: `{audit['prefix2_route_count']}`",
            f"- Cluster size min/max/mean: `{audit['cluster_size_min']}` / `{audit['cluster_size_max']}` / `{audit['cluster_size_mean']:.3f}`",
            f"- Singleton clusters: `{audit['singleton_cluster_count']}`",
            f"- Route entropy: `{audit['route_entropy']:.6f}`",
            f"- Backend: `{audit['backend']}`",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": manifest, "audit": audit}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
