#!/usr/bin/env python3
"""Audit reusable or generated memory assets for oracle-route experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from typing import Any, Dict, Mapping

import numpy as np

from genrec.memory.data_adapter import load_item_embeddings, load_item_metadata, load_item_sids
from genrec.memory.catalog_memory import CatalogMemory


def load_embeddings_any(path: str) -> Dict[str, np.ndarray]:
    p = Path(path)
    if p.suffix == ".sent_emb":
        sidecar = p.parent / "id_mapping.json"
        if not sidecar.exists():
            raise FileNotFoundError(f"Missing id_mapping.json beside {p}")
        mapping = json.loads(sidecar.read_text(encoding="utf-8"))
        id2item = mapping.get("id2item")
        if not isinstance(id2item, list):
            raise ValueError(f"id_mapping.json beside {p} does not contain id2item list")
        item_ids = [str(x) for x in id2item[1:]]
        size = os.path.getsize(p)
        dim_guess = None
        for candidate in (384, 768):
            if len(item_ids) * candidate * 4 == size:
                dim_guess = candidate
                break
        if dim_guess is None:
            raise ValueError(f"Could not infer embedding dim for {p}; file size={size}, item_count={len(item_ids)}")
        matrix = np.fromfile(p, dtype=np.float32).reshape(len(item_ids), dim_guess)
        return {item_id: matrix[idx] for idx, item_id in enumerate(item_ids)}
    return load_item_embeddings(p.parent, p)


def load_sids_any(path: str) -> Dict[str, Any]:
    p = Path(path)
    if p.suffix == ".sem_ids":
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): v for k, v in data.items()}
    return load_item_sids(p.parent, p)


def normalize_route(sid: Any):
    if sid is None:
        return None
    if isinstance(sid, list):
        return tuple(str(x) for x in sid)
    if isinstance(sid, tuple):
        return tuple(str(x) for x in sid)
    if isinstance(sid, str):
        if sid.startswith("[") and sid.endswith("]"):
            sid = sid[1:-1]
        if "," in sid:
            return tuple(x.strip() for x in sid.split(",") if x.strip())
        if "|" in sid:
            return tuple(x.strip() for x in sid.split("|") if x.strip())
        return tuple(list(sid))
    return (str(sid),)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--item-embedding-path", required=True)
    parser.add_argument("--item-sid-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_item_metadata(args.data_dir)
    embeddings = load_embeddings_any(args.item_embedding_path)
    sids = load_sids_any(args.item_sid_path)

    meta_ids = set(metadata.keys())
    emb_ids = set(embeddings.keys())
    sid_ids = set(sids.keys())
    triple = meta_ids & emb_ids & sid_ids

    dims = sorted({int(np.asarray(v).shape[-1]) for v in embeddings.values()})
    nan_count = 0
    inf_count = 0
    for vec in embeddings.values():
        arr = np.asarray(vec, dtype=np.float32)
        nan_count += int(np.isnan(arr).any())
        inf_count += int(np.isinf(arr).any())

    routes = {item_id: normalize_route(sid) for item_id, sid in sids.items()}
    route_depths = Counter(len(route) if route is not None else 0 for route in routes.values())
    prefix1 = Counter(route[:1] for route in routes.values() if route)
    prefix2 = Counter(route[:2] for route in routes.values() if route and len(route) >= 2)
    empty_route_count = sum(1 for route in routes.values() if not route)
    singleton_route_count = sum(1 for _, count in prefix2.items() if count == 1)
    route_collision_count = sum(count - 1 for count in prefix2.values() if count > 1)

    status = "ASSETS_REUSABLE"
    if len(triple) == 0:
        status = "ASSETS_INCOMPATIBLE"
    elif len(triple) < len(meta_ids) * 0.5:
        status = "ASSETS_PARTIALLY_REUSABLE"

    audit = {
        "status": status,
        "metadata_item_count": len(meta_ids),
        "embedding_item_count": len(emb_ids),
        "sid_item_count": len(sid_ids),
        "metadata_intersection_embedding": len(meta_ids & emb_ids),
        "metadata_intersection_sid": len(meta_ids & sid_ids),
        "embedding_intersection_sid": len(emb_ids & sid_ids),
        "metadata_embedding_sid_intersection": len(triple),
        "coverage_ratio": (len(triple) / len(meta_ids)) if meta_ids else 0.0,
        "embedding_dimensions": dims,
        "nan_item_count": nan_count,
        "inf_item_count": inf_count,
        "route_depth_distribution": dict(route_depths),
        "prefix1_route_count": len(prefix1),
        "prefix2_route_count": len(prefix2),
        "empty_route_count": empty_route_count,
        "singleton_route_count": singleton_route_count,
        "route_collision_count": route_collision_count,
        "item_embedding_path": str(Path(args.item_embedding_path).resolve()),
        "item_sid_path": str(Path(args.item_sid_path).resolve()),
    }

    with (output_dir / "coverage.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "in_metadata", "in_embedding", "in_sid"])
        writer.writeheader()
        for item_id in sorted(meta_ids | emb_ids | sid_ids):
            writer.writerow({
                "item_id": item_id,
                "in_metadata": int(item_id in meta_ids),
                "in_embedding": int(item_id in emb_ids),
                "in_sid": int(item_id in sid_ids),
            })

    with (output_dir / "route_prefix_distribution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "route", "count"])
        writer.writeheader()
        for route, count in prefix1.items():
            writer.writerow({"level": 1, "route": "|".join(route), "count": count})
        for route, count in prefix2.items():
            writer.writerow({"level": 2, "route": "|".join(route), "count": count})

    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# Memory Asset Audit",
        "",
        f"- Status: `{status}`",
        f"- Metadata items: `{audit['metadata_item_count']}`",
        f"- Embedding items: `{audit['embedding_item_count']}`",
        f"- SID items: `{audit['sid_item_count']}`",
        f"- Triple intersection: `{audit['metadata_embedding_sid_intersection']}`",
        f"- Coverage ratio: `{audit['coverage_ratio']:.4f}`",
        f"- Embedding dims: `{dims}`",
        f"- NaN item count: `{nan_count}`",
        f"- Inf item count: `{inf_count}`",
        f"- Route depth distribution: `{audit['route_depth_distribution']}`",
        f"- Prefix-1 route count: `{audit['prefix1_route_count']}`",
        f"- Prefix-2 route count: `{audit['prefix2_route_count']}`",
        f"- Empty route count: `{empty_route_count}`",
        f"- Singleton route count: `{singleton_route_count}`",
        f"- Route collision count: `{route_collision_count}`",
    ]
    (output_dir / "audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
