#!/usr/bin/env python3
"""Export metadata embeddings for oracle-route memory experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from typing import Any, Dict, List, Mapping

import numpy as np

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from sentence_transformers import SentenceTransformer

from genrec.memory.data_adapter import load_item_metadata

MODEL_CANDIDATES = [
    "/home/grf/crs/tools/all-MiniLM-L6-v2",
    "/home/grf/agent-rec/crs/tools/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "t"}


def choose_model(explicit: str | None) -> str:
    if explicit:
        return explicit
    for candidate in MODEL_CANDIDATES[:-1]:
        if Path(candidate).exists():
            return candidate
    return MODEL_CANDIDATES[-1]


def build_text(item_id: str, meta: Mapping[str, Any]) -> str:
    parts = []
    if meta.get("title"):
        parts.append(f"Title: {meta['title']}")
    if meta.get("categories"):
        parts.append(f"Categories: {meta['categories']}")
    if meta.get("description"):
        parts.append(f"Description: {meta['description']}")
    if meta.get("features"):
        parts.append(f"Features: {meta['features']}")
    if not parts:
        parts.append(f"ItemID: {item_id}")
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    item_id_path = output_dir / "item_ids.json"
    embedding_path = output_dir / "item_embeddings.npy"
    jsonl_path = output_dir / "item_embeddings.jsonl"

    metadata = load_item_metadata(args.data_dir)
    item_ids = sorted(metadata.keys())
    texts = [build_text(item_id, metadata[item_id]) for item_id in item_ids]

    if args.resume and item_id_path.exists() and embedding_path.exists():
        existing_ids = json.loads(item_id_path.read_text(encoding="utf-8"))
        arr = np.load(embedding_path)
        if existing_ids == item_ids and len(arr) == len(item_ids):
            embeddings = arr.astype(np.float32)
        else:
            embeddings = None
    else:
        embeddings = None

    model_name = choose_model(args.model_name_or_path)
    if embeddings is None:
        device = None if args.device == "auto" else args.device
        model = SentenceTransformer(model_name, device=device)
        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=args.normalize,
        ).astype(np.float32)
    if args.normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        embeddings = embeddings / norms

    nan_count = int(np.isnan(embeddings).any(axis=1).sum())
    inf_count = int(np.isinf(embeddings).any(axis=1).sum())
    if nan_count or inf_count:
        raise ValueError(f"Embedding export produced invalid vectors: nan_items={nan_count}, inf_items={inf_count}")

    np.save(embedding_path, embeddings)
    item_id_path.write_text(json.dumps(item_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item_id, text, emb in zip(item_ids, texts, embeddings):
            f.write(json.dumps({"item_id": item_id, "text": text, "embedding": emb.tolist()}, ensure_ascii=False) + "\n")

    manifest = {
        "git_commit": os.popen("cd /home/grf/agent-rec && git rev-parse HEAD").read().strip(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "model_name_or_path": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "item_count": len(item_ids),
        "batch_size": args.batch_size,
        "device": args.device,
        "normalization": args.normalize,
        "text_template": ["title", "category", "description", "features"],
        "source_files": sorted({meta.get("source_file", "") for meta in metadata.values()}),
    }
    audit = {
        "item_count": len(item_ids),
        "embedding_dimension": int(embeddings.shape[1]),
        "nan_item_count": nan_count,
        "inf_item_count": inf_count,
        "mean_norm": float(np.linalg.norm(embeddings, axis=1).mean()),
        "min_norm": float(np.linalg.norm(embeddings, axis=1).min()),
        "max_norm": float(np.linalg.norm(embeddings, axis=1).max()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "audit.md").write_text(
        "\n".join([
            "# Metadata Embedding Audit",
            "",
            f"- Model: `{model_name}`",
            f"- Item count: `{audit['item_count']}`",
            f"- Embedding dimension: `{audit['embedding_dimension']}`",
            f"- NaN item count: `{audit['nan_item_count']}`",
            f"- Inf item count: `{audit['inf_item_count']}`",
            f"- Mean norm: `{audit['mean_norm']:.6f}`",
            f"- Min norm: `{audit['min_norm']:.6f}`",
            f"- Max norm: `{audit['max_norm']:.6f}`",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": manifest, "audit": audit}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
