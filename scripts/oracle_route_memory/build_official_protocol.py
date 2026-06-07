#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.memory.data_adapter import load_item_embeddings
from genrec.training import (
    assert_leakage_audit_passed,
    build_blind_confirmation_protocol_manifest,
    build_leakage_audit,
    build_official_protocol_manifest,
    build_training_examples,
    filter_training_item_embeddings,
    load_route_mapping,
    write_protocol_bundle,
)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the official protocol bundle.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--item-embedding-path", required=True)
    parser.add_argument("--item-sid-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-version", choices=["v1", "v3"], default="v3")
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--cold-like-item-ratio", type=float, default=0.12)
    parser.add_argument("--confirmation-item-ratio", type=float, default=0.10)
    parser.add_argument("--warm-val-ratio", type=float, default=0.08)
    parser.add_argument("--max-val-examples", type=int, default=1200)
    parser.add_argument("--max-confirmation-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    item_embeddings = load_item_embeddings(args.data_dir, args.item_embedding_path)
    route_mapping = load_route_mapping(args.item_sid_path)
    examples = build_training_examples(args.data_dir, item_embeddings, route_mapping, max_history=args.max_history)

    metadata = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "hostname": socket.gethostname(),
            "git_commit": git_commit(),
            "data_dir": str(Path(args.data_dir).resolve()),
            "item_embedding_path": str(Path(args.item_embedding_path).resolve()),
            "item_sid_path": str(Path(args.item_sid_path).resolve()),
            "max_history": args.max_history,
            "num_item_embeddings": len(item_embeddings),
            "num_route_items": len(route_mapping),
    }
    if args.protocol_version == "v3":
        manifest = build_blind_confirmation_protocol_manifest(
            examples=examples,
            cold_like_item_ratio=args.cold_like_item_ratio,
            confirmation_item_ratio=args.confirmation_item_ratio,
            warm_val_ratio=args.warm_val_ratio,
            max_val_examples=args.max_val_examples,
            max_confirmation_examples=args.max_confirmation_examples,
            seed=args.seed,
            metadata=metadata,
        )
    else:
        manifest = build_official_protocol_manifest(
            examples=examples,
            cold_like_item_ratio=args.cold_like_item_ratio,
            warm_val_ratio=args.warm_val_ratio,
            max_val_examples=args.max_val_examples,
            seed=args.seed,
            metadata=metadata,
        )
    training_item_embeddings = filter_training_item_embeddings(item_embeddings, manifest)
    audit = build_leakage_audit(examples, manifest, hard_negative_items=training_item_embeddings.keys())
    assert_leakage_audit_passed(audit)
    paths = write_protocol_bundle(args.output_dir, manifest, audit)
    payload = {
        "status": "ok",
        "protocol_name": manifest["protocol_name"],
        "config_hash": manifest["config_hash"],
        "split_hash": manifest.get("split_hash"),
        "stats": manifest["stats"],
        "all_passed": audit["all_passed"],
        **paths,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
