#!/usr/bin/env python3
"""Discover reusable embedding and SID assets for oracle-route experiments."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.memory.data_adapter import load_item_metadata

PATTERNS = [
    "*embedding*.npy",
    "*embedding*.npz",
    "*embedding*.pt",
    "*embedding*.pkl",
    "*item_emb*",
    "*.sent_emb",
    "*semantic*id*",
    "*sid*",
    "*codebook*",
    "*rqvae*",
    "*residual*quant*",
    "*cluster*",
    "*mapping*.json",
    "*mapping*.jsonl",
]
NOTEBOOK_PATH = Path("user_simulator/embedding/item_embedding.ipynb")


def collect_candidates(roots: Iterable[str], maxdepth: int) -> List[Path]:
    seen = set()
    results: List[Path] = []
    for root_text in roots:
        root = Path(root_text)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                rel_depth = len(path.relative_to(root).parts)
            except Exception:
                continue
            if rel_depth > maxdepth:
                continue
            lower_name = path.name.lower()
            if any(path.match(pattern) or lower_name.endswith(pattern.lstrip("*")) for pattern in PATTERNS):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    results.append(resolved)
    if NOTEBOOK_PATH.exists():
        resolved = NOTEBOOK_PATH.resolve()
        if resolved not in seen:
            results.append(resolved)
    return sorted(results)


def infer_provenance(path: Path) -> str:
    text = str(path)
    if "RPG_KDD2025" in text and "Beauty" in text:
        return "RPG AmazonReviews2014 Beauty"
    if "GenRecEdit-main" in text and "Cell_Phones_and_Accessories" in text:
        return "GenRecEdit AmazonReviews2023 Cell Phones"
    if path == NOTEBOOK_PATH.resolve():
        return "agent-rec notebook logic"
    return f"filesystem:{path.parents[1].name if len(path.parents) > 1 else path.parent.name}"


def infer_item_ids_for_sent_emb(path: Path) -> Optional[List[str]]:
    sidecar = path.parent / "id_mapping.json"
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None
    id2item = payload.get("id2item")
    if not isinstance(id2item, list):
        return None
    return [str(value) for value in id2item[1:]]


def infer_sem_ids_path(path: Path) -> Optional[Path]:
    if path.suffix == ".sent_emb":
        siblings = sorted(path.parent.glob("*.sem_ids"))
        return siblings[0] if siblings else None
    return None


def describe_candidate(path: Path, metadata_ids: set[str]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "path": str(path),
        "type": "unknown",
        "size_bytes": path.stat().st_size,
        "mtime": path.stat().st_mtime,
        "item_count": None,
        "format": path.suffix.lstrip(".") or path.name,
        "readable": False,
        "contains_item_id": False,
        "contains_embedding": False,
        "contains_sid": False,
        "alignable_with_user_simulator": None,
        "alignment_overlap_count": None,
        "provenance": infer_provenance(path),
    }
    try:
        if path == NOTEBOOK_PATH.resolve():
            record.update({
                "type": "notebook_logic_only",
                "format": "ipynb",
                "readable": True,
            })
            return record

        if path.suffix == ".sent_emb":
            item_ids = infer_item_ids_for_sent_emb(path) or []
            sem_ids_path = infer_sem_ids_path(path)
            overlap = len(metadata_ids & set(item_ids)) if item_ids else 0
            coverage_ratio = (overlap / len(metadata_ids)) if metadata_ids else 0.0
            record.update({
                "type": "embedding+sid_candidate",
                "readable": True,
                "contains_item_id": bool(item_ids),
                "contains_embedding": True,
                "contains_sid": bool(sem_ids_path),
                "item_count": len(item_ids) if item_ids else None,
                "alignable_with_user_simulator": coverage_ratio >= 0.5,
                "alignment_overlap_count": overlap,
                "alignment_coverage_ratio": coverage_ratio,
                "paired_sid_path": str(sem_ids_path) if sem_ids_path else None,
            })
            return record

        if path.suffix in {".json", ".jsonl", ".sem_ids"}:
            preview = path.read_text(encoding="utf-8")[:1000]
            record["readable"] = True
            lowered = preview.lower()
            if "item_id" in lowered or "parentasin" in lowered or "businessid" in lowered:
                record["contains_item_id"] = True
            if '"sid"' in lowered or path.suffix == ".sem_ids":
                record["contains_sid"] = True
            if path.suffix == ".sem_ids":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        item_ids = {str(key) for key in payload.keys()}
                        overlap = len(metadata_ids & item_ids)
                        record["item_count"] = len(item_ids)
                        record["contains_item_id"] = True
                        record["alignable_with_user_simulator"] = (overlap / len(metadata_ids)) >= 0.5 if metadata_ids else False
                        record["alignment_overlap_count"] = overlap
                        record["alignment_coverage_ratio"] = (overlap / len(metadata_ids)) if metadata_ids else 0.0
                except Exception:
                    pass
            if path.name == "id_mapping.json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    id2item = payload.get("id2item")
                    if isinstance(id2item, list):
                        item_ids = {str(value) for value in id2item[1:]}
                        overlap = len(metadata_ids & item_ids)
                        record["item_count"] = len(item_ids)
                        record["contains_item_id"] = True
                        record["alignable_with_user_simulator"] = (overlap / len(metadata_ids)) >= 0.5 if metadata_ids else False
                        record["alignment_overlap_count"] = overlap
                        record["alignment_coverage_ratio"] = (overlap / len(metadata_ids)) if metadata_ids else 0.0
                except Exception:
                    pass
            return record

        if path.suffix in {".npy", ".npz"}:
            record["readable"] = True
            record["contains_embedding"] = True
            return record
    except Exception as exc:
        record["error"] = str(exc)
        return record

    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["/home/grf/agent-rec", "/home/grf/external_baselines", "/home/grf/GenRecEdit-main", "/data", "/mnt", "/share", "/home/share"],
    )
    parser.add_argument("--maxdepth", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_ids = set(load_item_metadata(args.data_dir).keys())
    candidates = [describe_candidate(path, metadata_ids) for path in collect_candidates(args.roots, args.maxdepth)]

    reusable_embedding = any(row.get("contains_embedding") and row.get("alignable_with_user_simulator") for row in candidates)
    reusable_sid = any(row.get("contains_sid") and row.get("alignable_with_user_simulator") for row in candidates)
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metadata_item_count": len(metadata_ids),
        "REUSABLE_EMBEDDING_FOUND": reusable_embedding,
        "REUSABLE_SID_FOUND": reusable_sid,
        "EMBEDDING_MISSING": not reusable_embedding,
        "SID_MISSING": not reusable_sid,
        "candidates": candidates,
    }
    (output_dir / "discovered_assets.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_lines = [
        "# Discovered Assets",
        "",
        f"- Metadata item count: `{summary['metadata_item_count']}`",
        f"- REUSABLE_EMBEDDING_FOUND: `{summary['REUSABLE_EMBEDDING_FOUND']}`",
        f"- REUSABLE_SID_FOUND: `{summary['REUSABLE_SID_FOUND']}`",
        f"- EMBEDDING_MISSING: `{summary['EMBEDDING_MISSING']}`",
        f"- SID_MISSING: `{summary['SID_MISSING']}`",
        "",
        "## Candidates",
    ]
    for row in candidates:
        md_lines.append(
            f"- `{row['path']}`: type=`{row['type']}`, format=`{row['format']}`, readable=`{row['readable']}`, contains_embedding=`{row['contains_embedding']}`, contains_sid=`{row['contains_sid']}`, item_count=`{row['item_count']}`, alignable=`{row['alignable_with_user_simulator']}`, overlap=`{row['alignment_overlap_count']}`, provenance=`{row['provenance']}`"
        )
    (output_dir / "discovered_assets.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
