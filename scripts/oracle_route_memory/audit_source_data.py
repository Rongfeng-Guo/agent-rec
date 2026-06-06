#!/usr/bin/env python3
"""Audit user_simulator source data for oracle-route memory experiments."""

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
from typing import Any, Dict, List, Mapping

from genrec.memory.data_adapter import build_eval_samples, load_interactions

ITEM_ID_KEYS = ("ItemID", "ParentASIN", "BusinessID", "item_id")
TITLE_KEYS = ("ItemName", "BusinessName", "title", "name")
DESC_KEYS = ("Description", "description")
CAT_KEYS = ("Categories", "categories")
FEATURE_KEYS = ("Features", "features", "Attributes", "attributes")


def extract_item_id(item: Mapping[str, Any]) -> str | None:
    for key in ITEM_ID_KEYS:
        if key in item and item[key] not in (None, ""):
            return str(item[key])
    return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    if isinstance(value, dict):
        return " ".join(f"{k}: {v}" for k, v in value.items())
    return str(value).strip()


def audit_metadata_file(path: Path, metadata_items: Dict[str, Dict[str, Any]], row_errors: List[dict], duplicate_counter: Counter):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                row_errors.append({"path": str(path), "line": line_no, "error": f"json_decode: {exc}", "raw": line[:400]})
                continue
            review_list = row.get("ReviewList") if isinstance(row, dict) else None
            if isinstance(review_list, list):
                items = review_list
            elif isinstance(review_list, dict):
                items = [review_list]
            elif isinstance(row, dict):
                items = [row]
            else:
                row_errors.append({"path": str(path), "line": line_no, "error": "unsupported_row_type"})
                continue
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                item_id = extract_item_id(item)
                if item_id is None:
                    row_errors.append({"path": str(path), "line": line_no, "error": "missing_item_id", "row_keys": sorted(item.keys())[:30]})
                    continue
                duplicate_counter[item_id] += 1
                if item_id in metadata_items:
                    continue
                title = next((normalize_text(item.get(k)) for k in TITLE_KEYS if normalize_text(item.get(k))), "")
                description = next((normalize_text(item.get(k)) for k in DESC_KEYS if normalize_text(item.get(k))), "")
                categories = next((normalize_text(item.get(k)) for k in CAT_KEYS if normalize_text(item.get(k))), "")
                features = next((normalize_text(item.get(k)) for k in FEATURE_KEYS if normalize_text(item.get(k))), "")
                metadata_items[item_id] = {
                    "title": title,
                    "description": description,
                    "categories": categories,
                    "features": features,
                    "source_file": str(path),
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summary = []
    row_errors: List[dict] = []
    unique_task_items = set()
    split_item_sets = {}

    for split in ("train", "test"):
        records = load_interactions(args.data_dir, split)
        domains = Counter(record["__domain__"] for record in records)
        item_count = 0
        item_ids = set()
        for record in records:
            items = record.get("Items") or []
            if isinstance(items, dict):
                items = [items]
            for item in items:
                if isinstance(item, Mapping):
                    item_id = extract_item_id(item)
                    if item_id:
                        item_ids.add(item_id)
                        unique_task_items.add(item_id)
                        item_count += 1
                    else:
                        row_errors.append({"path": record["__source_path__"], "error": "missing_item_id_in_task", "user_id": record.get("UserID")})
        split_item_sets[split] = item_ids
        split_summary.append({
            "split": split,
            "row_count": len(records),
            "unique_item_count": len(item_ids),
            "item_mentions": item_count,
            "domains": dict(domains),
        })

    cold_samples = build_eval_samples(args.data_dir, split="test", cold_only=True)

    metadata_items: Dict[str, Dict[str, Any]] = {}
    duplicate_counter: Counter = Counter()
    raw_root = Path(args.data_dir) / "raw_data"
    if not raw_root.exists():
        raw_root = Path(args.data_dir).resolve() / "raw_data"
    for path in sorted(raw_root.rglob("*.jsonl")):
        audit_metadata_file(path, metadata_items, row_errors, duplicate_counter)

    title_coverage = sum(1 for meta in metadata_items.values() if meta["title"])
    desc_coverage = sum(1 for meta in metadata_items.values() if meta["description"])
    cat_coverage = sum(1 for meta in metadata_items.values() if meta["categories"])
    feat_coverage = sum(1 for meta in metadata_items.values() if meta["features"])
    overlap = unique_task_items & set(metadata_items.keys())
    duplicate_items = sum(1 for _, count in duplicate_counter.items() if count > 1)

    audit = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "task_row_count": sum(row["row_count"] for row in split_summary),
        "metadata_row_count": int(sum(duplicate_counter.values())),
        "unique_task_item_count": len(unique_task_items),
        "unique_metadata_item_count": len(metadata_items),
        "train_unique_item_count": len(split_item_sets.get("train", set())),
        "test_unique_item_count": len(split_item_sets.get("test", set())),
        "cold_start_sample_count": len(cold_samples),
        "metadata_overlap_count": len(overlap),
        "metadata_overlap_ratio": (len(overlap) / len(unique_task_items)) if unique_task_items else 0.0,
        "title_coverage": title_coverage,
        "description_coverage": desc_coverage,
        "category_coverage": cat_coverage,
        "features_coverage": feat_coverage,
        "duplicate_item_count": duplicate_items,
        "invalid_row_count": len(row_errors),
        "split_summary": split_summary,
    }

    with (output_dir / "split_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "row_count", "unique_item_count", "item_mentions", "domains"])
        writer.writeheader()
        for row in split_summary:
            row = dict(row)
            row["domains"] = json.dumps(row["domains"], ensure_ascii=False)
            writer.writerow(row)

    with (output_dir / "metadata_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric in [
            "unique_metadata_item_count", "metadata_overlap_count", "metadata_overlap_ratio",
            "title_coverage", "description_coverage", "category_coverage", "features_coverage",
            "duplicate_item_count", "invalid_row_count", "cold_start_sample_count",
        ]:
            writer.writerow({"metric": metric, "value": audit[metric]})

    with (output_dir / "item_id_coverage.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "in_train", "in_test", "in_metadata", "title", "description", "categories", "features"])
        writer.writeheader()
        for item_id in sorted(unique_task_items):
            meta = metadata_items.get(item_id, {})
            writer.writerow({
                "item_id": item_id,
                "in_train": int(item_id in split_item_sets.get("train", set())),
                "in_test": int(item_id in split_item_sets.get("test", set())),
                "in_metadata": int(item_id in metadata_items),
                "title": int(bool(meta.get("title"))),
                "description": int(bool(meta.get("description"))),
                "categories": int(bool(meta.get("categories"))),
                "features": int(bool(meta.get("features"))),
            })

    (output_dir / "row_errors.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in row_errors), encoding="utf-8")
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        "# Source Data Audit",
        "",
        f"- Data dir: `{audit['data_dir']}`",
        f"- Task rows: `{audit['task_row_count']}`",
        f"- Unique task items: `{audit['unique_task_item_count']}`",
        f"- Unique metadata items: `{audit['unique_metadata_item_count']}`",
        f"- Cold-start samples: `{audit['cold_start_sample_count']}`",
        f"- Metadata overlap ratio: `{audit['metadata_overlap_ratio']:.4f}`",
        f"- Title coverage: `{audit['title_coverage']}`",
        f"- Description coverage: `{audit['description_coverage']}`",
        f"- Category coverage: `{audit['category_coverage']}`",
        f"- Features coverage: `{audit['features_coverage']}`",
        f"- Duplicate item count: `{audit['duplicate_item_count']}`",
        f"- Invalid row count: `{audit['invalid_row_count']}`",
        "",
        "## Split Summary",
    ]
    for row in split_summary:
        md.append(f"- `{row['split']}`: rows={row['row_count']}, unique_items={row['unique_item_count']}, item_mentions={row['item_mentions']}, domains={row['domains']}")
    (output_dir / "audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
