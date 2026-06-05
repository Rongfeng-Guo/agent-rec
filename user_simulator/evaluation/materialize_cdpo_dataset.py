"""Materialize CDPO bridge pairs into train/dev JSON datasets."""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Iterable, List

from user_simulator.evaluation.build_cdpo_dataset_manifest import build_split_dataset_info_snippet
from user_simulator.evaluation.build_cdpo_dataset_manifest import file_sha256
from user_simulator.evaluation.build_cdpo_dataset_manifest import git_value
from user_simulator.evaluation.validate_cdpo_pairs import validate_file as validate_cdpo_file


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_source_ref(row: dict) -> str:
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ["source_ref", "branch_id"]:
            value = metadata.get(key)
            if value:
                return str(value)
    for key in ["source_ref", "branch_id", "id"]:
        value = row.get(key)
        if value:
            return str(value)
    return "UNKNOWN"


def state_snapshot_hash(row: dict) -> str:
    payload = row.get("state_snapshot")
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return sha256(serialized.encode("utf-8")).hexdigest()


def trajectory_length(row: dict, key: str) -> int:
    text = str(row.get(key, "")).strip()
    if not text:
        return 0
    return len([line for line in text.splitlines() if line.strip()])


def uplift_value(row: dict) -> float | None:
    try:
        return float(row.get("score_delta"))
    except (TypeError, ValueError):
        return None


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(normalize_source_ref(row), []).append(row)
    for group in grouped.values():
        group.sort(key=lambda item: (item.get("id", ""), item.get("scenario", ""), item.get("method", ""), item.get("seed", 0)))
    return grouped


def split_groups(grouped: dict[str, list[dict]], seed: int, dev_ratio: float) -> dict[str, set[str]]:
    source_refs = sorted(grouped)
    if not source_refs:
        return {"train": set(), "dev": set()}
    rng = random.Random(seed)
    rng.shuffle(source_refs)
    dev_count = max(1, round(len(source_refs) * dev_ratio)) if dev_ratio > 0 else 0
    dev_refs = set(source_refs[:dev_count])
    train_refs = set(source_refs[dev_count:])
    return {"train": train_refs, "dev": dev_refs}


def enrich_row(row: dict, split: str, git_commit: str, git_branch: str) -> dict:
    enriched = copy.deepcopy(row)
    source_ref = normalize_source_ref(enriched)
    snapshot_hash = state_snapshot_hash(enriched)
    enriched["source_ref"] = source_ref
    enriched["state_snapshot_hash"] = snapshot_hash
    enriched["split"] = split
    enriched["git_commit"] = git_commit
    enriched["git_branch"] = git_branch
    metadata = enriched.get("metadata", {})
    if isinstance(metadata, dict):
        metadata["source_ref"] = source_ref
        metadata["state_snapshot_hash"] = snapshot_hash
        metadata["git_commit"] = git_commit
        metadata["git_branch"] = git_branch
        enriched["metadata"] = metadata
    return enriched


def summarize_numeric(values: list[float]) -> dict:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": mean(values), "max": max(values)}


def build_manifest(
    input_path: Path,
    rows: list[dict],
    train_rows: list[dict],
    dev_rows: list[dict],
    seed: int,
    dev_ratio: float,
    status: str,
) -> dict:
    task_types = Counter(str(row.get("task_type", "generic")).strip().lower() or "generic" for row in rows)
    source_refs = Counter(normalize_source_ref(row) for row in rows)
    chosen_lengths = [trajectory_length(row.get("chosen", {}), "trajectory") for row in rows]
    rejected_lengths = [trajectory_length(row.get("rejected", {}), "trajectory") for row in rows]
    uplifts = [value for value in (uplift_value(row) for row in rows) if value is not None]

    return {
        "dataset_name": input_path.parent.name + "_cdpo_dataset",
        "source": "CritiqueWorld",
        "status": status,
        "proxy": rows[0].get("metadata", {}).get("proxy", "controlled counterfactual rollout proxy") if rows else "controlled counterfactual rollout proxy",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "input_file": str(input_path),
        "input_sha256": file_sha256(input_path),
        "seed": seed,
        "dev_ratio": dev_ratio,
        "row_count": len(rows),
        "source_ref_count": len(source_refs),
        "task_type_distribution": dict(sorted(task_types.items())),
        "recommend_count": task_types.get("recommend", 0),
        "ask_count": task_types.get("ask", 0),
        "search_count": task_types.get("search", 0),
        "generic_count": task_types.get("generic", 0),
        "task_types": dict(sorted(task_types.items())),
        "source_refs": dict(sorted(source_refs.items())),
        "chosen_trajectory_length": summarize_numeric(chosen_lengths),
        "rejected_trajectory_length": summarize_numeric(rejected_lengths),
        "uplift_distribution": summarize_numeric(uplifts),
        "uplift_positive_count": sum(1 for value in uplifts if value > 0),
        "uplift_negative_count": sum(1 for value in uplifts if value < 0),
        "uplift_zero_count": sum(1 for value in uplifts if value == 0),
        "splits": {
            "train_count": len(train_rows),
            "dev_count": len(dev_rows),
            "train_source_refs": sorted({row["source_ref"] for row in train_rows}),
            "dev_source_refs": sorted({row["source_ref"] for row in dev_rows}),
            "train_file": str(input_path.parent / "train.json"),
            "dev_file": str(input_path.parent / "dev.json"),
        },
        "schema": {
            "format": "llamafactory_dpo_bridge",
            "required_fields": [
                "id",
                "scenario",
                "seed",
                "method",
                "parser_mode",
                "conversations",
                "chosen",
                "rejected",
                "score_delta",
                "metadata",
                "source_ref",
                "state_snapshot_hash",
                "git_commit",
            ],
        },
        "limitations": [
            "train/dev split is grouped by source_ref to prevent leakage",
            "fixture smoke output is not a substitute for real GPE/HAP output",
        ],
    }


def build_audit(manifest: dict, validation: dict, train_rows: list[dict], dev_rows: list[dict]) -> dict:
    train_source_refs = {row["source_ref"] for row in train_rows}
    dev_source_refs = {row["source_ref"] for row in dev_rows}
    overlap = sorted(train_source_refs & dev_source_refs)
    duplicate_ids = sorted(
        row_id for row_id, count in Counter(row["id"] for row in [*train_rows, *dev_rows]).items() if count > 1
    )
    status = "PASS"
    warnings = []
    critical = []
    if overlap:
        critical.append("train/dev source_ref overlap detected")
        status = "FAIL"
    if duplicate_ids:
        warnings.append("duplicate pair ids detected across splits")
    if validation.get("status") != "PASS":
        critical.append("cdpo validation failed")
        status = "FAIL"
    return {
        "status": status,
        "critical_errors": critical,
        "warnings": warnings,
        "row_count": manifest["row_count"],
        "train_count": manifest["splits"]["train_count"],
        "dev_count": manifest["splits"]["dev_count"],
        "train_source_ref_count": len(train_source_refs),
        "dev_source_ref_count": len(dev_source_refs),
        "source_ref_overlap_count": len(overlap),
        "duplicate_id_count": len(duplicate_ids),
        "validation_status": validation.get("status"),
    }


def relativize_dataset_info(dataset_info: dict) -> dict:
    normalized = copy.deepcopy(dataset_info)
    for value in normalized.values():
        if isinstance(value, dict) and "file_name" in value:
            value["file_name"] = Path(value["file_name"]).name
    return normalized


def render_readme(manifest: dict, audit: dict, output_dir: Path, input_path: Path) -> str:
    return "\n".join(
        [
            "# CDPO Dataset Materialization",
            "",
            f"- Input: `{input_path}`",
            f"- Output: `{output_dir}`",
            f"- Status: `{manifest['status']}`",
            f"- Rows: `{manifest['row_count']}`",
            f"- Train: `{manifest['splits']['train_count']}`",
            f"- Dev: `{manifest['splits']['dev_count']}`",
            f"- Train/dev overlap: `{audit['source_ref_overlap_count']}`",
            f"- Git commit: `{manifest['git_commit']}`",
            "",
            "## Notes",
            "- Splits are grouped by `source_ref` to avoid leakage.",
            "- `state_snapshot_hash` is preserved on each row.",
            "- This materialization is intended for bridge validation and smoke runs.",
        ]
    ) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to cdpo_pairs.jsonl or a directory containing it.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        input_path = input_path / "cdpo_pairs.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input CDPO pairs: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_cdpo_file(input_path)
    if validation["status"] != "PASS":
        raise ValueError("CDPO validation failed; refusing to materialize dataset")

    rows = read_jsonl(input_path)
    grouped = group_rows(rows)
    split_refs = split_groups(grouped, seed=args.seed, dev_ratio=args.dev_ratio)
    git_commit = git_value(["rev-parse", "HEAD"])
    git_branch = git_value(["branch", "--show-current"])

    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    for source_ref, source_rows in sorted(grouped.items()):
        split = "dev" if source_ref in split_refs["dev"] else "train"
        target = dev_rows if split == "dev" else train_rows
        for row in source_rows:
            target.append(enrich_row(row, split, git_commit, git_branch))

    train_rows.sort(key=lambda row: (row["source_ref"], row.get("id", "")))
    dev_rows.sort(key=lambda row: (row["source_ref"], row.get("id", "")))

    manifest_status = "COMPLETED_FIXTURE_SMOKE" if "fixture" in output_dir.name.lower() or "smoke" in output_dir.name.lower() else "COMPLETED_REAL_LOG_VALIDATION"
    manifest = build_manifest(input_path, rows, train_rows, dev_rows, args.seed, args.dev_ratio, manifest_status)
    audit = build_audit(manifest, validation, train_rows, dev_rows)
    dataset_info = relativize_dataset_info(build_split_dataset_info_snippet(manifest))

    write_json(output_dir / "train.json", train_rows)
    write_json(output_dir / "dev.json", dev_rows)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "dataset_info.json", dataset_info)
    write_json(output_dir / "audit.json", audit)
    (output_dir / "README.md").write_text(render_readme(manifest, audit, output_dir, input_path), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "train_count": len(train_rows),
                "dev_count": len(dev_rows),
                "source_ref_overlap_count": audit["source_ref_overlap_count"],
                "manifest": str(output_dir / "manifest.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
