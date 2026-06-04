"""Build manifest files for CritiqueWorld CDPO bridge datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from user_simulator.evaluation.validate_cdpo_pairs import validate_file


def git_value(args: List[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNKNOWN"


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_ids(rows: Iterable[dict], dev_fraction: float) -> dict:
    rows = sorted(rows, key=lambda row: (row.get("scenario", ""), row.get("method", ""), row.get("seed", 0), row.get("id", "")))
    if not rows:
        return {"train": [], "dev": []}
    dev_count = max(1, round(len(rows) * dev_fraction)) if dev_fraction > 0 else 0
    dev_indices = set()
    if dev_count:
        stride = max(1, len(rows) // dev_count)
        index = 0
        while len(dev_indices) < dev_count and index < len(rows):
            dev_indices.add(index)
            index += stride
    train = []
    dev = []
    for index, row in enumerate(rows):
        target = dev if index in dev_indices else train
        target.append(row["id"])
    return {"train": train, "dev": dev}


def materialize_splits(rows: List[dict], splits: dict, train_output: Path, dev_output: Path) -> dict:
    by_id = {row["id"]: row for row in rows}
    train_ids = splits["train"]
    dev_ids = splits["dev"]
    missing = [pair_id for pair_id in [*train_ids, *dev_ids] if pair_id not in by_id]
    if missing:
        raise ValueError(f"Split ids missing from input rows: {missing[:5]}")

    overlap = sorted(set(train_ids) & set(dev_ids))
    if overlap:
        raise ValueError(f"Train/dev split overlap detected: {overlap[:5]}")

    if len(train_ids) != len(set(train_ids)) or len(dev_ids) != len(set(dev_ids)):
        raise ValueError("Duplicate ids detected inside train/dev split")

    train_rows = [by_id[pair_id] for pair_id in train_ids]
    dev_rows = [by_id[pair_id] for pair_id in dev_ids]
    write_jsonl(train_output, train_rows)
    write_jsonl(dev_output, dev_rows)
    return {
        "train_file": str(train_output),
        "dev_file": str(dev_output),
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "train_sha256": file_sha256(train_output),
        "dev_sha256": file_sha256(dev_output),
    }


def write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_manifest(input_path: Path, validation_path: Path | None, dev_fraction: float) -> dict:
    validation = validate_file(input_path)
    if validation["status"] != "PASS":
        raise ValueError("CDPO pair validation failed; refusing to build dataset manifest")

    rows = read_jsonl(input_path)
    by_parser = Counter(row.get("parser_mode", "UNKNOWN") for row in rows)
    by_method = Counter(row.get("method", "UNKNOWN") for row in rows)
    by_scenario = Counter(row.get("scenario", "UNKNOWN") for row in rows)
    by_rejected = Counter(row.get("rejected", {}).get("branch", "UNKNOWN") for row in rows)
    splits = split_ids(rows, dev_fraction)

    return {
        "dataset_name": input_path.parent.name + "_cdpo",
        "source": "CritiqueWorld",
        "status": "SMOKE_TEST_ONLY",
        "proxy": "controlled counterfactual rollout proxy",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "input_file": str(input_path),
        "input_sha256": file_sha256(input_path),
        "validation_file": str(validation_path) if validation_path else None,
        "validation_status": validation["status"],
        "row_count": len(rows),
        "score_delta_min": validation["score_delta_min"],
        "score_delta_mean": validation["score_delta_mean"],
        "score_delta_max": validation["score_delta_max"],
        "by_parser_mode": dict(sorted(by_parser.items())),
        "by_method": dict(sorted(by_method.items())),
        "by_scenario": dict(sorted(by_scenario.items())),
        "by_rejected_branch": dict(sorted(by_rejected.items())),
        "splits": {
            "dev_fraction": dev_fraction,
            "train_count": len(splits["train"]),
            "dev_count": len(splits["dev"]),
            "train_ids": splits["train"],
            "dev_ids": splits["dev"],
            "train_file": None,
            "dev_file": None,
            "train_sha256": None,
            "dev_sha256": None,
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
            ],
            "chosen_branch": "follow",
            "rejected_branches": ["ignore", "over_apply"],
            "score_delta": "strictly_positive",
        },
        "limitations": [
            "controlled synthetic latent-state benchmark",
            "not human evaluation",
            "not complete causal inference",
            "requires downstream mapping before full GIMO/LLaMA-Factory training",
        ],
    }


def build_llamafactory_snippet(manifest: dict, input_path: Path) -> dict:
    dataset_name = manifest["dataset_name"]
    return {
        dataset_name: {
            "file_name": str(input_path).replace("\\", "/"),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
            },
            "metadata": {
                "source": manifest["source"],
                "status": manifest["status"],
                "proxy": manifest["proxy"],
                "row_count": manifest["row_count"],
                "manifest": "cdpo_dataset_manifest.json",
            },
        }
    }


def build_split_dataset_info_snippet(manifest: dict) -> dict:
    dataset_name = manifest["dataset_name"]
    splits = manifest.get("splits", {})
    return {
        f"{dataset_name}_train": {
            "file_name": str(splits.get("train_file", "")).replace("\\", "/"),
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"},
            "metadata": {
                "source": manifest["source"],
                "status": manifest["status"],
                "proxy": manifest["proxy"],
                "row_count": splits.get("train_count"),
                "manifest": "cdpo_dataset_manifest.json",
                "split": "train",
            },
        },
        f"{dataset_name}_dev": {
            "file_name": str(splits.get("dev_file", "")).replace("\\", "/"),
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"},
            "metadata": {
                "source": manifest["source"],
                "status": manifest["status"],
                "proxy": manifest["proxy"],
                "row_count": splits.get("dev_count"),
                "manifest": "cdpo_dataset_manifest.json",
                "split": "dev",
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--validation")
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--dataset-info-output", required=True)
    parser.add_argument("--train-output")
    parser.add_argument("--dev-output")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    args = parser.parse_args()

    input_path = Path(args.input)
    validation_path = Path(args.validation) if args.validation else None
    manifest = build_manifest(input_path, validation_path, args.dev_fraction)
    rows = read_jsonl(input_path)

    if args.train_output or args.dev_output:
        if not args.train_output or not args.dev_output:
            raise SystemExit("--train-output and --dev-output must be provided together")
        split_info = materialize_splits(
            rows,
            {"train": manifest["splits"]["train_ids"], "dev": manifest["splits"]["dev_ids"]},
            Path(args.train_output),
            Path(args.dev_output),
        )
        manifest["splits"].update(split_info)
        snippet = build_split_dataset_info_snippet(manifest)
    else:
        snippet = build_llamafactory_snippet(manifest, input_path)

    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    snippet_path = Path(args.dataset_info_output)
    snippet_path.parent.mkdir(parents=True, exist_ok=True)
    snippet_path.write_text(json.dumps(snippet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": str(manifest_path),
                "dataset_info": str(snippet_path),
                "rows": manifest["row_count"],
                "train_file": manifest["splits"].get("train_file"),
                "dev_file": manifest["splits"].get("dev_file"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
