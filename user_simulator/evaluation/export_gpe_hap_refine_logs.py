"""Export GPE_HAP refinement traces into adapter-ready rollout artifacts.

This helper bridges the repo's real GPE_HAP output format
(`*_refine_log_sample*.json`) to the CritiqueWorld-compatible adapter inputs.
It is intentionally thin: the normalization logic stays in
`critique_rollout_adapter.py`, while this module handles file discovery and
batch export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from user_simulator.evaluation.critique_rollout_adapter import (
    build_branch_pairs,
    build_cdpo_pair,
    load_rollouts,
    materialize_branch_rollouts,
    strip_adapter_fields,
    write_jsonl,
)


def discover_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    candidates = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file()
        and (
            path.name.endswith("_refine_log_sample.json")
            or path.suffix == ".jsonl"
            or path.suffix == ".json"
        )
    )
    if not candidates:
        raise FileNotFoundError(f"No trace files found under {input_path}")
    return candidates


def export_rollouts(input_paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in input_paths:
        rows.extend(load_rollouts(str(path)))
    return rows


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Trace file or directory of GPE_HAP refine logs.")
    parser.add_argument("--output-dir", required=True, help="Directory to write normalized artifacts.")
    parser.add_argument(
        "--write-source-jsonl",
        action="store_true",
        help="Also write the stripped adapter-ready rollout input JSONL.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    input_paths = discover_inputs(input_path)
    rows = export_rollouts(input_paths)
    branch_rows = materialize_branch_rollouts(rows)
    dpo_pairs = build_branch_pairs(rows)
    cdpo_pairs = [build_cdpo_pair(pair) for pair in dpo_pairs]

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_source_jsonl:
        write_jsonl(output_dir / "adapter_input.jsonl", [strip_adapter_fields(row) for row in rows])
    write_jsonl(output_dir / "branch_rollouts.jsonl", branch_rows)
    write_jsonl(output_dir / "dpo_pairs.jsonl", dpo_pairs)
    write_jsonl(output_dir / "cdpo_pairs.jsonl", cdpo_pairs)

    summary = {
        "status": "PASS",
        "input": str(input_path),
        "discovered_inputs": [str(path) for path in input_paths],
        "trace_count": len(rows),
        "branch_row_count": len(branch_rows),
        "dpo_pair_count": len(dpo_pairs),
        "cdpo_pair_count": len(cdpo_pairs),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "export_metadata.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
