"""Adapter for turning CritiqueScope rollouts into benchmark scenarios.

The adapter accepts JSONL rows with utterance, critiques, and branch utilities.
When no input is provided it emits the deterministic built-in scenarios. This is
the bridge for later real GIMO rollouts: keep the schema stable, replace the
utility source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from user_simulator.evaluation.critique_scope_eval import DEFAULT_SCENARIOS
from user_simulator.evaluation.critique_uplift_pairs import build_pairs
from user_simulator.evaluation.validate_critique_scenarios import validate_scenario


REQUIRED_BRANCHES = ["follow_value", "ignore_value", "over_apply_value"]


def load_rollouts(path: str | None) -> List[dict]:
    if not path:
        return DEFAULT_SCENARIOS
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            row = json.loads(line)
            validate_rollout(row, line_no)
            rows.append(row)
    return rows


def validate_rollout(row: dict, line_no: int):
    errors = validate_scenario(row, index=line_no)
    if errors:
        raise ValueError(f"line {line_no}: " + "; ".join(errors))
    for branch in REQUIRED_BRANCHES:
        if not isinstance(row[branch], list) or not all(isinstance(value, (int, float)) for value in row[branch]):
            raise ValueError(f"line {line_no}: {branch} must be a list of numbers")


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Optional real rollout JSONL.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    scenarios = load_rollouts(args.input)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "normalized_scenarios.jsonl", scenarios)
    write_jsonl(output_dir / "critique_pairs.jsonl", build_pairs(scenarios))
    metadata = {
        "status": "SMOKE_TEST_ONLY" if args.input is None else "REAL_ROLLOUT_INPUT",
        "input": args.input or "DEFAULT_SCENARIOS",
        "scenario_count": len(scenarios),
        "pair_count": len(build_pairs(scenarios)),
    }
    (output_dir / "adapter_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", **metadata, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
