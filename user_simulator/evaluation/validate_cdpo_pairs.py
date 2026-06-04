"""Validate CritiqueWorld CDPO/DPO bridge JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, List


REQUIRED_TOP_LEVEL = {
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
}
REQUIRED_BRANCH_FIELDS = {"branch", "policy", "trajectory"}
VALID_REJECTED_BRANCHES = {"ignore", "over_apply"}


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"__line_no__": line_no, "__decode_error__": str(exc)})
                continue
            row["__line_no__"] = line_no
            rows.append(row)
    return rows


def validate_pair(row: dict) -> list[str]:
    line_no = row.get("__line_no__", "?")
    if "__decode_error__" in row:
        return [f"line {line_no}: invalid JSON: {row['__decode_error__']}"]

    errors = []
    missing = REQUIRED_TOP_LEVEL - set(row)
    if missing:
        errors.append(f"line {line_no}: missing top-level fields {sorted(missing)}")

    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        errors.append(f"line {line_no}: conversations must be a non-empty list")
    elif not all(isinstance(item, dict) and item.get("from") and item.get("value") for item in conversations):
        errors.append(f"line {line_no}: each conversation item must include from and value")

    for branch_name in ["chosen", "rejected"]:
        branch = row.get(branch_name)
        if not isinstance(branch, dict):
            errors.append(f"line {line_no}: {branch_name} must be an object")
            continue
        missing_branch = REQUIRED_BRANCH_FIELDS - set(branch)
        if missing_branch:
            errors.append(f"line {line_no}: {branch_name} missing fields {sorted(missing_branch)}")
        if not str(branch.get("trajectory", "")).strip():
            errors.append(f"line {line_no}: {branch_name}.trajectory is empty")

    if row.get("chosen", {}).get("branch") != "follow":
        errors.append(f"line {line_no}: chosen.branch must be follow")
    if row.get("rejected", {}).get("branch") not in VALID_REJECTED_BRANCHES:
        errors.append(f"line {line_no}: rejected.branch must be one of {sorted(VALID_REJECTED_BRANCHES)}")

    try:
        score_delta = float(row.get("score_delta"))
    except (TypeError, ValueError):
        errors.append(f"line {line_no}: score_delta must be numeric")
    else:
        if score_delta <= 0:
            errors.append(f"line {line_no}: score_delta must be strictly positive")

    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        errors.append(f"line {line_no}: metadata must be an object")
    else:
        if metadata.get("format") != "llamafactory_dpo_bridge":
            errors.append(f"line {line_no}: metadata.format must be llamafactory_dpo_bridge")
        if metadata.get("source") != "CritiqueWorld":
            errors.append(f"line {line_no}: metadata.source must be CritiqueWorld")
        if metadata.get("proxy") != "controlled counterfactual rollout proxy":
            errors.append(f"line {line_no}: metadata.proxy must describe the controlled proxy")

    return errors


def summarize(rows: Iterable[dict], errors: list[str]) -> dict:
    rows = list(rows)
    score_deltas = []
    by_method = Counter()
    by_scenario = Counter()
    by_rejected = Counter()
    by_method_scenario: dict[str, Counter] = defaultdict(Counter)
    ids = Counter()

    for row in rows:
        if "__decode_error__" in row:
            continue
        ids[row.get("id", "")] += 1
        method = row.get("method", "UNKNOWN")
        scenario = row.get("scenario", "UNKNOWN")
        rejected = row.get("rejected", {}).get("branch", "UNKNOWN")
        by_method[method] += 1
        by_scenario[scenario] += 1
        by_rejected[rejected] += 1
        by_method_scenario[method][scenario] += 1
        try:
            score_deltas.append(float(row.get("score_delta")))
        except (TypeError, ValueError):
            pass

    duplicate_ids = sorted([pair_id for pair_id, count in ids.items() if pair_id and count > 1])
    for pair_id in duplicate_ids:
        errors.append(f"duplicate id: {pair_id}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "rows": len(rows),
        "errors": errors,
        "error_count": len(errors),
        "score_delta_min": min(score_deltas) if score_deltas else None,
        "score_delta_mean": mean(score_deltas) if score_deltas else None,
        "score_delta_max": max(score_deltas) if score_deltas else None,
        "by_method": dict(sorted(by_method.items())),
        "by_scenario": dict(sorted(by_scenario.items())),
        "by_rejected_branch": dict(sorted(by_rejected.items())),
        "by_method_scenario": {
            method: dict(sorted(counter.items())) for method, counter in sorted(by_method_scenario.items())
        },
        "unique_id_count": len(ids),
        "duplicate_ids": duplicate_ids,
    }


def validate_file(path: Path) -> dict:
    rows = read_jsonl(path)
    errors = []
    for row in rows:
        errors.extend(validate_pair(row))
    result = summarize(rows, errors)
    result["input"] = str(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    result = validate_file(Path(args.input))
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
