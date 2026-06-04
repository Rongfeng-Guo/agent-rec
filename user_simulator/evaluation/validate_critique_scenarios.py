"""Validate CritiqueScope scenario and rollout JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from user_simulator.evaluation.critique_scope_eval import load_scenarios
from user_simulator.state.critique_scope import (
    VALID_OBJECT_SCOPES,
    VALID_OPERATIONS,
    VALID_TEMPORAL_SCOPES,
)


REQUIRED_SCENARIO_KEYS = {
    "id",
    "critique_type",
    "utterance",
    "critiques",
    "follow_value",
    "ignore_value",
    "over_apply_value",
    "post_expiry_items",
}
REQUIRED_CRITIQUE_KEYS = {
    "target",
    "operation",
    "reason",
    "object_scope",
    "temporal_scope",
    "horizon",
    "hardness",
    "confidence",
    "promotion_condition",
}
VALID_HARDNESS = {"soft", "hard"}


class ScenarioValidationError(ValueError):
    pass


def validate_scenario(scenario: dict, index: int = 0) -> List[str]:
    errors: List[str] = []
    missing = sorted(REQUIRED_SCENARIO_KEYS - set(scenario))
    if missing:
        errors.append(f"scenario[{index}] missing keys: {missing}")

    critiques = scenario.get("critiques", [])
    if not isinstance(critiques, list) or not critiques:
        errors.append(f"scenario[{index}] critiques must be a non-empty list")
    else:
        for critique_index, critique in enumerate(critiques):
            errors.extend(validate_critique(critique, index, critique_index))

    for branch in ["follow_value", "ignore_value", "over_apply_value"]:
        values = scenario.get(branch)
        if not isinstance(values, list) or not values:
            errors.append(f"scenario[{index}] {branch} must be a non-empty list")
        elif not all(isinstance(value, (int, float)) for value in values):
            errors.append(f"scenario[{index}] {branch} must contain only numbers")

    post_expiry_items = scenario.get("post_expiry_items", [])
    if not isinstance(post_expiry_items, list):
        errors.append(f"scenario[{index}] post_expiry_items must be a list")
    else:
        for item_index, item in enumerate(post_expiry_items):
            if "target" not in item or "true_relevance" not in item:
                errors.append(f"scenario[{index}] post_expiry_items[{item_index}] missing target/true_relevance")

    return errors


def validate_critique(critique: dict, scenario_index: int, critique_index: int) -> List[str]:
    errors: List[str] = []
    prefix = f"scenario[{scenario_index}] critiques[{critique_index}]"
    missing = sorted(REQUIRED_CRITIQUE_KEYS - set(critique))
    if missing:
        errors.append(f"{prefix} missing keys: {missing}")
    if critique.get("operation") not in VALID_OPERATIONS:
        errors.append(f"{prefix} invalid operation: {critique.get('operation')}")
    if critique.get("object_scope") not in VALID_OBJECT_SCOPES:
        errors.append(f"{prefix} invalid object_scope: {critique.get('object_scope')}")
    if str(critique.get("temporal_scope", "")).replace("-", "_") not in VALID_TEMPORAL_SCOPES:
        errors.append(f"{prefix} invalid temporal_scope: {critique.get('temporal_scope')}")
    if critique.get("hardness") not in VALID_HARDNESS:
        errors.append(f"{prefix} invalid hardness: {critique.get('hardness')}")
    confidence = critique.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        errors.append(f"{prefix} confidence must be in [0, 1]")
    horizon = critique.get("horizon")
    if horizon is not None and (not isinstance(horizon, int) or horizon < 0):
        errors.append(f"{prefix} horizon must be a non-negative integer or null")
    return errors


def validate_scenarios(scenarios: Iterable[dict]) -> List[str]:
    errors: List[str] = []
    seen_ids = set()
    for index, scenario in enumerate(scenarios):
        scenario_id = scenario.get("id")
        if scenario_id in seen_ids:
            errors.append(f"scenario[{index}] duplicate id: {scenario_id}")
        seen_ids.add(scenario_id)
        errors.extend(validate_scenario(scenario, index))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-set", default="deterministic", choices=["deterministic", "noisy"])
    parser.add_argument("--scenario-file")
    parser.add_argument("--output")
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_file, scenario_set=args.scenario_set)
    errors = validate_scenarios(scenarios)
    result = {
        "status": "PASS" if not errors else "FAIL",
        "scenario_count": len(scenarios),
        "errors": errors,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
