"""Build counterfactual preference pairs from CritiqueScope scenarios.

The output can be used as a lightweight bridge from scope-aware critique
diagnostics to CDPO/DPO-style alignment: the follow branch is preferred over
ignore or over-apply when it produces higher counterfactual trajectory value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from user_simulator.evaluation.critique_scope_eval import DEFAULT_SCENARIOS, load_scenarios


def trajectory_value(values: Iterable[float]) -> float:
    return sum(values)


def build_pair(scenario: dict, rejected_branch: str) -> dict:
    chosen_value = trajectory_value(scenario["follow_value"])
    rejected_values = scenario[f"{rejected_branch}_value"]
    rejected_value = trajectory_value(rejected_values)
    critique = scenario["critiques"][0]
    return {
        "scenario_id": scenario["id"],
        "critique_type": scenario["critique_type"],
        "prompt": {
            "user_feedback": scenario["utterance"],
            "critique": critique,
            "instruction": "Choose the response policy that best follows the critique without over-correcting durable user memory.",
        },
        "chosen": {
            "branch": "follow",
            "policy": describe_policy(critique, branch="follow"),
            "trajectory_value": chosen_value,
        },
        "rejected": {
            "branch": rejected_branch,
            "policy": describe_policy(critique, branch=rejected_branch),
            "trajectory_value": rejected_value,
        },
        "uplift": chosen_value - rejected_value,
    }


def describe_policy(critique: dict, branch: str) -> str:
    target = critique["target"]
    operation = critique["operation"]
    temporal_scope = critique["temporal_scope"]
    horizon = critique["horizon"]

    if branch == "follow":
        if temporal_scope == "persistent":
            return f"Apply a durable {operation} intervention to {target}."
        return (
            f"Apply a scoped {operation} intervention to {target} for "
            f"{temporal_scope} horizon={horizon}, then let it expire unless new evidence appears."
        )
    if branch == "ignore":
        return f"Ignore the feedback and keep the existing recommendation policy for {target}."
    if operation in {"attenuate", "diversify", "explore"}:
        return f"Over-apply the feedback by turning it into a persistent filter against {target}."
    return f"Over-apply the feedback beyond its stated scope for {target}."


def build_pairs(scenarios: List[dict]) -> List[dict]:
    pairs = []
    for scenario in scenarios:
        for rejected_branch in ["ignore", "over_apply"]:
            pair = build_pair(scenario, rejected_branch)
            if pair["uplift"] > 0:
                pairs.append(pair)
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", help="Optional JSON/JSONL critique scenarios.")
    parser.add_argument("--output", help="Path to save JSONL preference pairs.")
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios) if args.scenarios else DEFAULT_SCENARIOS
    pairs = build_pairs(scenarios)
    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as file:
            for pair in pairs:
                file.write(json.dumps(pair, ensure_ascii=False) + "\n")
    else:
        print(json.dumps(pairs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
