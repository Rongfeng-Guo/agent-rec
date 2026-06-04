"""Offline drift-memory benchmark for GIMO.

This script gives a lightweight, model-free protocol for comparing memory
strategies when user preferences drift across turns. It is intentionally small:
research runs can replace the toy recommenders with real GIMO outputs while
keeping the metric calculation stable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from user_simulator.state.structured_memory import StructuredMemory


DEFAULT_SCENARIOS = [
    {
        "id": "yelp_group_to_family",
        "turns": [
            {
                "utterance": "I want a restaurant for a group dinner.",
                "updates": [
                    {
                        "bucket": "positive",
                        "key": "occasion",
                        "value": "group dinner",
                        "operation": "merge",
                        "confidence": 0.74,
                    }
                ],
            },
            {
                "utterance": "It should be quiet enough for conversation.",
                "updates": [
                    {
                        "bucket": "soft",
                        "key": "noise",
                        "value": "quiet",
                        "operation": "merge",
                        "confidence": 0.64,
                    }
                ],
            },
            {
                "utterance": "Actually plans changed. I am bringing kids, so make it family-friendly.",
                "drift_turn": True,
                "updates": [
                    {
                        "bucket": "positive",
                        "key": "occasion",
                        "value": "group dinner",
                        "operation": "retain",
                        "confidence": 0.78,
                    },
                    {
                        "bucket": "hard",
                        "key": "audience",
                        "value": "family-friendly",
                        "operation": "merge",
                        "confidence": 0.86,
                    },
                    {
                        "bucket": "soft",
                        "key": "noise",
                        "value": "quiet",
                        "operation": "forget",
                        "confidence": 0.0,
                    },
                ],
            },
        ],
        "target_terms": ["group dinner", "family-friendly"],
        "stale_terms": ["quiet"],
        "hard_terms": ["family-friendly"],
        "recommendations": {
            "full_history": [
                ["quiet restaurant for a group dinner"],
                ["quiet family-friendly restaurant"],
                ["family-friendly restaurant for a group dinner"],
            ],
            "summary_memory": [
                ["quiet restaurant for a group dinner"],
                ["family-friendly quiet restaurant"],
                ["family-friendly restaurant for a group dinner"],
            ],
            "retrieval_memory": [
                ["family-friendly casual restaurant"],
                ["family-friendly casual restaurant"],
                ["family-friendly casual restaurant"],
            ],
            "structured_memory": [
                ["family-friendly restaurant for a group dinner"],
                ["family-friendly restaurant for a group dinner"],
                ["family-friendly restaurant for a group dinner"],
            ],
        },
    }
]


def load_scenarios(path: str | None) -> List[dict]:
    if not path:
        return DEFAULT_SCENARIOS
    scenario_path = Path(path)
    if scenario_path.suffix == ".jsonl":
        with scenario_path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    with scenario_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else [data]


def contains_all(text: str, terms: Iterable[str]) -> bool:
    lower = text.lower()
    return all(term.lower() in lower for term in terms)


def contains_any(text: str, terms: Iterable[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def success_at_k(recommendations: List[str], target_terms: List[str], k: int) -> float:
    top_k = recommendations[:k]
    return 1.0 if any(contains_all(item, target_terms) for item in top_k) else 0.0


def recovery_turns(post_drift_recommendations: List[List[str]], target_terms: List[str]) -> int:
    for offset, recommendations in enumerate(post_drift_recommendations):
        if success_at_k(recommendations, target_terms, k=len(recommendations)):
            return offset
    return len(post_drift_recommendations)


def evaluate_method(scenario: dict, method: str, memory: StructuredMemory) -> Dict[str, float]:
    recommendations_by_turn = scenario["recommendations"][method]
    drift_index = next(
        (idx for idx, turn in enumerate(scenario["turns"]) if turn.get("drift_turn")),
        len(scenario["turns"]) - 1,
    )
    flat_recommendations = [item for turn_items in recommendations_by_turn for item in turn_items]
    post_drift = recommendations_by_turn[drift_index:]

    stale_violations = sum(
        1 for item in flat_recommendations if contains_any(item, scenario.get("stale_terms", []))
    )
    hard_terms = scenario.get("hard_terms", [])
    hard_checked = [item for turn_items in post_drift for item in turn_items]
    hard_satisfied = sum(1 for item in hard_checked if contains_all(item, hard_terms))

    return {
        "recovery_turns": recovery_turns(post_drift, scenario["target_terms"]),
        "stale_preference_violation_rate": stale_violations / max(1, len(flat_recommendations)),
        "constraint_satisfaction_rate": hard_satisfied / max(1, len(hard_checked)),
        "success_at_1": success_at_k(recommendations_by_turn[-1], scenario["target_terms"], k=1),
        "token_cost": memory.token_cost_estimate() if method == "structured_memory" else token_cost_proxy(method, scenario),
    }


def token_cost_proxy(method: str, scenario: dict) -> int:
    if method == "full_history":
        return sum(len(turn["utterance"].split()) for turn in scenario["turns"])
    if method == "summary_memory":
        return 28
    if method == "retrieval_memory":
        return max(len(turn["utterance"].split()) for turn in scenario["turns"])
    return 0


def run_benchmark(scenarios: List[dict]) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for scenario in scenarios:
        memory = StructuredMemory()
        for turn in scenario["turns"]:
            memory.apply_turn(turn["utterance"], updates=turn.get("updates"))

        scenario_results = {}
        for method in scenario["recommendations"]:
            scenario_results[method] = evaluate_method(scenario, method, memory)
        results[scenario["id"]] = scenario_results
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", help="Optional JSON/JSONL drift scenarios.")
    parser.add_argument("--output", help="Optional path to save metrics as JSON.")
    args = parser.parse_args()

    results = run_benchmark(load_scenarios(args.scenarios))
    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
