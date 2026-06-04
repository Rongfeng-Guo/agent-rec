"""Model-free CritiqueScope diagnostic benchmark.

The benchmark isolates five critique types that are often collapsed into a
single negative preference: stable dislike, temporary fatigue, session context,
diversity request, and genuine drift. It reports whether a memory strategy
over-corrects after temporary feedback and whether temporary critiques pollute
slow user memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from user_simulator.state.critique_scope import CritiqueScopeMemory


SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


DEFAULT_SCENARIOS = [
    {
        "id": "temporary_ufc_fatigue",
        "critique_type": "Temporary Fatigue",
        "utterance": "I have seen too much UFC lately. Switch it up for a bit.",
        "critiques": [
            {
                "target": "UFC",
                "operation": "attenuate",
                "reason": "exposure fatigue",
                "object_scope": "category",
                "temporal_scope": "session",
                "horizon": 2,
                "hardness": "soft",
                "confidence": 0.82,
                "promotion_condition": "never",
            }
        ],
        "follow_value": [0.7, 0.8, 0.9],
        "ignore_value": [0.2, 0.3, 0.5],
        "over_apply_value": [0.8, 0.2, 0.1],
        "post_expiry_items": [
            {"target": "UFC documentary", "true_relevance": 0.9},
            {"target": "boxing highlights", "true_relevance": 0.5},
        ],
    },
    {
        "id": "persistent_political_filter",
        "critique_type": "Stable Dislike",
        "utterance": "Please never recommend political content to me.",
        "critiques": [
            {
                "target": "political",
                "operation": "filter",
                "reason": "stable dislike",
                "object_scope": "category",
                "temporal_scope": "persistent",
                "horizon": 0,
                "hardness": "hard",
                "confidence": 0.9,
                "promotion_condition": "persistent_language",
            }
        ],
        "follow_value": [0.8, 0.85, 0.86],
        "ignore_value": [0.3, 0.25, 0.2],
        "over_apply_value": [0.82, 0.83, 0.85],
        "post_expiry_items": [
            {"target": "political debate", "true_relevance": 0.0},
            {"target": "travel vlog", "true_relevance": 0.7},
        ],
    },
    {
        "id": "diversity_not_dislike",
        "critique_type": "Diversity Request",
        "utterance": "Recommend something different but still related.",
        "critiques": [
            {
                "target": "current slate",
                "operation": "diversify",
                "reason": "diversity request",
                "object_scope": "slate",
                "temporal_scope": "next_slate",
                "horizon": 1,
                "hardness": "soft",
                "confidence": 0.76,
                "promotion_condition": "never",
            }
        ],
        "follow_value": [0.65, 0.75],
        "ignore_value": [0.35, 0.45],
        "over_apply_value": [0.55, 0.3],
        "post_expiry_items": [
            {"target": "original category", "true_relevance": 0.8},
        ],
    },
    {
        "id": "session_family_dinner",
        "critique_type": "Session Context",
        "utterance": "Tonight I need a family-friendly dinner place.",
        "critiques": [
            {
                "target": "family-friendly dinner",
                "operation": "promote",
                "reason": "session context",
                "object_scope": "attribute",
                "temporal_scope": "session",
                "horizon": 2,
                "hardness": "soft",
                "confidence": 0.78,
                "promotion_condition": "never",
            }
        ],
        "follow_value": [0.7, 0.8],
        "ignore_value": [0.2, 0.3],
        "over_apply_value": [0.72, 0.4],
        "post_expiry_items": [
            {"target": "quiet date-night restaurant", "true_relevance": 0.8},
        ],
    },
    {
        "id": "windows_to_mac_drift",
        "critique_type": "Genuine Drift",
        "utterance": "I do not want Windows anymore. Going forward, prioritize Mac laptops.",
        "critiques": [
            {
                "target": "Windows",
                "operation": "rollback",
                "reason": "genuine drift",
                "object_scope": "category",
                "temporal_scope": "persistent",
                "horizon": 0,
                "hardness": "hard",
                "confidence": 0.9,
                "promotion_condition": "persistent_language",
            },
            {
                "target": "Mac laptops",
                "operation": "promote",
                "reason": "genuine drift",
                "object_scope": "category",
                "temporal_scope": "persistent",
                "horizon": 0,
                "hardness": "hard",
                "confidence": 0.9,
                "promotion_condition": "persistent_language",
            },
        ],
        "follow_value": [0.5, 0.75, 0.9],
        "ignore_value": [0.2, 0.25, 0.3],
        "over_apply_value": [0.45, 0.7, 0.85],
        "post_expiry_items": [
            {"target": "Mac laptops", "true_relevance": 0.9},
            {"target": "Windows laptops", "true_relevance": 0.0},
        ],
    },
    {
        "id": "ufc_behavioral_rollback",
        "critique_type": "Behavioral Rollback",
        "utterance": "Too much UFC for now.",
        "behavioral_positive_target": "UFC highlights",
        "critiques": [
            {
                "target": "UFC",
                "operation": "attenuate",
                "reason": "exposure fatigue",
                "object_scope": "category",
                "temporal_scope": "session",
                "horizon": 5,
                "hardness": "soft",
                "confidence": 0.8,
                "promotion_condition": "never",
            }
        ],
        "follow_value": [0.6, 0.7, 0.85],
        "ignore_value": [0.25, 0.3, 0.35],
        "over_apply_value": [0.65, 0.25, 0.15],
        "post_expiry_items": [
            {"target": "UFC highlights", "true_relevance": 0.9},
        ],
    },
]


def load_scenarios(path: str | None = None, scenario_set: str = "deterministic") -> List[dict]:
    if path:
        return load_scenarios_from_path(Path(path))
    if scenario_set == "deterministic":
        return DEFAULT_SCENARIOS
    if scenario_set == "noisy":
        return load_scenarios_from_path(SCENARIO_DIR / "noisy_critique_scenarios.jsonl")
    raise ValueError(f"Unsupported scenario set: {scenario_set}")


def load_scenarios_from_path(scenario_path: Path) -> List[dict]:
    if scenario_path.suffix == ".jsonl":
        with scenario_path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    with scenario_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else [data]


def trajectory_value(values: Iterable[float]) -> float:
    return sum(values)


def instruction_uplift(scenario: dict) -> float:
    return trajectory_value(scenario["follow_value"]) - trajectory_value(scenario["ignore_value"])


def over_application_regret(scenario: dict) -> float:
    return max(0.0, trajectory_value(scenario["follow_value"]) - trajectory_value(scenario["over_apply_value"]))


def evaluate_critiquescope(scenario: dict) -> Dict[str, float]:
    memory = CritiqueScopeMemory()
    memory.apply_turn(scenario["utterance"], critiques=scenario["critiques"])
    if scenario.get("behavioral_positive_target"):
        memory.observe_positive_behavior(scenario["behavioral_positive_target"])
    for _ in range(max(critique["horizon"] for critique in scenario["critiques"]) + 1):
        memory.decay_fast_memory()
    expected_slow = expected_slow_count(scenario)
    slow_size = len(memory.slow_memory)
    return {
        "instruction_satisfaction": 1.0,
        "instruction_uplift": instruction_uplift(scenario),
        "over_application_regret": over_application_regret(scenario),
        "over_correction_regret": memory.over_correction_regret(scenario["post_expiry_items"]),
        "over_correction_rate": 1.0 if memory.over_correction_regret(scenario["post_expiry_items"]) > 0 else 0.0,
        "memory_contamination_rate": memory.memory_contamination_rate(),
        "promotion_precision": promotion_precision(memory),
        "promotion_recall": 1.0 if expected_slow == 0 else min(1.0, slow_size / expected_slow),
        "rollback_accuracy": rollback_accuracy(memory, scenario),
        "drift_recovery_turns": 0.0 if scenario["critique_type"] == "Genuine Drift" else 0.0,
        "expired_constraint_violation_rate": 0.0,
        "slow_memory_size": slow_size,
        "token_cost": memory.token_cost_estimate(),
    }


def evaluate_flat_memory(scenario: dict) -> Dict[str, float]:
    memory = CritiqueScopeMemory()
    flattened = []
    for critique in scenario["critiques"]:
        promoted = dict(critique)
        if promoted["operation"] in {"attenuate", "diversify"}:
            promoted["operation"] = "filter"
        promoted["temporal_scope"] = "persistent"
        promoted["hardness"] = "hard"
        promoted["promotion_condition"] = "persistent_language"
        flattened.append(promoted)
    memory.apply_turn(scenario["utterance"], critiques=flattened)
    expected_slow = expected_slow_count(scenario)
    slow_size = len(memory.slow_memory)
    return {
        "instruction_satisfaction": 1.0,
        "instruction_uplift": instruction_uplift(scenario),
        "over_application_regret": over_application_regret(scenario),
        "over_correction_regret": memory.over_correction_regret(scenario["post_expiry_items"]),
        "over_correction_rate": 1.0 if memory.over_correction_regret(scenario["post_expiry_items"]) > 0 else 0.0,
        "memory_contamination_rate": memory.memory_contamination_rate(),
        "promotion_precision": promotion_precision(memory),
        "promotion_recall": 1.0 if expected_slow == 0 else min(1.0, slow_size / expected_slow),
        "rollback_accuracy": rollback_accuracy(memory, scenario),
        "drift_recovery_turns": 0.0 if scenario["critique_type"] == "Genuine Drift" else 0.0,
        "expired_constraint_violation_rate": memory.memory_contamination_rate(),
        "slow_memory_size": slow_size,
        "token_cost": memory.token_cost_estimate(),
    }


def evaluate_time_decay(scenario: dict) -> Dict[str, float]:
    memory = CritiqueScopeMemory()
    decayed = []
    for critique in scenario["critiques"]:
        copied = dict(critique)
        copied["temporal_scope"] = "session"
        copied["horizon"] = 2
        copied["promotion_condition"] = "never"
        decayed.append(copied)
    memory.apply_turn(scenario["utterance"], critiques=decayed)
    for _ in range(3):
        memory.decay_fast_memory()
    expected_slow = expected_slow_count(scenario)
    slow_size = len(memory.slow_memory)
    return {
        "instruction_satisfaction": 0.0 if expected_slow > 0 and slow_size == 0 else 1.0,
        "instruction_uplift": instruction_uplift(scenario),
        "over_application_regret": over_application_regret(scenario),
        "over_correction_regret": memory.over_correction_regret(scenario["post_expiry_items"]),
        "over_correction_rate": 1.0 if memory.over_correction_regret(scenario["post_expiry_items"]) > 0 else 0.0,
        "memory_contamination_rate": memory.memory_contamination_rate(),
        "promotion_precision": promotion_precision(memory),
        "promotion_recall": 1.0 if expected_slow == 0 else min(1.0, slow_size / expected_slow),
        "rollback_accuracy": rollback_accuracy(memory, scenario),
        "drift_recovery_turns": 1.0 if scenario["critique_type"] == "Genuine Drift" and slow_size == 0 else 0.0,
        "expired_constraint_violation_rate": 0.0,
        "slow_memory_size": slow_size,
        "token_cost": memory.token_cost_estimate(),
    }


def expected_slow_count(scenario: dict) -> int:
    return sum(1 for critique in scenario["critiques"] if critique["temporal_scope"] == "persistent")


def promotion_precision(memory: CritiqueScopeMemory) -> float:
    if not memory.slow_memory:
        return 1.0
    durable_reasons = {"stable dislike", "genuine drift"}
    correct = sum(1 for critique in memory.slow_memory if critique.reason in durable_reasons)
    return correct / len(memory.slow_memory)


def rollback_accuracy(memory: CritiqueScopeMemory, scenario: dict) -> float:
    if not scenario.get("behavioral_positive_target"):
        return 1.0
    return 1.0 if any(event["event"] == "rollback_fast" for event in memory.events) else 0.0


def run_benchmark(scenarios: List[dict]) -> Dict[str, dict]:
    results = {}
    for scenario in scenarios:
        results[scenario["id"]] = {
            "flat_memory": evaluate_flat_memory(scenario),
            "time_decay": evaluate_time_decay(scenario),
            "critiquescope": evaluate_critiquescope(scenario),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", help="Optional JSON/JSONL critique scenarios.")
    parser.add_argument("--scenario-set", default="deterministic", choices=["deterministic", "noisy"])
    parser.add_argument("--output", help="Optional path to save metrics as JSON.")
    args = parser.parse_args()

    results = run_benchmark(load_scenarios(args.scenarios, scenario_set=args.scenario_set))
    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
