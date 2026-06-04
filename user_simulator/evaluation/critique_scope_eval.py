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
                "temporal_scope": "next-slate",
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


def trajectory_value(values: Iterable[float]) -> float:
    return sum(values)


def instruction_uplift(scenario: dict) -> float:
    return trajectory_value(scenario["follow_value"]) - trajectory_value(scenario["ignore_value"])


def over_application_regret(scenario: dict) -> float:
    return max(0.0, trajectory_value(scenario["follow_value"]) - trajectory_value(scenario["over_apply_value"]))


def evaluate_critiquescope(scenario: dict) -> Dict[str, float]:
    memory = CritiqueScopeMemory()
    memory.apply_turn(scenario["utterance"], critiques=scenario["critiques"])
    for _ in range(max(critique["horizon"] for critique in scenario["critiques"]) + 1):
        memory.decay_fast_memory()
    return {
        "instruction_uplift": instruction_uplift(scenario),
        "over_application_regret": over_application_regret(scenario),
        "over_correction_regret": memory.over_correction_regret(scenario["post_expiry_items"]),
        "memory_contamination_rate": memory.memory_contamination_rate(),
        "slow_memory_size": len(memory.slow_memory),
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
    return {
        "instruction_uplift": instruction_uplift(scenario),
        "over_application_regret": over_application_regret(scenario),
        "over_correction_regret": memory.over_correction_regret(scenario["post_expiry_items"]),
        "memory_contamination_rate": memory.memory_contamination_rate(),
        "slow_memory_size": len(memory.slow_memory),
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
    return {
        "instruction_uplift": instruction_uplift(scenario),
        "over_application_regret": over_application_regret(scenario),
        "over_correction_regret": memory.over_correction_regret(scenario["post_expiry_items"]),
        "memory_contamination_rate": memory.memory_contamination_rate(),
        "slow_memory_size": len(memory.slow_memory),
        "token_cost": memory.token_cost_estimate(),
    }


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
    parser.add_argument("--output", help="Optional path to save metrics as JSON.")
    args = parser.parse_args()

    results = run_benchmark(load_scenarios(args.scenarios))
    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
