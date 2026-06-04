"""Memory-aware reranking policy for CritiqueWorld."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from user_simulator.state.critique_scope import CritiqueScopeMemory
from user_simulator.state.structured_memory import StructuredMemory
from user_simulator.worlds.critique_world import CritiqueWorldConfig, Item, LatentUserState, score_item_utility


@dataclass
class RankedSlate:
    slate: List[Item]
    scores: Dict[str, float]
    score_breakdowns: Dict[str, dict]
    applied_interventions: List[dict]

    def to_dict(self) -> dict:
        return {
            "slate": [item.item_id for item in self.slate],
            "scores": self.scores,
            "score_breakdowns": self.score_breakdowns,
            "applied_interventions": self.applied_interventions,
        }


def _target_matches(item: Item, target: str, object_scope: str = "category") -> bool:
    target_lower = target.lower()
    if object_scope in {"category", "global"} and target_lower in item.category.lower():
        return True
    if object_scope == "item" and target_lower in item.item_id.lower():
        return True
    for value in item.attributes.values():
        values = value if isinstance(value, list) else [value]
        if any(target_lower in str(candidate).lower() for candidate in values):
            return True
    return False


def _diversify_adjustment(
    score: float,
    item: Item,
    user_state: LatentUserState,
    intervention: dict,
) -> tuple[float, bool, dict]:
    exposure = user_state.category_exposure_counts.get(item.category, 0)
    recent_penalty_weight = float(intervention.get("penalty", 0.35))
    novelty_weight = float(intervention.get("bonus", 0.75))
    recent_exposure_penalty = recent_penalty_weight * max(0, exposure - 1)
    diversity_bonus = 0.0

    if exposure == 0:
        diversity_bonus += novelty_weight
    elif exposure == 1:
        diversity_bonus += novelty_weight * 0.35

    if item.novelty_group and not user_state.exposure_counts.get(item.novelty_group, 0):
        diversity_bonus += novelty_weight * 0.5

    intervention_score_delta = diversity_bonus - recent_exposure_penalty
    adjusted = score + intervention_score_delta
    return adjusted, True, {
        "diversity_bonus": diversity_bonus,
        "recent_exposure_penalty": recent_exposure_penalty,
        "intervention_score_delta": intervention_score_delta,
    }


def _apply_intervention(
    score: float,
    item: Item,
    user_state: LatentUserState,
    intervention: dict,
) -> tuple[float, bool, dict]:
    operation = intervention.get("operation")
    target = intervention.get("target", "")
    object_scope = intervention.get("object_scope", "category")
    if operation == "diversify":
        return _diversify_adjustment(score, item, user_state, intervention)
    if not _target_matches(item, target, object_scope):
        return score, False, {
            "diversity_bonus": 0.0,
            "recent_exposure_penalty": 0.0,
            "intervention_score_delta": 0.0,
        }
    if operation == "filter":
        penalty = float(intervention.get("penalty", 100.0))
        return score - penalty, True, {
            "diversity_bonus": 0.0,
            "recent_exposure_penalty": 0.0,
            "intervention_score_delta": -penalty,
        }
    if operation == "attenuate":
        penalty = float(intervention.get("penalty", 1.6))
        return score - penalty, True, {
            "diversity_bonus": 0.0,
            "recent_exposure_penalty": 0.0,
            "intervention_score_delta": -penalty,
        }
    if operation == "rollback":
        penalty = float(intervention.get("penalty", 1.0))
        return score - penalty, True, {
            "diversity_bonus": 0.0,
            "recent_exposure_penalty": 0.0,
            "intervention_score_delta": -penalty,
        }
    if operation == "promote":
        bonus = float(intervention.get("bonus", 1.2))
        return score + bonus, True, {
            "diversity_bonus": 0.0,
            "recent_exposure_penalty": 0.0,
            "intervention_score_delta": bonus,
        }
    return score, False, {
        "diversity_bonus": 0.0,
        "recent_exposure_penalty": 0.0,
        "intervention_score_delta": 0.0,
    }


def _flat_interventions(memory: list[dict]) -> List[dict]:
    interventions = []
    for critique in memory:
        operation = critique.get("operation")
        if operation in {"attenuate", "diversify", "explore"}:
            interventions.append({**critique, "operation": "filter", "penalty": 3.0, "source": "flat_over_apply"})
        elif operation == "promote":
            interventions.append({**critique, "bonus": 1.2, "source": "flat"})
        else:
            interventions.append({**critique, "source": "flat"})
    return interventions


def _structured_interventions(memory: StructuredMemory) -> List[dict]:
    interventions = []
    for slot in memory.active_slots():
        operation = "filter" if slot.bucket in {"negative", "hard"} else "promote"
        interventions.append(
            {
                "target": slot.value,
                "operation": operation,
                "object_scope": "category",
                "penalty": 2.5,
                "bonus": 0.9,
                "source": "structured",
            }
        )
    return interventions


def _time_decay_interventions(memory: list[dict], current_turn: int) -> List[dict]:
    interventions = []
    for critique in memory:
        age = max(0, current_turn - int(critique.get("turn", 0)))
        strength = max(0.25, 1.0 - 0.18 * age)
        operation = critique.get("operation")
        if operation in {"attenuate", "filter", "rollback"}:
            interventions.append({**critique, "penalty": 1.7 * strength, "source": "time_decay"})
        elif operation == "diversify":
            interventions.append({**critique, "operation": "diversify", "bonus": 0.6 * strength, "source": "time_decay"})
        else:
            interventions.append({**critique, "bonus": 1.0 * strength, "source": "time_decay"})
    return interventions


def _critiquescope_interventions(memory: CritiqueScopeMemory) -> List[dict]:
    interventions = []
    for critique in memory.active_slow():
        operation = "filter" if critique.operation in {"filter", "rollback"} else critique.operation
        interventions.append(
            {
                "target": critique.target,
                "operation": operation,
                "object_scope": critique.object_scope,
                "penalty": 100.0 if operation == "filter" else 2.0,
                "bonus": 1.2,
                "source": "critiquescope_slow",
                "temporal_scope": critique.temporal_scope,
            }
        )
    for critique in memory.active_fast():
        if critique.operation == "diversify":
            interventions.append(
                {
                    "target": critique.target,
                    "operation": "diversify",
                    "object_scope": critique.object_scope,
                    "penalty": 0.35,
                    "bonus": 0.75,
                    "source": "critiquescope_fast",
                    "temporal_scope": critique.temporal_scope,
                }
            )
        else:
            interventions.append(
                {
                    "target": critique.target,
                    "operation": critique.operation,
                    "object_scope": critique.object_scope,
                    "penalty": 1.7,
                    "bonus": 1.0,
                    "source": "critiquescope_fast",
                    "temporal_scope": critique.temporal_scope,
                }
            )
    return interventions


def interventions_for_mode(memory: Any, memory_mode: str, current_turn: int) -> List[dict]:
    if memory_mode == "none" or memory is None:
        return []
    if memory_mode == "flat":
        return _flat_interventions(memory)
    if memory_mode == "structured":
        return _structured_interventions(memory)
    if memory_mode == "time_decay":
        return _time_decay_interventions(memory, current_turn)
    if memory_mode == "critiquescope":
        return _critiquescope_interventions(memory)
    raise ValueError(f"Unsupported memory mode: {memory_mode}")


def rank_items(
    items: Iterable[Item],
    user_state: LatentUserState,
    memory: Any,
    memory_mode: str,
    top_k: int,
    config: CritiqueWorldConfig | None = None,
) -> RankedSlate:
    config = config or CritiqueWorldConfig()
    interventions = interventions_for_mode(memory, memory_mode, user_state.turn)
    scored = []
    for item in items:
        breakdown = score_item_utility(item, user_state, config).to_dict()
        base_score = breakdown["total"]
        score = base_score
        breakdown["base_score"] = base_score
        breakdown["diversity_bonus"] = 0.0
        breakdown["recent_exposure_penalty"] = 0.0
        breakdown["intervention_score_delta"] = 0.0
        item_interventions = []
        for intervention in interventions:
            score, applied, delta_fields = _apply_intervention(score, item, user_state, intervention)
            breakdown["diversity_bonus"] += float(delta_fields["diversity_bonus"])
            breakdown["recent_exposure_penalty"] += float(delta_fields["recent_exposure_penalty"])
            breakdown["intervention_score_delta"] += float(delta_fields["intervention_score_delta"])
            if applied:
                item_interventions.append({"item_id": item.item_id, **intervention, **delta_fields})
        breakdown["final_score"] = score
        scored.append((item, score, breakdown, item_interventions))

    base_ranks = {
        item.item_id: index + 1
        for index, (item, _, _, _) in enumerate(sorted(scored, key=lambda row: (-row[2]["base_score"], row[0].item_id)))
    }
    scored.sort(key=lambda row: (-row[1], row[0].item_id))
    for index, (item, _, breakdown, _) in enumerate(scored, start=1):
        breakdown["rank_before"] = base_ranks[item.item_id]
        breakdown["rank_after"] = index
    selected = scored[:top_k]
    return RankedSlate(
        slate=[row[0] for row in selected],
        scores={row[0].item_id: row[1] for row in selected},
        score_breakdowns={row[0].item_id: row[2] for row in selected},
        applied_interventions=[event for row in selected for event in row[3]],
    )
