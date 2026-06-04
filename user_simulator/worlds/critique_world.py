"""CritiqueWorld: a deterministic closed-loop testbed for recommendation memory.

The world is intentionally small and transparent. It does not claim to simulate
real users; it exposes a controlled latent state so memory, reranking, and
counterfactual branch behavior can be audited without API calls or model runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from random import Random
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Item:
    item_id: str
    category: str
    attributes: Dict[str, str | float | List[str]]
    base_quality: float
    novelty_group: Optional[str] = None


@dataclass
class LatentUserState:
    stable_positive: Dict[str, Any] = field(default_factory=dict)
    stable_negative: Dict[str, Any] = field(default_factory=dict)
    contextual_positive: Dict[str, Any] = field(default_factory=dict)
    contextual_negative: Dict[str, Any] = field(default_factory=dict)
    drift_positive: Dict[str, Any] = field(default_factory=dict)
    drift_negative: Dict[str, Any] = field(default_factory=dict)
    exposure_counts: Dict[str, int] = field(default_factory=dict)
    category_exposure_counts: Dict[str, int] = field(default_factory=dict)
    clicked_items: List[str] = field(default_factory=list)
    skipped_items: List[str] = field(default_factory=list)
    patience: float = 1.0
    session_id: str = "session-0"
    turn: int = 0
    active: bool = True

    def reset_session(self, session_id: Optional[str] = None):
        self.contextual_positive = {}
        self.contextual_negative = {}
        self.category_exposure_counts = {}
        self.patience = 1.0
        self.session_id = session_id or f"{self.session_id}-next"
        self.active = True

    def snapshot(self) -> dict:
        return asdict(self)


@dataclass
class UtilityBreakdown:
    stable_match: float
    context_match: float
    drift_match: float
    negative_penalty: float
    fatigue_penalty: float
    novelty_bonus: float
    base_quality: float
    total: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CritiqueWorldConfig:
    stable_weight: float = 1.0
    context_weight: float = 0.8
    drift_weight: float = 1.15
    negative_weight: float = 1.25
    fatigue_weight: float = 0.28
    novelty_weight: float = 0.18
    low_slate_threshold: float = 0.7
    critique_threshold: float = 0.58
    leave_threshold: float = 0.18
    patience_decay: float = 0.22
    patience_recovery: float = 0.08
    click_temperature: float = 0.08
    default_branch_horizon: int = 5


def _values(preferences: Dict[str, Any], key: str) -> List[str]:
    value = preferences.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).lower() for item in value]
    return [str(value).lower()]


def _matches(item: Item, preferences: Dict[str, Any]) -> float:
    score = 0.0
    category = item.category.lower()
    if category in _values(preferences, "category"):
        score += 1.0
    for key, raw_value in item.attributes.items():
        preferred = _values(preferences, key)
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        lower_values = {str(value).lower() for value in values}
        if preferred and any(value in lower_values for value in preferred):
            score += 1.0
    return score


def score_item_utility(
    item: Item,
    user_state: LatentUserState,
    config: CritiqueWorldConfig | None = None,
) -> UtilityBreakdown:
    config = config or CritiqueWorldConfig()
    stable_match = config.stable_weight * _matches(item, user_state.stable_positive)
    context_match = config.context_weight * _matches(item, user_state.contextual_positive)
    drift_match = config.drift_weight * _matches(item, user_state.drift_positive)
    negative_penalty = config.negative_weight * (
        _matches(item, user_state.stable_negative)
        + _matches(item, user_state.contextual_negative)
        + _matches(item, user_state.drift_negative)
    )
    exposure = user_state.category_exposure_counts.get(item.category, 0)
    fatigue_penalty = config.fatigue_weight * max(0, exposure - 1)
    seen_group = bool(item.novelty_group and user_state.exposure_counts.get(item.novelty_group, 0))
    novelty_bonus = 0.0 if seen_group else config.novelty_weight
    total = (
        item.base_quality
        + stable_match
        + context_match
        + drift_match
        + novelty_bonus
        - negative_penalty
        - fatigue_penalty
    )
    return UtilityBreakdown(
        stable_match=stable_match,
        context_match=context_match,
        drift_match=drift_match,
        negative_penalty=negative_penalty,
        fatigue_penalty=fatigue_penalty,
        novelty_bonus=novelty_bonus,
        base_quality=item.base_quality,
        total=total,
    )


def deterministic_critique_for_slate(slate: Iterable[Item], user_state: LatentUserState) -> dict | None:
    categories = [item.category for item in slate]
    if not categories:
        return None
    most_common = max(set(categories), key=categories.count)
    if user_state.category_exposure_counts.get(most_common, 0) >= 3:
        return {
            "utterance": f"I have seen too much {most_common} lately. Switch it up for a bit.",
            "critiques": [
                {
                    "target": most_common,
                    "operation": "attenuate",
                    "reason": "exposure fatigue",
                    "object_scope": "category",
                    "temporal_scope": "session",
                    "horizon": 3,
                    "hardness": "soft",
                    "confidence": 0.78,
                    "promotion_condition": "never",
                }
            ],
        }
    return None


def simulate_user_response(
    slate: List[Item],
    user_state: LatentUserState,
    rng: Random,
    config: CritiqueWorldConfig | None = None,
) -> dict:
    config = config or CritiqueWorldConfig()
    if not user_state.active:
        return {"action": "leave", "item_id": None, "utility": 0.0, "critique": None}

    scored = [(item, score_item_utility(item, user_state, config).total) for item in slate]
    best_item, best_utility = max(scored, key=lambda pair: pair[1])
    avg_utility = sum(value for _, value in scored) / max(1, len(scored))

    for item in slate:
        user_state.exposure_counts[item.item_id] = user_state.exposure_counts.get(item.item_id, 0) + 1
        if item.novelty_group:
            user_state.exposure_counts[item.novelty_group] = user_state.exposure_counts.get(item.novelty_group, 0) + 1
        user_state.category_exposure_counts[item.category] = (
            user_state.category_exposure_counts.get(item.category, 0) + 1
        )

    if avg_utility < config.low_slate_threshold:
        user_state.patience = max(0.0, user_state.patience - config.patience_decay)
    else:
        user_state.patience = min(1.0, user_state.patience + config.patience_recovery)

    if user_state.patience < config.leave_threshold:
        user_state.active = False
        action = {"action": "leave", "item_id": None, "utility": avg_utility, "critique": None}
    elif user_state.patience < config.critique_threshold:
        action = {
            "action": "critique",
            "item_id": None,
            "utility": avg_utility,
            "critique": deterministic_critique_for_slate(slate, user_state),
        }
    else:
        click_cutoff = max(config.low_slate_threshold, avg_utility + rng.uniform(-config.click_temperature, config.click_temperature))
        if best_utility >= click_cutoff:
            user_state.clicked_items.append(best_item.item_id)
            action = {
                "action": "click",
                "item_id": best_item.item_id,
                "category": best_item.category,
                "utility": best_utility,
                "critique": None,
            }
        else:
            user_state.skipped_items.extend(item.item_id for item in slate)
            action = {"action": "skip", "item_id": None, "utility": avg_utility, "critique": None}

    user_state.turn += 1
    return action
