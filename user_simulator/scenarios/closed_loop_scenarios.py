"""Deterministic closed-loop scenarios for CritiqueWorld."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from user_simulator.worlds.critique_world import Item, LatentUserState


@dataclass
class Scenario:
    name: str
    items: List[Item]
    initial_user_state: LatentUserState
    injected_events: List[dict]
    max_turns: int = 12
    top_k: int = 5
    expected_properties: Dict[str, Any] = field(default_factory=dict)


def _catalog() -> List[Item]:
    return [
        Item("ufc_1", "UFC", {"tone": "intense", "platform": "video"}, 0.86, "combat"),
        Item("ufc_2", "UFC", {"tone": "technical", "platform": "video"}, 0.82, "combat"),
        Item("boxing_1", "Boxing", {"tone": "intense", "platform": "video"}, 0.78, "combat"),
        Item("fitness_1", "Fitness", {"tone": "practical", "platform": "video"}, 0.74, "fitness"),
        Item("politics_1", "Politics", {"tone": "debate", "platform": "article"}, 0.82, "news"),
        Item("science_1", "Science", {"tone": "calm", "platform": "article"}, 0.77, "science"),
        Item("jazz_1", "Jazz", {"tone": "calm", "platform": "audio"}, 0.70, "music"),
        Item("family_restaurant_1", "Restaurant", {"occasion": "family", "cuisine": "Chinese"}, 0.84, "food_family"),
        Item("bar_1", "Restaurant", {"occasion": "friends", "cuisine": "Bar"}, 0.76, "food_bar"),
        Item("windows_1", "Windows", {"device": "laptop", "os": "Windows"}, 0.83, "laptop_windows"),
        Item("windows_2", "Windows", {"device": "desktop", "os": "Windows"}, 0.78, "laptop_windows"),
        Item("mac_1", "Mac", {"device": "laptop", "os": "Mac"}, 0.84, "laptop_mac"),
        Item("mac_2", "Mac", {"device": "desktop", "os": "Mac"}, 0.76, "laptop_mac"),
        Item("linux_1", "Linux", {"device": "laptop", "os": "Linux"}, 0.73, "laptop_linux"),
    ]


def _critique(target: str, operation: str, reason: str, temporal_scope: str, horizon: int, hardness: str = "soft") -> dict:
    return {
        "target": target,
        "operation": operation,
        "reason": reason,
        "object_scope": "category" if target != "current slate" else "slate",
        "temporal_scope": temporal_scope,
        "horizon": horizon,
        "hardness": hardness,
        "confidence": 0.9 if temporal_scope == "persistent" else 0.78,
        "promotion_condition": "persistent_language" if temporal_scope == "persistent" else "never",
    }


def temporary_fatigue() -> Scenario:
    return Scenario(
        name="temporary_fatigue",
        items=_catalog(),
        initial_user_state=LatentUserState(stable_positive={"category": ["UFC", "Boxing"]}),
        injected_events=[
            {"turn": 2, "type": "critique", "utterance": "I have seen too much UFC lately. Switch it up for a bit.", "critiques": [_critique("UFC", "attenuate", "exposure fatigue", "session", 3)]}
        ],
        expected_properties={"temporary_target": "UFC", "horizon": 3},
    )


def stable_dislike() -> Scenario:
    return Scenario(
        name="stable_dislike",
        items=_catalog(),
        initial_user_state=LatentUserState(stable_positive={"category": ["Science", "Politics"]}),
        injected_events=[
            {
                "turn": 1,
                "type": "critique",
                "utterance": "Please never recommend political content to me.",
                "critiques": [_critique("Politics", "filter", "stable dislike", "persistent", 0, "hard")],
                "state_update": {"stable_positive": {"category": ["Science"]}, "stable_negative": {"category": ["Politics"]}},
            },
            {"turn": 5, "type": "session_reset"},
        ],
        expected_properties={"persistent_target": "Politics"},
    )


def diversity_request() -> Scenario:
    return Scenario(
        name="diversity_request",
        items=_catalog(),
        initial_user_state=LatentUserState(stable_positive={"category": ["UFC", "Boxing"]}),
        injected_events=[
            {"turn": 1, "type": "critique", "utterance": "Recommend something different but still related.", "critiques": [_critique("current slate", "diversify", "diversity request", "next_slate", 1)]}
        ],
        expected_properties={"must_not_pollute": "UFC"},
    )


def session_context() -> Scenario:
    return Scenario(
        name="session_context",
        items=_catalog(),
        initial_user_state=LatentUserState(stable_positive={"category": ["Science", "Restaurant"]}),
        injected_events=[
            {
                "turn": 1,
                "type": "critique",
                "utterance": "Tonight I need a family-friendly dinner place.",
                "critiques": [{"target": "family", "operation": "promote", "reason": "session context", "object_scope": "attribute", "temporal_scope": "session", "horizon": 4, "hardness": "soft", "confidence": 0.74, "promotion_condition": "never"}],
            },
            {"turn": 5, "type": "session_reset"},
        ],
        expected_properties={"session_target": "family"},
    )


def genuine_drift() -> Scenario:
    return Scenario(
        name="genuine_drift",
        items=_catalog(),
        initial_user_state=LatentUserState(stable_positive={"category": ["Windows"]}),
        injected_events=[
            {
                "turn": 2,
                "type": "drift",
                "utterance": "I do not want Windows anymore. Going forward, prioritize Mac laptops.",
                "critiques": [
                    _critique("Windows", "rollback", "genuine drift", "persistent", 0, "hard"),
                    _critique("Mac", "promote", "genuine drift", "persistent", 0, "hard"),
                ],
                "state_update": {"drift_positive": {"category": ["Mac"]}, "drift_negative": {"category": ["Windows"]}},
            }
        ],
        expected_properties={"drift_target": "Mac", "old_target": "Windows"},
    )


def behavioral_rollback() -> Scenario:
    return Scenario(
        name="behavioral_rollback",
        items=_catalog(),
        initial_user_state=LatentUserState(stable_positive={"category": ["UFC"]}),
        injected_events=[
            {"turn": 1, "type": "critique", "utterance": "I have seen too much UFC lately. Switch it up for a bit.", "critiques": [_critique("UFC", "attenuate", "exposure fatigue", "session", 5)]},
            {"turn": 3, "type": "behavioral_confirmation", "target": "UFC"},
        ],
        expected_properties={"rollback_target": "UFC"},
    )


def mixed_multi_turn() -> Scenario:
    scenario = temporary_fatigue()
    scenario.name = "mixed_multi_turn"
    scenario.injected_events.extend(
        [
            {
                "turn": 5,
                "type": "critique",
                "utterance": "Tonight I need a family-friendly dinner place.",
                "critiques": [{"target": "family", "operation": "promote", "reason": "session context", "object_scope": "attribute", "temporal_scope": "session", "horizon": 3, "hardness": "soft", "confidence": 0.74, "promotion_condition": "never"}],
            },
            {
                "turn": 7,
                "type": "drift",
                "utterance": "I do not want Windows anymore. Going forward, prioritize Mac laptops.",
                "critiques": [_critique("Windows", "rollback", "genuine drift", "persistent", 0, "hard"), _critique("Mac", "promote", "genuine drift", "persistent", 0, "hard")],
                "state_update": {"drift_positive": {"category": ["Mac"]}, "drift_negative": {"category": ["Windows"]}},
            },
        ]
    )
    return scenario


FACTORIES: Dict[str, Callable[[], Scenario]] = {
    "temporary_fatigue": temporary_fatigue,
    "stable_dislike": stable_dislike,
    "diversity_request": diversity_request,
    "session_context": session_context,
    "genuine_drift": genuine_drift,
    "behavioral_rollback": behavioral_rollback,
    "mixed_multi_turn": mixed_multi_turn,
}


def list_scenarios(names: str | List[str] = "all") -> List[Scenario]:
    if names == "all" or names == ["all"]:
        return [factory() for factory in FACTORIES.values()]
    selected = names if isinstance(names, list) else [names]
    return [get_scenario(name) for name in selected]


def get_scenario(name: str) -> Scenario:
    try:
        return FACTORIES[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown closed-loop scenario: {name}") from exc
