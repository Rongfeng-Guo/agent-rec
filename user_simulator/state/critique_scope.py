"""Scope-aware critique memory for interactive recommendation.

CritiqueScope treats user feedback as an exposure-conditioned intervention,
not as a stable preference label by default. Temporary complaints first enter
fast memory with an explicit scope and horizon. Only repeated, high-confidence,
or persistent critiques are promoted into slow memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


VALID_OPERATIONS = {"promote", "attenuate", "filter", "diversify", "explore", "rollback"}
VALID_OBJECT_SCOPES = {"item", "attribute", "category", "creator", "slate", "global"}
VALID_TEMPORAL_SCOPES = {"next_slate", "session", "contextual", "persistent"}
VALID_STATUSES = {"active_fast", "promoted_slow", "expired", "rolled_back"}


@dataclass
class Critique:
    target: str
    operation: str
    reason: str
    object_scope: str = "category"
    temporal_scope: str = "session"
    horizon: int = 5
    hardness: str = "soft"
    confidence: float = 0.7
    promotion_condition: Optional[str] = "repeat_across_sessions"
    source_turn: int = 0
    created_at_step: int = 0
    expires_at_step: Optional[int] = None
    status: str = "active_fast"
    created_turn: int = 0
    remaining_horizon: Optional[int] = None
    active: bool = True
    evidence_count: int = 1

    def __post_init__(self):
        self.temporal_scope = self.temporal_scope.replace("-", "_")
        if self.operation not in VALID_OPERATIONS:
            raise ValueError(f"Unsupported critique operation: {self.operation}")
        if self.object_scope not in VALID_OBJECT_SCOPES:
            raise ValueError(f"Unsupported object scope: {self.object_scope}")
        if self.temporal_scope not in VALID_TEMPORAL_SCOPES:
            raise ValueError(f"Unsupported temporal scope: {self.temporal_scope}")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Unsupported critique status: {self.status}")
        if self.source_turn == 0:
            self.source_turn = self.created_turn
        if self.created_at_step == 0:
            self.created_at_step = self.created_turn
        if self.remaining_horizon is None:
            self.remaining_horizon = self.horizon
        if self.expires_at_step is None and self.horizon is not None and not self.is_persistent:
            self.expires_at_step = self.created_at_step + self.horizon

    @property
    def is_persistent(self) -> bool:
        return self.temporal_scope == "persistent" or self.hardness == "hard"

    @property
    def key(self) -> str:
        return f"{self.operation}:{self.object_scope}:{self.target.lower()}"

    def to_prompt_line(self) -> str:
        return (
            f"- [{self.operation}/{self.object_scope}/{self.temporal_scope}/"
            f"h={self.remaining_horizon}/c={self.confidence:.2f}] "
            f"{self.target} | reason={self.reason}"
        )


class CritiqueScopeMemory:
    """Two-speed memory for scope-aware critique grounding."""

    def __init__(
        self,
        promotion_confidence: float = 0.84,
        promotion_evidence: int = 2,
        max_prompt_items: int = 12,
    ):
        self.promotion_confidence = promotion_confidence
        self.promotion_evidence = promotion_evidence
        self.max_prompt_items = max_prompt_items
        self.turn = 0
        self.fast_memory: List[Critique] = []
        self.slow_memory: List[Critique] = []
        self.events: List[dict] = []

    def reset(self):
        self.turn = 0
        self.fast_memory = []
        self.slow_memory = []
        self.events = []

    def apply_turn(self, user_utterance: str, critiques: Optional[Iterable[dict]] = None):
        self.turn += 1
        parsed = list(critiques) if critiques is not None else infer_critiques(user_utterance)
        for critique_data in parsed:
            self.add_critique(critique_data)
        self.decay_fast_memory()

    def add_critique(self, critique_data: dict) -> Critique:
        critique = Critique(created_turn=self.turn, **critique_data)
        existing = self._find_active_fast(critique.key)
        if existing:
            existing.evidence_count += 1
            existing.confidence = max(existing.confidence, critique.confidence)
            existing.remaining_horizon = max(existing.remaining_horizon or 0, critique.horizon)
            existing.reason = critique.reason or existing.reason
            critique = existing
            event_type = "refresh_fast"
        elif critique.is_persistent:
            critique.status = "promoted_slow"
            self.slow_memory.append(critique)
            event_type = "write_slow"
        else:
            critique.status = "active_fast"
            self.fast_memory.append(critique)
            event_type = "write_fast"

        if self.should_promote(critique):
            self.promote(critique)
            event_type = "promote_slow"

        self._record(event_type, critique)
        return critique

    def should_promote(self, critique: Critique) -> bool:
        if critique in self.slow_memory:
            return False
        if critique.promotion_condition == "never":
            return False
        if critique.promotion_condition == "persistent_language":
            return critique.temporal_scope == "persistent"
        return (
            critique.confidence >= self.promotion_confidence
            and critique.evidence_count >= self.promotion_evidence
        )

    def promote(self, critique: Critique):
        if critique in self.fast_memory:
            self.fast_memory.remove(critique)
        critique.temporal_scope = "persistent"
        critique.remaining_horizon = None
        critique.active = True
        critique.status = "promoted_slow"
        critique.expires_at_step = None
        self.slow_memory.append(critique)

    def observe_positive_behavior(self, target: str):
        """Rollback temporary attenuation when the user re-engages with a target."""
        target_lower = target.lower()
        for critique in self.fast_memory:
            if critique.active and critique.operation == "attenuate" and critique.target.lower() in target_lower:
                critique.active = False
                critique.status = "rolled_back"
                self._record("rollback_fast", critique)

    def decay_fast_memory(self):
        for critique in self.fast_memory:
            if not critique.active or critique.remaining_horizon is None:
                continue
            if critique.temporal_scope == "next_slate":
                critique.remaining_horizon -= 1
            elif critique.temporal_scope in {"session", "contextual"}:
                critique.remaining_horizon -= 1
            if critique.remaining_horizon <= 0:
                critique.active = False
                critique.status = "expired"
                self._record("expire_fast", critique)

    def end_session(self):
        for critique in self.fast_memory:
            if critique.temporal_scope in {"session", "contextual"}:
                critique.active = False
                critique.status = "expired"
                self._record("expire_session", critique)

    def active_fast(self) -> List[Critique]:
        return [critique for critique in self.fast_memory if critique.active]

    def active_slow(self) -> List[Critique]:
        return [critique for critique in self.slow_memory if critique.active]

    def memory_contamination_rate(self) -> float:
        if not self.slow_memory:
            return 0.0
        contaminated = [
            critique
            for critique in self.slow_memory
            if critique.reason in {"exposure fatigue", "diversity request", "session context"}
        ]
        return len(contaminated) / len(self.slow_memory)

    def over_correction_regret(self, recommendation_values: Iterable[dict]) -> float:
        regret = 0.0
        for value in recommendation_values:
            target = value.get("target", "").lower()
            is_relevant = float(value.get("true_relevance", 0.0))
            suppressed = any(
                critique.active
                and critique.operation in {"attenuate", "filter"}
                and critique.target.lower() in target
                for critique in self.active_slow()
            )
            if suppressed:
                regret += is_relevant
        return regret

    def to_prompt_context(self) -> str:
        lines = ["CritiqueScope memory:"]
        fast = self.active_fast()[: self.max_prompt_items]
        slow = self.active_slow()[: self.max_prompt_items]
        if fast:
            lines.append("Fast memory: temporary critique interventions")
            lines.extend(critique.to_prompt_line() for critique in fast)
        if slow:
            lines.append("Slow memory: durable user constraints/preferences")
            lines.extend(critique.to_prompt_line() for critique in slow)
        if not fast and not slow:
            lines.append("No active critique state yet.")
        lines.append(
            "Apply fast memory as scoped interventions; only slow memory should change the durable user profile."
        )
        return "\n".join(lines)

    def token_cost_estimate(self) -> int:
        return max(1, len(self.to_prompt_context().split()))

    def _find_active_fast(self, key: str) -> Optional[Critique]:
        for critique in self.fast_memory:
            if critique.active and critique.key == key:
                return critique
        return None

    def _record(self, event_type: str, critique: Critique):
        self.events.append(
            {
                "turn": self.turn,
                "event": event_type,
                "target": critique.target,
                "operation": critique.operation,
                "object_scope": critique.object_scope,
                "temporal_scope": critique.temporal_scope,
                "confidence": critique.confidence,
                "active": critique.active,
                "remaining_horizon": critique.remaining_horizon,
                "status": critique.status,
                "source_turn": critique.source_turn,
                "created_at_step": critique.created_at_step,
                "expires_at_step": critique.expires_at_step,
            }
        )


def infer_critiques(user_utterance: str) -> List[dict]:
    """Small cue-based parser for smoke tests.

    Full experiments should replace this with an LLM parser that emits the
    schema described in docs/critiquescope_gimo.md.
    """

    text = user_utterance.strip()
    lower = text.lower()
    critiques: List[dict] = []

    if ("windows" in lower and "mac" in lower) or ("不要 windows" in lower and "mac" in lower):
        critiques.extend(
            [
                {
                    "target": "Windows",
                    "operation": "rollback",
                    "reason": "genuine drift",
                    "object_scope": "category",
                    "temporal_scope": "persistent",
                    "horizon": 0,
                    "hardness": "hard",
                    "confidence": 0.86,
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
                    "confidence": 0.86,
                    "promotion_condition": "persistent_language",
                },
            ]
        )
    elif any(cue in lower for cue in ["too much", "too many", "看太多", "换换口味", "少来点"]):
        critiques.append(
            {
                "target": extract_target(text),
                "operation": "attenuate",
                "reason": "exposure fatigue",
                "object_scope": "category",
                "temporal_scope": "session",
                "horizon": 5,
                "hardness": "soft",
                "confidence": 0.78,
                "promotion_condition": "never",
            }
        )
    elif any(cue in lower for cue in ["never", "以后不要", "do not recommend", "不要再"]):
        critiques.append(
            {
                "target": extract_target(text),
                "operation": "filter",
                "reason": "stable dislike",
                "object_scope": "category",
                "temporal_scope": "persistent",
                "horizon": 0,
                "hardness": "hard",
                "confidence": 0.88,
                "promotion_condition": "persistent_language",
            }
        )
    elif any(cue in lower for cue in ["different", "不一样", "diverse", "换点"]):
        critiques.append(
            {
                "target": "current slate",
                "operation": "diversify",
                "reason": "diversity request",
                "object_scope": "slate",
                "temporal_scope": "next_slate",
                "horizon": 1,
                "hardness": "soft",
                "confidence": 0.72,
                "promotion_condition": "never",
            }
        )
    elif any(cue in lower for cue in ["today", "tonight", "今天", "今晚"]):
        critiques.append(
            {
                "target": extract_target(text),
                "operation": "promote",
                "reason": "session context",
                "object_scope": "attribute",
                "temporal_scope": "session",
                "horizon": 6,
                "hardness": "soft",
                "confidence": 0.74,
                "promotion_condition": "never",
            }
        )
    return critiques


def extract_target(text: str) -> str:
    lowered = text.lower()
    for marker in ["ufc", "politics", "political", "sweet", "dessert", "甜食", "政治"]:
        if marker in lowered:
            return marker
    return text
