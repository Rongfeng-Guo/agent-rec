"""Structured preference memory for drift-aware recommendation dialogs.

The memory separates user signals into positive preferences, negative
preferences, hard constraints, and soft preferences. Each slot keeps a
confidence score and the turn where it was last updated so drift can be
evaluated explicitly instead of being hidden in the full dialogue history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


VALID_BUCKETS = {"positive", "negative", "hard", "soft"}


@dataclass
class PreferenceSlot:
    """One normalized user preference or constraint."""

    key: str
    value: str
    bucket: str = "soft"
    confidence: float = 0.6
    turn: int = 0
    source: str = ""
    active: bool = True
    supersedes: List[str] = field(default_factory=list)

    @property
    def slot_id(self) -> str:
        return f"{self.bucket}:{self.key}"

    def to_prompt_line(self) -> str:
        status = "active" if self.active else "inactive"
        return (
            f"- [{self.bucket}/{status}/c={self.confidence:.2f}/t={self.turn}] "
            f"{self.key}: {self.value}"
        )


class StructuredMemory:
    """Maintains explicit preference state under multi-turn interest drift."""

    def __init__(
        self,
        decay: float = 0.92,
        conflict_threshold: float = 0.55,
        max_prompt_items: int = 12,
    ):
        self.decay = decay
        self.conflict_threshold = conflict_threshold
        self.max_prompt_items = max_prompt_items
        self.turn = 0
        self.slots: Dict[str, PreferenceSlot] = {}
        self.events: List[dict] = []

    def reset(self):
        self.turn = 0
        self.slots = {}
        self.events = []

    def active_slots(self) -> List[PreferenceSlot]:
        return sorted(
            [slot for slot in self.slots.values() if slot.active],
            key=lambda slot: (slot.bucket != "hard", -slot.confidence, -slot.turn, slot.key),
        )

    def update(
        self,
        bucket: str,
        key: str,
        value: str,
        operation: str = "merge",
        confidence: float = 0.7,
        source: str = "",
    ) -> PreferenceSlot:
        if bucket not in VALID_BUCKETS:
            raise ValueError(f"Unsupported memory bucket: {bucket}")
        if operation not in {"retain", "merge", "overwrite", "forget"}:
            raise ValueError(f"Unsupported memory operation: {operation}")

        self.turn += 1
        slot_id = f"{bucket}:{key}"
        existing = self.slots.get(slot_id)

        if operation == "forget":
            if existing:
                existing.active = False
                existing.confidence = max(0.0, existing.confidence * self.decay - 0.2)
                existing.turn = self.turn
                self._record(operation, existing)
                return existing
            slot = PreferenceSlot(
                key=key,
                value=value,
                bucket=bucket,
                confidence=0.0,
                turn=self.turn,
                source=source,
                active=False,
            )
            self.slots[slot_id] = slot
            self._record(operation, slot)
            return slot

        if operation == "retain" and existing:
            existing.confidence = min(1.0, existing.confidence + 0.08)
            existing.turn = self.turn
            self._record(operation, existing)
            return existing

        if operation == "overwrite" and existing:
            existing.active = False
            existing.confidence = max(0.0, existing.confidence * self.decay - 0.1)
            old_slot_id = existing.slot_id
            slot = PreferenceSlot(
                key=key,
                value=value,
                bucket=bucket,
                confidence=confidence,
                turn=self.turn,
                source=source,
                active=True,
                supersedes=[old_slot_id],
            )
            self.slots[slot_id] = slot
            self._record(operation, slot)
            return slot

        if existing:
            existing.value = self._merge_values(existing.value, value)
            existing.confidence = min(1.0, max(existing.confidence * self.decay, confidence))
            existing.turn = self.turn
            existing.source = source or existing.source
            existing.active = True
            self._record(operation, existing)
            return existing

        slot = PreferenceSlot(
            key=key,
            value=value,
            bucket=bucket,
            confidence=confidence,
            turn=self.turn,
            source=source,
            active=True,
        )
        self.slots[slot_id] = slot
        self._record(operation, slot)
        return slot

    def update_many(self, updates: Iterable[dict]):
        for update in updates:
            self.update(
                bucket=update.get("bucket", "soft"),
                key=update["key"],
                value=update["value"],
                operation=update.get("operation", "merge"),
                confidence=float(update.get("confidence", 0.7)),
                source=update.get("source", ""),
            )

    def apply_turn(self, user_utterance: str, updates: Optional[Iterable[dict]] = None):
        """Apply explicit updates, or a conservative cue-based fallback."""
        if updates is not None:
            self.update_many(updates)
            return

        inferred_updates = infer_preference_updates(user_utterance)
        if inferred_updates:
            self.update_many(inferred_updates)

    def to_prompt_context(self) -> str:
        active = self.active_slots()[: self.max_prompt_items]
        if not active:
            return "Structured memory: no active preferences yet."

        lines = ["Structured memory for the current user:"]
        for bucket in ["hard", "positive", "negative", "soft"]:
            bucket_slots = [slot for slot in active if slot.bucket == bucket]
            if bucket_slots:
                lines.append(f"{bucket.title()} preferences:")
                lines.extend(slot.to_prompt_line() for slot in bucket_slots)
        lines.append(
            "Use active hard constraints first; treat low-confidence soft preferences as tentative."
        )
        return "\n".join(lines)

    def token_cost_estimate(self) -> int:
        return max(1, len(self.to_prompt_context().split()))

    def stale_preference_violations(self, recommended_texts: Iterable[str]) -> int:
        stale_slots = [slot for slot in self.slots.values() if not slot.active and slot.value]
        count = 0
        for text in recommended_texts:
            text_lower = text.lower()
            if any(slot.value.lower() in text_lower for slot in stale_slots):
                count += 1
        return count

    def constraint_satisfaction_rate(self, recommended_texts: Iterable[str]) -> float:
        hard_slots = [slot for slot in self.active_slots() if slot.bucket == "hard"]
        texts = list(recommended_texts)
        if not hard_slots or not texts:
            return 1.0
        satisfied = 0
        for text in texts:
            text_lower = text.lower()
            if all(slot.value.lower() in text_lower for slot in hard_slots):
                satisfied += 1
        return satisfied / len(texts)

    def _merge_values(self, old_value: str, new_value: str) -> str:
        if old_value == new_value:
            return old_value
        parts = [part.strip() for part in f"{old_value}; {new_value}".split(";") if part.strip()]
        deduped = list(dict.fromkeys(parts))
        return "; ".join(deduped)

    def _record(self, operation: str, slot: PreferenceSlot):
        self.events.append(
            {
                "turn": self.turn,
                "operation": operation,
                "bucket": slot.bucket,
                "key": slot.key,
                "value": slot.value,
                "confidence": slot.confidence,
                "active": slot.active,
            }
        )


def infer_preference_updates(user_utterance: str) -> List[dict]:
    """Conservative keyword fallback for lightweight experiments.

    Production runs should pass explicit updates from an LLM extractor or task
    annotation. This function only captures common drift cues so local tests can
    run without model calls.
    """

    text = user_utterance.strip()
    lower = text.lower()
    updates: List[dict] = []

    overwrite_cues = ["instead", "change to", "switch to", "forget", "no longer", "算了", "改成", "不要"]
    hard_cues = ["must", "need", "required", "必须", "一定要", "需要"]
    negative_cues = ["not", "avoid", "don't", "do not", "不想", "不要", "避免"]
    soft_cues = ["prefer", "like", "最好", "更喜欢", "适合", "想"]

    operation = "overwrite" if any(cue in lower for cue in overwrite_cues) else "merge"
    bucket = "soft"
    if any(cue in lower for cue in hard_cues):
        bucket = "hard"
    elif any(cue in lower for cue in negative_cues):
        bucket = "negative"
    elif any(cue in lower for cue in soft_cues):
        bucket = "positive"

    if any(cue in lower for cue in hard_cues + negative_cues + soft_cues + overwrite_cues):
        updates.append(
            {
                "bucket": bucket,
                "key": "utterance_preference",
                "value": text,
                "operation": operation,
                "confidence": 0.58 if operation == "overwrite" else 0.52,
                "source": "cue_inference",
            }
        )
    return updates
