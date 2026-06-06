from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from user_simulator.state.critique_scope import Critique, CritiqueScopeMemory
from user_simulator.state.structured_memory import PreferenceSlot, StructuredMemory

PROVENANCE_REAL_TRACE = "REAL_TRACE"
PROVENANCE_REAL_USER_SIM_REPLAY = "REAL_USER_SIM_REPLAY"
PROVENANCE_CONTROLLED_SIMULATOR_REPLAY_PROXY = "CONTROLLED_SIMULATOR_REPLAY_PROXY"
PROVENANCE_SYNTHETIC_CRITIQUEWORLD = "SYNTHETIC_CRITIQUEWORLD"

BRANCH_TYPES = {"follow", "ignore", "over_apply"}
PROVENANCE_TYPES = {
    PROVENANCE_REAL_TRACE,
    PROVENANCE_REAL_USER_SIM_REPLAY,
    PROVENANCE_CONTROLLED_SIMULATOR_REPLAY_PROXY,
    PROVENANCE_SYNTHETIC_CRITIQUEWORLD,
}


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_snapshot_id(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:16]
    episode_id = str(payload.get("episode_id", "episode"))
    turn = payload.get("turn", "x")
    task_type = str(payload.get("task_type", "unknown"))
    return f"{episode_id}:turn{turn}:{task_type}:{digest}"


def stable_branch_id(snapshot_id: str, branch_type: str, turn_index: int = 0) -> str:
    digest = hashlib.sha256(f"{snapshot_id}:{branch_type}:{turn_index}".encode("utf-8")).hexdigest()[:12]
    return f"{snapshot_id}:{branch_type}:{turn_index}:{digest}"


def normalize_provenance(provenance: str | None) -> str:
    provenance = provenance or PROVENANCE_CONTROLLED_SIMULATOR_REPLAY_PROXY
    if provenance not in PROVENANCE_TYPES:
        raise ValueError(f"Unsupported provenance: {provenance}")
    return provenance


def action_text(action: Any | None) -> str:
    if action is None:
        return ""
    if isinstance(action, str):
        return action
    if isinstance(action, dict):
        for key in ["assistant_message", "text", "response", "content", "value", "action"]:
            value = action.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return _stable_json(action)
    return str(action)


def serialize_dialogue_history(history: Any) -> list[dict]:
    if history is None:
        return []
    if isinstance(history, list):
        return copy.deepcopy(history)
    if hasattr(history, "get_history"):
        return copy.deepcopy(history.get_history())
    return copy.deepcopy(list(history))


def serialize_critique_memory(memory: CritiqueScopeMemory | None) -> dict | None:
    if memory is None:
        return None
    return {
        "kind": "critiquescope",
        "promotion_confidence": memory.promotion_confidence,
        "promotion_evidence": memory.promotion_evidence,
        "max_prompt_items": memory.max_prompt_items,
        "turn": memory.turn,
        "fast_memory": [asdict(item) for item in memory.fast_memory],
        "slow_memory": [asdict(item) for item in memory.slow_memory],
        "events": copy.deepcopy(memory.events),
    }


def serialize_structured_memory(memory: StructuredMemory | None) -> dict | None:
    if memory is None:
        return None
    return {
        "kind": "structured",
        "decay": memory.decay,
        "conflict_threshold": memory.conflict_threshold,
        "max_prompt_items": memory.max_prompt_items,
        "turn": memory.turn,
        "slots": {key: asdict(value) for key, value in memory.slots.items()},
        "events": copy.deepcopy(memory.events),
    }


def serialize_memory_state(memory_mode: str, memory: Any) -> dict | None:
    if memory is None:
        return None
    if memory_mode == "critiquescope":
        return serialize_critique_memory(memory)
    if memory_mode == "structured":
        return serialize_structured_memory(memory)
    if memory_mode in {"flat", "time_decay"}:
        return {"kind": memory_mode, "items": copy.deepcopy(memory)}
    return {"kind": memory_mode, "repr": repr(memory)}


def restore_critique_memory(payload: dict | None) -> CritiqueScopeMemory | None:
    if not payload:
        return None
    memory = CritiqueScopeMemory(
        promotion_confidence=payload.get("promotion_confidence", 0.84),
        promotion_evidence=payload.get("promotion_evidence", 2),
        max_prompt_items=payload.get("max_prompt_items", 12),
    )
    memory.turn = payload.get("turn", 0)
    memory.fast_memory = [Critique(**item) for item in payload.get("fast_memory", [])]
    memory.slow_memory = [Critique(**item) for item in payload.get("slow_memory", [])]
    memory.events = copy.deepcopy(payload.get("events", []))
    return memory


def restore_structured_memory(payload: dict | None) -> StructuredMemory | None:
    if not payload:
        return None
    memory = StructuredMemory(
        decay=payload.get("decay", 0.92),
        conflict_threshold=payload.get("conflict_threshold", 0.55),
        max_prompt_items=payload.get("max_prompt_items", 12),
    )
    memory.turn = payload.get("turn", 0)
    memory.slots = {key: PreferenceSlot(**value) for key, value in payload.get("slots", {}).items()}
    memory.events = copy.deepcopy(payload.get("events", []))
    return memory


def restore_memory_state(memory_mode: str, payload: dict | None):
    if payload is None:
        return None
    if memory_mode == "critiquescope":
        return restore_critique_memory(payload)
    if memory_mode == "structured":
        return restore_structured_memory(payload)
    if memory_mode in {"flat", "time_decay"}:
        return copy.deepcopy(payload.get("items", []))
    return copy.deepcopy(payload)


@dataclass
class ReplaySnapshot:
    snapshot_id: str
    episode_id: str
    turn: int
    task_type: str
    user_state: dict
    persona: dict
    conversation_history: list[dict]
    candidate_state: dict
    original_action: dict
    refined_action: dict | None
    critique: dict | None
    source_trace: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    provenance: str = PROVENANCE_REAL_TRACE

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReplayBranch:
    snapshot_id: str
    branch_type: str
    task_type: str
    policy_action: dict
    trajectory: list[dict]
    utility_breakdown: dict
    utility_total: float
    status: str
    snapshot: dict = field(default_factory=dict)
    source_trace: dict = field(default_factory=dict)
    provenance: str = PROVENANCE_REAL_USER_SIM_REPLAY
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReplayPair:
    snapshot_id: str
    chosen_branch: str
    rejected_branch: str
    chosen_utility: float
    rejected_utility: float
    uplift: float
    source_trace: dict = field(default_factory=dict)
    provenance: str = PROVENANCE_REAL_USER_SIM_REPLAY
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
