"""Adapter for normalizing critique rollouts into benchmark-ready artifacts.

The adapter supports two input shapes:

1. The existing lightweight scenario schema with `follow_value`,
   `ignore_value`, and `over_apply_value`.
2. A richer "real rollout" schema that includes branch trajectories and an
   optional CritiqueWorld-style state snapshot. This lets us connect future
   GIMO rollout logs to the same branch-rollout and preference-pair outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List

from user_simulator.evaluation.critique_parser import parse_deterministic
from user_simulator.evaluation.critique_scope_eval import DEFAULT_SCENARIOS
from user_simulator.evaluation.critique_uplift_pairs import build_pairs
from user_simulator.evaluation.validate_critique_scenarios import validate_scenario
from user_simulator.state.critique_scope import TARGET_ALIASES


REQUIRED_BRANCHES = ["follow", "ignore", "over_apply"]
VALUE_KEYS = {branch: f"{branch}_value" for branch in REQUIRED_BRANCHES}
CRITIQUEWORLD_PROXY = "controlled counterfactual rollout proxy"
GPE_HAP_PROXY = "gpe_hap_refinement_proxy"
KNOWN_TASK_TYPES = {"recommend", "ask", "search"}


def load_rollouts(path: str | None) -> List[dict]:
    if not path:
        rows = DEFAULT_SCENARIOS
    else:
        rows = read_rollout_records(Path(path))
    return [normalize_rollout(row, index) for index, row in enumerate(rows, start=1)]


def read_rollout_records(path: Path) -> List[dict]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                validate_rollout(row, line_no)
                rows.append(row)
        return rows

    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            if "logs" in payload and isinstance(payload["logs"], list):
                rows = payload["logs"]
            else:
                rows = [payload]
        else:
            raise ValueError(f"Unsupported JSON payload in {path}")
        for line_no, row in enumerate(rows, start=1):
            validate_rollout(row, line_no)
        return rows

    raise ValueError(f"Unsupported rollout file suffix: {path.suffix}")


def validate_rollout(row: dict, line_no: int):
    if _has_trace_payload(row):
        errors = validate_trace_rollout(row, line_no)
    elif _has_branch_payload(row):
        errors = validate_branch_rollout(row, line_no)
    else:
        errors = validate_value_rollout(row, line_no)
    if errors:
        raise ValueError(f"line {line_no}: " + "; ".join(errors))


def validate_value_rollout(row: dict, line_no: int) -> List[str]:
    errors = validate_scenario(row, index=line_no)
    for key in VALUE_KEYS.values():
        values = row.get(key)
        if not isinstance(values, list) or not all(isinstance(value, (int, float)) for value in values):
            errors.append(f"{key} must be a list of numbers")
    return errors


def validate_branch_rollout(row: dict, line_no: int) -> List[str]:
    errors: List[str] = []
    required = {"id", "critique_type", "utterance", "critiques"}
    missing = sorted(required - set(row))
    if missing:
        errors.append(f"missing keys: {missing}")

    critiques = row.get("critiques", [])
    if not isinstance(critiques, list) or not critiques:
        errors.append("critiques must be a non-empty list")

    branches = extract_branch_payloads(row)
    missing_branches = [branch for branch in REQUIRED_BRANCHES if branch not in branches]
    if missing_branches:
        errors.append(f"missing branches: {missing_branches}")

    for branch in REQUIRED_BRANCHES:
        payload = branches.get(branch)
        if payload is None:
            continue
        trajectory = payload.get("trajectory")
        values = extract_branch_values(payload)
        if trajectory is None and values is None:
            errors.append(f"{branch} branch must include trajectory, rows, value_sequence, values, or value")
            continue
        if trajectory is not None and not isinstance(trajectory, list):
            errors.append(f"{branch} trajectory must be a list")
        if values is None or not isinstance(values, list) or not values:
            errors.append(f"{branch} branch must resolve to a non-empty numeric value sequence")
        elif not all(isinstance(value, (int, float)) for value in values):
            errors.append(f"{branch} branch values must contain only numbers")

    snapshot = row.get("state_snapshot")
    if snapshot is not None and not isinstance(snapshot, dict):
        errors.append("state_snapshot must be an object when provided")
    return errors


def validate_trace_rollout(row: dict, line_no: int) -> List[str]:
    errors: List[str] = []
    required = {"task_type", "input", "original_response", "ground_truth"}
    missing = sorted(required - set(row))
    if missing:
        errors.append(f"missing keys: {missing}")
    if not str(row.get("task_type", "")).strip():
        errors.append("task_type must be a non-empty string")
    if not str(row.get("input", "")).strip():
        errors.append("input must be a non-empty string")
    if not str(row.get("original_response", "")).strip():
        errors.append("original_response must be a non-empty string")
    if "best_refinement" not in row and "policy_improvement_output" not in row and "evaluation_error" not in row:
        errors.append("trace log should include a refinement output or an error field")
    return errors


def _has_branch_payload(row: dict) -> bool:
    return "branches" in row or any(branch in row for branch in REQUIRED_BRANCHES)


def _has_trace_payload(row: dict) -> bool:
    return "task_type" in row and "original_response" in row and "ground_truth" in row


def extract_branch_payloads(row: dict) -> dict[str, dict]:
    if isinstance(row.get("branches"), dict):
        return {branch: row["branches"].get(branch, {}) for branch in REQUIRED_BRANCHES}
    return {branch: row.get(branch, {}) for branch in REQUIRED_BRANCHES if branch in row}


def extract_branch_values(payload: Any) -> list[float] | None:
    if isinstance(payload, list):
        return [float(value) for value in payload]
    if not isinstance(payload, dict):
        return None

    for key in ["value_sequence", "values", "value"]:
        raw = payload.get(key)
        if isinstance(raw, list) and raw:
            return [float(value) for value in raw]
        if isinstance(raw, (int, float)):
            return [float(raw)]

    trajectory = payload.get("trajectory") or payload.get("rows")
    if isinstance(trajectory, list) and trajectory:
        values = []
        for step in trajectory:
            if not isinstance(step, dict):
                return None
            for key in ["instant_utility", "utility", "reward", "value"]:
                raw_value = step.get(key)
                if isinstance(raw_value, (int, float)):
                    values.append(float(raw_value))
                    break
            else:
                return None
        return values if values else None
    return None


def normalize_rollout(row: dict, index: int) -> dict:
    if _has_trace_payload(row):
        return normalize_trace_rollout(row, index)
    if not _has_branch_payload(row):
        return {
            **row,
            "_adapter_source": "value_only",
            "_adapter_point_index": 0,
            "_adapter_branch_rows": materialize_branch_rows_from_values(row, index=index),
        }

    branches = extract_branch_payloads(row)
    normalized = {
        "id": row["id"],
        "critique_type": row["critique_type"],
        "utterance": row["utterance"],
        "critiques": row["critiques"],
        "post_expiry_items": row.get("post_expiry_items", []),
        "follow_value": extract_branch_values(branches["follow"]) or [],
        "ignore_value": extract_branch_values(branches["ignore"]) or [],
        "over_apply_value": extract_branch_values(branches["over_apply"]) or [],
        "_adapter_source": "branch_rollout",
        "_adapter_point_index": int(row.get("critique_point_index", 0)),
        "_adapter_branch_id": row.get("branch_id") or f"{row.get('method', 'gimo_real_rollout')}:{row['id']}:{row.get('seed', 0)}:0",
        "_adapter_branch_rows": materialize_branch_rows_from_payloads(row, branches),
        "_adapter_metadata": {
            "method": row.get("method", "gimo_real_rollout"),
            "scenario": row.get("scenario", row["id"]),
            "seed": int(row.get("seed", 0)),
            "parser_mode": row.get("parser_mode", "external"),
            "branch_id": row.get("branch_id") or f"{row.get('method', 'gimo_real_rollout')}:{row['id']}:{row.get('seed', 0)}:0",
            "snapshot_turn": _snapshot_turn(row.get("state_snapshot")),
            "proxy": row.get("proxy", CRITIQUEWORLD_PROXY),
            "source": row.get("source", "GIMO_real_rollout"),
        },
        "_adapter_state_snapshot": normalize_state_snapshot(row),
    }
    return normalized


def normalize_trace_rollout(row: dict, index: int) -> dict:
    task_type = normalize_task_type(row.get("task_type", "gpe_hap_trace"))
    best_refinement = row.get("best_refinement")
    improved_output = best_refinement or _extract_refinement_output(row.get("policy_improvement_output")) or row.get("original_response", "")
    original_output = row.get("original_response", "")
    original_is_best = bool(row.get("is_original_best", False))
    follow_better = bool(improved_output and improved_output != original_output and not original_is_best)

    follow_value = [1.0] if follow_better else [0.0]
    ignore_value = [0.0] if follow_better else [1.0]
    over_apply_value = [0.2 if follow_better else 0.0]

    branch_id = f"gpe_hap:{task_type}:{index}"
    state_snapshot = {
        "scenario": task_type,
        "method": row.get("task_type", task_type),
        "seed": int(row.get("sample_num", row.get("seed", 0)) or 0),
        "turn": 0,
        "user_state": {
            "input": row.get("input", ""),
            "original_response": original_output,
            "ground_truth": row.get("ground_truth", ""),
            "original_action": row.get("original_action", ""),
            "original_query": row.get("original_query", ""),
            "refined_queries": row.get("refined_queries", []),
            "original_rank": row.get("original_rank"),
        },
        "memory_state": {
            "potential_reward_output": row.get("potential_reward_output"),
            "policy_improvement_output": row.get("policy_improvement_output"),
            "eval_response": row.get("eval_response"),
        },
        "event": {
            "turn": 0,
            "type": task_type,
            "utterance": row.get("input", ""),
            "critiques": [
                {
                    "target": row.get("original_query") or row.get("original_action") or task_type,
                    "operation": "promote" if follow_better else "attenuate",
                    "reason": "gpe_hap_refinement_trace",
                    "object_scope": "global",
                    "temporal_scope": "session",
                    "horizon": 1,
                    "hardness": "soft",
                    "confidence": 1.0,
                    "promotion_condition": "never",
                }
            ],
        },
    }

    branches = {
        "follow": {
            "trajectory": [
                {
                    "turn": 0,
                    "action": "refine",
                    "response": improved_output,
                    "utility": follow_value[0],
                    "prompt": row.get("policy_improvement_prompt"),
                }
            ]
        },
        "ignore": {
            "trajectory": [
                {
                    "turn": 0,
                    "action": "preserve",
                    "response": original_output,
                    "utility": ignore_value[0],
                    "prompt": row.get("original_response"),
                }
            ]
        },
        "over_apply": {
            "trajectory": [
                {
                    "turn": 0,
                    "action": "over_apply",
                    "response": row.get("policy_improvement_output") or improved_output,
                    "utility": over_apply_value[0],
                    "prompt": row.get("potential_reward_output") or row.get("policy_improvement_output"),
                }
            ]
        },
    }

    normalized = {
        "id": row.get("id") or f"{task_type}:{index}",
        "critique_type": row.get("critique_type", task_type),
        "utterance": row.get("input", ""),
        "critiques": state_snapshot["event"]["critiques"],
        "post_expiry_items": row.get("post_expiry_items", []),
        "follow_value": follow_value,
        "ignore_value": ignore_value,
        "over_apply_value": over_apply_value,
        "_adapter_source": "gpe_hap_refinement_trace",
        "_adapter_point_index": 0,
        "_adapter_branch_id": branch_id,
        "_adapter_branch_rows": materialize_branch_rows_from_payloads(
            {
                "scenario": task_type,
                "method": row.get("task_type", task_type),
                "seed": int(row.get("sample_num", row.get("seed", 0)) or 0),
                "parser_mode": "gpe_hap_trace",
                "branch_id": branch_id,
                "task_type": task_type,
                "state_snapshot": state_snapshot,
            },
        branches,
        ),
        "_adapter_metadata": {
            "method": row.get("task_type", task_type),
            "scenario": task_type,
            "seed": int(row.get("sample_num", row.get("seed", 0)) or 0),
            "parser_mode": "gpe_hap_trace",
            "branch_id": branch_id,
            "source_ref": row.get("source_ref") or row.get("id") or branch_id,
            "snapshot_turn": 0,
            "proxy": GPE_HAP_PROXY,
            "source": row.get("source", "gpe_hap_refinement_trace"),
            "task_type": task_type,
        },
        "_adapter_state_snapshot": state_snapshot,
        "_adapter_trace_fields": {
            "best_refinement": best_refinement,
            "is_original_best": original_is_best,
            "potential_reward_output": row.get("potential_reward_output"),
            "policy_improvement_output": row.get("policy_improvement_output"),
            "eval_response": row.get("eval_response"),
        },
    }
    return normalized


def _snapshot_turn(snapshot: Any) -> int:
    if isinstance(snapshot, dict) and isinstance(snapshot.get("turn"), int):
        return snapshot["turn"]
    return 0


def normalize_task_type(task_type: Any) -> str:
    normalized = str(task_type or "").strip().lower()
    if normalized in KNOWN_TASK_TYPES:
        return normalized
    if normalized:
        return "generic"
    return "generic"


def normalize_state_snapshot(row: dict) -> dict:
    row_id = row.get("id") or f"{row.get('scenario', 'scenario')}"
    snapshot = row.get("state_snapshot")
    if isinstance(snapshot, dict):
        event = snapshot.get("event")
        if not isinstance(event, dict):
            event = {
                "turn": _snapshot_turn(snapshot),
                "type": row.get("event_type", "critique"),
                "utterance": row.get("utterance", ""),
                "critiques": row.get("critiques", []),
            }
        return {
            "scenario": row.get("scenario", row_id),
            "method": row.get("method", "gimo_real_rollout"),
            "seed": int(row.get("seed", 0)),
            "turn": _snapshot_turn(snapshot),
            "user_state": snapshot.get("user_state", {}),
            "memory_state": snapshot.get("memory_state", {}),
            "event": event,
        }
    return {
        "scenario": row.get("scenario", row_id),
        "method": row.get("method", "gimo_real_rollout"),
        "seed": int(row.get("seed", 0)),
        "turn": int(row.get("turn", 0)),
        "user_state": {},
        "memory_state": {},
        "event": {
            "turn": int(row.get("turn", 0)),
            "type": row.get("event_type", "critique"),
            "utterance": row.get("utterance", ""),
            "critiques": row.get("critiques", []),
        },
    }


def _extract_refinement_output(raw: Any) -> str | None:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(parsed, dict):
            refinement = parsed.get("refinement_output")
            if isinstance(refinement, list) and refinement:
                first = refinement[0]
                return first if isinstance(first, str) else json.dumps(first, ensure_ascii=False)
            if isinstance(refinement, str):
                return refinement
        return text
    if isinstance(raw, dict):
        refinement = raw.get("refinement_output")
        if isinstance(refinement, list) and refinement:
            first = refinement[0]
            return first if isinstance(first, str) else json.dumps(first, ensure_ascii=False)
        if isinstance(refinement, str):
            return refinement
    return None


def materialize_branch_rows_from_values(row: dict, index: int) -> list[dict]:
    branch_id = f"adapter:{row.get('id', f'scenario_{index}')}:{index}"
    state_snapshot = {
        "scenario": row.get("id", f"scenario_{index}"),
        "method": "rollout_adapter",
        "seed": 0,
        "turn": 0,
        "user_state": {},
        "memory_state": {},
        "event": {
            "turn": 0,
            "type": "critique",
            "utterance": row.get("utterance", ""),
            "critiques": row.get("critiques", []),
        },
    }
    rows: list[dict] = []
    for branch in REQUIRED_BRANCHES:
        values = [float(value) for value in row[VALUE_KEYS[branch]]]
        rows.extend(
            build_branch_rows(
                scenario=row.get("id", f"scenario_{index}"),
                method="rollout_adapter",
                seed=0,
                parser_mode="adapter",
                branch=branch,
                branch_id=branch_id,
                values=values,
                state_snapshot=state_snapshot,
                trajectory=None,
                task_type="",
            )
        )
    return rows


def materialize_branch_rows_from_payloads(row: dict, branches: dict[str, dict]) -> list[dict]:
    row_id = row.get("id") or f"{row.get('scenario', 'scenario')}"
    task_type = row.get("task_type", "")
    metadata = {
        "scenario": row.get("scenario", row_id),
        "method": row.get("method", "gimo_real_rollout"),
        "seed": int(row.get("seed", 0)),
        "parser_mode": row.get("parser_mode", "external"),
        "branch_id": row.get("branch_id") or f"{row.get('method', 'gimo_real_rollout')}:{row_id}:{row.get('seed', 0)}:0",
        "state_snapshot": normalize_state_snapshot(row),
        "task_type": task_type,
    }
    rows: list[dict] = []
    for branch in REQUIRED_BRANCHES:
        payload = branches[branch]
        values = extract_branch_values(payload) or []
        trajectory = payload.get("trajectory") or payload.get("rows")
        rows.extend(
            build_branch_rows(
                scenario=metadata["scenario"],
                method=metadata["method"],
                seed=metadata["seed"],
                parser_mode=metadata["parser_mode"],
                branch=branch,
                branch_id=metadata["branch_id"],
                values=values,
                state_snapshot=metadata["state_snapshot"],
                trajectory=trajectory,
                task_type=metadata["task_type"],
            )
        )
    return rows


def materialize_branch_rows_from_trace(row: dict) -> list[dict]:
    metadata = {
        "scenario": row.get("task_type", "gpe_hap_trace"),
        "method": row.get("task_type", "gpe_hap_trace"),
        "seed": int(row.get("sample_num", row.get("seed", 0)) or 0),
        "parser_mode": "gpe_hap_trace",
        "branch_id": row.get("_adapter_branch_id") or f"gpe_hap:{row.get('task_type', 'trace')}:0",
        "state_snapshot": row.get("_adapter_state_snapshot") or normalize_state_snapshot(row),
    }
    trace_fields = row.get("_adapter_trace_fields", {})
    branches = {
        "follow": {
            "trajectory": [
                {
                    "turn": 0,
                    "action": "refine",
                    "response": trace_fields.get("best_refinement") or row.get("original_response", ""),
                    "utility": row.get("follow_value", [0.0])[0] if row.get("follow_value") else 0.0,
                    "prompt": row.get("policy_improvement_prompt"),
                }
            ]
        },
        "ignore": {
            "trajectory": [
                {
                    "turn": 0,
                    "action": "preserve",
                    "response": row.get("original_response", ""),
                    "utility": row.get("ignore_value", [0.0])[0] if row.get("ignore_value") else 0.0,
                    "prompt": row.get("input", ""),
                }
            ]
        },
        "over_apply": {
            "trajectory": [
                {
                    "turn": 0,
                    "action": "over_apply",
                    "response": row.get("policy_improvement_output") or trace_fields.get("best_refinement") or row.get("original_response", ""),
                    "utility": row.get("over_apply_value", [0.0])[0] if row.get("over_apply_value") else 0.0,
                    "prompt": row.get("potential_reward_output") or row.get("policy_improvement_output"),
                }
            ]
        },
    }
    return materialize_branch_rows_from_payloads(
        {
            "scenario": metadata["scenario"],
            "method": metadata["method"],
            "seed": metadata["seed"],
            "parser_mode": metadata["parser_mode"],
            "branch_id": metadata["branch_id"],
            "task_type": row.get("task_type", ""),
            "state_snapshot": metadata["state_snapshot"],
        },
        branches,
    )


def build_branch_rows(
    scenario: str,
    method: str,
    seed: int,
    parser_mode: str,
    branch: str,
    branch_id: str,
    values: list[float],
    state_snapshot: dict,
    trajectory: list[dict] | None,
    task_type: str = "",
) -> list[dict]:
    task_type = normalize_task_type(task_type)
    rows: list[dict] = []
    cumulative = 0.0
    snapshot_turn = int(state_snapshot.get("turn", 0))
    trajectory = trajectory if isinstance(trajectory, list) else []

    for index, value in enumerate(values):
        step = trajectory[index] if index < len(trajectory) and isinstance(trajectory[index], dict) else {}
        instant_utility = float(step.get("instant_utility", step.get("utility", step.get("reward", value))))
        cumulative += instant_utility
        turn = int(step.get("turn", snapshot_turn + index + 1))
        slate = step.get("ranked_slate")
        if isinstance(slate, dict):
            ranked_slate = slate
        else:
            ranked_slate = {
                "slate": list(step.get("slate", [])),
                "scores": step.get("scores", {}),
                "score_breakdowns": step.get("score_breakdowns", {}),
                "applied_interventions": step.get("applied_interventions", []),
            }
        action = step.get("user_action")
        if not isinstance(action, dict):
            action = {
                "action": step.get("action", "unknown"),
                "item_id": step.get("item_id"),
                "category": step.get("category"),
                "utility": instant_utility,
                "critique": step.get("critique"),
                "response": step.get("response"),
                "prompt": step.get("prompt"),
            }
        if task_type == "search" and not action.get("response"):
            action["response"] = step.get("refinement") or step.get("query") or step.get("response")
        if task_type == "search" and not action.get("prompt"):
            action["prompt"] = step.get("original_query") or step.get("original_action") or step.get("prompt")
        rows.append(
            {
                "run_id": step.get("run_id", f"{method}:{scenario}:{seed}:{parser_mode}"),
                "method": method,
                "scenario": scenario,
                "seed": seed,
                "parser_mode": parser_mode,
                "turn": turn,
                "branch": branch,
                "branch_id": branch_id,
                "user_state_before": step.get("user_state_before", {}),
                "memory_state_before": step.get("memory_state_before", {}),
                "ranked_slate": ranked_slate,
                "score_breakdowns": step.get("score_breakdowns", ranked_slate.get("score_breakdowns", {})),
                "user_action": action,
                "generated_critique": step.get("generated_critique"),
                "memory_update": step.get("memory_update", {"applied": []}),
                "user_state_after": step.get("user_state_after", {}),
                "memory_state_after": step.get("memory_state_after", {}),
                "instant_utility": instant_utility,
                "cumulative_utility": cumulative,
                "patience": float(step.get("patience", 1.0)),
                "active": bool(step.get("active", True)),
                "response": step.get("response"),
                "prompt": step.get("prompt"),
                "parser_scope_error": float(step.get("parser_scope_error", 0.0)),
                "memory_update_error": float(step.get("memory_update_error", 0.0)),
                "policy_application_error": float(step.get("policy_application_error", 0.0)),
                "candidate_coverage_error": float(step.get("candidate_coverage_error", 0.0)),
                "state_snapshot": state_snapshot,
                "task_type": task_type,
                "source_ref": f"{method}:{scenario}:{seed}:{branch_id}",
            }
        )
    return rows


def trajectory_value(values: list[float]) -> float:
    return float(sum(values))


def _normalize_target(target: str) -> str:
    lowered = str(target).strip().lower()
    if lowered in TARGET_ALIASES:
        return TARGET_ALIASES[lowered].lower()
    if "family" in lowered:
        return "family"
    if "politic" in lowered:
        return "politics"
    if "mac" in lowered:
        return "mac"
    if "window" in lowered:
        return "windows"
    return lowered


def _signature(critique: dict) -> tuple[str, str, str]:
    return (
        _normalize_target(str(critique.get("target", ""))),
        str(critique.get("operation", "")).lower(),
        str(critique.get("temporal_scope", "")).replace("-", "_").lower(),
    )


def audit_rollouts(rows: List[dict]) -> List[dict]:
    findings: List[dict] = []
    for row in rows:
        row_id = row.get("id", "UNKNOWN")
        branch_lengths = {branch: len(row.get(VALUE_KEYS[branch], [])) for branch in REQUIRED_BRANCHES}
        same_length = len(set(branch_lengths.values())) == 1
        findings.append(
            {
                "scenario_id": row_id,
                "check": "branch_length_consistency",
                "passed": same_length,
                "observed": branch_lengths,
                "expected": {"all_equal": True},
            }
        )

        follow_sum = trajectory_value(row["follow_value"])
        ignore_sum = trajectory_value(row["ignore_value"])
        over_apply_sum = trajectory_value(row["over_apply_value"])
        findings.append(
            {
                "scenario_id": row_id,
                "check": "follow_outperforms_at_least_one_counterfactual",
                "passed": bool(follow_sum > ignore_sum or follow_sum > over_apply_sum),
                "observed": {
                    "follow_sum": follow_sum,
                    "ignore_sum": ignore_sum,
                    "over_apply_sum": over_apply_sum,
                },
                "expected": {"follow_gt_ignore_or_over_apply": True},
            }
        )

        branch_rows = row.get("_adapter_branch_rows", [])
        findings.append(
            {
                "scenario_id": row_id,
                "check": "branch_schema_rows_present",
                "passed": bool(branch_rows),
                "observed": {"row_count": len(branch_rows)},
                "expected": {"row_count_gt_zero": True},
            }
        )

        if row.get("_adapter_source") == "branch_rollout":
            parsed = parse_deterministic(row.get("utterance", ""))
            provided = row.get("critiques", [])
            parsed_signatures = {_signature(item) for item in parsed}
            provided_signatures = {_signature(item) for item in provided}
            findings.append(
                {
                    "scenario_id": row_id,
                    "check": "deterministic_parser_alignment",
                    "passed": parsed_signatures == provided_signatures,
                    "observed": {
                        "parsed": sorted(parsed_signatures),
                        "provided": sorted(provided_signatures),
                    },
                    "expected": {"parsed_equals_provided": True},
                }
            )
            snapshot = row.get("_adapter_state_snapshot", {})
            findings.append(
                {
                    "scenario_id": row_id,
                    "check": "state_snapshot_present_for_real_rollout",
                    "passed": bool(snapshot.get("event")),
                    "observed": {"snapshot_keys": sorted(snapshot.keys()) if isinstance(snapshot, dict) else []},
                    "expected": {"contains_event": True},
                }
            )
        elif row.get("_adapter_source") == "gpe_hap_refinement_trace":
            trace_fields = row.get("_adapter_trace_fields", {})
            follow_text = row.get("follow_value", [])
            ignore_text = row.get("ignore_value", [])
            over_text = row.get("over_apply_value", [])
            findings.append(
                {
                    "scenario_id": row_id,
                    "check": "gpe_hap_trace_fields_present",
                    "passed": bool(row.get("utterance")) and bool(row.get("critiques")),
                    "observed": {
                        "has_best_refinement": bool(trace_fields.get("best_refinement")),
                        "is_original_best": trace_fields.get("is_original_best"),
                        "has_potential_reward": bool(trace_fields.get("potential_reward_output")),
                        "has_policy_output": bool(trace_fields.get("policy_improvement_output")),
                    },
                    "expected": {"has_task_trace": True},
                }
            )
            findings.append(
                {
                    "scenario_id": row_id,
                    "check": "gpe_hap_proxy_preference_order",
                    "passed": trajectory_value(follow_text) >= trajectory_value(ignore_text)
                    and trajectory_value(follow_text) >= trajectory_value(over_text),
                    "observed": {
                        "follow_sum": trajectory_value(follow_text),
                        "ignore_sum": trajectory_value(ignore_text),
                        "over_apply_sum": trajectory_value(over_text),
                    },
                    "expected": {"follow_not_worse_than_counterfactuals": True},
                }
            )
        else:
            parsed = parse_deterministic(row.get("utterance", ""))
            provided = row.get("critiques", [])
            parsed_signatures = {_signature(item) for item in parsed}
            provided_signatures = {_signature(item) for item in provided}
            findings.append(
                {
                    "scenario_id": row_id,
                    "check": "deterministic_parser_alignment",
                    "passed": parsed_signatures == provided_signatures,
                    "observed": {
                        "parsed": sorted(parsed_signatures),
                        "provided": sorted(provided_signatures),
                    },
                    "expected": {"parsed_equals_provided": True},
                }
            )
    return findings


def summarize_audit(findings: List[dict]) -> dict:
    by_check = Counter(finding["check"] for finding in findings)
    failed_by_check = Counter(finding["check"] for finding in findings if not finding["passed"])
    failed_scenarios = sorted({finding["scenario_id"] for finding in findings if not finding["passed"]})
    return {
        "total_checks": len(findings),
        "failed_checks": sum(1 for finding in findings if not finding["passed"]),
        "checks_by_type": dict(sorted(by_check.items())),
        "failed_by_type": dict(sorted(failed_by_check.items())),
        "failed_scenarios": failed_scenarios,
    }


def render_report(summary: dict) -> str:
    lines = [
        "# Rollout Adapter Audit",
        "",
        f"- Total checks: `{summary['total_checks']}`",
        f"- Failed checks: `{summary['failed_checks']}`",
        f"- Failed scenarios: `{', '.join(summary['failed_scenarios']) if summary['failed_scenarios'] else 'none'}`",
        "",
        "## Check Summary",
        "| Check | Count | Failed |",
        "| --- | ---: | ---: |",
    ]
    for check, count in summary["checks_by_type"].items():
        lines.append(f"| {check} | {count} | {summary['failed_by_type'].get(check, 0)} |")
    return "\n".join(lines) + "\n"


def write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario_id", "check", "passed"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def materialize_branch_rollouts(rows: List[dict]) -> List[dict]:
    branch_rows: list[dict] = []
    for row in rows:
        branch_rows.extend(row.get("_adapter_branch_rows", []))
    return branch_rows


def build_branch_pairs(rows: List[dict]) -> List[dict]:
    pairs: list[dict] = []
    for row in rows:
        branch_rows = row.get("_adapter_branch_rows", [])
        by_branch = {
            branch: [item for item in branch_rows if item.get("branch") == branch]
            for branch in REQUIRED_BRANCHES
        }
        metadata = row.get("_adapter_metadata", {})
        task_type = normalize_task_type(metadata.get("task_type", row.get("task_type", "")))
        source_ref = metadata.get("source_ref") or row.get("source_ref") or metadata.get("branch_id") or row.get("id")
        point_index = int(row.get("_adapter_point_index", 0))
        snapshot = row.get("_adapter_state_snapshot") or {
            "scenario": metadata.get("scenario", row.get("id")),
            "method": metadata.get("method", "rollout_adapter"),
            "seed": metadata.get("seed", 0),
            "turn": metadata.get("snapshot_turn", 0),
            "user_state": {},
            "memory_state": {},
            "event": {
                "turn": metadata.get("snapshot_turn", 0),
                "type": "critique",
                "utterance": row.get("utterance", ""),
                "critiques": row.get("critiques", []),
            },
        }
        chosen_value = trajectory_value(row["follow_value"])
        for rejected in ["ignore", "over_apply"]:
            rejected_value = trajectory_value(row[VALUE_KEYS[rejected]])
            if chosen_value > rejected_value:
                pairs.append(
                    {
                        "scenario": metadata.get("scenario", row.get("id")),
                        "seed": metadata.get("seed", 0),
                        "method": metadata.get("method", "rollout_adapter"),
                        "parser_mode": metadata.get("parser_mode", "adapter"),
                        "state_snapshot": snapshot,
                        "critique": {"utterance": row.get("utterance", ""), "critiques": row.get("critiques", [])},
                        "chosen_branch": "follow",
                        "rejected_branch": rejected,
                        "chosen_trajectory": by_branch["follow"],
                        "rejected_trajectory": by_branch[rejected],
                        "uplift": chosen_value - rejected_value,
                        "task_type": task_type,
                        "source_ref": source_ref,
                        "metadata": {
                            "proxy": metadata.get("proxy", CRITIQUEWORLD_PROXY),
                            "horizon": len(row["follow_value"]),
                            "critique_point_index": point_index,
                            "snapshot_turn": snapshot.get("turn", 0),
                            "branch_id": metadata.get("branch_id", f"adapter:{row.get('id')}:{point_index}"),
                            "source": metadata.get("source", "rollout_adapter"),
                            "task_type": task_type,
                            "source_ref": source_ref,
                        },
                    }
                )
    return pairs


def branch_policy_text(branch: str, critique: dict, source: str = "", task_type: str = "") -> str:
    target = critique.get("target", "")
    operation = critique.get("operation", "")
    temporal_scope = critique.get("temporal_scope", "")
    trace_task_type = normalize_task_type(task_type or critique.get("task_type", ""))

    def recommend_text(kind: str) -> str:
        if kind == "ask":
            if branch == "follow":
                return "Ask a clarifying question to reduce uncertainty in the current preference."
            if branch == "ignore":
                return "Keep the current response and skip the clarifying question."
            return "Over-apply the clarification and ask beyond the observed uncertainty."
        if kind == "search":
            if branch == "follow":
                return "Rewrite the query or run retrieval to expand the candidate set."
            if branch == "ignore":
                return "Keep the original query and ignore the refinement."
            return "Over-apply the query rewrite beyond the observed scope."
        if kind == "recommend":
            if branch == "follow":
                return "Directly rerank or recommend the best candidate under current preferences."
            if branch == "ignore":
                return "Keep the current recommendation order and ignore the critique."
            return "Over-apply the critique and over-constrain the recommendation set."
        if branch == "follow":
            return "Apply the critique in a generic policy update."
        if branch == "ignore":
            return "Ignore the critique and preserve the current policy."
        return "Over-apply the critique beyond the observed scope."

    if source == "gpe_hap_refinement_trace":
        if trace_task_type == "search":
            return recommend_text("search")
        if trace_task_type == "ask":
            return recommend_text("ask")
        if trace_task_type == "recommend":
            return recommend_text("recommend")
        return recommend_text("generic")
    return recommend_text(trace_task_type)


def trajectory_to_training_text(rows: list[dict]) -> str:
    parts = []
    for row in rows:
        slate = ", ".join(row.get("ranked_slate", {}).get("slate", []))
        action = row.get("user_action", {}).get("action")
        utility = float(row.get("instant_utility", row.get("utility", 0.0)))
        if slate:
            parts.append(f"turn={row.get('turn')} slate=[{slate}] action={action} utility={utility:.3f}")
            continue
        response = row.get("user_action", {}).get("response") or row.get("response") or row.get("text") or row.get("output") or ""
        prompt = row.get("prompt") or row.get("user_action", {}).get("prompt") or ""
        parts.append(
            f"turn={row.get('turn')} action={action} utility={utility:.3f}"
            + (f" prompt={prompt}" if prompt else "")
            + (f" response={response}" if response else "")
        )
    return "\n".join(parts)


def build_cdpo_pair(pair: dict) -> dict:
    critique_list = pair.get("critique", {}).get("critiques", [])
    critique = critique_list[0] if critique_list else {}
    metadata = pair.get("metadata", {})
    point_index = metadata.get("critique_point_index", "x")
    snapshot_turn = metadata.get("snapshot_turn", pair.get("state_snapshot", {}).get("turn", "x"))
    task_type = normalize_task_type(metadata.get("task_type", critique.get("task_type", "")))
    origin_source = metadata.get("source", "rollout_adapter")
    origin_proxy = metadata.get("proxy", CRITIQUEWORLD_PROXY)
    if task_type and isinstance(critique, dict) and not critique.get("task_type"):
        critique = {**critique, "task_type": task_type}
    return {
        "id": (
            f"{pair.get('method')}:{pair.get('scenario')}:{pair.get('seed')}:"
            f"turn{snapshot_turn}:point{point_index}:{pair.get('rejected_branch')}"
        ),
        "scenario": pair.get("scenario"),
        "seed": pair.get("seed"),
        "method": pair.get("method"),
        "parser_mode": pair.get("parser_mode"),
        "task_type": task_type,
        "source_ref": pair.get("source_ref") or metadata.get("source_ref") or metadata.get("branch_id"),
        "conversations": [
            {
                "from": "human",
                "value": (
                    "Given the state snapshot and user critique, choose the recommendation policy "
                    "that follows the instruction without over-correcting durable memory.\n"
                    f"Critique: {json.dumps(pair.get('critique'), ensure_ascii=False)}\n"
                    f"State snapshot turn: {pair.get('state_snapshot', {}).get('turn')}"
                ),
            }
        ],
        "chosen": {
            "branch": pair.get("chosen_branch"),
            "policy": branch_policy_text(pair.get("chosen_branch", "follow"), critique, metadata.get("source", ""), task_type=task_type),
            "trajectory": trajectory_to_training_text(pair.get("chosen_trajectory", [])),
        },
        "rejected": {
            "branch": pair.get("rejected_branch"),
            "policy": branch_policy_text(pair.get("rejected_branch", "ignore"), critique, metadata.get("source", ""), task_type=task_type),
            "trajectory": trajectory_to_training_text(pair.get("rejected_trajectory", [])),
        },
        "score_delta": pair.get("uplift"),
        "metadata": {
            **metadata,
            "format": "llamafactory_dpo_bridge",
            "source": "CritiqueWorld",
            "proxy": CRITIQUEWORLD_PROXY,
            "origin_source": origin_source,
            "origin_proxy": origin_proxy,
            "task_type": task_type,
            "source_ref": pair.get("source_ref") or metadata.get("source_ref") or metadata.get("branch_id"),
        },
    }


def strip_adapter_fields(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_adapter_")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Optional real rollout JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fail-on-audit-error", action="store_true")
    args = parser.parse_args()

    scenarios = load_rollouts(args.input)
    output_dir = Path(args.output_dir)
    audit = audit_rollouts(scenarios)
    failures = [finding for finding in audit if not finding["passed"]]
    summary = summarize_audit(audit)
    branch_rollouts = materialize_branch_rollouts(scenarios)
    dpo_pairs = build_branch_pairs(scenarios)
    cdpo_pairs = [build_cdpo_pair(pair) for pair in dpo_pairs]
    lightweight_pairs = build_pairs([strip_adapter_fields(row) for row in scenarios])

    write_jsonl(output_dir / "normalized_scenarios.jsonl", [strip_adapter_fields(row) for row in scenarios])
    write_jsonl(output_dir / "critique_pairs.jsonl", lightweight_pairs)
    write_jsonl(output_dir / "branch_rollouts.jsonl", branch_rollouts)
    write_jsonl(output_dir / "dpo_pairs.jsonl", dpo_pairs)
    write_jsonl(output_dir / "cdpo_pairs.jsonl", cdpo_pairs)
    write_jsonl(output_dir / "adapter_audit.jsonl", audit)
    write_jsonl(output_dir / "adapter_failures.jsonl", failures)
    write_csv(output_dir / "adapter_audit_summary.csv", audit)
    (output_dir / "adapter_report.md").write_text(render_report(summary), encoding="utf-8")
    metadata = {
        "status": "PASS" if not failures else "FAIL",
        "input_status": "SMOKE_TEST_ONLY" if args.input is None else "REAL_ROLLOUT_INPUT",
        "input": args.input or "DEFAULT_SCENARIOS",
        "scenario_count": len(scenarios),
        "pair_count": len(lightweight_pairs),
        "branch_rollout_count": len(branch_rollouts),
        "dpo_pair_count": len(dpo_pairs),
        "cdpo_pair_count": len(cdpo_pairs),
        "audit_checks": summary["total_checks"],
        "audit_failures": summary["failed_checks"],
        "audit_failed_scenarios": summary["failed_scenarios"],
        "audit_failed_by_type": summary["failed_by_type"],
    }
    (output_dir / "adapter_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", **metadata, "output_dir": str(output_dir)}, indent=2))
    if failures and args.fail_on_audit_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
