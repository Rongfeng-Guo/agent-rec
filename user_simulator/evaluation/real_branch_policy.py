
from __future__ import annotations

import re
from typing import Any

from user_simulator.evaluation.real_branch_replay_schema import (
    PROVENANCE_CONTROLLED_SIMULATOR_REPLAY_PROXY,
    ReplaySnapshot,
    action_text,
)

ACTION_PATTERN = re.compile(r"^(Ask|Recommend|Search)\[(.*)\]$", re.IGNORECASE | re.DOTALL)


def parse_action(text: str) -> tuple[str | None, str | None]:
    match = ACTION_PATTERN.match(text.strip())
    if not match:
        return None, None
    return match.group(1).lower(), match.group(2).strip()


def _as_action_text(action: Any | None) -> str:
    return action_text(action).strip()


def _recommend_over_apply(base_text: str) -> str:
    action_type, payload = parse_action(base_text)
    if action_type == 'recommend' and payload:
        if '|' in payload:
            item, reason = payload.split('|', 1)
            item = item.strip()
            reason = reason.strip()
            return f'Recommend[{item} | {reason} Over-applied: turn this scoped preference into a permanent filter.]'
        return f'Recommend[{payload} | Over-applied: turn this scoped preference into a permanent filter.]'
    return f'{base_text} [OVER_APPLY: escalate this scoped change into a durable rule.]'


def _ask_over_apply(base_text: str) -> str:
    action_type, payload = parse_action(base_text)
    if action_type == 'ask' and payload:
        return f'Ask[{payload} Also make this a hard standing rule that persists beyond the current conversation.]'
    return f'{base_text} [OVER_APPLY: keep asking and enforce the answer as a permanent rule.]'


def _search_over_apply(base_text: str) -> str:
    action_type, payload = parse_action(base_text)
    if action_type == 'search' and payload:
        return f'Search[{payload} AND apply this rewrite as a permanent route/filter.]'
    return f'{base_text} [OVER_APPLY: treat this local rewrite as a permanent retrieval filter.]'


def build_branch_policy(snapshot: ReplaySnapshot, branch_type: str, turn_index: int = 0) -> dict:
    branch_type = branch_type.strip().lower()
    if branch_type not in {'follow', 'ignore', 'over_apply'}:
        raise ValueError(f'Unsupported branch type: {branch_type}')

    original_text = _as_action_text(snapshot.original_action)
    refined_text = _as_action_text(snapshot.refined_action)
    base_text = refined_text or original_text
    task_type = snapshot.task_type.lower()

    if branch_type == 'follow':
        assistant_message = base_text or original_text
        policy_description = 'Use the refined action when available; otherwise replay the original action.'
        source = 'refined_action' if refined_text else 'original_action'
    elif branch_type == 'ignore':
        assistant_message = original_text or base_text
        policy_description = 'Ignore refinement and replay the original action unchanged.'
        source = 'original_action'
    else:
        if task_type == 'recommend':
            assistant_message = _recommend_over_apply(base_text or original_text)
        elif task_type == 'ask':
            assistant_message = _ask_over_apply(base_text or original_text)
        elif task_type == 'search':
            assistant_message = _search_over_apply(base_text or original_text)
        else:
            assistant_message = f'{base_text or original_text} [OVER_APPLY: escalate this branch into a durable rule.]'
        policy_description = 'Escalate a scoped correction into a durable, over-applied rule.'
        source = 'over_apply_transform'

    action_type = parse_action(assistant_message)[0] or task_type or 'generic'
    return {
        'snapshot_id': snapshot.snapshot_id,
        'branch_type': branch_type,
        'task_type': snapshot.task_type,
        'action_type': action_type,
        'assistant_message': assistant_message,
        'policy_description': policy_description,
        'source_action': original_text,
        'refined_action': refined_text or None,
        'policy_source': source,
        'turn_index': turn_index,
        'provenance': PROVENANCE_CONTROLLED_SIMULATOR_REPLAY_PROXY,
    }


def branch_policy_to_string(policy_action: dict) -> str:
    return policy_action.get('assistant_message', '')
