from __future__ import annotations

import copy
import json
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_WEIGHTS = {
    'task_success': 2.0,
    'satisfaction_signal': 1.0,
    'constraint_satisfaction': 1.0,
    'continuation_signal': 0.75,
    'recommendation_relevance': 1.25,
    'extra_turn_cost': 0.45,
    'repetition_penalty': 0.65,
    'tool_failure': 1.5,
    'parse_failure': 1.0,
}

DEFAULT_CONFIG = {
    'weights': copy.deepcopy(DEFAULT_WEIGHTS),
}


def _parse_scalar(value: str):
    value = value.strip()
    if not value:
        return ''
    lower = value.lower()
    if lower in {'true', 'false'}:
        return lower == 'true'
    try:
        if '.' in value or 'e' in lower:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_simple_yaml(text: str) -> dict:
    result: dict[str, Any] = {}
    current_map: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith('#'):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.strip()
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        if indent == 0 and not value:
            current_map = {}
            result[key] = current_map
            continue
        if indent == 0:
            result[key] = _parse_scalar(value)
            current_map = None
            continue
        if current_map is None:
            current_map = result.setdefault('weights', {})
        current_map[key] = _parse_scalar(value)
    return result


def load_weight_config(path: str | Path | None = None) -> dict:
    if not path:
        return copy.deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)
    if not config_path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    text = config_path.read_text(encoding='utf-8')
    data: dict[str, Any] | None = None
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            data = loaded
    except Exception:
        data = None
    if data is None:
        data = _parse_simple_yaml(text)
    weights = data.get('weights', data)
    config = copy.deepcopy(DEFAULT_CONFIG)
    if isinstance(weights, dict):
        for key, value in weights.items():
            try:
                config['weights'][key] = float(value)
            except (TypeError, ValueError, KeyError):
                continue
    return config


def _parse_rating(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                return float(text)
            except Exception:
                return None
        if isinstance(parsed, dict):
            rating = parsed.get('rating')
            try:
                return float(rating)
            except Exception:
                return None
        try:
            return float(parsed)
        except Exception:
            return None
    return None


def _normalize_rating(value: Any) -> float:
    rating = _parse_rating(value)
    if rating is None:
        return 0.0
    return max(0.0, min(1.0, (rating - 1.0) / 4.0))


def _assistant_messages(trajectory: list[dict]) -> list[str]:
    return [str(step.get('assistant_message', '')) for step in trajectory]


def _task_target(snapshot: dict) -> str:
    candidate_state = snapshot.get('candidate_state', {})
    target = candidate_state.get('ground_truth_item', {})
    if isinstance(target, dict):
        return str(target.get('ItemName', '')).lower()
    return ''


def _action_type(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith('Ask['):
        return 'ask'
    if stripped.startswith('Recommend['):
        return 'recommend'
    if stripped.startswith('Search['):
        return 'search'
    return 'generic'


def compute_step_breakdown(step: dict, snapshot: dict, branch_type: str) -> dict:
    ratings = [
        _normalize_rating(step.get('recommendation_satisfaction')),
        _normalize_rating(step.get('action_satisfaction')),
        _normalize_rating(step.get('expression_satisfaction')),
    ]
    ratings = [value for value in ratings if value is not None]
    satisfaction_signal = mean(ratings) if ratings else 0.0

    assistant_message = str(step.get('assistant_message', ''))
    action_type = _action_type(assistant_message)
    task_type = str(snapshot.get('task_type', 'generic')).lower()
    target = _task_target(snapshot)
    assistant_lower = assistant_message.lower()

    task_success = 0.0
    recommendation_relevance = 0.0
    if task_type == 'recommend':
        task_success = 1.0 if target and target in assistant_lower else 0.0
        recommendation_relevance = task_success
    elif task_type == 'ask':
        task_success = 1.0 if action_type == 'ask' else 0.0
        recommendation_relevance = 0.7 if action_type == 'ask' else 0.0
    elif task_type == 'search':
        task_success = 1.0 if action_type == 'search' else 0.0
        recommendation_relevance = 0.7 if action_type == 'search' else 0.0
    else:
        task_success = 0.5 if action_type != 'generic' else 0.0
        recommendation_relevance = 0.5 if action_type != 'generic' else 0.0

    permanent_markers = ['permanent', 'standing rule', 'forever', 'durable', 'always', 'hard filter']
    constraint_satisfaction = 1.0
    if branch_type == 'over_apply' or any(marker in assistant_lower for marker in permanent_markers):
        constraint_satisfaction = 0.0 if branch_type == 'over_apply' else 0.5

    user_active = bool(step.get('user_active', True))
    continuation_signal = 1.0 if user_active else 0.0
    tool_failure = 1.0 if str(step.get('tool_status', 'NO_TOOL')).upper() == 'FAIL' else 0.0
    parse_failure = 1.0 if str(step.get('parser_status', 'OK')).upper() not in {'OK', 'RECOVERED'} else 0.0

    return {
        'task_success': task_success,
        'satisfaction_signal': satisfaction_signal,
        'constraint_satisfaction': constraint_satisfaction,
        'continuation_signal': continuation_signal,
        'recommendation_relevance': recommendation_relevance,
        'tool_failure': tool_failure,
        'parse_failure': parse_failure,
    }


def compute_rollout_utility(branch_row: dict, config: dict | None = None) -> dict:
    config = config or load_weight_config(None)
    weights = config.get('weights', DEFAULT_WEIGHTS)
    trajectory = list(branch_row.get('trajectory', []))
    snapshot = dict(branch_row.get('snapshot', {}))
    branch_type = str(branch_row.get('branch_type', 'generic'))

    step_breakdowns = [compute_step_breakdown(step, snapshot, branch_type) for step in trajectory]
    if step_breakdowns:
        task_success = mean(item['task_success'] for item in step_breakdowns)
        satisfaction_signal = mean(item['satisfaction_signal'] for item in step_breakdowns)
        constraint_satisfaction = mean(item['constraint_satisfaction'] for item in step_breakdowns)
        continuation_signal = mean(item['continuation_signal'] for item in step_breakdowns)
        recommendation_relevance = mean(item['recommendation_relevance'] for item in step_breakdowns)
        tool_failure = max(item['tool_failure'] for item in step_breakdowns)
        parse_failure = max(item['parse_failure'] for item in step_breakdowns)
    else:
        task_success = 0.0
        satisfaction_signal = 0.0
        constraint_satisfaction = 0.0
        continuation_signal = 0.0
        recommendation_relevance = 0.0
        tool_failure = 1.0
        parse_failure = 1.0

    horizon = max(1, int(branch_row.get('horizon', len(trajectory) or 1)))
    extra_turn_cost = max(0, len(trajectory) - 1) / float(horizon)
    assistant_messages = _assistant_messages(trajectory)
    unique_assistant_messages = len(set(assistant_messages))
    repetition_penalty = 0.0
    if assistant_messages:
        repetition_penalty = 1.0 - (unique_assistant_messages / float(len(assistant_messages)))

    utility_total = (
        weights.get('task_success', DEFAULT_WEIGHTS['task_success']) * task_success
        + weights.get('satisfaction_signal', DEFAULT_WEIGHTS['satisfaction_signal']) * satisfaction_signal
        + weights.get('constraint_satisfaction', DEFAULT_WEIGHTS['constraint_satisfaction']) * constraint_satisfaction
        + weights.get('continuation_signal', DEFAULT_WEIGHTS['continuation_signal']) * continuation_signal
        + weights.get('recommendation_relevance', DEFAULT_WEIGHTS['recommendation_relevance']) * recommendation_relevance
        - weights.get('extra_turn_cost', DEFAULT_WEIGHTS['extra_turn_cost']) * extra_turn_cost
        - weights.get('repetition_penalty', DEFAULT_WEIGHTS['repetition_penalty']) * repetition_penalty
        - weights.get('tool_failure', DEFAULT_WEIGHTS['tool_failure']) * tool_failure
        - weights.get('parse_failure', DEFAULT_WEIGHTS['parse_failure']) * parse_failure
    )

    breakdown = {
        'task_success': task_success,
        'satisfaction_signal': satisfaction_signal,
        'constraint_satisfaction': constraint_satisfaction,
        'continuation_signal': continuation_signal,
        'recommendation_relevance': recommendation_relevance,
        'extra_turn_cost': extra_turn_cost,
        'repetition_penalty': repetition_penalty,
        'tool_failure': tool_failure,
        'parse_failure': parse_failure,
        'utility_total': utility_total,
        'weights': copy.deepcopy(weights),
        'step_breakdowns': step_breakdowns,
    }
    return breakdown
