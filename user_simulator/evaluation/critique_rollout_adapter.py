"""Adapter for turning CritiqueWorld rollouts into benchmark scenarios.

The adapter keeps the deterministic built-in scenario path intact, and also
supports real GIMO replay branch rows produced by the new replay pipeline.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import List

from user_simulator.evaluation.critique_scope_eval import DEFAULT_SCENARIOS
from user_simulator.evaluation.critique_uplift_pairs import build_pairs
from user_simulator.evaluation.validate_critique_scenarios import validate_scenario

REQUIRED_BRANCHES = ['follow_value', 'ignore_value', 'over_apply_value']


def load_rollouts(path: str | None) -> List[dict]:
    if not path:
        return DEFAULT_SCENARIOS
    rows: list[dict] = []
    with Path(path).open('r', encoding='utf-8') as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if 'snapshot_id' not in row or 'branch_type' not in row:
                validate_rollout(row, line_no)
            rows.append(row)
    return rows


def validate_rollout(row: dict, line_no: int) -> None:
    errors = validate_scenario(row, index=line_no)
    if errors:
        raise ValueError(f'line {line_no}: ' + '; '.join(errors))
    for branch in REQUIRED_BRANCHES:
        if not isinstance(row.get(branch), list) or not all(isinstance(value, (int, float)) for value in row[branch]):
            raise ValueError(f'line {line_no}: {branch} must be a list of numbers')


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def conversation_to_sharegpt(history: list[dict]) -> list[dict]:
    converted = []
    for message in history:
        role = str(message.get('role', 'user')).lower()
        value = message.get('content', message.get('value', ''))
        converted.append({'from': 'human' if role == 'user' else 'gpt' if role == 'assistant' else role, 'value': value})
    return converted


def trajectory_to_training_text(rows: list[dict]) -> str:
    parts = []
    for row in rows:
        assistant_message = str(row.get('assistant_message', '')).strip()
        user_response = row.get('user_response', {})
        if isinstance(user_response, dict):
            user_text = user_response.get('response', '')
        else:
            user_text = str(user_response)
        utility = float(row.get('step_utility', row.get('instant_utility', 0.0)) or 0.0)
        parts.append(
            f"turn={row.get('turn_index', row.get('turn', 0))} assistant={assistant_message} user={user_text} utility={utility:.3f}"
        )
    return '\n'.join(parts)


def build_scenario_artifacts(scenarios: list[dict], output_dir: Path) -> dict:
    normalized_path = output_dir / 'normalized_scenarios.jsonl'
    pair_path = output_dir / 'critique_pairs.jsonl'
    write_jsonl(normalized_path, scenarios)
    pairs = build_pairs(scenarios)
    write_jsonl(pair_path, pairs)
    metadata = {
        'status': 'SMOKE_TEST_ONLY',
        'source_mode': 'CritiqueWorld',
        'input': 'DEFAULT_SCENARIOS',
        'scenario_count': len(scenarios),
        'pair_count': len(pairs),
        'output_dir': str(output_dir),
    }
    write_json(output_dir / 'adapter_metadata.json', metadata)
    return metadata


def build_replay_pair_rows(branch_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    by_snapshot: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in branch_rows:
        by_snapshot[str(row.get('snapshot_id', ''))][str(row.get('branch_type', ''))] = row

    replay_pairs: list[dict] = []
    dpo_pairs: list[dict] = []
    cdpo_pairs: list[dict] = []

    for snapshot_id, branch_map in by_snapshot.items():
        follow = branch_map.get('follow')
        if not follow:
            continue
        snapshot = follow.get('snapshot', {}) or next(iter(branch_map.values())).get('snapshot', {})
        source_trace = follow.get('source_trace', {}) or snapshot.get('source_trace', {})
        task_type = snapshot.get('task_type') or follow.get('task_type', 'unknown')
        history = conversation_to_sharegpt(snapshot.get('conversation_history', []))

        for rejected_branch in ['ignore', 'over_apply']:
            rejected = branch_map.get(rejected_branch)
            if not rejected:
                continue
            chosen_utility = float(follow.get('utility_total', 0.0))
            rejected_utility = float(rejected.get('utility_total', 0.0))
            uplift = chosen_utility - rejected_utility
            pair_status = 'positive' if uplift > 0 else 'zero' if uplift == 0 else 'negative'
            pair_id = f'{snapshot_id}:{rejected_branch}'
            pair_row = {
                'id': pair_id,
                'scenario': task_type or 'real_branch_replay',
                'seed': follow.get('metadata', {}).get('seed', 0),
                'method': 'real_branch_replay',
                'parser_mode': 'real_user_sim_replay',
                'conversations': history,
                'chosen': {
                    'branch': 'follow',
                    'policy': follow.get('policy_action', {}).get('assistant_message', ''),
                    'trajectory': trajectory_to_training_text(follow.get('trajectory', [])),
                },
                'rejected': {
                    'branch': rejected_branch,
                    'policy': rejected.get('policy_action', {}).get('assistant_message', ''),
                    'trajectory': trajectory_to_training_text(rejected.get('trajectory', [])),
                },
                'score_delta': uplift,
                'provenance': follow.get('provenance', 'REAL_USER_SIM_REPLAY'),
                'source_trace': copy.deepcopy(source_trace),
                'metadata': {
                    'format': 'llamafactory_dpo_bridge',
                    'source': 'RealBranchReplay',
                    'proxy': 'controlled real user simulator replay proxy',
                    'provenance': follow.get('provenance', 'REAL_USER_SIM_REPLAY'),
                    'snapshot_id': snapshot_id,
                    'task_type': task_type,
                    'chosen_utility': chosen_utility,
                    'rejected_utility': rejected_utility,
                    'pair_status': pair_status,
                    'utility_breakdown': {
                        'follow': follow.get('utility_breakdown', {}),
                        'rejected': rejected.get('utility_breakdown', {}),
                    },
                    'source_trace': source_trace,
                },
            }
            replay_pairs.append(pair_row)
            dpo_pairs.append(pair_row)
            if uplift > 0:
                cdpo_pairs.append(pair_row)

    return replay_pairs, dpo_pairs, cdpo_pairs


def build_replay_artifacts(branch_rows: list[dict], output_dir: Path) -> dict:
    replay_pairs, dpo_pairs, cdpo_pairs = build_replay_pair_rows(branch_rows)
    write_jsonl(output_dir / 'adapter_input.jsonl', branch_rows)
    write_jsonl(output_dir / 'branch_rollouts.jsonl', branch_rows)
    write_jsonl(output_dir / 'replay_pairs.jsonl', replay_pairs)
    write_jsonl(output_dir / 'dpo_pairs.jsonl', dpo_pairs)
    write_jsonl(output_dir / 'cdpo_pairs.jsonl', cdpo_pairs)
    metadata = {
        'status': 'SMOKE_TEST_ONLY',
        'source_mode': 'RealBranchReplay',
        'input': 'branch_rollouts.jsonl',
        'branch_count': len(branch_rows),
        'pair_count': len(replay_pairs),
        'positive_pair_count': len(cdpo_pairs),
        'zero_pair_count': sum(1 for row in replay_pairs if row.get('score_delta', 0) == 0),
        'negative_pair_count': sum(1 for row in replay_pairs if row.get('score_delta', 0) < 0),
        'output_dir': str(output_dir),
    }
    write_json(output_dir / 'adapter_metadata.json', metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Optional real rollout JSONL.')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input:
        scenarios = load_rollouts(None)
        metadata = build_scenario_artifacts(scenarios, output_dir)
        print(json.dumps({'status': 'ok', **metadata, 'output_dir': str(output_dir)}, indent=2))
        return

    rows = load_rollouts(args.input)
    if rows and all('branch_type' in row and 'snapshot_id' in row for row in rows):
        metadata = build_replay_artifacts(rows, output_dir)
        print(json.dumps({'status': 'ok', **metadata, 'output_dir': str(output_dir)}, indent=2))
    else:
        metadata = build_scenario_artifacts(rows, output_dir)
        print(json.dumps({'status': 'ok', **metadata, 'output_dir': str(output_dir)}, indent=2))


if __name__ == '__main__':
    main()
