from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from user_simulator.evaluation.real_branch_policy import build_branch_policy
from user_simulator.evaluation.real_branch_replay_schema import (
    PROVENANCE_REAL_USER_SIM_REPLAY,
    ReplayBranch,
    ReplayPair,
    ReplaySnapshot,
    action_text,
    restore_memory_state,
)
from user_simulator.evaluation.real_branch_utility import compute_rollout_utility, load_weight_config
from user_simulator.user_agent_env_v1 import UserAgentEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--branches', nargs='+', default=['follow', 'ignore', 'over_apply'])
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--max-snapshots', type=int, default=10)
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--utility-config', default='configs/server184/real_branch_utility.yaml')
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')


def parse_user_response(payload: str) -> tuple[dict, str]:
    try:
        return json.loads(payload), 'OK'
    except Exception:
        return {'response': payload}, 'PARSE_FAIL'


def parse_user_policy(payload: str | None) -> tuple[dict | None, str]:
    if not payload:
        return None, 'NO_POLICY'
    try:
        return json.loads(payload), 'OK'
    except Exception:
        return {'raw': payload}, 'PARSE_FAIL'


def build_effective_config(snapshot: ReplaySnapshot, args: argparse.Namespace, output_dir: Path) -> Path:
    source_trace = snapshot.source_trace or {}
    config_path = Path(source_trace.get('config_path', '')) if source_trace.get('config_path') else None
    if args.base_url or args.model or args.api_key:
        effective = output_dir / f'{snapshot.snapshot_id.replace(":", "_")}_runtime_api_config.json'
        effective.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            effective,
            {
                'vllm': {
                    'base_url': args.base_url or source_trace.get('base_url') or 'http://127.0.0.1:8000/v1',
                    'api_key': args.api_key or source_trace.get('api_key') or 'EMPTY',
                    'model_path': args.model or source_trace.get('model_name') or 'qwen2.5-3b-instruct',
                }
            },
        )
        return effective
    if config_path and config_path.exists():
        return config_path
    raise FileNotFoundError('Missing runtime_api_config path for replay')


def restore_env_from_snapshot(snapshot: ReplaySnapshot, args: argparse.Namespace, output_dir: Path) -> UserAgentEnv:
    source_trace = snapshot.source_trace or {}
    persona_path = source_trace.get('persona_path')
    if not persona_path:
        raise FileNotFoundError('snapshot is missing persona_path in source_trace')
    domain = source_trace.get('domain', 'Book')
    memory_mode = source_trace.get('memory_mode', 'critiquescope')
    config_path = build_effective_config(snapshot, args, output_dir)
    format_path = source_trace.get('format_path', 'configs/server184')
    user_id = int(source_trace.get('user_id', snapshot.metadata.get('user_id', 0)))
    item_id = int(source_trace.get('item_id', snapshot.metadata.get('item_id', 0)))

    env = UserAgentEnv(
        persona_path=persona_path,
        user_id=user_id,
        item_id=item_id,
        config_path=str(config_path),
        format_path=format_path,
        domain=domain,
        model_type='vllm',
        memory_mode=memory_mode,
    )
    env.persona = copy.deepcopy(snapshot.persona)
    env.item = copy.deepcopy(snapshot.candidate_state.get('selected_item', env.item))
    env.user_simulator.persona = env.persona
    env.user_simulator.item = env.item
    if hasattr(env.dialogue_history, 'history'):
        env.dialogue_history.history = copy.deepcopy(snapshot.conversation_history)
    env.user_simulator.raw_history = copy.deepcopy(snapshot.user_state.get('raw_history', []))
    restored_memory = restore_memory_state(
        memory_mode,
        snapshot.user_state.get('critique_memory') or snapshot.user_state.get('structured_memory'),
    )
    if memory_mode == 'critiquescope':
        env.user_simulator.critique_memory = restored_memory
        env.user_simulator.structured_memory = None
    elif memory_mode == 'structured':
        env.user_simulator.structured_memory = restored_memory
        env.user_simulator.critique_memory = None
    elif memory_mode in {'flat', 'time_decay'}:
        env.user_simulator.critique_memory = None
        env.user_simulator.structured_memory = None
    return env


def run_branch_rollout(
    snapshot: ReplaySnapshot,
    branch_type: str,
    args: argparse.Namespace,
    output_dir: Path,
    config: dict,
) -> tuple[dict, list[dict], list[dict]]:
    env = restore_env_from_snapshot(snapshot, args, output_dir)
    policy_action = build_branch_policy(snapshot, branch_type)
    assistant_message = policy_action['assistant_message']

    trajectory: list[dict] = []
    raw_requests: list[dict] = []
    raw_responses: list[dict] = []
    branch_status = 'COMPLETED'

    for turn_index in range(args.horizon):
        step_start = time.perf_counter()
        step_result = env.step(assistant_message)
        latency_seconds = time.perf_counter() - step_start

        user_response_raw = step_result.get('user_response', '')
        parsed_user_response, parser_status = parse_user_response(user_response_raw)
        user_policy_raw = step_result.get('user_policy')
        parsed_user_policy, policy_status = parse_user_policy(user_policy_raw)

        terminal_status = 'ACTIVE'
        if parsed_user_policy and parsed_user_policy.get('policy') == 'end_conversation':
            terminal_status = 'END_CONVERSATION'
        if not getattr(env.user_simulator, 'active', True):
            terminal_status = 'TERMINATED'

        step_row = {
            'snapshot_id': snapshot.snapshot_id,
            'episode_id': snapshot.episode_id,
            'turn_index': turn_index,
            'branch_type': branch_type,
            'task_type': snapshot.task_type,
            'assistant_message': assistant_message,
            'user_response_raw': user_response_raw,
            'user_response': parsed_user_response,
            'user_policy_raw': user_policy_raw,
            'user_policy': parsed_user_policy,
            'recommendation_satisfaction': step_result.get('recommendation_satisfaction'),
            'action_satisfaction': step_result.get('action_satisfaction'),
            'expression_satisfaction': step_result.get('expression_satisfaction'),
            'latency_seconds': latency_seconds,
            'parser_status': parser_status,
            'policy_parser_status': policy_status,
            'tool_status': 'NO_TOOL',
            'terminal_status': terminal_status,
            'user_active': bool(getattr(env.user_simulator, 'active', True)),
            'source_trace': copy.deepcopy(snapshot.source_trace),
            'provenance': PROVENANCE_REAL_USER_SIM_REPLAY,
        }
        trajectory.append(step_row)
        raw_requests.append(
            {
                'snapshot_id': snapshot.snapshot_id,
                'branch_type': branch_type,
                'turn_index': turn_index,
                'assistant_message': assistant_message,
                'source_trace': copy.deepcopy(snapshot.source_trace),
            }
        )
        raw_responses.append(
            {
                'snapshot_id': snapshot.snapshot_id,
                'branch_type': branch_type,
                'turn_index': turn_index,
                'response': user_response_raw,
                'user_policy': user_policy_raw,
                'parser_status': parser_status,
                'policy_parser_status': policy_status,
                'source_trace': copy.deepcopy(snapshot.source_trace),
            }
        )

        if parser_status != 'OK' and policy_status != 'OK':
            branch_status = 'PARTIAL_PARSE_FAILURE'
        if terminal_status != 'ACTIVE':
            break

    branch_dict = ReplayBranch(
        snapshot_id=snapshot.snapshot_id,
        branch_type=branch_type,
        task_type=snapshot.task_type,
        policy_action=policy_action,
        trajectory=trajectory,
        utility_breakdown={},
        utility_total=0.0,
        status=branch_status if trajectory else 'BLOCKED_NO_TRAJECTORY',
        snapshot=snapshot.to_dict(),
        source_trace=copy.deepcopy(snapshot.source_trace),
        provenance=PROVENANCE_REAL_USER_SIM_REPLAY,
        metadata={
            'episode_id': snapshot.episode_id,
            'turn': snapshot.turn,
            'horizon': args.horizon,
            'seed': args.seed,
            'branch_policy_source': policy_action.get('policy_source'),
            'branch_policy_description': policy_action.get('policy_description'),
        },
    ).to_dict()
    branch_dict['utility_breakdown'] = compute_rollout_utility(branch_dict, config)
    branch_dict['utility_total'] = branch_dict['utility_breakdown']['utility_total']
    branch_dict['status'] = branch_status if trajectory else 'BLOCKED_NO_TRAJECTORY'
    return branch_dict, raw_requests, raw_responses


def build_pairs(branch_rows: list[dict]) -> list[dict]:
    by_snapshot: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in branch_rows:
        by_snapshot[str(row.get('snapshot_id', ''))][str(row.get('branch_type', ''))] = row

    pairs: list[dict] = []
    for snapshot_id, branch_map in by_snapshot.items():
        follow = branch_map.get('follow')
        if not follow:
            continue
        for rejected_branch in ['ignore', 'over_apply']:
            rejected = branch_map.get(rejected_branch)
            if not rejected:
                continue
            uplift = float(follow['utility_total']) - float(rejected['utility_total'])
            pair_status = 'positive' if uplift > 0 else 'zero' if uplift == 0 else 'negative'
            pair = ReplayPair(
                snapshot_id=snapshot_id,
                chosen_branch='follow',
                rejected_branch=rejected_branch,
                chosen_utility=float(follow['utility_total']),
                rejected_utility=float(rejected['utility_total']),
                uplift=uplift,
                source_trace=copy.deepcopy(follow.get('source_trace', {})),
                provenance=PROVENANCE_REAL_USER_SIM_REPLAY,
                metadata={
                    'task_type': follow.get('task_type'),
                    'pair_status': pair_status,
                    'branch_utilities': {
                        'follow': float(follow['utility_total']),
                        'rejected': float(rejected['utility_total']),
                    },
                },
            ).to_dict()
            pairs.append(pair)
    return pairs


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_weight_config(args.utility_config)

    snapshot_rows = read_jsonl(Path(args.snapshots))
    snapshots = [ReplaySnapshot(**row) for row in snapshot_rows[: args.max_snapshots]]
    branch_rows: list[dict] = []
    raw_requests: list[dict] = []
    raw_responses: list[dict] = []
    failures: list[dict] = []

    for snapshot in snapshots:
        for branch_type in args.branches:
            try:
                branch_dict, requests, responses = run_branch_rollout(snapshot, branch_type, args, output_dir, config)
                branch_rows.append(branch_dict)
                raw_requests.extend(requests)
                raw_responses.extend(responses)
            except Exception as exc:
                failures.append(
                    {
                        'snapshot_id': snapshot.snapshot_id,
                        'branch_type': branch_type,
                        'error': str(exc),
                        'exception_type': type(exc).__name__,
                        'provenance': PROVENANCE_REAL_USER_SIM_REPLAY,
                    }
                )

    replay_pairs = build_pairs(branch_rows)
    write_jsonl(output_dir / 'branch_rollouts.jsonl', branch_rows)
    write_jsonl(output_dir / 'replay_pairs.jsonl', replay_pairs)
    write_jsonl(output_dir / 'raw_requests.jsonl', raw_requests)
    write_jsonl(output_dir / 'raw_responses.jsonl', raw_responses)
    write_jsonl(output_dir / 'replay_failures.jsonl', failures)

    summary = {
        'status': 'PASS' if branch_rows and not failures else 'PARTIAL' if branch_rows else 'BLOCKED_REAL_USER_SIM_REPLAY',
        'snapshot_count': len(snapshots),
        'branch_count': len(branch_rows),
        'pair_count': len(replay_pairs),
        'follow_count': sum(1 for row in branch_rows if row.get('branch_type') == 'follow'),
        'ignore_count': sum(1 for row in branch_rows if row.get('branch_type') == 'ignore'),
        'over_apply_count': sum(1 for row in branch_rows if row.get('branch_type') == 'over_apply'),
        'positive_pair_count': sum(1 for row in replay_pairs if row.get('uplift', 0) > 0),
        'zero_pair_count': sum(1 for row in replay_pairs if row.get('uplift', 0) == 0),
        'negative_pair_count': sum(1 for row in replay_pairs if row.get('uplift', 0) < 0),
        'failure_count': len(failures),
        'provenance_counts': dict(Counter(row.get('provenance', 'UNKNOWN') for row in branch_rows)),
        'utility_config': args.utility_config,
        'branches': args.branches,
        'horizon': args.horizon,
        'seed': args.seed,
        'snapshots_path': args.snapshots,
        'output_dir': str(output_dir),
    }
    write_json(output_dir / 'run_metadata.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not branch_rows:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
