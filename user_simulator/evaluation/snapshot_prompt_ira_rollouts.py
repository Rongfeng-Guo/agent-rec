from __future__ import annotations

import argparse
import copy
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from user_simulator.evaluation.real_branch_replay_schema import (
    PROVENANCE_REAL_TRACE,
    ReplaySnapshot,
    action_text,
    serialize_critique_memory,
    serialize_dialogue_history,
    serialize_memory_state,
    stable_snapshot_id,
)
from user_simulator.user_agent_env_v1 import UserAgentEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-ira-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config-path', default=None)
    parser.add_argument('--format-path', default='configs/server184')
    parser.add_argument('--persona-path', default=None)
    parser.add_argument('--domain', default=None)
    parser.add_argument('--user-start', type=int, default=None)
    parser.add_argument('--item-id', type=int, default=None)
    parser.add_argument('--max-episodes', type=int, default=1)
    parser.add_argument('--max-snapshots', type=int, default=8)
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--memory-mode', default=None)
    parser.add_argument('--gpe-log-dir', default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return rows
    if text.startswith('['):
        loaded = json.loads(text)
        return loaded if isinstance(loaded, list) else []
    for line in text.splitlines():
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


def load_runtime_config(prompt_ira_dir: Path) -> dict:
    config_path = prompt_ira_dir / 'runtime_api_config.json'
    if config_path.exists():
        return read_json(config_path)
    return {
        'vllm': {
            'base_url': 'http://127.0.0.1:8000/v1',
            'api_key': 'EMPTY',
            'model_path': 'qwen2.5-3b-instruct',
        }
    }


def load_refine_log(gpe_log_dir: Path | None) -> dict[str, dict]:
    if not gpe_log_dir:
        return {}
    candidates = [gpe_log_dir / 'Book_refine_log_sample1.json']
    candidates.extend(sorted(gpe_log_dir.glob('*_refine_log_*.json')))
    refine_log = next((path for path in candidates if path.exists()), None)
    if not refine_log:
        return {}
    rows = read_jsonl(refine_log)
    mapping: dict[str, dict] = {}
    for row in rows:
        task_type = str(row.get('task_type', '')).lower()
        if task_type and task_type not in mapping:
            mapping[task_type] = row
    return mapping


def build_candidate_state(*, prompt_row: dict, response_row: dict, source_label: str) -> dict:
    ground_truth = copy.deepcopy(prompt_row.get('ground_truth', {}))
    candidate_state = {
        'source_label': source_label,
        'ground_truth_item': ground_truth,
        'retrieved_items': copy.deepcopy(prompt_row.get('retrieved_items', [])),
        'instruction': prompt_row.get('Instruction', prompt_row.get('input', '')),
        'output': prompt_row.get('output', prompt_row.get('original_response', '')),
        'user_feedback': response_row.get('feedback_user', {}).get('response'),
        'feedback_metrics': copy.deepcopy(response_row.get('feedback_metrics', {})),
        'is_correct': response_row.get('is_correct'),
        'is_recall': response_row.get('is_recall'),
    }
    return candidate_state


def build_snapshot_payload(
    *,
    episode_id: str,
    turn: int,
    task_type: str,
    env: UserAgentEnv,
    original_action: str,
    refined_action: str | None,
    critique: dict | None,
    source_trace: dict,
    prompt_ira_dir: Path,
    persona_path: str,
    memory_mode: str,
    candidate_state: dict,
) -> ReplaySnapshot:
    user_memory = env.user_simulator.critique_memory if env.user_simulator.critique_memory is not None else env.user_simulator.structured_memory
    user_state = {
        'memory_mode': memory_mode,
        'active': getattr(env.user_simulator, 'active', True),
        'turn': getattr(user_memory, 'turn', None),
        'raw_history': copy.deepcopy(getattr(env.user_simulator, 'raw_history', [])),
        'critique_memory': serialize_critique_memory(env.user_simulator.critique_memory),
        'structured_memory': serialize_memory_state('structured', env.user_simulator.structured_memory),
    }
    conversation_history = serialize_dialogue_history(env.get_dialogue_history())
    candidate_state = copy.deepcopy(candidate_state)
    payload = {
        'episode_id': episode_id,
        'turn': turn,
        'task_type': task_type,
        'user_state': user_state,
        'persona': copy.deepcopy(env.persona),
        'conversation_history': conversation_history,
        'candidate_state': candidate_state,
        'original_action': {
            'text': original_action,
            'action_type': task_type,
            'source': 'PROMPT_IRA_REAL_TRACE',
            'task_type': task_type,
        },
        'refined_action': (
            {
                'text': refined_action,
                'action_type': task_type,
                'source': 'GPE_HAP_REAL_TRACE',
                'task_type': task_type,
            }
            if refined_action
            else None
        ),
        'critique': critique,
        'source_trace': copy.deepcopy(source_trace),
        'metadata': {
            'prompt_ira_dir': str(prompt_ira_dir),
            'persona_path': persona_path,
            'memory_mode': memory_mode,
        },
        'provenance': PROVENANCE_REAL_TRACE,
    }
    snapshot_id = stable_snapshot_id(payload)
    return ReplaySnapshot(
        snapshot_id=snapshot_id,
        episode_id=episode_id,
        turn=turn,
        task_type=task_type,
        user_state=user_state,
        persona=copy.deepcopy(env.persona),
        conversation_history=conversation_history,
        candidate_state=candidate_state,
        original_action=payload['original_action'],
        refined_action=payload['refined_action'],
        critique=critique,
        source_trace=copy.deepcopy(source_trace),
        metadata=payload['metadata'],
        provenance=PROVENANCE_REAL_TRACE,
    )


def replay_prompt_trace(
    *,
    prompt_ira_dir: Path,
    request: dict,
    response: dict,
    runtime_cfg: dict,
    gpe_refine_map: dict[str, dict],
    args: argparse.Namespace,
) -> tuple[list[ReplaySnapshot], list[dict], dict]:
    vllm_cfg = runtime_cfg.get('vllm', {})
    domain = args.domain or request.get('domain', 'Book')
    persona_path = args.persona_path or request.get('persona_path')
    if not persona_path:
        raise SystemExit('BLOCKED_REPLAY_SNAPSHOT_INCOMPLETE: persona_path is missing')
    memory_mode = args.memory_mode or request.get('memory_mode', 'critiquescope')
    config_path = args.config_path or str(prompt_ira_dir / 'runtime_api_config.json')
    format_path = args.format_path
    user_id = args.user_start if args.user_start is not None else int(request.get('user_id', 0))
    item_id = args.item_id if args.item_id is not None else int(request.get('item_id', 0))
    base_url = args.base_url or vllm_cfg.get('base_url', 'http://127.0.0.1:8000/v1')
    api_key = args.api_key or vllm_cfg.get('api_key', 'EMPTY')
    model_name = args.model_name or vllm_cfg.get('model_path', 'qwen2.5-3b-instruct')
    gpe_log_dir = Path(args.gpe_log_dir) if args.gpe_log_dir else prompt_ira_dir.parent.parent / 'gpe_hap_smoke' / 'latest_real'

    source_trace = {
        'source_kind': 'REAL_TRACE',
        'prompt_ira_dir': str(prompt_ira_dir),
        'request_path': str(prompt_ira_dir / 'request.json'),
        'response_path': str(prompt_ira_dir / 'response.json'),
        'runtime_api_config_path': str(prompt_ira_dir / 'runtime_api_config.json'),
        'gpe_log_dir': str(gpe_log_dir),
        'gpe_refine_log_path': str(next((p for p in [gpe_log_dir / 'Book_refine_log_sample1.json'] if p.exists()), gpe_log_dir)),
        'config_path': config_path,
        'format_path': format_path,
        'base_url': base_url,
        'api_key': api_key,
        'model_name': model_name,
        'domain': domain,
        'memory_mode': memory_mode,
        'persona_path': persona_path,
        'user_id': user_id,
        'item_id': item_id,
    }

    ask_rows = read_jsonl(prompt_ira_dir / 'bpo' / 'ask_data.json')
    recommend_rows = read_jsonl(prompt_ira_dir / 'bpo' / 'recommend_data.json')
    ask_row = ask_rows[0] if ask_rows else {}
    recommend_row = recommend_rows[0] if recommend_rows else {}

    ask_refine = action_text(gpe_refine_map.get('ask', {}).get('best_refinement')) or None
    recommend_refine = action_text(gpe_refine_map.get('recommend', {}).get('best_refinement')) or None

    env = UserAgentEnv(
        persona_path=persona_path,
        user_id=user_id,
        item_id=item_id,
        config_path=config_path,
        format_path=format_path,
        domain=domain,
        model_type='vllm',
        memory_mode=memory_mode,
    )

    snapshots: list[ReplaySnapshot] = []
    raw_rows: list[dict] = []

    first_start = datetime.now().timestamp()
    first_turn = env.step(None)
    first_user = json.loads(first_turn['user_response'])
    raw_rows.append(
        {
            'episode_id': f'{domain}:{user_id}:{item_id}:prompt_ira',
            'turn': 0,
            'event': 'initial_user',
            'latency_seconds': datetime.now().timestamp() - first_start,
            'user_response': first_user,
            'source_trace': copy.deepcopy(source_trace),
        }
    )

    ask_snapshot = build_snapshot_payload(
        episode_id=f'{domain}:{user_id}:{item_id}:prompt_ira',
        turn=0,
        task_type='ask',
        env=env,
        original_action=response.get('ask_response', ''),
        refined_action=ask_refine,
        critique=response.get('first_user'),
        source_trace=source_trace,
        prompt_ira_dir=prompt_ira_dir,
        persona_path=persona_path,
        memory_mode=memory_mode,
        candidate_state=build_candidate_state(prompt_row=ask_row, response_row=response, source_label='prompt_ira_ask'),
    )
    snapshots.append(ask_snapshot)

    ask_start = datetime.now().timestamp()
    ask_turn = env.step(response.get('ask_response', ''))
    ask_user = json.loads(ask_turn['user_response'])
    raw_rows.append(
        {
            'episode_id': f'{domain}:{user_id}:{item_id}:prompt_ira',
            'turn': 1,
            'event': 'ask_action',
            'assistant_message': response.get('ask_response', ''),
            'latency_seconds': datetime.now().timestamp() - ask_start,
            'user_response': ask_user,
            'source_trace': copy.deepcopy(source_trace),
        }
    )

    recommend_snapshot = build_snapshot_payload(
        episode_id=f'{domain}:{user_id}:{item_id}:prompt_ira',
        turn=1,
        task_type='recommend',
        env=env,
        original_action=response.get('recommend_response', ''),
        refined_action=recommend_refine,
        critique=response.get('feedback_user'),
        source_trace=source_trace,
        prompt_ira_dir=prompt_ira_dir,
        persona_path=persona_path,
        memory_mode=memory_mode,
        candidate_state=build_candidate_state(prompt_row=recommend_row, response_row=response, source_label='prompt_ira_recommend'),
    )
    snapshots.append(recommend_snapshot)

    recommend_start = datetime.now().timestamp()
    recommend_turn = env.step(response.get('recommend_response', ''))
    recommend_user = json.loads(recommend_turn['user_response'])
    raw_rows.append(
        {
            'episode_id': f'{domain}:{user_id}:{item_id}:prompt_ira',
            'turn': 2,
            'event': 'recommend_action',
            'assistant_message': response.get('recommend_response', ''),
            'latency_seconds': datetime.now().timestamp() - recommend_start,
            'user_response': recommend_user,
            'feedback_metrics': copy.deepcopy(response.get('feedback_metrics', {})),
            'source_trace': copy.deepcopy(source_trace),
        }
    )

    episode_summary = {
        'episode_id': f'{domain}:{user_id}:{item_id}:prompt_ira',
        'status': 'COMPLETED_REAL_PROMPT_IRA_SMOKE',
        'domain': domain,
        'user_id': user_id,
        'item_id': item_id,
        'ask_action': response.get('ask_response', ''),
        'recommend_action': response.get('recommend_response', ''),
        'ask_refined_action': ask_refine,
        'recommend_refined_action': recommend_refine,
        'source_trace': copy.deepcopy(source_trace),
    }
    return snapshots, raw_rows, episode_summary


def main() -> None:
    args = parse_args()
    prompt_ira_dir = Path(args.prompt_ira_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    request = read_json(prompt_ira_dir / 'request.json')
    response = read_json(prompt_ira_dir / 'response.json')
    runtime_cfg = load_runtime_config(prompt_ira_dir)
    gpe_refine_map = load_refine_log(Path(args.gpe_log_dir) if args.gpe_log_dir else prompt_ira_dir.parent.parent / 'gpe_hap_smoke' / 'latest_real')

    snapshots, raw_rows, episode_summary = replay_prompt_trace(
        prompt_ira_dir=prompt_ira_dir,
        request=request,
        response=response,
        runtime_cfg=runtime_cfg,
        gpe_refine_map=gpe_refine_map,
        args=args,
    )

    snapshot_dicts = [snapshot.to_dict() for snapshot in snapshots[: args.max_snapshots]]
    required = {
        'snapshot_id',
        'episode_id',
        'turn',
        'task_type',
        'user_state',
        'persona',
        'conversation_history',
        'candidate_state',
        'original_action',
        'source_trace',
        'provenance',
    }
    missing_rows = []
    for row in snapshot_dicts:
        missing = [field for field in required if field not in row or row.get(field) in (None, '')]
        if missing:
            missing_rows.append(
                {
                    'snapshot_id': row.get('snapshot_id'),
                    'task_type': row.get('task_type'),
                    'missing_fields': missing,
                    'provenance': row.get('provenance'),
                }
            )

    status = 'PASS' if snapshot_dicts and not missing_rows else 'PARTIAL'
    if not snapshot_dicts:
        status = 'BLOCKED_REPLAY_SNAPSHOT_INCOMPLETE'

    write_jsonl(output_dir / 'replay_snapshots.jsonl', snapshot_dicts)
    write_jsonl(output_dir / 'raw_episode_trace.jsonl', raw_rows)
    write_jsonl(output_dir / 'missing_fields.jsonl', missing_rows)

    audit = {
        'status': status,
        'prompt_ira_dir': str(prompt_ira_dir),
        'snapshot_count': len(snapshot_dicts),
        'episode_count': 1,
        'task_type_counts': dict(Counter(row.get('task_type', 'UNKNOWN') for row in snapshot_dicts)),
        'provenance_counts': dict(Counter(row.get('provenance', 'UNKNOWN') for row in snapshot_dicts)),
        'missing_field_count': len(missing_rows),
        'missing_by_field': dict(Counter(field for row in missing_rows for field in row.get('missing_fields', []))),
        'episode_summary': episode_summary,
        'source_trace': episode_summary.get('source_trace', {}),
    }
    write_json(output_dir / 'snapshot_audit.json', audit)

    md_lines = [
        '# Prompt IRA Snapshot Audit',
        '',
        f'- status: `{status}`',
        f'- snapshot_count: `{audit["snapshot_count"]}`',
        f'- missing_field_count: `{audit["missing_field_count"]}`',
        f'- provenance_counts: `{json.dumps(audit["provenance_counts"], ensure_ascii=False)}`',
        f'- task_type_counts: `{json.dumps(audit["task_type_counts"], ensure_ascii=False)}`',
        '',
        '## Missing Fields',
    ]
    if not missing_rows:
        md_lines.append('- none')
    else:
        for row in missing_rows:
            md_lines.append(f'- `{row["snapshot_id"]}`: {", ".join(row["missing_fields"])}')
    (output_dir / 'snapshot_audit.md').write_text('\n'.join(md_lines) + '\n', encoding='utf-8')

    run_metadata = {
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'prompt_ira_dir': str(prompt_ira_dir),
        'output_dir': str(output_dir),
        'snapshot_count': len(snapshot_dicts),
        'missing_field_count': len(missing_rows),
        'source_trace': episode_summary.get('source_trace', {}),
    }
    write_json(output_dir / 'run_metadata.json', run_metadata)
    print(json.dumps(run_metadata, indent=2, ensure_ascii=False))
    if status == 'BLOCKED_REPLAY_SNAPSHOT_INCOMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
