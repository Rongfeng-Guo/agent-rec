import json
import subprocess
import sys
from pathlib import Path


def test_audit_script_passes_on_minimal_real_branch_input(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    snapshot = {
        'snapshot_id': 's1',
        'episode_id': 'ep1',
        'turn': 0,
        'task_type': 'recommend',
        'provenance': 'REAL_TRACE',
        'user_state': {},
        'persona': {},
        'conversation_history': [],
        'candidate_state': {},
        'original_action': {'text': 'Recommend[A | reason]'},
        'refined_action': {'text': 'Recommend[A | refined]'},
        'critique': None,
        'source_trace': {'persona_path': 'user_simulator/task/Book_test.jsonl'},
        'metadata': {},
    }
    branches = []
    for branch_type, utility in [('follow', 2.0), ('ignore', 1.0), ('over_apply', 1.5)]:
        branches.append(
            {
                'snapshot_id': 's1',
                'branch_type': branch_type,
                'task_type': 'recommend',
                'status': 'COMPLETED',
                'snapshot': snapshot,
                'provenance': 'REAL_USER_SIM_REPLAY',
                'utility_breakdown': {'utility_total': utility},
                'utility_total': utility,
                'trajectory': [
                    {
                        'assistant_message': f'Recommend[A | {branch_type}]',
                        'user_response': {'response': f'ok {branch_type}'},
                        'parser_status': 'OK',
                        'tool_status': 'NO_TOOL',
                        'terminal_status': 'ACTIVE',
                    }
                ],
                'metadata': {'seed': 42},
            }
        )
    pairs = [
        {
            'id': 's1:ignore',
            'scenario': 'recommend',
            'seed': 42,
            'method': 'real_branch_replay',
            'parser_mode': 'real_user_sim_replay',
            'conversations': [],
            'chosen': {'branch': 'follow', 'policy': 'x', 'trajectory': 'y'},
            'rejected': {'branch': 'ignore', 'policy': 'z', 'trajectory': 'w'},
            'score_delta': 1.0,
            'provenance': 'REAL_USER_SIM_REPLAY',
            'metadata': {
                'format': 'llamafactory_dpo_bridge',
                'source': 'RealBranchReplay',
                'proxy': 'controlled real user simulator replay proxy',
                'provenance': 'REAL_USER_SIM_REPLAY',
            },
        }
    ]
    (input_dir / 'replay_snapshots.jsonl').write_text('\n'.join(json.dumps(row) for row in [snapshot]) + '\n', encoding='utf-8')
    (input_dir / 'branch_rollouts.jsonl').write_text('\n'.join(json.dumps(row) for row in branches) + '\n', encoding='utf-8')
    (input_dir / 'replay_pairs.jsonl').write_text('\n'.join(json.dumps(row) for row in pairs) + '\n', encoding='utf-8')
    (input_dir / 'replay_failures.jsonl').write_text('[]\n', encoding='utf-8')

    result = subprocess.run(
        [sys.executable, '-m', 'user_simulator.evaluation.audit_real_branch_replay', '--input-dir', str(input_dir), '--output-dir', str(output_dir), '--fail-on-critical-error'],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    summary = json.loads((output_dir / 'audit.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'PASS'
    assert summary['snapshot_count'] == 1
