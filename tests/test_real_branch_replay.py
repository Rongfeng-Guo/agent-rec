from user_simulator.evaluation.run_real_branch_replay import build_pairs, parse_user_policy, parse_user_response


def test_replay_pair_building_and_parsers():
    branch_rows = [
        {
            'snapshot_id': 's1',
            'branch_type': 'follow',
            'task_type': 'recommend',
            'utility_total': 2.0,
            'snapshot': {'task_type': 'recommend', 'conversation_history': []},
            'source_trace': {'persona_path': 'user_simulator/task/Book_test.jsonl'},
            'trajectory': [],
        },
        {
            'snapshot_id': 's1',
            'branch_type': 'ignore',
            'task_type': 'recommend',
            'utility_total': 1.0,
            'snapshot': {'task_type': 'recommend', 'conversation_history': []},
            'source_trace': {'persona_path': 'user_simulator/task/Book_test.jsonl'},
            'trajectory': [],
        },
        {
            'snapshot_id': 's1',
            'branch_type': 'over_apply',
            'task_type': 'recommend',
            'utility_total': 1.5,
            'snapshot': {'task_type': 'recommend', 'conversation_history': []},
            'source_trace': {'persona_path': 'user_simulator/task/Book_test.jsonl'},
            'trajectory': [],
        },
    ]
    pairs = build_pairs(branch_rows)
    assert len(pairs) == 2
    assert all(pair['chosen_branch'] == 'follow' for pair in pairs)
    assert {pair['rejected_branch'] for pair in pairs} == {'ignore', 'over_apply'}
    assert parse_user_response('{"response": "hi"}')[1] == 'OK'
    assert parse_user_policy('{"policy": "end_conversation"}')[1] == 'OK'
