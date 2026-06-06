from user_simulator.evaluation.real_branch_utility import DEFAULT_WEIGHTS, compute_rollout_utility


def test_rollout_utility_rewards_relevant_follow():
    branch_row = {
        'branch_type': 'follow',
        'horizon': 1,
        'snapshot': {
            'task_type': 'recommend',
            'candidate_state': {
                'ground_truth_item': {'ItemName': 'The Silent Patient'},
            },
        },
        'trajectory': [
            {
                'assistant_message': "Recommend['The Silent Patient' by Alex Michaelides | a fit]",
                'recommendation_satisfaction': '{"rating": "5"}',
                'action_satisfaction': '{"rating": "5"}',
                'expression_satisfaction': '{"rating": "4"}',
                'user_active': True,
                'tool_status': 'NO_TOOL',
                'parser_status': 'OK',
            }
        ],
    }
    breakdown = compute_rollout_utility(branch_row, {'weights': DEFAULT_WEIGHTS})
    assert breakdown['task_success'] == 1.0
    assert breakdown['utility_total'] > 0
    assert breakdown['parse_failure'] == 0.0
