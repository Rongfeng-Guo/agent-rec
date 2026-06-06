from user_simulator.evaluation.real_branch_policy import build_branch_policy
from user_simulator.evaluation.real_branch_replay_schema import ReplaySnapshot


def _snapshot(task_type: str = 'recommend'):
    return ReplaySnapshot(
        snapshot_id='ep1:turn0:recommend:abc',
        episode_id='ep1',
        turn=0,
        task_type=task_type,
        user_state={},
        persona={},
        conversation_history=[],
        candidate_state={},
        original_action={'text': "Recommend['The Silent Patient' by Alex Michaelides | reason]"},
        refined_action={'text': "Recommend['The Silent Patient' by Alex Michaelides | refined reason]"},
        critique=None,
        source_trace={},
        metadata={},
    )


def test_branch_policy_variants():
    snapshot = _snapshot()
    follow = build_branch_policy(snapshot, 'follow')
    ignore = build_branch_policy(snapshot, 'ignore')
    over_apply = build_branch_policy(snapshot, 'over_apply')

    assert follow['assistant_message'] == snapshot.refined_action['text']
    assert ignore['assistant_message'] == snapshot.original_action['text']
    assert 'Over-applied' in over_apply['assistant_message'] or 'OVER_APPLY' in over_apply['assistant_message']
    assert follow['provenance'] == ignore['provenance'] == over_apply['provenance']
