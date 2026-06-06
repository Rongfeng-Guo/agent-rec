from user_simulator.evaluation.real_branch_replay_schema import (
    PROVENANCE_CONTROLLED_SIMULATOR_REPLAY_PROXY,
    PROVENANCE_REAL_TRACE,
    ReplayBranch,
    ReplayPair,
    ReplaySnapshot,
    action_text,
    normalize_provenance,
    stable_branch_id,
    stable_snapshot_id,
)


def test_schema_ids_and_provenance_roundtrip():
    payload = {'episode_id': 'ep1', 'turn': 0, 'task_type': 'ask', 'extra': 'value'}
    snapshot_id = stable_snapshot_id(payload)
    assert snapshot_id.startswith('ep1:turn0:ask:')
    assert stable_branch_id(snapshot_id, 'follow', 1).startswith(snapshot_id)
    assert normalize_provenance(None) == PROVENANCE_CONTROLLED_SIMULATOR_REPLAY_PROXY
    assert normalize_provenance(PROVENANCE_REAL_TRACE) == PROVENANCE_REAL_TRACE

    snapshot = ReplaySnapshot(
        snapshot_id=snapshot_id,
        episode_id='ep1',
        turn=0,
        task_type='ask',
        user_state={'memory_mode': 'critiquescope'},
        persona={'name': 'persona'},
        conversation_history=[{'role': 'user', 'content': 'hi'}],
        candidate_state={'ground_truth_item': {'ItemName': 'Target'}},
        original_action={'text': 'Ask[Question]'},
        refined_action={'text': 'Ask[Better question]'},
        critique={'response': 'ok'},
        source_trace={'persona_path': 'user_simulator/task/Book_test.jsonl'},
        metadata={'source': 'real'},
        provenance=PROVENANCE_REAL_TRACE,
    )
    branch = ReplayBranch(
        snapshot_id=snapshot.snapshot_id,
        branch_type='follow',
        task_type='ask',
        policy_action={'assistant_message': 'Ask[Better question]'},
        trajectory=[],
        utility_breakdown={},
        utility_total=0.0,
        status='COMPLETED',
        snapshot=snapshot.to_dict(),
        source_trace={'persona_path': 'user_simulator/task/Book_test.jsonl'},
    )
    pair = ReplayPair(
        snapshot_id=snapshot.snapshot_id,
        chosen_branch='follow',
        rejected_branch='ignore',
        chosen_utility=1.0,
        rejected_utility=0.5,
        uplift=0.5,
        source_trace={'persona_path': 'user_simulator/task/Book_test.jsonl'},
    )

    assert action_text(snapshot.original_action) == 'Ask[Question]'
    assert branch.to_dict()['snapshot']['snapshot_id'] == snapshot.snapshot_id
    assert pair.to_dict()['uplift'] == 0.5
