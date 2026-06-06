from pathlib import Path


def test_pipeline_script_mentions_real_replay_commands():
    script = Path('scripts/server184/run_real_branch_replay_pipeline.sh').read_text(encoding='utf-8')
    assert 'snapshot_prompt_ira_rollouts' in script
    assert 'run_real_branch_replay' in script
    assert 'critique_rollout_adapter' in script
    assert 'validate_cdpo_pairs' in script
    assert 'audit_real_branch_replay' in script
