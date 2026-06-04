import inspect

from user_simulator.evaluation.critique_scope_eval import (
    DEFAULT_SCENARIOS,
    evaluate_critiquescope,
    evaluate_flat_memory,
)
from user_simulator.evaluation.critique_uplift_pairs import build_pairs
from user_simulator.state.critique_scope import CritiqueScopeMemory
from user_simulator.state.critique_scope_memory import CritiqueScopeMemory as CompatMemory


def scenario_by_id(scenario_id):
    return next(scenario for scenario in DEFAULT_SCENARIOS if scenario["id"] == scenario_id)


def test_temporary_fatigue_does_not_pollute_slow_memory():
    result = evaluate_critiquescope(scenario_by_id("temporary_ufc_fatigue"))
    assert result["memory_contamination_rate"] == 0.0
    assert result["slow_memory_size"] == 0


def test_persistent_dislike_is_promoted():
    result = evaluate_critiquescope(scenario_by_id("persistent_political_filter"))
    assert result["slow_memory_size"] == 1
    assert result["promotion_precision"] == 1.0


def test_diversity_request_is_not_negative_preference():
    result = evaluate_critiquescope(scenario_by_id("diversity_not_dislike"))
    assert result["memory_contamination_rate"] == 0.0
    assert result["slow_memory_size"] == 0


def test_session_scope_expires():
    memory = CritiqueScopeMemory()
    scenario = scenario_by_id("session_family_dinner")
    memory.apply_turn(scenario["utterance"], critiques=scenario["critiques"])
    memory.end_session()
    assert not memory.active_fast()
    assert memory.fast_memory[0].status == "expired"


def test_genuine_drift_replaces_old_preference():
    result = evaluate_critiquescope(scenario_by_id("windows_to_mac_drift"))
    assert result["promotion_recall"] == 1.0
    assert result["slow_memory_size"] == 2


def test_behavioral_confirmation_triggers_rollback():
    result = evaluate_critiquescope(scenario_by_id("ufc_behavioral_rollback"))
    assert result["rollback_accuracy"] == 1.0
    assert result["over_correction_regret"] == 0.0


def test_flat_memory_has_expected_contamination_failure():
    result = evaluate_flat_memory(scenario_by_id("temporary_ufc_fatigue"))
    assert result["memory_contamination_rate"] == 1.0
    assert result["over_correction_regret"] > 0.0


def test_counterfactual_uplift_prefers_follow_branch():
    pairs = build_pairs([scenario_by_id("temporary_ufc_fatigue")])
    assert pairs
    assert all(pair["chosen"]["branch"] == "follow" for pair in pairs)
    assert all(pair["uplift"] > 0 for pair in pairs)


def test_user_agent_env_supports_critiquescope_mode():
    import user_simulator.user_agent_env_v1 as env_module

    signature = inspect.signature(env_module.UserAgentEnv.__init__)
    assert "memory_mode" in signature.parameters


def test_compatibility_import_path():
    assert CompatMemory is CritiqueScopeMemory
