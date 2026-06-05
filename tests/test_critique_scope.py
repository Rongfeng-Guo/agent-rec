import inspect

from user_simulator.evaluation.critique_scope_eval import (
    DEFAULT_SCENARIOS,
    evaluate_critiquescope,
    evaluate_flat_memory,
)
from user_simulator.evaluation.critique_uplift_pairs import build_pairs
from user_simulator.evaluation.critique_parser import parse_deterministic
from user_simulator.evaluation.critique_rollout_adapter import load_rollouts
from user_simulator.evaluation.critique_rollout_adapter import audit_rollouts
from user_simulator.evaluation.critique_rollout_adapter import summarize_audit
from user_simulator.evaluation.critique_rollout_adapter import render_report
from user_simulator.evaluation.critique_scope_eval import load_scenarios
from user_simulator.evaluation.summarize_memory_baselines import aggregate
from user_simulator.evaluation.validate_critique_scenarios import validate_scenarios
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


def test_deterministic_parser_outputs_schema():
    critiques = parse_deterministic("I have seen too much UFC lately. Switch it up for a bit.")
    assert critiques
    assert critiques[0]["operation"] == "attenuate"
    assert critiques[0]["temporal_scope"] == "session"
    assert critiques[0]["target"] == "UFC"
    assert critiques[0]["horizon"] == 3


def test_deterministic_parser_detects_genuine_drift():
    critiques = parse_deterministic("I do not want Windows anymore. Going forward, prioritize Mac laptops.")
    operations = {critique["operation"] for critique in critiques}
    targets = {critique["target"] for critique in critiques}
    assert {"rollback", "promote"} <= operations
    assert {"Windows", "Mac"} <= targets


def test_deterministic_parser_normalizes_persistent_dislike_target():
    critiques = parse_deterministic("Please never recommend political content to me.")
    assert critiques
    assert critiques[0]["target"] == "Politics"


def test_deterministic_parser_extracts_family_session_context():
    critiques = parse_deterministic("Tonight I need a family-friendly dinner place.")
    assert critiques
    assert critiques[0]["target"] == "family"
    assert critiques[0]["horizon"] == 4


def test_rollout_adapter_loads_default_scenarios():
    scenarios = load_rollouts(None)
    assert len(scenarios) >= 6
    assert all("follow_value" in scenario for scenario in scenarios)


def test_rollout_adapter_audit_passes_default_scenarios():
    findings = audit_rollouts(load_rollouts(None))
    assert findings
    assert all(finding["passed"] for finding in findings)


def test_rollout_adapter_audit_flags_misaligned_parser_input():
    scenario = dict(load_rollouts(None)[0])
    scenario["critiques"] = [dict(scenario["critiques"][0], target="Politics")]
    findings = audit_rollouts([scenario])
    failed = [finding for finding in findings if not finding["passed"]]
    assert any(finding["check"] == "deterministic_parser_alignment" for finding in failed)

def test_rollout_adapter_audit_summary_counts_failures():
    scenario = dict(load_rollouts(None)[0])
    scenario["critiques"] = [dict(scenario["critiques"][0], target="Politics")]
    summary = summarize_audit(audit_rollouts([scenario]))
    assert summary["failed_checks"] == 1
    assert summary["failed_scenarios"] == [scenario["id"]]
    assert summary["failed_by_type"] == {"deterministic_parser_alignment": 1}


def test_rollout_adapter_report_mentions_failed_scenarios():
    report = render_report({
        "total_checks": 3,
        "failed_checks": 1,
        "checks_by_type": {"deterministic_parser_alignment": 1, "branch_length_consistency": 1, "follow_outperforms_at_least_one_counterfactual": 1},
        "failed_by_type": {"deterministic_parser_alignment": 1},
        "failed_scenarios": ["temporary_ufc_fatigue"],
    })
    assert "temporary_ufc_fatigue" in report
    assert "deterministic_parser_alignment" in report


def test_noisy_scenario_set_validates():
    scenarios = load_scenarios(scenario_set="noisy")
    assert len(scenarios) >= 5
    assert validate_scenarios(scenarios) == []


def test_summary_aggregation_groups_by_method():
    rows = [
        {"method": "flat", "scenario": "a", "memory_contamination_rate": "1.0"},
        {"method": "flat", "scenario": "b", "memory_contamination_rate": "0.0"},
    ]
    result = aggregate(rows, ["method"], ["memory_contamination_rate"])
    assert result[0]["method"] == "flat"
    assert result[0]["n"] == 2
    assert result[0]["memory_contamination_rate_mean"] == 0.5
