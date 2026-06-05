import inspect
import json

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
from user_simulator.evaluation.critique_rollout_adapter import materialize_branch_rollouts
from user_simulator.evaluation.critique_rollout_adapter import build_branch_pairs
from user_simulator.evaluation.critique_rollout_adapter import build_cdpo_pair
from user_simulator.evaluation.export_gpe_hap_refine_logs import discover_inputs
from user_simulator.evaluation.export_gpe_hap_refine_logs import main as export_gpe_hap_refine_logs_main
from user_simulator.evaluation.critique_scope_eval import load_scenarios
from user_simulator.evaluation.summarize_memory_baselines import aggregate
from user_simulator.evaluation.validate_critique_scenarios import validate_scenarios
from user_simulator.state.critique_scope import CritiqueScopeMemory
from user_simulator.state.critique_scope_memory import CritiqueScopeMemory as CompatMemory


def scenario_by_id(scenario_id):
    return next(scenario for scenario in DEFAULT_SCENARIOS if scenario["id"] == scenario_id)


def gpe_trace_row(task_type, input_text, original_response, ground_truth, best_refinement, sample_num=1, **extra):
    row = {
        "task_type": task_type,
        "input": input_text,
        "original_response": original_response,
        "ground_truth": ground_truth,
        "potential_reward_output": "{\"reward\": 0.2}",
        "policy_improvement_output": json.dumps({"refinement_output": [best_refinement]}, ensure_ascii=False),
        "best_refinement": best_refinement,
        "is_original_best": False,
        "sample_num": sample_num,
    }
    row.update(extra)
    return row


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


def test_json_array_refine_log_is_supported(tmp_path):
    input_path = tmp_path / "recommend_refine_log_sample1.json"
    input_path.write_text(
        json.dumps(
            [
                gpe_trace_row(
                    "recommend",
                    "Recommend a better response for the scratchpad.",
                    "Recommend[old answer]",
                    "Recommend[new answer]",
                    "Recommend[new answer]",
                    sample_num=2,
                    combined_log={"task_type": "recommend"},
                )
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scenarios = load_rollouts(str(input_path))
    assert len(scenarios) == 1
    assert scenarios[0]["_adapter_source"] == "gpe_hap_refinement_trace"


def test_jsonl_refine_log_is_supported(tmp_path):
    input_path = tmp_path / "ask_refine_log_sample1.jsonl"
    input_path.write_text(
        json.dumps(
            gpe_trace_row(
                "ask",
                "Ask a clarifying question to reduce uncertainty.",
                "What do you like?",
                "Please ask about budget and cuisine.",
                "What budget and cuisine do you prefer?",
                sample_num=3,
                combined_log={"task_type": "ask"},
            ),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    scenarios = load_rollouts(str(input_path))
    assert len(scenarios) == 1
    assert scenarios[0]["_adapter_metadata"]["task_type"] == "ask"


def test_trace_directory_input_is_supported(tmp_path):
    traces_dir = tmp_path / "real_logs"
    traces_dir.mkdir()

    (traces_dir / "recommend_refine_log_sample1.json").write_text(
        json.dumps(
            [
                gpe_trace_row(
                    "recommend",
                    "Recommend a better response for the scratchpad.",
                    "Recommend[old answer]",
                    "Recommend[new answer]",
                    "Recommend[new answer]",
                    sample_num=2,
                )
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (traces_dir / "real_rollouts.jsonl").write_text(
        json.dumps(
            {
                "id": "real_ufc_fatigue",
                "scenario": "real_ufc_fatigue",
                "method": "gimo_real_rollout",
                "seed": 7,
                "parser_mode": "external",
                "critique_type": "Temporary Fatigue",
                "utterance": "I have seen too much UFC lately. Switch it up for a bit.",
                "critiques": [
                    {
                        "target": "UFC",
                        "operation": "attenuate",
                        "reason": "exposure fatigue",
                        "object_scope": "category",
                        "temporal_scope": "session",
                        "horizon": 3,
                        "hardness": "soft",
                        "confidence": 0.78,
                        "promotion_condition": "never",
                    }
                ],
                "branches": {
                    "follow": {"trajectory": [{"turn": 5, "action": "click", "utility": 1.2}]},
                    "ignore": {"trajectory": [{"turn": 5, "action": "click", "utility": 0.8}]},
                    "over_apply": {"trajectory": [{"turn": 5, "action": "click", "utility": 0.6}]},
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    scenarios = load_rollouts(str(traces_dir))
    assert len(scenarios) == 2
    assert {scenario["_adapter_metadata"]["input_name"] for scenario in scenarios} == {
        "recommend_refine_log_sample1.json",
        "real_rollouts.jsonl",
    }


def test_nested_combined_log_trace_is_supported_without_top_level_aliases(tmp_path):
    input_path = tmp_path / "wrapped_refine_log_sample1.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "log": {
                        "combined_log": {
                            "task_type": "search",
                            "input": "Find a better query for the current search task.",
                            "original_output": "cheap hiking boots",
                            "ground_truth": "waterproof hiking boots",
                            "policy_improvement_output": "{\"refinement_output\": [\"waterproof hiking boots\"]}",
                            "best_refinement": "waterproof hiking boots",
                            "sample_id": 9,
                            "query_text": "cheap hiking boots",
                            "action": "search",
                        }
                    }
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scenarios = load_rollouts(str(input_path))
    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario["_adapter_source"] == "gpe_hap_refinement_trace"
    assert scenario["_adapter_metadata"]["task_type"] == "search"
    assert scenario["_adapter_metadata"]["input_name"] == "wrapped_refine_log_sample1.json"
    assert scenario["follow_value"] == [1.0]


def test_gpe_hap_exporter_discovers_and_exports_json_traces(tmp_path):
    traces_dir = tmp_path / "traces"
    nested_dir = traces_dir / "run-1"
    nested_dir.mkdir(parents=True)
    trace_path = nested_dir / "book_refine_log_sample1.json"
    trace_path.write_text(
        json.dumps(
            [
                {
                    "task_type": "recommend",
                    "input": "Recommend a better response for the scratchpad.",
                    "original_response": "Recommend[old answer]",
                    "ground_truth": "Recommend[new answer]",
                    "potential_reward_output": "{\"reward\": 0.2}",
                    "policy_improvement_output": "{\"refinement_output\": [\"Recommend[new answer]\"]}",
                    "best_refinement": "Recommend[new answer]",
                    "is_original_best": False,
                    "sample_num": 2,
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    discovered = discover_inputs(traces_dir)
    assert discovered == [trace_path]

    output_dir = tmp_path / "exported"
    import sys

    old_argv = sys.argv
    sys.argv = [
        "export_gpe_hap_refine_logs",
        "--input",
        str(traces_dir),
        "--output-dir",
        str(output_dir),
        "--write-source-jsonl",
    ]
    try:
        export_gpe_hap_refine_logs_main()
    finally:
        sys.argv = old_argv

    assert (output_dir / "export_metadata.json").exists()
    assert (output_dir / "adapter_input.jsonl").exists()
    assert (output_dir / "branch_rollouts.jsonl").exists()
    assert (output_dir / "cdpo_pairs.jsonl").exists()
    metadata = json.loads((output_dir / "export_metadata.json").read_text(encoding="utf-8"))
    assert metadata["trace_count"] == 1
    assert metadata["branch_row_count"] == 3
    assert metadata["cdpo_pair_count"] == 2


def test_gpe_hap_recommend_trace_exports_pairs(tmp_path):
    input_path = tmp_path / "recommend_refine_log_sample1.json"
    input_path.write_text(
        json.dumps(
            [
                gpe_trace_row(
                    "recommend",
                    "Recommend a better response for the scratchpad.",
                    "Recommend[old answer]",
                    "Recommend[new answer]",
                    "Recommend[new answer]",
                    sample_num=2,
                    combined_log={"task_type": "recommend"},
                )
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scenarios = load_rollouts(str(input_path))
    branch_rows = materialize_branch_rollouts(scenarios)
    pairs = build_branch_pairs(scenarios)
    cdpo = build_cdpo_pair(pairs[0])

    assert scenarios[0]["_adapter_metadata"]["task_type"] == "recommend"
    assert all(row["task_type"] == "recommend" for row in branch_rows)
    assert pairs[0]["task_type"] == "recommend"
    assert cdpo["task_type"] == "recommend"
    assert "recommend" in cdpo["chosen"]["policy"].lower()


def test_gpe_hap_ask_trace_exports_pairs(tmp_path):
    input_path = tmp_path / "ask_refine_log_sample1.json"
    input_path.write_text(
        json.dumps(
            [
                gpe_trace_row(
                    "ask",
                    "Ask a clarifying question to reduce uncertainty.",
                    "What do you like?",
                    "Please ask about budget and cuisine.",
                    "What budget and cuisine do you prefer?",
                    sample_num=3,
                    combined_log={"task_type": "ask"},
                )
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scenarios = load_rollouts(str(input_path))
    pairs = build_branch_pairs(scenarios)
    cdpo = build_cdpo_pair(pairs[0])
    assert scenarios[0]["_adapter_metadata"]["task_type"] == "ask"
    assert pairs[0]["task_type"] == "ask"
    assert cdpo["task_type"] == "ask"
    assert "clarifying" in cdpo["chosen"]["policy"].lower()


def test_gpe_hap_search_trace_exports_pairs(tmp_path):
    input_path = tmp_path / "search_refine_log_sample1.json"
    input_path.write_text(
        json.dumps(
            [
                gpe_trace_row(
                    "search",
                    "Find a better query for the current search task.",
                    "cheap hiking boots",
                    "waterproof hiking boots",
                    "waterproof hiking boots",
                    sample_num=4,
                    original_action="search",
                    original_query="cheap hiking boots",
                    original_rank=12,
                    refined_queries=["waterproof hiking boots"],
                    refined_ranks=[{"query": "waterproof hiking boots", "rank": 3}],
                    combined_log={"task_type": "search"},
                )
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scenarios = load_rollouts(str(input_path))
    pairs = build_branch_pairs(scenarios)
    cdpo = build_cdpo_pair(pairs[0])
    assert scenarios[0]["_adapter_metadata"]["task_type"] == "search"
    assert pairs[0]["task_type"] == "search"
    assert cdpo["task_type"] == "search"
    assert "query" in cdpo["chosen"]["policy"].lower()


def test_task_type_is_preserved_in_branch_rows(tmp_path):
    input_path = tmp_path / "ask_refine_log_sample1.json"
    input_path.write_text(
        json.dumps(
            [
                gpe_trace_row(
                    "ask",
                    "Ask a clarifying question to reduce uncertainty.",
                    "What do you like?",
                    "Please ask about budget and cuisine.",
                    "What budget and cuisine do you prefer?",
                    sample_num=3,
                    combined_log={"task_type": "ask"},
                )
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scenarios = load_rollouts(str(input_path))
    branch_rows = materialize_branch_rollouts(scenarios)
    assert all(row["task_type"] == "ask" for row in branch_rows)


def test_task_type_is_preserved_in_dpo_pairs(tmp_path):
    input_path = tmp_path / "search_refine_log_sample1.json"
    input_path.write_text(
        json.dumps(
            [
                gpe_trace_row(
                    "search",
                    "Find a better query for the current search task.",
                    "cheap hiking boots",
                    "waterproof hiking boots",
                    "waterproof hiking boots",
                    sample_num=4,
                    original_action="search",
                    original_query="cheap hiking boots",
                    original_rank=12,
                    refined_queries=["waterproof hiking boots"],
                    refined_ranks=[{"query": "waterproof hiking boots", "rank": 3}],
                    combined_log={"task_type": "search"},
                )
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scenarios = load_rollouts(str(input_path))
    pairs = build_branch_pairs(scenarios)
    assert all(pair["task_type"] == "search" for pair in pairs)


def test_task_type_is_preserved_in_cdpo_pairs(tmp_path):
    input_path = tmp_path / "recommend_refine_log_sample1.json"
    input_path.write_text(
        json.dumps(
            [
                gpe_trace_row(
                    "recommend",
                    "Recommend a better response for the scratchpad.",
                    "Recommend[old answer]",
                    "Recommend[new answer]",
                    "Recommend[new answer]",
                    sample_num=2,
                    combined_log={"task_type": "recommend"},
                )
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scenarios = load_rollouts(str(input_path))
    cdpo = build_cdpo_pair(build_branch_pairs(scenarios)[0])
    assert cdpo["task_type"] == "recommend"
    assert cdpo["metadata"]["task_type"] == "recommend"


def test_unknown_task_type_uses_generic_fallback(tmp_path):
    input_path = tmp_path / "unknown_refine_log_sample1.json"
    input_path.write_text(
        json.dumps(
            [
                gpe_trace_row(
                    "unknown_task",
                    "Do something ambiguous.",
                    "foo",
                    "bar",
                    "bar",
                    sample_num=5,
                    combined_log={"task_type": "unknown_task"},
                )
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scenarios = load_rollouts(str(input_path))
    branch_rows = materialize_branch_rollouts(scenarios)
    pairs = build_branch_pairs(scenarios)
    cdpo = build_cdpo_pair(pairs[0])
    assert all(row["task_type"] == "generic" for row in branch_rows)
    assert all(pair["task_type"] == "generic" for pair in pairs)
    assert cdpo["task_type"] == "generic"
    assert "generic" in cdpo["chosen"]["policy"].lower()


def test_value_only_schema_remains_backward_compatible():
    scenarios = load_rollouts(None)
    branch_rows = materialize_branch_rollouts(scenarios)
    pairs = build_branch_pairs(scenarios)
    cdpo = build_cdpo_pair(pairs[0])
    assert scenarios
    assert branch_rows
    assert pairs
    assert cdpo["metadata"]["format"] == "llamafactory_dpo_bridge"
    assert cdpo["task_type"] == "generic"


def test_real_rollout_branch_schema_normalizes_to_values_and_branch_rows(tmp_path):
    input_path = tmp_path / "real_rollouts.jsonl"
    input_row = {
        "id": "real_ufc_fatigue",
        "scenario": "real_ufc_fatigue",
        "method": "gimo_real_rollout",
        "seed": 7,
        "parser_mode": "external",
        "critique_type": "Temporary Fatigue",
        "utterance": "I have seen too much UFC lately. Switch it up for a bit.",
        "critiques": [
            {
                "target": "UFC",
                "operation": "attenuate",
                "reason": "exposure fatigue",
                "object_scope": "category",
                "temporal_scope": "session",
                "horizon": 3,
                "hardness": "soft",
                "confidence": 0.78,
                "promotion_condition": "never",
            }
        ],
        "state_snapshot": {
            "turn": 4,
            "user_state": {"session_id": "gimo-session"},
            "memory_state": {"fast": [], "slow": []},
        },
        "branches": {
            "follow": {
                "trajectory": [
                    {"turn": 5, "slate": ["boxing_1", "fitness_1"], "action": "click", "utility": 1.2},
                    {"turn": 6, "slate": ["ufc_1", "boxing_1"], "action": "click", "utility": 1.3},
                ]
            },
            "ignore": {
                "trajectory": [
                    {"turn": 5, "slate": ["ufc_1", "ufc_2"], "action": "click", "utility": 0.8},
                    {"turn": 6, "slate": ["ufc_1", "boxing_1"], "action": "click", "utility": 0.9},
                ]
            },
            "over_apply": {
                "trajectory": [
                    {"turn": 5, "slate": ["boxing_1", "fitness_1"], "action": "click", "utility": 0.7},
                    {"turn": 6, "slate": ["boxing_1", "science_1"], "action": "click", "utility": 0.6},
                ]
            },
        },
    }
    input_path.write_text(json.dumps(input_row, ensure_ascii=False) + "\n", encoding="utf-8")

    scenarios = load_rollouts(str(input_path))
    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario["follow_value"] == [1.2, 1.3]
    assert scenario["ignore_value"] == [0.8, 0.9]
    assert scenario["over_apply_value"] == [0.7, 0.6]

    branch_rows = materialize_branch_rollouts(scenarios)
    assert len(branch_rows) == 6
    assert {row["branch"] for row in branch_rows} == {"follow", "ignore", "over_apply"}
    assert all("state_snapshot" in row for row in branch_rows)
    assert branch_rows[0]["state_snapshot"]["event"]["utterance"] == input_row["utterance"]

    pairs = build_branch_pairs(scenarios)
    assert len(pairs) == 2
    assert all(pair["chosen_branch"] == "follow" for pair in pairs)
    cdpo = build_cdpo_pair(pairs[0])
    assert cdpo["metadata"]["format"] == "llamafactory_dpo_bridge"
    assert cdpo["metadata"]["source"] == "CritiqueWorld"
    assert cdpo["metadata"]["origin_source"] == "GIMO_real_rollout"


def test_real_rollout_audit_requires_branch_schema_rows():
    scenarios = load_rollouts(None)
    findings = audit_rollouts(scenarios)
    assert any(finding["check"] == "branch_schema_rows_present" and finding["passed"] for finding in findings)


def test_gpe_hap_trace_log_normalizes_to_branch_schema(tmp_path):
    input_path = tmp_path / "gpe_trace.json"
    input_row = {
        "task_type": "recommend",
        "input": "Recommend a better response for the scratchpad.",
        "original_response": "Recommend[old answer]",
        "ground_truth": "Recommend[new answer]",
        "potential_reward_output": "{\"reward\": 0.2}",
        "policy_improvement_output": "{\"refinement_output\": [\"Recommend[new answer]\"]}",
        "best_refinement": "Recommend[new answer]",
        "is_original_best": False,
        "sample_num": 2,
    }
    input_path.write_text(json.dumps([input_row], ensure_ascii=False, indent=2), encoding="utf-8")

    scenarios = load_rollouts(str(input_path))
    assert len(scenarios) == 1
    row = scenarios[0]
    assert row["_adapter_source"] == "gpe_hap_refinement_trace"
    assert row["follow_value"] == [1.0]
    assert row["ignore_value"] == [0.0]
    assert row["over_apply_value"] == [0.2]
    assert len(row["_adapter_branch_rows"]) == 3
    assert all(branch_row["branch_id"].startswith("gpe_hap:recommend") for branch_row in row["_adapter_branch_rows"])

    findings = audit_rollouts(scenarios)
    assert any(finding["check"] == "gpe_hap_trace_fields_present" and finding["passed"] for finding in findings)
    pairs = build_branch_pairs(scenarios)
    assert len(pairs) == 2
    cdpo = build_cdpo_pair(pairs[0])
    assert "Recommend[new answer]" in cdpo["chosen"]["trajectory"]
    assert "recommend" in cdpo["chosen"]["policy"].lower()


def test_gpe_hap_search_trace_uses_query_policy_text(tmp_path):
    input_path = tmp_path / "gpe_search_trace.json"
    input_row = {
        "task_type": "search",
        "input": "Find a better query for the current search task.",
        "original_action": "search",
        "original_response": "cheap hiking boots",
        "original_query": "cheap hiking boots",
        "original_rank": 12,
        "ground_truth": "waterproof hiking boots",
        "refinements": ["add waterproof", "add trail grip"],
        "refined_queries": ["waterproof hiking boots"],
        "refined_ranks": [3],
        "potential_reward_output": "{\"reward\": 0.3}",
        "policy_improvement_output": "{\"refinement_output\": [\"waterproof hiking boots\"]}",
        "best_refinement": "waterproof hiking boots",
        "is_original_best": False,
        "sample_num": 4,
    }
    input_path.write_text(json.dumps([input_row], ensure_ascii=False, indent=2), encoding="utf-8")

    scenarios = load_rollouts(str(input_path))
    assert len(scenarios) == 1
    row = scenarios[0]
    assert row["_adapter_source"] == "gpe_hap_refinement_trace"
    assert row["_adapter_metadata"]["task_type"] == "search"
    assert row["_adapter_trace_fields"]["best_refinement"] == "waterproof hiking boots"
    assert len(row["_adapter_branch_rows"]) == 3
    assert all(branch_row["task_type"] == "search" for branch_row in row["_adapter_branch_rows"])

    pairs = build_branch_pairs(scenarios)
    assert len(pairs) == 2
    cdpo = build_cdpo_pair(pairs[0])
    assert "query" in cdpo["chosen"]["policy"].lower()
    assert "retrieval" in cdpo["chosen"]["policy"].lower() or "expand" in cdpo["chosen"]["policy"].lower()
    assert "original query" in cdpo["rejected"]["policy"].lower()


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
