import json

from user_simulator.evaluation.build_cdpo_dataset_manifest import build_manifest, build_llamafactory_snippet
from user_simulator.evaluation.run_closed_loop_benchmark import (
    apply_critiques,
    build_cdpo_pair,
    candidate_coverage_error,
    make_memory,
    parse_event_critiques,
    rollout,
    run_branch_rollouts,
)
from user_simulator.evaluation.validate_cdpo_pairs import validate_file
from user_simulator.policies.memory_rerank_policy import rank_items
from user_simulator.scenarios.closed_loop_scenarios import get_scenario
from user_simulator.worlds.critique_world import CritiqueWorldConfig


def slate(rows, turn):
    return next(row["ranked_slate"]["slate"] for row in rows if row["turn"] == turn)


def run_scenario(name, mode="critiquescope", seed=0, parser_mode="oracle", max_turns=8):
    scenario = get_scenario(name)
    return rollout(scenario, mode, seed, parser_mode, max_turns, 5, CritiqueWorldConfig())


def test_same_seed_produces_same_trajectory():
    rows_a, *_ = run_scenario("mixed_multi_turn", seed=3)
    rows_b, *_ = run_scenario("mixed_multi_turn", seed=3)
    assert [row["ranked_slate"]["slate"] for row in rows_a] == [row["ranked_slate"]["slate"] for row in rows_b]
    assert [row["user_action"] for row in rows_a] == [row["user_action"] for row in rows_b]


def test_temporary_fatigue_recovers_after_horizon():
    rows, *_ = run_scenario("temporary_fatigue", mode="critiquescope", max_turns=8)
    assert not any("ufc" in item for item in slate(rows, 3)[:2])
    assert any("ufc" in item for item in slate(rows, 7)[:2])


def test_flat_memory_over_applies_temporary_fatigue():
    rows, *_ = run_scenario("temporary_fatigue", mode="flat", max_turns=8)
    assert not any("ufc" in item for item in slate(rows, 6))


def test_critiquescope_reduces_over_correction_regret():
    scenario = get_scenario("temporary_fatigue")
    rows, _, _, points = rollout(scenario, "critiquescope", 0, "oracle", 6, 5, CritiqueWorldConfig())
    branches, pairs = run_branch_rollouts(scenario, "critiquescope", 0, "oracle", points, CritiqueWorldConfig(), 5, 5)
    assert branches
    over_pairs = [pair for pair in pairs if pair["rejected_branch"] == "over_apply"]
    assert over_pairs
    assert max(pair["uplift"] for pair in over_pairs) > 0


def test_persistent_dislike_survives_session_reset():
    rows, memory, *_ = run_scenario("stable_dislike", mode="critiquescope", max_turns=7)
    assert any(item.target == "Politics" and item.active for item in memory.active_slow())
    assert not any("politics" in item for item in slate(rows, 6))


def test_diversity_request_changes_slate_without_long_term_pollution():
    rows, memory, *_ = run_scenario("diversity_request", mode="critiquescope", max_turns=5)
    assert len(set(item.split("_")[0] for item in slate(rows, 2))) > 1
    assert memory.memory_contamination_rate() == 0.0
    assert not memory.active_fast()


def test_behavioral_click_triggers_rollback():
    _, memory, *_ = run_scenario("behavioral_rollback", mode="critiquescope", max_turns=5)
    assert any(event["event"] == "rollback_fast" for event in memory.events)


def test_genuine_drift_recovers_within_bounded_turns():
    rows, *_ = run_scenario("genuine_drift", mode="critiquescope", max_turns=7)
    mac_turns = [row["turn"] for row in rows if any("mac" in item for item in row["ranked_slate"]["slate"][:2])]
    assert mac_turns and min(mac_turns) <= 4


def test_counterfactual_branches_start_from_identical_snapshot():
    scenario = get_scenario("temporary_fatigue")
    _, _, _, points = rollout(scenario, "critiquescope", 0, "oracle", 4, 5, CritiqueWorldConfig())
    branches, _ = run_branch_rollouts(scenario, "critiquescope", 0, "oracle", points, CritiqueWorldConfig(), 5, 3)
    first_by_branch = {}
    for row in branches:
        first_by_branch.setdefault(row["branch"], row["state_snapshot"])
    assert first_by_branch["follow"]["user_state"] == first_by_branch["ignore"]["user_state"]
    assert first_by_branch["follow"]["memory_state"] == first_by_branch["over_apply"]["memory_state"]


def test_follow_branch_beats_over_apply_on_temporary_fatigue():
    scenario = get_scenario("temporary_fatigue")
    _, _, _, points = rollout(scenario, "critiquescope", 0, "oracle", 4, 5, CritiqueWorldConfig())
    _, pairs = run_branch_rollouts(scenario, "critiquescope", 0, "oracle", points, CritiqueWorldConfig(), 5, 5)
    assert any(pair["rejected_branch"] == "over_apply" and pair["uplift"] > 0 for pair in pairs)


def test_runner_exports_expected_files(tmp_path):
    from user_simulator.evaluation.run_closed_loop_benchmark import main
    import sys

    old_argv = sys.argv
    sys.argv = [
        "run_closed_loop_benchmark",
        "--modes",
        "none",
        "critiquescope",
        "--scenarios",
        "temporary_fatigue",
        "--seeds",
        "0",
        "--max-turns",
        "4",
        "--top-k",
        "5",
        "--parser-mode",
        "oracle",
        "--output-dir",
        str(tmp_path),
    ]
    try:
        main()
    finally:
        sys.argv = old_argv

    expected = {
        "trajectories.jsonl",
        "branch_rollouts.jsonl",
        "cdpo_pairs.jsonl",
        "dpo_pairs.jsonl",
        "summary.csv",
        "summary.json",
        "method_summary.csv",
        "method_scenario_summary.csv",
        "run_metadata.json",
        "tables.tex",
        "README.md",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}


def test_dpo_pairs_are_valid_jsonl(tmp_path):
    from user_simulator.evaluation.run_closed_loop_benchmark import main
    import sys

    old_argv = sys.argv
    sys.argv = [
        "run_closed_loop_benchmark",
        "--modes",
        "critiquescope",
        "--scenarios",
        "temporary_fatigue",
        "--seeds",
        "0",
        "--max-turns",
        "4",
        "--top-k",
        "5",
        "--parser-mode",
        "oracle",
        "--output-dir",
        str(tmp_path),
    ]
    try:
        main()
    finally:
        sys.argv = old_argv

    rows = [json.loads(line) for line in (tmp_path / "dpo_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows
    assert {"scenario", "seed", "state_snapshot", "critique", "chosen_branch", "rejected_branch", "chosen_trajectory", "rejected_trajectory", "uplift", "metadata"} <= set(rows[0])


def test_cdpo_pair_bridge_contains_training_fields(tmp_path):
    from user_simulator.evaluation.run_closed_loop_benchmark import main
    import sys

    old_argv = sys.argv
    sys.argv = [
        "run_closed_loop_benchmark",
        "--modes",
        "critiquescope",
        "--scenarios",
        "temporary_fatigue",
        "--seeds",
        "0",
        "--max-turns",
        "4",
        "--top-k",
        "5",
        "--parser-mode",
        "oracle",
        "--output-dir",
        str(tmp_path),
    ]
    try:
        main()
    finally:
        sys.argv = old_argv

    rows = [json.loads(line) for line in (tmp_path / "cdpo_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows
    assert {"conversations", "chosen", "rejected", "score_delta", "metadata"} <= set(rows[0])
    assert rows[0]["metadata"]["format"] == "llamafactory_dpo_bridge"


def test_cdpo_validator_accepts_exported_pairs(tmp_path):
    from user_simulator.evaluation.run_closed_loop_benchmark import main
    import sys

    old_argv = sys.argv
    sys.argv = [
        "run_closed_loop_benchmark",
        "--modes",
        "critiquescope",
        "--scenarios",
        "temporary_fatigue",
        "--seeds",
        "0",
        "--max-turns",
        "4",
        "--top-k",
        "5",
        "--parser-mode",
        "oracle",
        "--output-dir",
        str(tmp_path),
    ]
    try:
        main()
    finally:
        sys.argv = old_argv

    result = validate_file(tmp_path / "cdpo_pairs.jsonl")
    assert result["status"] == "PASS"
    assert result["rows"] > 0
    assert result["score_delta_min"] > 0


def test_cdpo_validator_rejects_non_positive_delta(tmp_path):
    row = {
        "id": "bad",
        "scenario": "temporary_fatigue",
        "seed": 0,
        "method": "critiquescope",
        "parser_mode": "oracle",
        "conversations": [{"from": "human", "value": "x"}],
        "chosen": {"branch": "follow", "policy": "x", "trajectory": "x"},
        "rejected": {"branch": "ignore", "policy": "y", "trajectory": "y"},
        "score_delta": 0,
        "metadata": {
            "format": "llamafactory_dpo_bridge",
            "source": "CritiqueWorld",
            "proxy": "controlled counterfactual rollout proxy",
        },
    }
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = validate_file(path)
    assert result["status"] == "FAIL"
    assert any("strictly positive" in error for error in result["errors"])


def test_cdpo_dataset_manifest_summarizes_valid_pairs(tmp_path):
    from user_simulator.evaluation.run_closed_loop_benchmark import main
    import sys

    old_argv = sys.argv
    sys.argv = [
        "run_closed_loop_benchmark",
        "--modes",
        "critiquescope",
        "--scenarios",
        "temporary_fatigue",
        "--seeds",
        "0",
        "--max-turns",
        "4",
        "--top-k",
        "5",
        "--parser-mode",
        "oracle",
        "--output-dir",
        str(tmp_path),
    ]
    try:
        main()
    finally:
        sys.argv = old_argv

    validation = validate_file(tmp_path / "cdpo_pairs.jsonl")
    validation_path = tmp_path / "cdpo_validation.json"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    manifest = build_manifest(tmp_path / "cdpo_pairs.jsonl", validation_path, dev_fraction=0.5)
    snippet = build_llamafactory_snippet(manifest, tmp_path / "cdpo_pairs.jsonl")

    assert manifest["validation_status"] == "PASS"
    assert manifest["row_count"] == validation["rows"]
    assert manifest["splits"]["train_count"] + manifest["splits"]["dev_count"] == manifest["row_count"]
    assert manifest["schema"]["score_delta"] == "strictly_positive"
    assert manifest["dataset_name"] in snippet


def test_cdpo_dataset_manifest_rejects_invalid_pairs(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{\"id\":\"bad\"}\n", encoding="utf-8")
    try:
        build_manifest(path, None, dev_fraction=0.2)
    except ValueError as exc:
        assert "validation failed" in str(exc)
    else:
        raise AssertionError("invalid CDPO pairs should not build a manifest")


def test_candidate_coverage_error_flags_missing_targets():
    rows = [
        {
            "generated_critique": {"critiques": [{"target": "missing_category"}]},
            "ranked_slate": {"slate": ["ufc_1", "boxing_1"]},
        }
    ]
    assert candidate_coverage_error(rows) == 1.0


def test_memory_update_error_is_reported_for_parser_miss():
    rows, *_ = run_scenario("genuine_drift", mode="critiquescope", parser_mode="deterministic", max_turns=4)
    assert any(row["generated_critique"] for row in rows)
    assert any("memory_update_error" in row for row in rows)
