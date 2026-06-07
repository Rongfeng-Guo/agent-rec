import json

from user_simulator.evaluation.build_cdpo_dataset_manifest import (
    build_manifest,
    build_llamafactory_snippet,
    build_split_dataset_info_snippet,
    materialize_splits,
)
from user_simulator.evaluation.summarize_closed_loop_outputs import audit_output_dir, build_report
from user_simulator.evaluation.run_closed_loop_benchmark import (
    apply_critiques,
    build_cdpo_pair,
    candidate_coverage_error,
    make_memory,
    parse_event_critiques,
    rollout,
    run_branch_rollouts,
)
from user_simulator.evaluation.run_closed_loop_pipeline import main as run_pipeline_main
from user_simulator.evaluation.run_validity_gate import main as run_validity_gate_main
from user_simulator.evaluation.validate_cdpo_pairs import validate_file
from user_simulator.policies.memory_rerank_policy import rank_items
from user_simulator.scenarios.closed_loop_scenarios import get_scenario
from user_simulator.state.critique_scope import CritiqueScopeMemory
from user_simulator.worlds.critique_world import CritiqueWorldConfig, Item, LatentUserState, deterministic_critique_for_slate


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


def test_cdpo_validator_rejects_duplicate_ids(tmp_path):
    row = {
        "id": "duplicate",
        "scenario": "temporary_fatigue",
        "seed": 0,
        "method": "critiquescope",
        "parser_mode": "oracle",
        "conversations": [{"from": "human", "value": "x"}],
        "chosen": {"branch": "follow", "policy": "x", "trajectory": "x"},
        "rejected": {"branch": "ignore", "policy": "y", "trajectory": "y"},
        "score_delta": 0.1,
        "metadata": {
            "format": "llamafactory_dpo_bridge",
            "source": "CritiqueWorld",
            "proxy": "controlled counterfactual rollout proxy",
        },
    }
    path = tmp_path / "dupes.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    result = validate_file(path)
    assert result["status"] == "FAIL"
    assert result["duplicate_ids"] == ["duplicate"]


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


def test_cdpo_dataset_materializes_train_dev_splits(tmp_path):
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

    manifest = build_manifest(tmp_path / "cdpo_pairs.jsonl", None, dev_fraction=0.5)
    rows = [json.loads(line) for line in (tmp_path / "cdpo_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    split_info = materialize_splits(
        rows,
        {"train": manifest["splits"]["train_ids"], "dev": manifest["splits"]["dev_ids"]},
        tmp_path / "cdpo_train.jsonl",
        tmp_path / "cdpo_dev.jsonl",
    )
    manifest["splits"].update(split_info)
    snippet = build_split_dataset_info_snippet(manifest)
    train_ids = {json.loads(line)["id"] for line in (tmp_path / "cdpo_train.jsonl").read_text(encoding="utf-8").splitlines()}
    dev_ids = {json.loads(line)["id"] for line in (tmp_path / "cdpo_dev.jsonl").read_text(encoding="utf-8").splitlines()}

    assert split_info["train_count"] + split_info["dev_count"] == manifest["row_count"]
    assert not (train_ids & dev_ids)
    assert f"{manifest['dataset_name']}_train" in snippet
    assert f"{manifest['dataset_name']}_dev" in snippet


def test_cdpo_dataset_manifest_rejects_invalid_pairs(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{\"id\":\"bad\"}\n", encoding="utf-8")
    try:
        build_manifest(path, None, dev_fraction=0.2)
    except ValueError as exc:
        assert "validation failed" in str(exc)
    else:
        raise AssertionError("invalid CDPO pairs should not build a manifest")


def test_cdpo_dataset_manifest_summarizes_real_branch_replay_rows(tmp_path):
    row = {
        "id": "real:s1:ignore",
        "scenario": "recommend",
        "seed": 42,
        "method": "real_branch_replay",
        "parser_mode": "real_user_sim_replay",
        "conversations": [{"from": "human", "value": "need a recommendation"}],
        "chosen": {"branch": "follow", "policy": "Recommend[A]", "trajectory": "turn=0 assistant=Recommend[A]"},
        "rejected": {"branch": "ignore", "policy": "Ignore[A]", "trajectory": "turn=0 assistant=Ignore[A]"},
        "score_delta": 0.8,
        "provenance": "REAL_USER_SIM_REPLAY",
        "metadata": {
            "format": "llamafactory_dpo_bridge",
            "source": "RealBranchReplay",
            "proxy": "controlled real user simulator replay proxy",
            "provenance": "REAL_USER_SIM_REPLAY",
        },
    }
    path = tmp_path / "real_cdpo_pairs.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    manifest = build_manifest(path, None, dev_fraction=0.5)
    snippet = build_llamafactory_snippet(manifest, path)

    assert manifest["source"] == "RealBranchReplay"
    assert manifest["proxy"] == "controlled real user simulator replay proxy"
    assert manifest["by_source"] == {"RealBranchReplay": 1}
    assert manifest["by_proxy"] == {"controlled real user simulator replay proxy": 1}
    assert manifest["by_provenance"] == {"REAL_USER_SIM_REPLAY": 1}
    assert manifest["schema"]["real_replay_requires_provenance"] is True
    assert manifest["dataset_name"] in snippet


def test_closed_loop_report_audits_valid_output_dir():
    output_dir = __import__("pathlib").Path("outputs/closed_loop_oracle")
    audit = audit_output_dir(output_dir)
    assert audit["status"] == "PASS"
    report = build_report(output_dir, audit)
    assert "CritiqueWorld Closed-Loop Report" in report
    assert "controlled counterfactual rollout proxy" in report
    assert "closed_loop_oracle_cdpo" in report


def test_closed_loop_report_audit_fails_on_missing_files(tmp_path):
    audit = audit_output_dir(tmp_path)
    assert audit["status"] == "FAIL"
    assert any("missing" in error for error in audit["errors"])


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


def test_fatigue_critique_tie_breaks_by_slate_order():
    slate = [
        Item("windows_a", "Windows", {}, 0.8),
        Item("mac_a", "Mac", {}, 0.8),
        Item("mac_b", "Mac", {}, 0.8),
        Item("windows_b", "Windows", {}, 0.8),
    ]
    state = LatentUserState(category_exposure_counts={"Windows": 4, "Mac": 4})
    critique = deterministic_critique_for_slate(slate, state)
    assert critique["critiques"][0]["target"] == "Windows"


def test_next_slate_horizon_one_applies_exactly_once():
    memory = CritiqueScopeMemory()
    state = LatentUserState(stable_positive={"category": ["UFC", "Boxing"]})
    scenario = get_scenario("diversity_request")
    memory.apply_turn(
        "Recommend something different but still related.",
        critiques=scenario.injected_events[0]["critiques"],
        current_turn=1,
    )

    first_rank = rank_items(scenario.items, state, memory, "critiquescope", 5, CritiqueWorldConfig())
    assert memory.active_fast()
    assert any(
        breakdown["intervention_score_delta"] != 0.0 for breakdown in first_rank.score_breakdowns.values()
    )

    memory.turn = 2
    memory.decay_fast_memory()
    assert not memory.active_fast()

    second_rank = rank_items(scenario.items, state, memory, "critiquescope", 5, CritiqueWorldConfig())
    assert all(
        breakdown["intervention_score_delta"] == 0.0 for breakdown in second_rank.score_breakdowns.values()
    )


def test_session_scope_survives_until_session_end():
    memory = CritiqueScopeMemory()
    critique = {
        "target": "UFC",
        "operation": "attenuate",
        "reason": "exposure fatigue",
        "object_scope": "category",
        "temporal_scope": "session",
        "horizon": 5,
        "hardness": "soft",
        "confidence": 0.78,
        "promotion_condition": "never",
    }
    memory.apply_turn("too much UFC", critiques=[critique], current_turn=1)
    for _ in range(3):
        memory.decay_fast_memory()
    assert memory.active_fast()
    memory.end_session()
    assert not memory.active_fast()


def test_contextual_scope_expires_after_reset():
    memory = CritiqueScopeMemory()
    critique = {
        "target": "family",
        "operation": "promote",
        "reason": "session context",
        "object_scope": "attribute",
        "temporal_scope": "contextual",
        "horizon": 6,
        "hardness": "soft",
        "confidence": 0.74,
        "promotion_condition": "never",
    }
    memory.apply_turn("family dinner tonight", critiques=[critique], current_turn=1)
    assert memory.active_fast()
    memory.end_session()
    assert memory.fast_memory[0].status == "expired"
    assert not memory.active_fast()


def test_fast_memory_decay_occurs_after_effective_application():
    scenario = get_scenario("temporary_fatigue")
    rows, memory, *_ = rollout(scenario, "critiquescope", 0, "oracle", 6, 5, CritiqueWorldConfig())
    turn_two = next(row for row in rows if row["turn"] == 2)
    turn_three = next(row for row in rows if row["turn"] == 3)
    assert turn_two["memory_update"]["applied"]
    assert any("ufc" not in item.lower() for item in turn_three["ranked_slate"]["slate"][:2])
    expire_events = [event for event in memory.events if event["event"] == "expire_fast"]
    assert expire_events
    assert expire_events[0]["turn"] >= 5


def test_diversify_changes_ranking():
    scenario = get_scenario("diversity_request")
    state = LatentUserState(stable_positive={"category": ["UFC", "Boxing"]}, category_exposure_counts={"UFC": 4, "Boxing": 2})
    baseline = rank_items(scenario.items, state, CritiqueScopeMemory(), "critiquescope", 5, CritiqueWorldConfig())

    memory = CritiqueScopeMemory()
    memory.apply_turn(
        "Recommend something different but still related.",
        critiques=scenario.injected_events[0]["critiques"],
        current_turn=1,
    )
    diversified = rank_items(scenario.items, state, memory, "critiquescope", 5, CritiqueWorldConfig())
    assert baseline.slate != diversified.slate


def test_diversify_increases_slate_diversity():
    rows, *_ = run_scenario("diversity_request", mode="critiquescope", max_turns=4)
    before = next(row for row in rows if row["turn"] == 1)["ranked_slate"]["slate"]
    after = next(row for row in rows if row["turn"] == 2)["ranked_slate"]["slate"]
    before_diversity = len({item.split("_")[0] for item in before})
    after_diversity = len({item.split("_")[0] for item in after})
    assert after_diversity >= before_diversity


def test_diversify_does_not_pollute_slow_memory():
    _, memory, *_ = run_scenario("diversity_request", mode="critiquescope", max_turns=5)
    assert memory.memory_contamination_rate() == 0.0
    assert not memory.active_slow()


def test_diversify_preserves_relevance_floor():
    scenario = get_scenario("diversity_request")
    state = LatentUserState(stable_positive={"category": ["UFC", "Boxing"]}, category_exposure_counts={"UFC": 4, "Boxing": 2})
    memory = CritiqueScopeMemory()
    memory.apply_turn(
        "Recommend something different but still related.",
        critiques=scenario.injected_events[0]["critiques"],
        current_turn=1,
    )
    diversified = rank_items(scenario.items, state, memory, "critiquescope", 5, CritiqueWorldConfig())
    top_categories = {item.category for item in diversified.slate[:3]}
    assert top_categories & {"Boxing", "Fitness", "Linux", "Mac", "Science", "Jazz", "Restaurant"}


def test_closed_loop_pipeline_creates_gated_artifacts(tmp_path):
    import sys

    old_argv = sys.argv
    sys.argv = [
        "run_closed_loop_pipeline",
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
        "--branch-horizon",
        "3",
        "--dev-fraction",
        "0.5",
        "--output-dir",
        str(tmp_path),
    ]
    try:
        run_pipeline_main()
    finally:
        sys.argv = old_argv

    metadata = json.loads((tmp_path / "pipeline_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "PASS"
    assert metadata["counts"]["cdpo_train"] + metadata["counts"]["cdpo_dev"] == metadata["counts"]["cdpo_pairs"]
    assert metadata["unique_cdpo_ids"] == metadata["counts"]["cdpo_pairs"]
    assert len(metadata["steps"]) == 4
    assert (tmp_path / "closed_loop_report.md").exists()


def test_closed_loop_pipeline_rejects_openai_compatible(tmp_path):
    import sys

    old_argv = sys.argv
    sys.argv = [
        "run_closed_loop_pipeline",
        "--parser-mode",
        "openai_compatible",
        "--output-dir",
        str(tmp_path),
    ]
    try:
        try:
            run_pipeline_main()
        except SystemExit as exc:
            assert "BLOCKED_NO_API_KEY" in str(exc)
        else:
            raise AssertionError("openai_compatible pipeline should be blocked without API config")
    finally:
        sys.argv = old_argv


def test_validity_gate_exports_expected_files(tmp_path):
    import sys

    old_argv = sys.argv
    sys.argv = [
        "run_validity_gate",
        "--modes",
        "critiquescope",
        "--scenarios",
        "diversity_request",
        "--seeds",
        "0",
        "--max-turns",
        "4",
        "--top-k",
        "5",
        "--output-dir",
        str(tmp_path),
        "--fail-on-critical-invariant",
    ]
    try:
        run_validity_gate_main()
    finally:
        sys.argv = old_argv

    expected = {
        "invariant_results.csv",
        "invariant_failures.jsonl",
        "lifecycle_trace.jsonl",
        "score_delta_trace.jsonl",
        "method_scenario_invariants.csv",
        "scenario_report.md",
        "tables.tex",
        "run_metadata.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}


def test_closed_loop_pipeline_can_run_validity_gate(tmp_path):
    import sys

    old_argv = sys.argv
    sys.argv = [
        "run_closed_loop_pipeline",
        "--modes",
        "critiquescope",
        "--scenarios",
        "diversity_request",
        "--seeds",
        "0",
        "--max-turns",
        "4",
        "--top-k",
        "5",
        "--parser-mode",
        "oracle",
        "--run-validity-gate",
        "--fail-on-critical-invariant",
        "--output-dir",
        str(tmp_path),
    ]
    try:
        run_pipeline_main()
    finally:
        sys.argv = old_argv

    metadata = json.loads((tmp_path / "pipeline_metadata.json").read_text(encoding="utf-8"))
    assert any(step["name"] == "run_validity_gate" for step in metadata["steps"])
    assert (tmp_path / "validity_gate" / "scenario_report.md").exists()
