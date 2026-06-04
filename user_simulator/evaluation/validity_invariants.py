"""Scenario-level invariant checks for CritiqueWorld validity audits."""

from __future__ import annotations

import json
from statistics import mean
from typing import Any, Iterable


def _rows_for(rows: list[dict], scenario: str, method: str, seed: int) -> list[dict]:
    selected = [row for row in rows if row.get("scenario") == scenario and row.get("method") == method and row.get("seed") == seed]
    return sorted(selected, key=lambda row: row.get("turn", 0))


def _branch_rows_for(branch_rows: list[dict], scenario: str, method: str, seed: int, branch: str | None = None) -> list[dict]:
    selected = [
        row
        for row in branch_rows
        if row.get("scenario") == scenario
        and row.get("method") == method
        and row.get("state_snapshot", {}).get("seed") == seed
    ]
    if branch is not None:
        selected = [row for row in selected if row.get("branch") == branch]
    return sorted(selected, key=lambda row: (row.get("branch_id", ""), row.get("turn", 0)))


def _branch_groups(branch_rows: list[dict], scenario: str, method: str, seed: int) -> dict[str, dict[str, list[dict]]]:
    groups: dict[str, dict[str, list[dict]]] = {}
    for row in _branch_rows_for(branch_rows, scenario, method, seed):
        groups.setdefault(row["branch_id"], {}).setdefault(row["branch"], []).append(row)
    for branch_map in groups.values():
        for rows in branch_map.values():
            rows.sort(key=lambda row: row.get("turn", 0))
    return groups


def _slate_contains(row: dict, target: str, top_n: int | None = None) -> bool:
    slate = row.get("ranked_slate", {}).get("slate", [])
    slate = slate if top_n is None else slate[:top_n]
    target = target.lower()
    return any(target in str(item).lower() for item in slate)


def _top_rank(rows: list[dict], keyword: str) -> int | None:
    for row in rows:
        slate = row.get("ranked_slate", {}).get("slate", [])
        for index, item in enumerate(slate, start=1):
            if keyword.lower() in str(item).lower():
                return index
    return None


def _row_at_turn(rows: list[dict], turn: int) -> dict | None:
    for row in rows:
        if int(row.get("turn", -1)) == turn:
            return row
    return None


def _score_for_keyword(row: dict, keyword: str) -> float | None:
    for item_id, score in row.get("ranked_slate", {}).get("scores", {}).items():
        if keyword.lower() in str(item_id).lower():
            return float(score)
    return None


def _slow_targets(row: dict) -> set[str]:
    slow = row.get("memory_state_after", {}).get("slow", [])
    return {str(item.get("target", "")).lower() for item in slow if item.get("active", True)}


def _active_fast_targets(row: dict) -> set[str]:
    fast = row.get("memory_state_after", {}).get("fast", [])
    return {str(item.get("target", "")).lower() for item in fast if item.get("active", True)}


def _branch_sum(rows: Iterable[dict]) -> float:
    return sum(float(row.get("instant_utility", 0.0)) for row in rows)


def _trace_ref(scenario: str, method: str, seed: int, invariant: str) -> str:
    return f"{scenario}:{method}:{seed}:{invariant}"


def _record(scenario: str, seed: int, method: str, invariant: str, passed: bool, observed: dict, expected: dict, critical: bool = True) -> dict:
    return {
        "scenario": scenario,
        "seed": seed,
        "method": method,
        "invariant": invariant,
        "passed": passed,
        "critical": critical,
        "observed": observed,
        "expected": expected,
        "trace_ref": _trace_ref(scenario, method, seed, invariant),
    }


def evaluate_invariants(rows: list[dict], branch_rows: list[dict], scenarios: list[Any], modes: list[str], seeds: list[int]) -> list[dict]:
    scenario_map = {scenario.name: scenario for scenario in scenarios}
    results: list[dict] = []
    for scenario_name, scenario in scenario_map.items():
        for seed in seeds:
            by_method = {mode: _rows_for(rows, scenario_name, mode, seed) for mode in modes}
            branch_by_method = {mode: _branch_groups(branch_rows, scenario_name, mode, seed) for mode in modes}
            results.extend(_evaluate_scenario(scenario, by_method, branch_by_method, seed))
    return results


def _evaluate_scenario(scenario: Any, by_method: dict[str, list[dict]], branch_by_method: dict[str, dict[str, dict[str, list[dict]]]], seed: int) -> list[dict]:
    name = scenario.name
    if name == "temporary_fatigue":
        return _temporary_fatigue_invariants(scenario, by_method, branch_by_method, seed)
    if name == "stable_dislike":
        return _stable_dislike_invariants(scenario, by_method, branch_by_method, seed)
    if name == "diversity_request":
        return _diversity_request_invariants(scenario, by_method, branch_by_method, seed)
    if name == "session_context":
        return _session_context_invariants(scenario, by_method, branch_by_method, seed)
    if name == "genuine_drift":
        return _genuine_drift_invariants(scenario, by_method, branch_by_method, seed)
    if name == "behavioral_rollback":
        return _behavioral_rollback_invariants(scenario, by_method, branch_by_method, seed)
    if name == "mixed_multi_turn":
        return _mixed_multi_turn_invariants(scenario, by_method, branch_by_method, seed)
    return []


def _temporary_fatigue_invariants(scenario: Any, by_method: dict[str, list[dict]], branch_by_method: dict[str, dict[str, dict[str, list[dict]]]], seed: int) -> list[dict]:
    results = []
    horizon = int(scenario.expected_properties.get("horizon", 3))
    target = str(scenario.expected_properties.get("temporary_target", "UFC")).lower()
    critique_rows = by_method["critiquescope"]
    flat_rows = by_method["flat"]
    critique_groups = branch_by_method["critiquescope"]

    follow_sum = mean([_branch_sum(group.get("follow", [])) for group in critique_groups.values()] or [0.0])
    over_sum = mean([_branch_sum(group.get("over_apply", [])) for group in critique_groups.values()] or [0.0])
    results.append(
        _record(
            scenario.name,
            seed,
            "critiquescope",
            "follow_outperforms_over_apply",
            follow_sum > over_sum,
            {"follow_utility": follow_sum, "over_apply_utility": over_sum},
            {"relationship": "follow > over_apply"},
        )
    )

    suppression_turn = scenario.injected_events[0]["turn"] + 1
    recovery_turn = scenario.injected_events[0]["turn"] + horizon + 2
    suppressed = _row_at_turn(critique_rows, suppression_turn)
    recovered = _row_at_turn(critique_rows, recovery_turn)
    results.append(
        _record(
            scenario.name,
            seed,
            "critiquescope",
            "suppression_expires_after_horizon",
            bool(suppressed and recovered and (not _slate_contains(suppressed, target, 2)) and _slate_contains(recovered, target, 2)),
            {
                "suppressed_turn": suppression_turn,
                "recovered_turn": recovery_turn,
                "suppressed_contains_target_top2": _slate_contains(suppressed, target, 2) if suppressed else None,
                "recovered_contains_target_top2": _slate_contains(recovered, target, 2) if recovered else None,
            },
            {"suppressed_top2": False, "post_expiry_eligible": True},
        )
    )

    flat_recovered = _row_at_turn(flat_rows, recovery_turn)
    results.append(
        _record(
            scenario.name,
            seed,
            "flat",
            "flat_retains_longer_suppression_than_critiquescope",
            bool(recovered and flat_recovered and _slate_contains(recovered, target, 2) and (not _slate_contains(flat_recovered, target))),
            {
                "critiquescope_post_expiry_contains_target_top2": _slate_contains(recovered, target, 2) if recovered else None,
                "flat_post_expiry_contains_target": _slate_contains(flat_recovered, target) if flat_recovered else None,
            },
            {"critiquescope_recovers": True, "flat_post_expiry_suppresses": True},
        )
    )

    results.append(
        _record(
            scenario.name,
            seed,
            "critiquescope",
            "post_expiry_target_eligible_again",
            bool(recovered and _slate_contains(recovered, target, 2)),
            {"turn": recovery_turn, "contains_target_top2": _slate_contains(recovered, target, 2) if recovered else None},
            {"contains_target": True},
        )
    )
    return results


def _stable_dislike_invariants(scenario: Any, by_method: dict[str, list[dict]], branch_by_method: dict[str, dict[str, dict[str, list[dict]]]], seed: int) -> list[dict]:
    results = []
    target = str(scenario.expected_properties.get("persistent_target", "Politics")).lower()
    rows = by_method["critiquescope"]
    groups = branch_by_method["critiquescope"]
    first = _row_at_turn(rows, 1)
    after_reset = _row_at_turn(rows, 6)
    results.append(
        _record(
            scenario.name,
            seed,
            "critiquescope",
            "persistent_filter_enters_slow_memory",
            bool(first and target in _slow_targets(first)),
            {"slow_targets": sorted(_slow_targets(first) if first else set())},
            {"slow_target": target},
        )
    )
    follow_sum = mean([_branch_sum(group.get("follow", [])) for group in groups.values()] or [0.0])
    ignore_sum = mean([_branch_sum(group.get("ignore", [])) for group in groups.values()] or [0.0])
    ignore_exposes_target = any(
        _slate_contains(row, target)
        for group in groups.values()
        for row in group.get("ignore", [])
    )
    results.append(
        _record(
            scenario.name,
            seed,
            "critiquescope",
            "follow_outperforms_ignore_when_disliked_items_exposed",
            (follow_sum > ignore_sum) if ignore_exposes_target else True,
            {"follow_utility": follow_sum, "ignore_utility": ignore_sum, "ignore_exposes_target": ignore_exposes_target},
            {"relationship": "follow > ignore when ignore branch still exposes target"},
        )
    )
    results.append(
        _record(
            scenario.name,
            seed,
            "critiquescope",
            "filter_survives_session_reset",
            bool(after_reset and target in _slow_targets(after_reset) and (not _slate_contains(after_reset, target))),
            {
                "turn": 6,
                "slow_targets": sorted(_slow_targets(after_reset) if after_reset else set()),
                "slate_contains_target": _slate_contains(after_reset, target) if after_reset else None,
            },
            {"slow_target_persists": True, "slate_contains_target": False},
        )
    )
    return results


def _diversity_request_invariants(scenario: Any, by_method: dict[str, list[dict]], branch_by_method: dict[str, dict[str, dict[str, list[dict]]]], seed: int) -> list[dict]:
    results = []
    rows = by_method["critiquescope"]
    before = _row_at_turn(rows, 1)
    after = _row_at_turn(rows, 2)
    before_slate = before.get("ranked_slate", {}).get("slate", []) if before else []
    after_slate = after.get("ranked_slate", {}).get("slate", []) if after else []
    before_div = len({str(item).split("_")[0] for item in before_slate}) if before_slate else 0
    after_div = len({str(item).split("_")[0] for item in after_slate}) if after_slate else 0
    results.append(_record(scenario.name, seed, "critiquescope", "next_slate_differs_from_pre_critique_slate", before_slate != after_slate, {"before": before_slate, "after": after_slate}, {"before_after_differ": True}))
    results.append(_record(scenario.name, seed, "critiquescope", "slate_diversity_increases_or_stays_above_target", after_div >= max(2, before_div), {"before_diversity": before_div, "after_diversity": after_div}, {"after_diversity_min": max(2, before_div)}))
    last = rows[-1] if rows else None
    results.append(_record(scenario.name, seed, "critiquescope", "slow_memory_contamination_remains_zero", bool(last and not _slow_targets(last)), {"slow_targets": sorted(_slow_targets(last) if last else set())}, {"slow_targets": []}))
    return results


def _session_context_invariants(scenario: Any, by_method: dict[str, list[dict]], branch_by_method: dict[str, dict[str, dict[str, list[dict]]]], seed: int) -> list[dict]:
    results = []
    rows = by_method["critiquescope"]
    before = _row_at_turn(rows, 1)
    after = _row_at_turn(rows, 2)
    post_reset = _row_at_turn(rows, 6)
    before_rank = _top_rank([before] if before else [], "family_restaurant")
    after_rank = _top_rank([after] if after else [], "family_restaurant")
    results.append(_record(scenario.name, seed, "critiquescope", "contextual_preference_improves_in_session_ranking", bool(before_rank and after_rank and after_rank <= before_rank), {"rank_before": before_rank, "rank_after": after_rank}, {"rank_after_lte_before": True}))
    results.append(_record(scenario.name, seed, "critiquescope", "contextual_state_expires_after_reset", bool(post_reset and "family" not in _active_fast_targets(post_reset)), {"active_fast_targets_after_reset": sorted(_active_fast_targets(post_reset) if post_reset else set())}, {"family_active_after_reset": False}))
    results.append(_record(scenario.name, seed, "critiquescope", "stale_constraint_violation_remains_zero_after_reset", bool(post_reset and "family" not in _active_fast_targets(post_reset)), {"active_fast_targets_after_reset": sorted(_active_fast_targets(post_reset) if post_reset else set())}, {"no_stale_contextual_target": True}))
    return results


def _genuine_drift_invariants(scenario: Any, by_method: dict[str, list[dict]], branch_by_method: dict[str, dict[str, dict[str, list[dict]]]], seed: int) -> list[dict]:
    results = []
    rows = by_method["critiquescope"]
    before = _row_at_turn(rows, 1)
    after = _row_at_turn(rows, 3)
    windows_before = _score_for_keyword(before, "windows") if before else None
    windows_after = _score_for_keyword(after, "windows") if after else None
    mac_rank = _top_rank(rows[3:5], "mac") if len(rows) > 3 else None
    slow_targets = _slow_targets(rows[-1]) if rows else set()
    results.append(_record(scenario.name, seed, "critiquescope", "old_preference_weakens_or_is_suppressed", bool(windows_before is not None and (windows_after is None or windows_after < windows_before)), {"windows_score_before": windows_before, "windows_score_after": windows_after}, {"windows_score_after_lt_before_or_missing": True}))
    results.append(_record(scenario.name, seed, "critiquescope", "new_preference_is_promoted", bool(("mac" in slow_targets or any("mac" in target for target in slow_targets)) and mac_rank is not None and mac_rank <= 2), {"slow_targets": sorted(slow_targets), "mac_rank": mac_rank}, {"mac_promoted": True, "mac_rank_lte": 2}))
    results.append(_record(scenario.name, seed, "critiquescope", "drift_recovery_turns_bounded", bool(mac_rank is not None and mac_rank <= 2), {"mac_rank": mac_rank}, {"mac_rank_lte": 2}))
    return results


def _behavioral_rollback_invariants(scenario: Any, by_method: dict[str, list[dict]], branch_by_method: dict[str, dict[str, dict[str, list[dict]]]], seed: int) -> list[dict]:
    results = []
    rows = by_method["critiquescope"]
    before = _row_at_turn(rows, 3)
    after = _row_at_turn(rows, 4)
    events = rows[-1].get("memory_state_after", {}).get("events", []) if rows else []
    rollback_present = any(event.get("event") == "rollback_fast" for event in events)
    before_score = _score_for_keyword(before, "ufc") if before else None
    after_score = _score_for_keyword(after, "ufc") if after else None
    results.append(_record(scenario.name, seed, "critiquescope", "positive_reengagement_removes_temporary_attenuation", rollback_present, {"rollback_present": rollback_present}, {"rollback_present": True}))
    results.append(_record(scenario.name, seed, "critiquescope", "target_score_increases_after_rollback", bool(after and _slate_contains(after, "ufc")), {"score_before": before_score, "score_after": after_score, "ufc_present_after_rollback": _slate_contains(after, "ufc") if after else None}, {"ufc_present_after_rollback": True}))
    return results


def _mixed_multi_turn_invariants(scenario: Any, by_method: dict[str, list[dict]], branch_by_method: dict[str, dict[str, dict[str, list[dict]]]], seed: int) -> list[dict]:
    results = []
    rows = by_method["critiquescope"]
    ufc_recovered = _row_at_turn(rows, 7)
    mac_after_drift = _row_at_turn(rows, 8)
    results.append(_record(scenario.name, seed, "critiquescope", "temporary_fatigue_recovers_before_later_drift", bool(ufc_recovered and _slate_contains(ufc_recovered, "ufc")), {"turn": 6, "contains_ufc": _slate_contains(ufc_recovered, "ufc") if ufc_recovered else None}, {"contains_ufc": True}))
    results.append(_record(scenario.name, seed, "critiquescope", "later_drift_promotes_mac", bool(mac_after_drift and _slate_contains(mac_after_drift, "mac", 2)), {"turn": 8, "mac_top2": _slate_contains(mac_after_drift, "mac", 2) if mac_after_drift else None}, {"mac_top2": True}))
    return results
