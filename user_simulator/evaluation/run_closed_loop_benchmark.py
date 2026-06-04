"""Run closed-loop CritiqueWorld benchmark and counterfactual rollouts."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any, Dict, Iterable, List

from user_simulator.evaluation.closed_loop_metrics import METRIC_FIELDS, aggregate, summarize_trajectory
from user_simulator.evaluation.critique_parser import parse_deterministic
from user_simulator.policies.memory_rerank_policy import rank_items
from user_simulator.scenarios.closed_loop_scenarios import Scenario, list_scenarios
from user_simulator.state.critique_scope import CritiqueScopeMemory
from user_simulator.state.structured_memory import StructuredMemory
from user_simulator.worlds.critique_world import CritiqueWorldConfig, LatentUserState, simulate_user_response


SUMMARY_FIELDS = ["method", "scenario", "seed", "parser_mode", *METRIC_FIELDS, "status"]


def git_value(args: List[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNKNOWN"


def make_memory(mode: str):
    if mode == "critiquescope":
        return CritiqueScopeMemory()
    if mode == "structured":
        return StructuredMemory()
    if mode in {"flat", "time_decay"}:
        return []
    if mode == "none":
        return None
    raise ValueError(f"Unsupported mode: {mode}")


def memory_snapshot(memory: Any, mode: str) -> dict:
    if mode == "critiquescope":
        return {
            "fast": [asdict(item) for item in memory.fast_memory],
            "slow": [asdict(item) for item in memory.slow_memory],
            "events": list(memory.events),
        }
    if mode == "structured":
        return {"slots": {key: asdict(value) for key, value in memory.slots.items()}, "events": list(memory.events)}
    if mode in {"flat", "time_decay"}:
        return {"critiques": copy.deepcopy(memory)}
    return {}


def parse_event_critiques(event: dict, parser_mode: str) -> tuple[list[dict], dict]:
    oracle = copy.deepcopy(event.get("critiques", []))
    if parser_mode == "oracle":
        return oracle, {"parser_scope_error": 0.0}
    if parser_mode == "deterministic":
        parsed = parse_deterministic(event.get("utterance", ""))
        oracle_scopes = {(item.get("target"), item.get("temporal_scope"), item.get("operation")) for item in oracle}
        parsed_scopes = {(item.get("target"), item.get("temporal_scope"), item.get("operation")) for item in parsed}
        return parsed, {"parser_scope_error": 0.0 if parsed_scopes == oracle_scopes else 1.0}
    if parser_mode == "openai_compatible":
        return [], {"parser_scope_error": 1.0, "blocked": "BLOCKED_NO_API_KEY"}
    raise ValueError(f"Unsupported parser mode: {parser_mode}")


def critique_matches(a: dict, b: dict) -> bool:
    return (
        str(a.get("target", "")).lower() == str(b.get("target", "")).lower()
        and a.get("operation") == b.get("operation")
        and a.get("temporal_scope") == b.get("temporal_scope")
    )


def target_in_catalog(scenario: Scenario, critique: dict) -> bool:
    target = str(critique.get("target", "")).lower()
    if not target or target == "current slate":
        return True
    for item in scenario.items:
        if target in item.category.lower() or target in item.item_id.lower():
            return True
        for value in item.attributes.values():
            values = value if isinstance(value, list) else [value]
            if any(target in str(candidate).lower() for candidate in values):
                return True
    return False


def memory_contains(memory: Any, mode: str, critique: dict) -> bool:
    target = str(critique.get("target", "")).lower()
    if mode == "none":
        return True
    if mode == "critiquescope":
        active = memory.active_fast() + memory.active_slow()
        return any(target in item.target.lower() for item in active)
    if mode == "structured":
        return any(target in slot.value.lower() for slot in memory.active_slots())
    if mode in {"flat", "time_decay"}:
        return any(target in str(item.get("target", "")).lower() for item in memory)
    return False


def active_memory_targets(memory: Any, mode: str) -> list[str]:
    if mode == "critiquescope":
        return [item.target.lower() for item in memory.active_fast() + memory.active_slow()]
    if mode == "structured":
        return [slot.value.lower() for slot in memory.active_slots()]
    if mode in {"flat", "time_decay"}:
        return [str(item.get("target", "")).lower() for item in memory]
    return []


def policy_application_error_for_rank(memory: Any, mode: str, ranked: Any) -> float:
    targets = [target for target in active_memory_targets(memory, mode) if target and target != "current slate"]
    if not targets:
        return 0.0
    slate_text = " ".join(ranked.scores.keys()).lower()
    if not any(target in slate_text for target in targets):
        return 0.0
    interventions = ranked.applied_interventions
    return 0.0 if interventions else 1.0


def apply_critiques(memory: Any, mode: str, critiques: Iterable[dict], utterance: str, turn: int, branch: str = "follow") -> dict:
    critiques = [copy.deepcopy(critique) for critique in critiques]
    if branch == "ignore" or mode == "none":
        return {"applied": [], "branch": branch}
    if branch == "over_apply":
        for critique in critiques:
            if critique.get("temporal_scope") != "persistent":
                critique["temporal_scope"] = "persistent"
                critique["operation"] = "filter" if critique.get("operation") in {"attenuate", "diversify", "explore"} else critique.get("operation")
                critique["hardness"] = "hard"
                critique["promotion_condition"] = "persistent_language"

    if mode == "critiquescope":
        memory.apply_turn(utterance, critiques=critiques)
    elif mode == "structured":
        for critique in critiques:
            bucket = "hard" if critique.get("hardness") == "hard" or critique.get("operation") in {"filter", "rollback"} else "soft"
            operation = "forget" if critique.get("operation") == "rollback" else "merge"
            memory.update(bucket=bucket, key=critique.get("object_scope", "category"), value=critique.get("target", ""), operation=operation, confidence=critique.get("confidence", 0.7), source="closed_loop")
    elif mode in {"flat", "time_decay"}:
        for critique in critiques:
            memory.append({**critique, "turn": turn})
    return {"applied": critiques, "branch": branch}


def apply_event_to_state(user_state: LatentUserState, event: dict):
    if event.get("type") == "session_reset":
        user_state.reset_session()
    for key, value in event.get("state_update", {}).items():
        setattr(user_state, key, copy.deepcopy(value))


def handle_behavioral_confirmation(memory: Any, mode: str, target: str):
    if mode == "critiquescope":
        memory.observe_positive_behavior(target)


def event_by_turn(scenario: Scenario) -> dict[int, list[dict]]:
    events: dict[int, list[dict]] = {}
    for event in scenario.injected_events:
        events.setdefault(int(event["turn"]), []).append(event)
    return events


def rollout(
    scenario: Scenario,
    mode: str,
    seed: int,
    parser_mode: str,
    max_turns: int,
    top_k: int,
    config: CritiqueWorldConfig,
    memory: Any | None = None,
    user_state: LatentUserState | None = None,
    start_turn: int = 0,
    branch: str | None = None,
    branch_id: str | None = None,
    suppress_events: bool = False,
) -> tuple[list[dict], Any, LatentUserState, list[dict]]:
    rng = Random(seed)
    memory = memory if memory is not None else make_memory(mode)
    user_state = user_state if user_state is not None else copy.deepcopy(scenario.initial_user_state)
    events = {} if suppress_events else event_by_turn(scenario)
    rows = []
    critique_points = []
    cumulative = 0.0

    for local_index in range(max_turns):
        turn = start_turn + local_index
        if not user_state.active:
            break
        before_state = user_state.snapshot()
        before_memory = memory_snapshot(memory, mode)
        ranked = rank_items(scenario.items, user_state, memory, mode, top_k, config)
        action = simulate_user_response(ranked.slate, user_state, rng, config)
        instant_utility = float(action.get("utility", 0.0))
        cumulative += instant_utility
        memory_update = {"applied": []}
        generated_critique = None
        attribution = {
            "parser_scope_error": 0.0,
            "memory_update_error": 0.0,
            "policy_application_error": policy_application_error_for_rank(memory, mode, ranked),
            "candidate_coverage_error": 0.0,
        }

        for event in events.get(turn, []):
            if event.get("type") in {"critique", "drift"}:
                parsed, parser_attr = parse_event_critiques(event, parser_mode)
                attribution.update({key: value for key, value in parser_attr.items() if key in attribution})
                snapshot = {
                    "scenario": scenario.name,
                    "method": mode,
                    "seed": seed,
                    "turn": turn,
                    "user_state": copy.deepcopy(before_state),
                    "memory_state": copy.deepcopy(before_memory),
                    "_user_state_obj": copy.deepcopy(user_state),
                    "_memory_obj": copy.deepcopy(memory),
                    "event": copy.deepcopy(event),
                }
                critique_points.append(snapshot)
                memory_update = apply_critiques(memory, mode, parsed, event.get("utterance", ""), turn)
                generated_critique = {"utterance": event.get("utterance", ""), "critiques": parsed}
                if parsed and any(not target_in_catalog(scenario, critique) for critique in parsed):
                    attribution["candidate_coverage_error"] = 1.0
                if parsed and any(not memory_contains(memory, mode, critique) for critique in parsed):
                    attribution["memory_update_error"] = 1.0
                apply_event_to_state(user_state, event)
            elif event.get("type") == "session_reset":
                apply_event_to_state(user_state, event)
                if mode == "critiquescope":
                    memory.end_session()
            elif event.get("type") == "behavioral_confirmation":
                handle_behavioral_confirmation(memory, mode, event.get("target", ""))
                action = {"action": "click", "item_id": f"confirm_{event.get('target')}", "category": event.get("target"), "utility": instant_utility, "critique": None}

        if action.get("action") == "click" and mode == "critiquescope":
            handle_behavioral_confirmation(memory, mode, action.get("category", ""))
        if mode == "critiquescope":
            memory.decay_fast_memory()

        row = {
            "run_id": f"{mode}:{scenario.name}:{seed}:{parser_mode}",
            "method": mode,
            "scenario": scenario.name,
            "seed": seed,
            "parser_mode": parser_mode,
            "turn": turn,
            "branch": branch,
            "branch_id": branch_id,
            "user_state_before": before_state,
            "memory_state_before": before_memory,
            "ranked_slate": ranked.to_dict(),
            "score_breakdowns": ranked.score_breakdowns,
            "user_action": action,
            "generated_critique": generated_critique,
            "memory_update": memory_update,
            "user_state_after": user_state.snapshot(),
            "memory_state_after": memory_snapshot(memory, mode),
            "instant_utility": instant_utility,
            "cumulative_utility": cumulative,
            "patience": user_state.patience,
            "active": user_state.active,
            **attribution,
        }
        rows.append(row)
    return rows, memory, user_state, critique_points


def run_branch_rollouts(
    scenario: Scenario,
    mode: str,
    seed: int,
    parser_mode: str,
    critique_points: List[dict],
    config: CritiqueWorldConfig,
    top_k: int,
    horizon: int,
) -> tuple[list[dict], list[dict]]:
    rows = []
    pairs = []
    for point_index, point in enumerate(critique_points):
        event = point["event"]
        parsed, _ = parse_event_critiques(event, parser_mode)
        branch_trajectories = {}
        for branch in ["follow", "ignore", "over_apply"]:
            branch_memory = copy.deepcopy(point["_memory_obj"])
            branch_state = copy.deepcopy(point["_user_state_obj"])
            apply_critiques(branch_memory, mode, parsed, event.get("utterance", ""), point["turn"], branch=branch)
            apply_event_to_state(branch_state, event)
            branch_rows, _, _, _ = rollout(
                scenario,
                mode,
                seed + 1000 + point_index,
                parser_mode,
                max_turns=horizon,
                top_k=top_k,
                config=config,
                memory=branch_memory,
                user_state=branch_state,
                start_turn=point["turn"] + 1,
                branch=branch,
                branch_id=f"{mode}:{scenario.name}:{seed}:{point_index}",
                suppress_events=True,
            )
            for row in branch_rows:
                row["state_snapshot"] = public_snapshot(point)
            rows.extend(branch_rows)
            branch_trajectories[branch] = branch_rows

        for rejected in ["ignore", "over_apply"]:
            chosen_value = sum(row["instant_utility"] for row in branch_trajectories["follow"])
            rejected_value = sum(row["instant_utility"] for row in branch_trajectories[rejected])
            if chosen_value > rejected_value:
                pairs.append(
                    {
                        "scenario": scenario.name,
                        "seed": seed,
                        "method": mode,
                        "parser_mode": parser_mode,
                        "state_snapshot": public_snapshot(point),
                        "critique": {"utterance": event.get("utterance", ""), "critiques": parsed},
                        "chosen_branch": "follow",
                        "rejected_branch": rejected,
                        "chosen_trajectory": branch_trajectories["follow"],
                        "rejected_trajectory": branch_trajectories[rejected],
                        "uplift": chosen_value - rejected_value,
                        "metadata": {
                            "proxy": "controlled counterfactual rollout proxy",
                            "horizon": horizon,
                            "critique_point_index": point_index,
                            "snapshot_turn": point["turn"],
                            "branch_id": f"{mode}:{scenario.name}:{seed}:{point_index}",
                        },
                    }
                )
    return rows, pairs


def public_snapshot(point: dict) -> dict:
    return {key: value for key, value in point.items() if not key.startswith("_")}


def write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_latex(path: Path, rows: list[dict]):
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Method & N & Utility & ClickRate & Regret@H \\\\", "\\midrule"]
    for row in rows:
        lines.append(
            f"{row['method']} & {row['n']} & {row['CumulativeUtility_mean']:.3f} & "
            f"{row['ClickRate_mean']:.3f} & {row['OverCorrectionRegret@H_mean']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def branch_policy_text(branch: str, critique: dict) -> str:
    target = critique.get("target", "")
    operation = critique.get("operation", "")
    temporal_scope = critique.get("temporal_scope", "")
    if branch == "follow":
        return f"Apply {operation} to {target} with temporal scope {temporal_scope}."
    if branch == "ignore":
        return f"Do not update memory for critique target {target}."
    return f"Over-apply critique target {target} as a persistent hard constraint."


def trajectory_to_training_text(rows: list[dict]) -> str:
    parts = []
    for row in rows:
        slate = ", ".join(row.get("ranked_slate", {}).get("slate", []))
        action = row.get("user_action", {}).get("action")
        utility = float(row.get("instant_utility", 0.0))
        parts.append(f"turn={row.get('turn')} slate=[{slate}] action={action} utility={utility:.3f}")
    return "\n".join(parts)


def build_cdpo_pair(pair: dict) -> dict:
    critique_list = pair.get("critique", {}).get("critiques", [])
    critique = critique_list[0] if critique_list else {}
    metadata = pair.get("metadata", {})
    point_index = metadata.get("critique_point_index", "x")
    snapshot_turn = metadata.get("snapshot_turn", pair.get("state_snapshot", {}).get("turn", "x"))
    return {
        "id": (
            f"{pair.get('method')}:{pair.get('scenario')}:{pair.get('seed')}:"
            f"turn{snapshot_turn}:point{point_index}:{pair.get('rejected_branch')}"
        ),
        "scenario": pair.get("scenario"),
        "seed": pair.get("seed"),
        "method": pair.get("method"),
        "parser_mode": pair.get("parser_mode"),
        "conversations": [
            {
                "from": "human",
                "value": (
                    "Given the state snapshot and user critique, choose the recommendation policy "
                    "that follows the instruction without over-correcting durable memory.\n"
                    f"Critique: {json.dumps(pair.get('critique'), ensure_ascii=False)}\n"
                    f"State snapshot turn: {pair.get('state_snapshot', {}).get('turn')}"
                ),
            }
        ],
        "chosen": {
            "branch": pair.get("chosen_branch"),
            "policy": branch_policy_text(pair.get("chosen_branch", "follow"), critique),
            "trajectory": trajectory_to_training_text(pair.get("chosen_trajectory", [])),
        },
        "rejected": {
            "branch": pair.get("rejected_branch"),
            "policy": branch_policy_text(pair.get("rejected_branch", "ignore"), critique),
            "trajectory": trajectory_to_training_text(pair.get("rejected_trajectory", [])),
        },
        "score_delta": pair.get("uplift"),
        "metadata": {
            **metadata,
            "format": "llamafactory_dpo_bridge",
            "source": "CritiqueWorld",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["none", "flat", "structured", "time_decay", "critiquescope"])
    parser.add_argument("--scenarios", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--parser-mode", choices=["oracle", "deterministic", "openai_compatible"], default="oracle")
    parser.add_argument("--branch-horizon", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.parser_mode == "openai_compatible":
        raise SystemExit("BLOCKED_NO_API_KEY: openai_compatible parser is optional and not run without API config")

    config = CritiqueWorldConfig(default_branch_horizon=args.branch_horizon)
    scenarios = list_scenarios(args.scenarios)
    trajectories = []
    branch_rollouts = []
    dpo_pairs = []
    summaries = []

    for seed in args.seeds:
        for scenario in scenarios:
            scenario.max_turns = args.max_turns
            scenario.top_k = args.top_k
            for mode in args.modes:
                rows, _, _, critique_points = rollout(
                    scenario, mode, seed, args.parser_mode, args.max_turns, args.top_k, config
                )
                branches, pairs = run_branch_rollouts(
                    scenario, mode, seed, args.parser_mode, critique_points, config, args.top_k, args.branch_horizon
                )
                trajectories.extend(rows)
                branch_rollouts.extend(branches)
                dpo_pairs.extend(pairs)
                metrics = summarize_trajectory(rows, branches)
                metrics.update(
                    {
                        "ExpiredConstraintViolationRate": expired_violation_rate(rows),
                        "DriftRecoveryTurns": drift_recovery_turns(rows),
                        "RollbackAccuracy": rollback_accuracy(rows),
                        "MemoryContaminationRate": memory_contamination_rate(rows),
                        "PromotionPrecision": promotion_precision(rows),
                        "PromotionRecall": promotion_recall(rows),
                        "ScopeClassificationAccuracy": 1.0 - mean_attr(rows, "parser_scope_error"),
                        "parser_scope_error": mean_attr(rows, "parser_scope_error"),
                        "memory_update_error": mean_attr(rows, "memory_update_error"),
                        "policy_application_error": mean_attr(rows, "policy_application_error"),
                        "candidate_coverage_error": mean_attr(rows, "candidate_coverage_error"),
                    }
                )
                summaries.append(
                    {
                        "method": mode,
                        "scenario": scenario.name,
                        "seed": seed,
                        "parser_mode": args.parser_mode,
                        **metrics,
                        "status": "SMOKE_TEST_ONLY",
                    }
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "trajectories.jsonl", trajectories)
    write_jsonl(output_dir / "branch_rollouts.jsonl", branch_rollouts)
    write_jsonl(output_dir / "dpo_pairs.jsonl", dpo_pairs)
    cdpo_pairs = [build_cdpo_pair(pair) for pair in dpo_pairs]
    write_jsonl(output_dir / "cdpo_pairs.jsonl", cdpo_pairs)
    write_csv(output_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    method_summary = aggregate(summaries, ["method"])
    method_scenario_summary = aggregate(summaries, ["method", "scenario"])
    write_csv(output_dir / "method_summary.csv", method_summary, list(method_summary[0].keys()) if method_summary else [])
    write_csv(output_dir / "method_scenario_summary.csv", method_scenario_summary, list(method_scenario_summary[0].keys()) if method_scenario_summary else [])
    write_latex(output_dir / "tables.tex", method_summary)
    metadata = {
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "modes": args.modes,
        "scenarios": [scenario.name for scenario in scenarios],
        "seeds": args.seeds,
        "parser_mode": args.parser_mode,
        "status": "SMOKE_TEST_ONLY",
        "proxy": "controlled counterfactual rollout proxy",
        "dpo_pair_count": len(dpo_pairs),
        "cdpo_pair_count": len(cdpo_pairs),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Closed-loop CritiqueWorld Run\n\n"
        f"- Status: SMOKE_TEST_ONLY\n"
        f"- Parser mode: `{args.parser_mode}`\n"
        f"- Proxy: controlled counterfactual rollout proxy\n"
        f"- Trajectories: `{len(trajectories)}` rows\n"
        f"- Branch rollouts: `{len(branch_rollouts)}` rows\n"
        f"- DPO/CDPO pairs: `{len(dpo_pairs)}` rows\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "summary_rows": len(summaries), "trajectory_rows": len(trajectories), "branch_rows": len(branch_rollouts), "dpo_pairs": len(dpo_pairs), "output_dir": str(output_dir)}, indent=2))


def mean_attr(rows: list[dict], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return sum(values) / max(1, len(values))


def expired_violation_rate(rows: list[dict]) -> float:
    violations = 0
    checks = 0
    for row in rows:
        for event in row.get("memory_state_after", {}).get("events", []):
            if event.get("event", "").startswith("expire"):
                checks += 1
                target = event.get("target", "").lower()
                slate = row.get("ranked_slate", {}).get("slate", [])
                violations += int(any(target in item.lower() for item in slate))
    return violations / max(1, checks)


def drift_recovery_turns(rows: list[dict]) -> float:
    drift_turn = None
    for row in rows:
        critique = row.get("generated_critique") or {}
        if any(item.get("reason") == "genuine drift" for item in critique.get("critiques", [])):
            drift_turn = row["turn"]
        if drift_turn is not None and any("mac" in item.lower() for item in row.get("ranked_slate", {}).get("slate", [])[:2]):
            return float(max(0, row["turn"] - drift_turn))
    return 0.0


def rollback_accuracy(rows: list[dict]) -> float:
    for row in rows:
        events = row.get("memory_state_after", {}).get("events", [])
        if any(event.get("event") == "rollback_fast" for event in events):
            return 1.0
    return 0.0


def memory_contamination_rate(rows: list[dict]) -> float:
    last = rows[-1] if rows else {}
    slow = last.get("memory_state_after", {}).get("slow", [])
    if not slow:
        return 0.0
    contaminated = [item for item in slow if item.get("reason") in {"exposure fatigue", "diversity request", "session context"}]
    return len(contaminated) / len(slow)


def promotion_precision(rows: list[dict]) -> float:
    return 1.0 - memory_contamination_rate(rows)


def promotion_recall(rows: list[dict]) -> float:
    generated = []
    slow = []
    for row in rows:
        critique = row.get("generated_critique") or {}
        generated.extend(item for item in critique.get("critiques", []) if item.get("temporal_scope") == "persistent")
        slow = row.get("memory_state_after", {}).get("slow", slow)
    if not generated:
        return 1.0
    slow_targets = {item.get("target") for item in slow}
    return sum(1 for item in generated if item.get("target") in slow_targets) / len(generated)


def candidate_coverage_error(rows: list[dict]) -> float:
    for row in rows:
        critique = row.get("generated_critique") or {}
        targets = [item.get("target", "").lower() for item in critique.get("critiques", [])]
        if targets and not any(any(target in slate_item.lower() for slate_item in row.get("ranked_slate", {}).get("slate", [])) for target in targets):
            return 1.0
    return 0.0


if __name__ == "__main__":
    main()
