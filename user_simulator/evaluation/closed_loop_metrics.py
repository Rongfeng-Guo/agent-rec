"""Metrics for CritiqueWorld closed-loop rollouts."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable, List


METRIC_FIELDS = [
    "CumulativeUtility",
    "AverageSlateUtility",
    "ClickRate",
    "LeaveRate",
    "AverageSessionLength",
    "SlateDiversity",
    "CategoryCoverage",
    "InstructionUplift@1",
    "InstructionUplift@H",
    "OverCorrectionRegret@1",
    "OverCorrectionRegret@H",
    "ExpiredConstraintViolationRate",
    "DriftRecoveryTurns",
    "RollbackAccuracy",
    "MemoryContaminationRate",
    "PromotionPrecision",
    "PromotionRecall",
    "ScopeClassificationAccuracy",
    "parser_scope_error",
    "memory_update_error",
    "policy_application_error",
    "candidate_coverage_error",
]


def slate_diversity(slate: list[dict] | list[str], score_breakdowns: dict | None = None) -> float:
    if not slate:
        return 0.0
    categories = []
    for item in slate:
        if isinstance(item, dict):
            categories.append(item.get("category", item.get("item_id", "")))
        else:
            categories.append(str(item).split("_")[0])
    return len(set(categories)) / len(categories)


def summarize_trajectory(rows: List[dict], branch_rows: List[dict] | None = None) -> dict:
    branch_rows = branch_rows or []
    utilities = [float(row.get("instant_utility", 0.0)) for row in rows]
    actions = [row.get("user_action", {}).get("action") for row in rows]
    categories = set()
    diversities = []
    for row in rows:
        slate = row.get("ranked_slate", {}).get("slate", [])
        categories.update(str(item_id).split("_")[0] for item_id in slate)
        diversities.append(slate_diversity(slate))

    follow = [row for row in branch_rows if row.get("branch") == "follow"]
    ignore = [row for row in branch_rows if row.get("branch") == "ignore"]
    over = [row for row in branch_rows if row.get("branch") == "over_apply"]

    def branch_sum(branch: List[dict], horizon: int | None = None) -> float:
        selected = branch if horizon is None else branch[:horizon]
        return sum(float(row.get("instant_utility", 0.0)) for row in selected)

    return {
        "CumulativeUtility": sum(utilities),
        "AverageSlateUtility": mean(utilities) if utilities else 0.0,
        "ClickRate": actions.count("click") / max(1, len(actions)),
        "LeaveRate": actions.count("leave") / max(1, len(actions)),
        "AverageSessionLength": len(rows),
        "SlateDiversity": mean(diversities) if diversities else 0.0,
        "CategoryCoverage": len(categories),
        "InstructionUplift@1": branch_sum(follow, 1) - branch_sum(ignore, 1),
        "InstructionUplift@H": branch_sum(follow) - branch_sum(ignore),
        "OverCorrectionRegret@1": branch_sum(follow, 1) - branch_sum(over, 1),
        "OverCorrectionRegret@H": branch_sum(follow) - branch_sum(over),
    }


def aggregate(rows: Iterable[dict], group_keys: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for key, group_rows in sorted(groups.items()):
        result = dict(zip(group_keys, key))
        result["n"] = len(group_rows)
        for metric in METRIC_FIELDS:
            values = [float(row[metric]) for row in group_rows if row.get(metric) not in {"", None}]
            result[f"{metric}_mean"] = mean(values) if values else 0.0
        output.append(result)
    return output
