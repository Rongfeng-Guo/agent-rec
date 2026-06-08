#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.oracle_route_memory.train_candidate_level_source_ranker import summarize_eval_rows

try:
    from scripts.oracle_route_memory.handoff_io import ensure_empty_output_dir
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from handoff_io import ensure_empty_output_dir

DEFAULT_COMPARE_FIELDS = (
    "sample_id",
    "domain",
    "target_item_id",
    "match_rank",
    "candidate_pool_hit",
    "candidate_pool_match_rank",
    "selected_source",
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [dict(row) for row in payload]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compare_rows(
    locked_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] = DEFAULT_COMPARE_FIELDS,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    if len(locked_rows) != len(replay_rows):
        mismatches.append({"type": "length", "locked": len(locked_rows), "replay": len(replay_rows)})
        return mismatches
    for idx, (locked, replay) in enumerate(zip(locked_rows, replay_rows)):
        for field in fields:
            if locked.get(field) != replay.get(field):
                mismatches.append(
                    {
                        "type": "field",
                        "index": idx,
                        "field": field,
                        "sample_id": locked.get("sample_id"),
                        "locked": locked.get(field),
                        "replay": replay.get(field),
                    }
                )
                break
    return mismatches


def validate_replay(
    *,
    locked_outputs: Path,
    replay_outputs: Path,
    topk: int = 50,
    fields: Sequence[str] = DEFAULT_COMPARE_FIELDS,
) -> dict[str, Any]:
    locked_rows = read_rows(locked_outputs)
    replay_rows = read_rows(replay_outputs)
    mismatches = compare_rows(locked_rows, replay_rows, fields=fields)
    locked_metric = summarize_eval_rows(locked_rows, topk=topk)
    replay_metric = summarize_eval_rows(replay_rows, topk=topk)
    metric_errors: list[str] = []
    for key in ("sample_count", f"Recall@{topk}", "CandidatePoolHitRate", "ConditionalRecall@50GivenPoolHit"):
        if locked_metric.get(key) != replay_metric.get(key):
            metric_errors.append(f"{key} mismatch: locked={locked_metric.get(key)!r} replay={replay_metric.get(key)!r}")
    return {
        "status": "ok" if not mismatches and not metric_errors else "failed",
        "locked_outputs": str(locked_outputs),
        "replay_outputs": str(replay_outputs),
        "compare_fields": list(fields),
        "locked_metric": locked_metric,
        "replay_metric": replay_metric,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "metric_errors": metric_errors,
        "next_target": (
            "If status is ok, use this replay validation as the readiness input for the locked H5-D "
            "fresh-confirmation gate; if status is failed, stop and resolve row or metric mismatches "
            "before registering or scoring any fresh split."
        ),
    }


def render_report(result: Mapping[str, Any], topk: int) -> str:
    replay_metric = result["replay_metric"]
    lines = [
        "# H5 Loaded-Model Replay Validation",
        "",
        f"Status: `{result['status']}`",
        "",
        "## Inputs",
        "",
        f"- locked outputs: `{result['locked_outputs']}`",
        f"- replay outputs: `{result['replay_outputs']}`",
        "",
        "## Replay Metric",
        "",
        f"- sample_count: `{replay_metric['sample_count']}`",
        f"- Recall@{topk}: `{replay_metric[f'Recall@{topk}']:.6f}`",
        f"- CandidatePoolHitRate: `{replay_metric['CandidatePoolHitRate']:.6f}`",
        f"- ConditionalRecall@50GivenPoolHit: `{replay_metric['ConditionalRecall@50GivenPoolHit']:.6f}`",
        "",
        "## Row Comparison",
        "",
        f"- mismatch_count: `{result['mismatch_count']}`",
    ]
    if result.get("metric_errors"):
        lines.extend(["", "## Metric Errors", ""])
        lines.extend(f"- {error}" for error in result["metric_errors"])
    if result.get("mismatches"):
        lines.extend(["", "## First Mismatches", ""])
        lines.extend(f"- {mismatch}" for mismatch in result["mismatches"])
    lines.extend(["", "## Next Target", "", str(result["next_target"]), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate loaded-model replay outputs against locked H5-D outputs.")
    parser.add_argument("--locked-outputs", required=True)
    parser.add_argument("--replay-outputs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--field", action="append", default=None, help="Field to compare. Repeatable; defaults to locked fields.")
    args = parser.parse_args()

    fields = tuple(args.field) if args.field else DEFAULT_COMPARE_FIELDS
    result = validate_replay(
        locked_outputs=Path(args.locked_outputs),
        replay_outputs=Path(args.replay_outputs),
        topk=int(args.topk),
        fields=fields,
    )
    output_dir = Path(args.output_dir)
    ensure_empty_output_dir(output_dir)
    write_json(output_dir / "loaded_model_replay_validation.json", result)
    (output_dir / "loaded_model_replay_validation.md").write_text(render_report(result, int(args.topk)), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
