#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--outputs-root")
    parser.add_argument("--index-json")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_smoke_only_status_counts(status_counts: dict | None) -> bool:
    counts = status_counts or {}
    return bool(counts) and set(counts).issubset({"SMOKE_TEST_ONLY"})


def summarize_closed_loop_readiness(rows: list[dict]) -> dict:
    if not rows:
        return {
            "run_count": 0,
            "smoke_only_run_count": 0,
            "non_smoke_run_count": 0,
            "best_run": None,
            "best_method": None,
            "best_avg_cumulative_utility": None,
            "status": "NO_CLOSED_LOOP_RUNS",
        }

    smoke_only_rows = [row for row in rows if is_smoke_only_status_counts(row.get("status_counts"))]
    non_smoke_rows = [row for row in rows if not is_smoke_only_status_counts(row.get("status_counts"))]
    ranked = sorted(
        rows,
        key=lambda row: (
            is_smoke_only_status_counts(row.get("status_counts")),
            -(row.get("best_avg_cumulative_utility") if isinstance(row.get("best_avg_cumulative_utility"), (int, float)) else float("-inf")),
            row.get("name", ""),
        ),
    )
    best_row = ranked[0] if ranked else None
    return {
        "run_count": len(rows),
        "smoke_only_run_count": len(smoke_only_rows),
        "non_smoke_run_count": len(non_smoke_rows),
        "best_run": best_row.get("name") if best_row else None,
        "best_method": best_row.get("best_method") if best_row else None,
        "best_avg_cumulative_utility": best_row.get("best_avg_cumulative_utility") if best_row else None,
        "status": "ONLY_SMOKE_CLOSED_LOOP_RUNS" if smoke_only_rows and not non_smoke_rows else "HAS_NON_SMOKE_CLOSED_LOOP_RUNS",
    }


def build_decision_payload(index_payload: dict) -> dict:
    env = index_payload.get("env") or {}
    bridge = index_payload.get("bridge_latest_real") or {}
    replay = index_payload.get("real_branch_replay") or {}
    closed_loop_rows = index_payload.get("closed_loop_runs") or []
    closed_loop = summarize_closed_loop_readiness(closed_loop_rows)

    blockers: list[str] = []
    next_steps: list[str] = []

    if not env.get("exists") or not env.get("model_exists") or not env.get("vllm_python_exists"):
        blockers.append("ENV_NOT_READY")
        next_steps.append("Repair the server184 model/vLLM environment before interpreting evaluation outputs.")

    if not bridge.get("exists") or bridge.get("status") != "OK" or bridge.get("validation_status") != "PASS" or bridge.get("audit_status") != "PASS":
        blockers.append("BRIDGE_NOT_READY")
        next_steps.append("Rebuild the real rollout bridge until validation and audit both pass.")

    if not replay.get("exists") or not replay.get("latest_ok_run"):
        blockers.append("REPLAY_NOT_READY")
        next_steps.append("Produce at least one successful real_branch_replay run before comparing methods.")

    if closed_loop["status"] == "NO_CLOSED_LOOP_RUNS":
        blockers.append("NO_CLOSED_LOOP_RUNS")
        next_steps.append("Run a closed-loop evaluation and export summary.json outputs under outputs/server184_gimo/closed_loop_*.")
    elif closed_loop["status"] == "ONLY_SMOKE_CLOSED_LOOP_RUNS":
        blockers.append("ONLY_SMOKE_CLOSED_LOOP_RUNS")
        next_steps.append("Promote server184 from smoke-only closed-loop runs to at least one non-smoke metric run.")

    overall_status = "READY_FOR_METRIC_REVIEW" if not blockers else "BLOCKED"
    summary = (
        "Server184 has enough validated replay/bridge/closed-loop evidence for metric review."
        if overall_status == "READY_FOR_METRIC_REVIEW"
        else "Server184 is not yet ready for final metric review."
    )

    return {
        "overall_status": overall_status,
        "summary": summary,
        "blockers": blockers,
        "next_steps": next_steps,
        "env_ready": not any(code == "ENV_NOT_READY" for code in blockers),
        "bridge_ready": not any(code == "BRIDGE_NOT_READY" for code in blockers),
        "replay_ready": not any(code == "REPLAY_NOT_READY" for code in blockers),
        "closed_loop_readiness": closed_loop,
        "index_path": index_payload.get("index_path"),
        "source_outputs_root": index_payload.get("outputs_root"),
        "latest_ok_replay_run": replay.get("latest_ok_run"),
        "latest_bridge_run_dir": bridge.get("latest_run_dir"),
    }


def build_report(payload: dict) -> str:
    closed_loop = payload["closed_loop_readiness"]
    lines = [
        "# Server184 Decision Report",
        "",
        f"- overall_status: `{payload['overall_status']}`",
        f"- summary: {payload['summary']}",
        f"- env_ready: `{payload['env_ready']}`",
        f"- bridge_ready: `{payload['bridge_ready']}`",
        f"- replay_ready: `{payload['replay_ready']}`",
        f"- latest_ok_replay_run: `{payload.get('latest_ok_replay_run')}`",
        f"- latest_bridge_run_dir: `{payload.get('latest_bridge_run_dir')}`",
        "",
        "## Closed Loop Readiness",
        f"- status: `{closed_loop['status']}`",
        f"- run_count: `{closed_loop['run_count']}`",
        f"- smoke_only_run_count: `{closed_loop['smoke_only_run_count']}`",
        f"- non_smoke_run_count: `{closed_loop['non_smoke_run_count']}`",
        f"- best_run: `{closed_loop['best_run']}`",
        f"- best_method: `{closed_loop['best_method']}`",
        f"- best_avg_cumulative_utility: `{closed_loop['best_avg_cumulative_utility']}`",
        "",
        "## Blockers",
    ]
    if not payload["blockers"]:
        lines.append("- none")
    else:
        lines.extend(f"- `{blocker}`" for blocker in payload["blockers"])
    lines.extend(["", "## Next Steps"])
    if not payload["next_steps"]:
        lines.append("- none")
    else:
        lines.extend(f"- {step}" for step in payload["next_steps"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    outputs_root = Path(args.outputs_root).resolve() if args.outputs_root else root_dir / "outputs/server184_gimo"
    index_json = Path(args.index_json).resolve() if args.index_json else outputs_root / "index" / "index.json"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else outputs_root / "decision"

    index_payload = read_json(index_json) or {}
    index_payload["index_path"] = str(index_json)
    if not index_payload.get("outputs_root"):
        index_payload["outputs_root"] = str(outputs_root)
    payload = build_decision_payload(index_payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decision.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "decision.md").write_text(build_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
