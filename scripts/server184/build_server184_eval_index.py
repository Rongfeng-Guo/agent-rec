#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--outputs-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_env(outputs_root: Path) -> dict:
    payload = read_json(outputs_root / "env" / "env_report.json") or {}
    return {
        "exists": bool(payload),
        "model_exists": payload.get("model_exists"),
        "model_path": payload.get("model_path"),
        "vllm_python_exists": payload.get("vllm_python_exists"),
        "vllm_base_url": payload.get("vllm_base_url"),
        "vllm_model_alias": payload.get("vllm_model_alias"),
        "smoke_sample_limit": payload.get("smoke_sample_limit"),
    }


def summarize_bridge(outputs_root: Path) -> dict:
    payload = read_json(outputs_root / "bridge" / "latest_real" / "bridge_metadata.json") or {}
    return {
        "exists": bool(payload),
        "status": payload.get("status"),
        "latest_run_dir": payload.get("latest_run_dir"),
        "validation_status": payload.get("validation_status"),
        "audit_status": payload.get("audit_status"),
        "cdpo_pair_count": payload.get("cdpo_pair_count"),
        "issues": payload.get("issues") or [],
    }


def summarize_replay(outputs_root: Path) -> dict:
    payload = read_json(outputs_root / "real_branch_replay_summary" / "summary.json") or {}
    return {
        "exists": bool(payload),
        "run_count": payload.get("run_count"),
        "latest_run": payload.get("latest_run"),
        "latest_run_status": payload.get("latest_run_status"),
        "latest_ok_run": payload.get("latest_ok_run"),
        "latest_ok_path": payload.get("latest_ok_path"),
        "status_counts": payload.get("status_counts") or {},
    }


def summarize_closed_loop_dir(run_dir: Path) -> dict:
    rows = read_json(run_dir / "summary.json") or []
    if not isinstance(rows, list):
        rows = []
    status_counts = Counter(str(row.get("status")) for row in rows)
    scenarios = sorted({str(row.get("scenario")) for row in rows if row.get("scenario") is not None})
    parser_modes = sorted({str(row.get("parser_mode")) for row in rows if row.get("parser_mode") is not None})
    method_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        method = row.get("method")
        utility = row.get("CumulativeUtility")
        if method is not None and isinstance(utility, (int, float)):
            method_scores[str(method)].append(float(utility))
    best_method = None
    best_avg = None
    if method_scores:
        scored = sorted(
            ((method, sum(values) / len(values)) for method, values in method_scores.items()),
            key=lambda item: (-item[1], item[0]),
        )
        best_method, best_avg = scored[0]
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "row_count": len(rows),
        "scenario_count": len(scenarios),
        "parser_modes": parser_modes,
        "status_counts": dict(sorted(status_counts.items())),
        "best_method": best_method,
        "best_avg_cumulative_utility": best_avg,
    }


def collect_closed_loop(outputs_root: Path) -> list[dict]:
    rows = []
    if not outputs_root.exists():
        return rows
    for child in sorted(outputs_root.iterdir()):
        if child.is_dir() and child.name.startswith("closed_loop_") and (child / "summary.json").exists():
            rows.append(summarize_closed_loop_dir(child))
    return rows


def load_decision_module():
    module_path = Path(__file__).with_name("build_server184_decision_report.py")
    spec = importlib.util.spec_from_file_location("build_server184_decision_report", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load decision report module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_decision_artifacts(payload: dict, outputs_root: Path) -> dict:
    module = load_decision_module()
    decision_payload = dict(payload)
    decision_payload["index_path"] = str(outputs_root / "index" / "index.json")
    built = module.build_decision_payload(decision_payload)
    decision_dir = outputs_root / "decision"
    decision_dir.mkdir(parents=True, exist_ok=True)
    (decision_dir / "decision.json").write_text(json.dumps(built, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (decision_dir / "decision.md").write_text(module.build_report(built), encoding="utf-8")
    return {
        "decision_dir": str(decision_dir),
        "overall_status": built.get("overall_status"),
        "blockers": built.get("blockers") or [],
    }


def load_metric_table_module():
    module_path = Path(__file__).with_name("build_server184_metric_table.py")
    spec = importlib.util.spec_from_file_location("build_server184_metric_table", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load metric table module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_metric_table_artifacts(outputs_root: Path) -> dict:
    module = load_metric_table_module()
    run_rows = module.collect_runs(outputs_root)
    method_rows = module.flatten_method_rows(run_rows)
    metric_dir = outputs_root / "metric_table"
    payload = {
        "outputs_root": str(outputs_root),
        "run_count": len(run_rows),
        "method_row_count": len(method_rows),
        "run_rows": run_rows,
        "method_rows": method_rows,
    }
    metric_dir.mkdir(parents=True, exist_ok=True)
    (metric_dir / "metric_table.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (metric_dir / "metric_table.md").write_text(module.build_markdown(run_rows, method_rows), encoding="utf-8")
    module.write_csv(metric_dir / "metric_table.csv", method_rows)
    ranked_runs = sorted(
        run_rows,
        key=lambda row: (
            -(row.get("best_cumulative_utility") if isinstance(row.get("best_cumulative_utility"), (int, float)) else float("-inf")),
            row.get("run", ""),
        ),
    )
    top_run = ranked_runs[0] if ranked_runs else None
    return {
        "metric_dir": str(metric_dir),
        "run_count": len(run_rows),
        "method_row_count": len(method_rows),
        "top_run": top_run.get("run") if top_run else None,
        "top_run_best_method": top_run.get("best_method") if top_run else None,
        "top_run_best_cumulative_utility": top_run.get("best_cumulative_utility") if top_run else None,
    }


def build_report(payload: dict) -> str:
    env = payload["env"]
    bridge = payload["bridge_latest_real"]
    replay = payload["real_branch_replay"]
    closed_loop = payload["closed_loop_runs"]
    lines = [
        "# Server184 Eval Index",
        "",
        "## Environment",
        f"- env_report: `{'present' if env['exists'] else 'missing'}`",
        f"- model_exists: `{env['model_exists']}`",
        f"- vllm_python_exists: `{env['vllm_python_exists']}`",
        f"- vllm_base_url: `{env['vllm_base_url']}`",
        f"- vllm_model_alias: `{env['vllm_model_alias']}`",
        "",
        "## Latest Real Bridge",
        f"- bridge_status: `{bridge['status']}`",
        f"- latest_run_dir: `{bridge['latest_run_dir']}`",
        f"- validation/audit: `{bridge['validation_status']}` / `{bridge['audit_status']}`",
        f"- cdpo_pair_count: `{bridge['cdpo_pair_count']}`",
        "",
        "## Real Branch Replay",
        f"- run_count: `{replay['run_count']}`",
        f"- latest_run: `{replay['latest_run']}`",
        f"- latest_run_status: `{replay['latest_run_status']}`",
        f"- latest_ok_run: `{replay['latest_ok_run']}`",
        "",
        "## Replay Status Counts",
    ]
    status_counts = replay.get("status_counts") or {}
    if not status_counts:
        lines.append("- none")
    else:
        for status, count in sorted(status_counts.items()):
            lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Closed Loop Runs"])
    if not closed_loop:
        lines.append("- none")
    else:
        for row in closed_loop:
            lines.append(
                f"- `{row['name']}` rows=`{row['row_count']}` scenarios=`{row['scenario_count']}` "
                f"best_method=`{row['best_method']}` best_avg_cu=`{row['best_avg_cumulative_utility']}` statuses={row['status_counts']}"
            )
    metric_table = payload.get("metric_table") or {}
    lines.extend(["", "## Metric Table"])
    if not metric_table:
        lines.append("- none")
    else:
        lines.append(f"- run_count: `{metric_table.get('run_count')}`")
        lines.append(f"- method_row_count: `{metric_table.get('method_row_count')}`")
        lines.append(f"- top_run: `{metric_table.get('top_run')}`")
        lines.append(f"- top_run_best_method: `{metric_table.get('top_run_best_method')}`")
        lines.append(f"- top_run_best_cumulative_utility: `{metric_table.get('top_run_best_cumulative_utility')}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    outputs_root = Path(args.outputs_root).resolve() if args.outputs_root else root_dir / "outputs/server184_gimo"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else outputs_root / "index"
    payload = {
        "outputs_root": str(outputs_root),
        "env": summarize_env(outputs_root),
        "bridge_latest_real": summarize_bridge(outputs_root),
        "real_branch_replay": summarize_replay(outputs_root),
        "closed_loop_runs": collect_closed_loop(outputs_root),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    payload["metric_table"] = build_metric_table_artifacts(outputs_root)
    (output_dir / "index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "index.md").write_text(build_report(payload), encoding="utf-8")
    payload["decision_report"] = build_decision_artifacts(payload, outputs_root)
    (output_dir / "index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
