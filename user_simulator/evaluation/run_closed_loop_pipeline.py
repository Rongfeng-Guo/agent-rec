"""Run the full CritiqueWorld closed-loop artifact pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from user_simulator.evaluation.build_cdpo_dataset_manifest import read_jsonl
from user_simulator.evaluation.summarize_closed_loop_outputs import audit_output_dir, count_jsonl

FULL_EVAL_MODES = {"none", "flat", "structured", "time_decay", "critiquescope"}


def run_command(command: List[str]) -> dict:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def enrich_output_readme(output_dir: Path):
    readme = output_dir / "README.md"
    if not readme.exists():
        return
    lines = readme.read_text(encoding="utf-8").rstrip().splitlines()
    additions = [
        "- CDPO train/dev splits: `cdpo_train.jsonl`, `cdpo_dev.jsonl`",
        "- CDPO dataset manifest: `cdpo_dataset_manifest.json`",
        "- LLaMA-Factory dataset-info snippet: `llamafactory_dataset_info_snippet.json`",
        "- Closed-loop report: `closed_loop_report.md`",
        "- Pipeline metadata: `pipeline_metadata.json`",
    ]
    existing = set(lines)
    lines.extend(line for line in additions if line not in existing)
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def command_with_values(prefix: list[str], flag: str, values: Iterable[object]) -> list[str]:
    return [*prefix, flag, *[str(value) for value in values]]


def build_commands(args: argparse.Namespace, output_dir: Path) -> list[tuple[str, list[str]]]:
    python = sys.executable
    benchmark = [
        python,
        "-B",
        "-m",
        "user_simulator.evaluation.run_closed_loop_benchmark",
        "--max-turns",
        str(args.max_turns),
        "--top-k",
        str(args.top_k),
        "--parser-mode",
        args.parser_mode,
        "--branch-horizon",
        str(args.branch_horizon),
        "--output-dir",
        str(output_dir),
    ]
    benchmark = command_with_values(benchmark, "--modes", args.modes)
    benchmark = command_with_values(benchmark, "--scenarios", args.scenarios)
    benchmark = command_with_values(benchmark, "--seeds", args.seeds)

    validation = [
        python,
        "-B",
        "-m",
        "user_simulator.evaluation.validate_cdpo_pairs",
        "--input",
        str(output_dir / "cdpo_pairs.jsonl"),
        "--output",
        str(output_dir / "cdpo_validation.json"),
    ]
    manifest = [
        python,
        "-B",
        "-m",
        "user_simulator.evaluation.build_cdpo_dataset_manifest",
        "--input",
        str(output_dir / "cdpo_pairs.jsonl"),
        "--validation",
        str(output_dir / "cdpo_validation.json"),
        "--manifest-output",
        str(output_dir / "cdpo_dataset_manifest.json"),
        "--dataset-info-output",
        str(output_dir / "llamafactory_dataset_info_snippet.json"),
        "--train-output",
        str(output_dir / "cdpo_train.jsonl"),
        "--dev-output",
        str(output_dir / "cdpo_dev.jsonl"),
        "--dev-fraction",
        str(args.dev_fraction),
    ]
    report = [
        python,
        "-B",
        "-m",
        "user_simulator.evaluation.summarize_closed_loop_outputs",
        "--output-dir",
        str(output_dir),
        "--report-output",
        str(output_dir / "closed_loop_report.md"),
    ]
    commands = [
        ("benchmark", benchmark),
        ("validate_cdpo_pairs", validation),
        ("build_cdpo_dataset_manifest", manifest),
        ("summarize_closed_loop_outputs", report),
    ]
    if args.run_validity_gate:
        validity_gate = [
            python,
            "-B",
            "-m",
            "user_simulator.evaluation.run_validity_gate",
            "--output-dir",
            str(output_dir / "validity_gate"),
            "--parser-mode",
            args.parser_mode,
            "--branch-horizon",
            str(args.branch_horizon),
            "--max-turns",
            str(args.max_turns),
            "--top-k",
            str(args.top_k),
        ]
        validity_gate = command_with_values(validity_gate, "--modes", args.modes)
        validity_gate = command_with_values(validity_gate, "--scenarios", args.scenarios)
        validity_gate = command_with_values(validity_gate, "--seeds", args.seeds)
        if args.fail_on_critical_invariant:
            validity_gate.append("--fail-on-critical-invariant")
        commands.append(("run_validity_gate", validity_gate))
    return commands


def has_full_eval_coverage(args: argparse.Namespace) -> bool:
    scenario_values = {str(value) for value in args.scenarios}
    mode_values = {str(value) for value in args.modes}
    return scenario_values == {"all"} and mode_values == FULL_EVAL_MODES


def validity_gate_passed(validity_metadata: dict | None) -> bool:
    if not validity_metadata:
        return False
    failed = int(validity_metadata.get("failed_invariants", 0) or 0)
    critical_failed = int(validity_metadata.get("critical_failed_invariants", 0) or 0)
    return failed == 0 and critical_failed == 0


def classify_run_status(args: argparse.Namespace, audit: dict, validity_metadata: dict | None) -> str:
    if audit["status"] != "PASS":
        return audit["status"]
    if has_full_eval_coverage(args) and validity_gate_passed(validity_metadata):
        return "PASS"
    return "SMOKE_TEST_ONLY"


def rewrite_json(path: Path, transform) -> None:
    payload = read_json(path)
    if payload is None:
        return
    updated = transform(payload)
    path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rewrite_summary_csv(path: Path, run_status: str) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    header = lines[0]
    headers = header.split(",")
    if "status" not in headers:
        return
    status_index = headers.index("status")
    rewritten = [header]
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) <= status_index:
            rewritten.append(line)
            continue
        fields[status_index] = run_status
        rewritten.append(",".join(fields))
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def rewrite_readme_status(path: Path, run_status: str) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    rewritten = []
    replaced = False
    for line in lines:
        if line.startswith("- Status: "):
            rewritten.append(f"- Status: {run_status}")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        rewritten.insert(2 if len(rewritten) >= 2 else len(rewritten), f"- Status: {run_status}")
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def normalize_closed_loop_artifacts(output_dir: Path, run_status: str) -> None:
    rewrite_json(
        output_dir / "summary.json",
        lambda rows: [{**row, "status": run_status} for row in rows] if isinstance(rows, list) else rows,
    )
    rewrite_summary_csv(output_dir / "summary.csv", run_status)
    rewrite_json(output_dir / "run_metadata.json", lambda payload: {**payload, "status": run_status})
    rewrite_json(output_dir / "cdpo_dataset_manifest.json", lambda payload: {**payload, "status": run_status})
    rewrite_readme_status(output_dir / "README.md", run_status)


def build_summary(output_dir: Path, args: argparse.Namespace, steps: list[dict]) -> dict:
    audit = audit_output_dir(output_dir)
    counts = {
        "trajectories": count_jsonl(output_dir / "trajectories.jsonl"),
        "branch_rollouts": count_jsonl(output_dir / "branch_rollouts.jsonl"),
        "dpo_pairs": count_jsonl(output_dir / "dpo_pairs.jsonl"),
        "cdpo_pairs": count_jsonl(output_dir / "cdpo_pairs.jsonl"),
        "cdpo_train": count_jsonl(output_dir / "cdpo_train.jsonl"),
        "cdpo_dev": count_jsonl(output_dir / "cdpo_dev.jsonl"),
    }
    rows = read_jsonl(output_dir / "cdpo_pairs.jsonl")
    unique_ids = len({row.get("id") for row in rows})
    validity_metadata = None
    validity_path = output_dir / "validity_gate" / "run_metadata.json"
    if validity_path.exists():
        validity_metadata = json.loads(validity_path.read_text(encoding="utf-8"))
    run_status = classify_run_status(args, audit, validity_metadata)
    return {
        "status": audit["status"],
        "run_status": run_status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "parser_mode": args.parser_mode,
        "modes": args.modes,
        "scenarios": args.scenarios,
        "seeds": args.seeds,
        "max_turns": args.max_turns,
        "top_k": args.top_k,
        "branch_horizon": args.branch_horizon,
        "dev_fraction": args.dev_fraction,
        "counts": counts,
        "unique_cdpo_ids": unique_ids,
        "audit_errors": audit["errors"],
        "validity_gate": validity_metadata,
        "steps": steps,
    }


def run_pipeline(args: argparse.Namespace) -> dict:
    if args.parser_mode == "openai_compatible":
        raise SystemExit("BLOCKED_NO_API_KEY: openai_compatible parser is optional and not run without API config")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = []
    try:
        for name, command in build_commands(args, output_dir):
            result = run_command(command)
            steps.append({"name": name, "returncode": result["returncode"], "command": command})
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "steps": steps,
            "error": str(exc),
        }
        write_json(output_dir / "pipeline_metadata.json", failure)
        raise

    enrich_output_readme(output_dir)
    summary = build_summary(output_dir, args, steps)
    normalize_closed_loop_artifacts(output_dir, summary["run_status"])
    write_json(output_dir / "pipeline_metadata.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["none", "flat", "structured", "time_decay", "critiquescope"])
    parser.add_argument("--scenarios", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--parser-mode", choices=["oracle", "deterministic", "openai_compatible"], default="oracle")
    parser.add_argument("--branch-horizon", type=int, default=5)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--run-validity-gate", action="store_true")
    parser.add_argument("--fail-on-critical-invariant", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
