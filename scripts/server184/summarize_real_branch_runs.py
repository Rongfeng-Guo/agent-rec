#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
REQUIRED_ARTIFACTS = {
    "validation": "cdpo_validation.json",
    "manifest": "cdpo_dataset_manifest.json",
    "audit": "audit/audit.json",
    "adapter": "adapter/adapter_metadata.json",
    "cdpo_pairs": "adapter/cdpo_pairs.jsonl",
    "train": "train.jsonl",
    "dev": "dev.jsonl",
}
EXPECTED_SOURCE = "RealBranchReplay"
EXPECTED_PROXY = "controlled real user simulator replay proxy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--runs-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1
    return count


def summarize_run(run_dir: Path) -> dict:
    validation = read_json(run_dir / "cdpo_validation.json")
    manifest = read_json(run_dir / "cdpo_dataset_manifest.json")
    audit = read_json(run_dir / "audit" / "audit.json")
    adapter = read_json(run_dir / "adapter" / "adapter_metadata.json")
    train_count = count_jsonl(run_dir / "train.jsonl")
    dev_count = count_jsonl(run_dir / "dev.jsonl")
    cdpo_pair_count = count_jsonl(run_dir / "adapter" / "cdpo_pairs.jsonl")

    missing_artifacts = [rel_path for rel_path in REQUIRED_ARTIFACTS.values() if not (run_dir / rel_path).exists()]
    issues: list[str] = []
    bridge_status = "OK"

    if missing_artifacts:
        bridge_status = "MISSING_REQUIRED_ARTIFACTS"
        issues.append("missing required artifacts")
    else:
        if validation is not None and validation.get("status") != "PASS":
            bridge_status = "VALIDATION_NOT_PASS"
            issues.append(f"validation status={validation.get('status')!r}")
        if manifest is not None:
            if manifest.get("source") != EXPECTED_SOURCE:
                if bridge_status == "OK":
                    bridge_status = "MANIFEST_NOT_REAL_REPLAY"
                issues.append(f"manifest source={manifest.get('source')!r}")
            if manifest.get("proxy") != EXPECTED_PROXY:
                if bridge_status == "OK":
                    bridge_status = "MANIFEST_PROXY_MISMATCH"
                issues.append(f"manifest proxy={manifest.get('proxy')!r}")
        if audit is not None and audit.get("status") != "PASS":
            if bridge_status == "OK":
                bridge_status = "AUDIT_NOT_PASS"
            issues.append(f"audit status={audit.get('status')!r}")
        if train_count is not None and dev_count is not None and cdpo_pair_count is not None and train_count + dev_count != cdpo_pair_count:
            if bridge_status == "OK":
                bridge_status = "SPLIT_COUNT_MISMATCH"
            issues.append("train/dev counts do not sum to cdpo_pairs count")

    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "validation_status": validation.get("status") if validation else None,
        "validation_rows": validation.get("rows") if validation else None,
        "manifest_source": manifest.get("source") if manifest else None,
        "manifest_proxy": manifest.get("proxy") if manifest else None,
        "manifest_row_count": manifest.get("row_count") if manifest else None,
        "audit_status": audit.get("status") if audit else None,
        "audit_positive_uplift_count": audit.get("positive_uplift_count") if audit else None,
        "adapter_status": adapter.get("status") if adapter else None,
        "adapter_positive_pair_count": adapter.get("positive_pair_count") if adapter else None,
        "cdpo_pair_count": cdpo_pair_count,
        "train_count": train_count,
        "dev_count": dev_count,
        "missing_artifacts": missing_artifacts,
        "bridge_status": bridge_status,
        "bridge_issues": issues,
    }


def collect_runs(runs_root: Path) -> list[dict]:
    rows = []
    if not runs_root.exists():
        return rows
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and TIMESTAMP_RE.match(p.name)):
        rows.append(summarize_run(run_dir))
    return rows


def build_report(rows: list[dict]) -> str:
    latest_run = rows[-1] if rows else None
    latest_ok = next((row for row in reversed(rows) if row["bridge_status"] == "OK"), None)
    status_counts = Counter(row["bridge_status"] for row in rows)
    lines = [
        "# Real Branch Replay Run Summary",
        "",
        f"- total_runs: `{len(rows)}`",
        f"- latest_run: `{latest_run['run'] if latest_run else 'NONE'}`",
        f"- latest_run_status: `{latest_run['bridge_status'] if latest_run else 'NONE'}`",
        f"- latest_ok_run: `{latest_ok['run'] if latest_ok else 'NONE'}`",
        "",
        "## Status Counts",
    ]
    if not status_counts:
        lines.append("- none")
    else:
        for status, count in sorted(status_counts.items()):
            lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Runs"])
    if not rows:
        lines.append("- none")
        return "\n".join(lines) + "\n"
    for row in reversed(rows):
        issue_text = ", ".join(row["bridge_issues"]) if row["bridge_issues"] else "none"
        missing_text = ", ".join(row["missing_artifacts"]) if row["missing_artifacts"] else "none"
        lines.append(
            f"- `{row['run']}` bridge=`{row['bridge_status']}` audit=`{row['audit_status']}` "
            f"validation=`{row['validation_status']}` pairs=`{row['cdpo_pair_count']}` missing={missing_text} issues={issue_text}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    runs_root = Path(args.runs_root).resolve() if args.runs_root else root_dir / "outputs/server184_gimo/real_branch_replay"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root_dir / "outputs/server184_gimo/real_branch_replay_summary"
    rows = collect_runs(runs_root)
    latest_run = rows[-1] if rows else None
    latest_ok = next((row for row in reversed(rows) if row["bridge_status"] == "OK"), None)
    payload = {
        "runs_root": str(runs_root),
        "run_count": len(rows),
        "latest_run": latest_run["run"] if latest_run else None,
        "latest_run_status": latest_run["bridge_status"] if latest_run else None,
        "latest_ok_run": latest_ok["run"] if latest_ok else None,
        "latest_ok_path": latest_ok["path"] if latest_ok else None,
        "status_counts": dict(sorted(Counter(row["bridge_status"] for row in rows).items())),
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(build_report(rows), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
