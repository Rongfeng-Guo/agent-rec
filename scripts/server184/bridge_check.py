#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
REQUIRED_FILES = {
    "validation": "cdpo_validation.json",
    "manifest": "cdpo_dataset_manifest.json",
    "audit": "audit/audit.json",
    "adapter_metadata": "adapter/adapter_metadata.json",
    "cdpo_pairs": "adapter/cdpo_pairs.jsonl",
    "train": "train.jsonl",
    "dev": "dev.jsonl",
}
EXPECTED_SOURCE = "RealBranchReplay"
EXPECTED_PROXY = "controlled real user simulator replay proxy"
EXPECTED_PROVENANCE = "REAL_USER_SIM_REPLAY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--real-branch-root")
    parser.add_argument("--run-dir")
    parser.add_argument("--bridge-dir")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1
    return count


def find_latest_run(real_branch_root: Path) -> Path | None:
    if not real_branch_root.exists():
        return None
    candidates = [child for child in real_branch_root.iterdir() if child.is_dir() and TIMESTAMP_RE.match(child.name)]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def build_bridge_metadata(root_dir: Path, real_branch_root: Path, run_dir: Path | None) -> dict:
    resolved_run_dir = run_dir or find_latest_run(real_branch_root)
    if resolved_run_dir is None:
        return {
            "status": "BLOCKED_NO_REAL_BRANCH_RUN",
            "root_dir": str(root_dir),
            "real_branch_root": str(real_branch_root),
            "latest_run_dir": None,
            "missing_files": list(REQUIRED_FILES.values()),
            "issues": ["No timestamped real_branch_replay run directory found."],
        }

    file_paths = {name: resolved_run_dir / rel_path for name, rel_path in REQUIRED_FILES.items()}
    missing_files = [rel_path for name, rel_path in REQUIRED_FILES.items() if not file_paths[name].exists()]
    payload = {
        "status": "OK",
        "root_dir": str(root_dir),
        "real_branch_root": str(real_branch_root),
        "latest_run_dir": str(resolved_run_dir),
        "missing_files": missing_files,
        "issues": [],
    }
    if missing_files:
        payload["status"] = "BLOCKED_MISSING_BRIDGE_ARTIFACTS"
        payload["issues"].append("Required bridge artifacts are missing.")
        return payload

    validation = load_json(file_paths["validation"])
    manifest = load_json(file_paths["manifest"])
    audit = load_json(file_paths["audit"])
    adapter_metadata = load_json(file_paths["adapter_metadata"])
    cdpo_pair_count = count_jsonl(file_paths["cdpo_pairs"])
    train_count = count_jsonl(file_paths["train"])
    dev_count = count_jsonl(file_paths["dev"])

    payload.update(
        {
            "validation_status": validation.get("status"),
            "validation_rows": validation.get("rows"),
            "validation_error_count": validation.get("error_count"),
            "manifest_status": manifest.get("status"),
            "manifest_source": manifest.get("source"),
            "manifest_proxy": manifest.get("proxy"),
            "manifest_row_count": manifest.get("row_count"),
            "manifest_by_provenance": manifest.get("by_provenance", {}),
            "audit_status": audit.get("status"),
            "audit_snapshot_count": audit.get("snapshot_count"),
            "audit_branch_count": audit.get("branch_count"),
            "audit_positive_uplift_count": audit.get("positive_uplift_count"),
            "adapter_status": adapter_metadata.get("status"),
            "adapter_pair_count": adapter_metadata.get("pair_count"),
            "adapter_positive_pair_count": adapter_metadata.get("positive_pair_count"),
            "cdpo_pair_count": cdpo_pair_count,
            "train_count": train_count,
            "dev_count": dev_count,
        }
    )

    if validation.get("status") != "PASS":
        payload["issues"].append(f"CDPO validation status is {validation.get('status')!r}, expected 'PASS'.")
    if audit.get("status") == "FAIL":
        payload["issues"].append("Replay audit failed.")
    if manifest.get("source") != EXPECTED_SOURCE:
        payload["issues"].append(f"Manifest source is {manifest.get('source')!r}, expected {EXPECTED_SOURCE!r}.")
    if manifest.get("proxy") != EXPECTED_PROXY:
        payload["issues"].append(f"Manifest proxy is {manifest.get('proxy')!r}, expected {EXPECTED_PROXY!r}.")
    if manifest.get("validation_status") != validation.get("status"):
        payload["issues"].append("Manifest validation_status does not match cdpo_validation.json.")
    if manifest.get("row_count") != cdpo_pair_count:
        payload["issues"].append("Manifest row_count does not match adapter/cdpo_pairs.jsonl line count.")
    if validation.get("rows") != cdpo_pair_count:
        payload["issues"].append("Validation rows do not match adapter/cdpo_pairs.jsonl line count.")
    if train_count + dev_count != cdpo_pair_count:
        payload["issues"].append("Train/dev split counts do not sum to cdpo_pairs row count.")
    if int(adapter_metadata.get("positive_pair_count", 0) or 0) <= 0:
        payload["issues"].append("Adapter positive_pair_count is not greater than zero.")
    if int(audit.get("positive_uplift_count", 0) or 0) <= 0:
        payload["issues"].append("Replay audit reports no positive uplift pairs.")
    if EXPECTED_PROVENANCE not in manifest.get("by_provenance", {}):
        payload["issues"].append(f"Manifest provenance summary does not contain {EXPECTED_PROVENANCE!r}.")

    if payload["issues"]:
        payload["status"] = "BLOCKED_BRIDGE_VALIDATION_FAILED"
    return payload


def build_report(payload: dict) -> str:
    lines = [
        "# Real Rollout Bridge Check",
        "",
        f"- status: `{payload.get('status')}`",
        f"- latest_run_dir: `{payload.get('latest_run_dir')}`",
        f"- validation_status: `{payload.get('validation_status')}`",
        f"- audit_status: `{payload.get('audit_status')}`",
        f"- manifest_source: `{payload.get('manifest_source')}`",
        f"- manifest_proxy: `{payload.get('manifest_proxy')}`",
        f"- cdpo_pair_count: `{payload.get('cdpo_pair_count')}`",
        f"- train/dev: `{payload.get('train_count')}` / `{payload.get('dev_count')}`",
        f"- positive_pairs(adapter/audit): `{payload.get('adapter_positive_pair_count')}` / `{payload.get('audit_positive_uplift_count')}`",
        "",
        "## Issues",
    ]
    issues = payload.get("issues") or []
    if not issues:
        lines.append("- none")
    else:
        lines.extend(f"- {issue}" for issue in issues)
    missing_files = payload.get("missing_files") or []
    lines.extend(["", "## Missing Files"])
    if not missing_files:
        lines.append("- none")
    else:
        lines.extend(f"- `{path}`" for path in missing_files)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    real_branch_root = Path(args.real_branch_root).resolve() if args.real_branch_root else root_dir / "outputs/server184_gimo/real_branch_replay"
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    bridge_dir = Path(args.bridge_dir).resolve() if args.bridge_dir else root_dir / "outputs/server184_gimo/bridge/latest_real"

    payload = build_bridge_metadata(root_dir, real_branch_root, run_dir)
    bridge_dir.mkdir(parents=True, exist_ok=True)
    (bridge_dir / "bridge_metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (bridge_dir / "bridge_report.md").write_text(build_report(payload), encoding="utf-8")
    (bridge_dir / "latest_run.txt").write_text((payload.get("latest_run_dir") or "") + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["status"] == "OK" else 2)


if __name__ == "__main__":
    main()
