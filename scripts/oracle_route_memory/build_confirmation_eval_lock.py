#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.training import build_confirmation_eval_lock_payload


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def render_markdown(payload: dict) -> str:
    return "\n".join(
        [
            "# Confirmation Eval Lock",
            "",
            "- Config is locked before blind confirmation metrics are read.",
            "- `confirmation_eval_consumed`: `false`",
            "- Do not tune checkpoints, split, selector candidates, or fusion config after this file is created.",
            "",
            "## Hashes",
            "",
            f"- Split manifest: `{payload['split_manifest_hash']}`",
            f"- Leakage audit: `{payload['leakage_audit_hash']}`",
            f"- Router checkpoint: `{payload['router_checkpoint_hash']}`",
            f"- Query-head checkpoint: `{payload['query_head_checkpoint_hash']}`",
            f"- Fusion config: `{payload['fusion_config_hash']}`",
            f"- Selector rows: `{payload['selector_rows_hash']}`",
            f"- Lock hash: `{payload['lock_hash']}`",
            "",
        ]
    )


def ensure_lock_output_dir(output_dir: Path, *, force: bool = False) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Lock output path exists but is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Lock output directory already contains files: {output_dir}. "
            "Use --force only for an intentional dry-run overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a one-shot blind confirmation eval lock.")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--leakage-audit", required=True)
    parser.add_argument("--router-checkpoint-dir", required=True)
    parser.add_argument("--query-head-checkpoint-dir", required=True)
    parser.add_argument("--fusion-config", required=True)
    parser.add_argument("--selector-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite files in an existing non-empty lock output directory.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_lock_output_dir(output_dir, force=args.force)
    payload = build_confirmation_eval_lock_payload(
        split_manifest_path=args.split_manifest,
        leakage_audit_path=args.leakage_audit,
        router_checkpoint_dir=args.router_checkpoint_dir,
        query_head_checkpoint_dir=args.query_head_checkpoint_dir,
        fusion_config_path=args.fusion_config,
        selector_rows_path=args.selector_rows,
        git_commit=git_commit(),
    )
    json_path = output_dir / "confirmation_eval_lock.json"
    md_path = output_dir / "confirmation_eval_lock.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "ok", "lock_json": str(json_path.resolve()), "lock_hash": payload["lock_hash"]}, indent=2))


if __name__ == "__main__":
    main()
