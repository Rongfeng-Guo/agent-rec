#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${1:-${ROOT_DIR}/outputs/cdpo_dataset_fixture_smoke}"
PYTHON_BIN="$(command -v python.exe || command -v python)"
to_win_path() { wslpath -w "$1"; }
DATASET_DIR_WIN="$(to_win_path "${DATASET_DIR}")"
ROOT_DIR_WIN="$(to_win_path "${ROOT_DIR}")"

"${PYTHON_BIN}" - "${DATASET_DIR_WIN}" "${ROOT_DIR_WIN}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

dataset_dir = Path(sys.argv[1])
root_dir = Path(sys.argv[2])

required_files = {
    "dataset_info": dataset_dir / "dataset_info.json",
    "train": dataset_dir / "train.json",
    "dev": dataset_dir / "dev.json",
    "manifest": dataset_dir / "manifest.json",
    "audit": dataset_dir / "audit.json",
}

missing = [name for name, path in required_files.items() if not path.exists()]
if missing:
    raise SystemExit(f"BLOCKED_DATASET_SCHEMA: missing dataset files: {missing}")

dataset_info = json.loads(required_files["dataset_info"].read_text(encoding="utf-8"))
train_rows = json.loads(required_files["train"].read_text(encoding="utf-8"))
dev_rows = json.loads(required_files["dev"].read_text(encoding="utf-8"))
manifest = json.loads(required_files["manifest"].read_text(encoding="utf-8"))

required_row_fields = {
    "id",
    "scenario",
    "seed",
    "method",
    "parser_mode",
    "conversations",
    "chosen",
    "rejected",
    "score_delta",
    "metadata",
    "source_ref",
    "state_snapshot_hash",
    "git_commit",
}

def check_rows(rows: list[dict], split: str) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        missing = sorted(required_row_fields - set(row))
        if missing:
            errors.append(f"{split}[{index}] missing fields: {missing}")
        if Path(str(row.get("source_ref", ""))).is_absolute():
            errors.append(f"{split}[{index}] source_ref must not be an absolute path")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            errors.append(f"{split}[{index}] metadata must be an object")
        else:
            for key in ["source_ref", "state_snapshot_hash", "git_commit"]:
                if key not in metadata:
                    errors.append(f"{split}[{index}] metadata missing {key}")
        for branch in ["chosen", "rejected"]:
            branch_row = row.get(branch, {})
            if not isinstance(branch_row, dict):
                errors.append(f"{split}[{index}] {branch} must be an object")
            elif not str(branch_row.get("trajectory", "")).strip():
                errors.append(f"{split}[{index}] {branch}.trajectory must not be empty")
    return errors

errors = []
errors.extend(check_rows(train_rows, "train"))
errors.extend(check_rows(dev_rows, "dev"))

dataset_paths = []
for key, value in dataset_info.items():
    if isinstance(value, dict):
        file_name = value.get("file_name")
        if file_name:
            dataset_paths.append(file_name)
            if os.path.isabs(file_name):
                errors.append(f"{key} file_name must be relative, got {file_name}")

if errors:
    raise SystemExit("BLOCKED_DATASET_SCHEMA:\n" + "\n".join(errors[:50]))

model_path = os.environ.get("LLAMAFACTORY_MODEL_NAME_OR_PATH") or os.environ.get("MODEL_NAME_OR_PATH")
model_status = "BLOCKED_MODEL_MISSING"
if model_path and Path(model_path).exists():
    model_status = "READY_FOR_GPU_CDPO_SMOKE"

llama_factory_dir = root_dir / "LLaMA-Factory"
if not llama_factory_dir.exists():
    raise SystemExit("BLOCKED_LLAMAFACTORY_CONFIG: LLaMA-Factory directory is missing")

validation = {
    "status": model_status,
    "dataset_dir": str(dataset_dir),
    "llama_factory_dir": str(llama_factory_dir),
    "dataset_files": dataset_paths,
    "train_count": len(train_rows),
    "dev_count": len(dev_rows),
    "manifest_row_count": manifest.get("row_count"),
    "model_path": model_path,
}

md = [
    "# LLaMA-Factory CDPO Validation",
    "",
    f"- Status: `{validation['status']}`",
    f"- Dataset dir: `{validation['dataset_dir']}`",
    f"- Train rows: `{validation['train_count']}`",
    f"- Dev rows: `{validation['dev_count']}`",
    f"- Manifest rows: `{validation['manifest_row_count']}`",
    f"- Model path: `{validation['model_path'] or 'missing'}`",
    "",
    "## Checks",
    "- dataset_info.json is valid JSON and points to relative files",
    "- train/dev JSON files are readable",
    "- required row fields exist",
    "- LLaMA-Factory repository path exists",
]

out = dataset_dir / "llamafactory_validation.md"
out.write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps(validation, indent=2, ensure_ascii=False))
PY
