#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_INPUT="${1:-}"
OUTPUT_DIR="${2:-}"
PYTHON_BIN="$(command -v python.exe || command -v python)"
to_win_path() { wslpath -w "$1"; }
TRACE_INPUT_WIN="$(to_win_path "${TRACE_INPUT}")"
OUTPUT_DIR_WIN="$(to_win_path "${OUTPUT_DIR}")"

if [[ -z "${TRACE_INPUT}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: bash scripts/run_real_rollout_bridge.sh <TRACE_DIR_OR_FILE> <OUTPUT_DIR>" >&2
  exit 1
fi

if [[ -d "${TRACE_INPUT}" ]]; then
  if ! find "${TRACE_INPUT}" -type f \( -name '*refine_log_sample*.json' -o -name '*refine_log_sample*.jsonl' -o -name '*refine_log*.json' -o -name '*refine_log*.jsonl' \) | grep -q .; then
    echo "BLOCKED_REAL_LOG_MISSING" >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_DIR}/export" "${OUTPUT_DIR}/audit" "${OUTPUT_DIR}/cdpo_dataset"

"${PYTHON_BIN}" -B -m user_simulator.evaluation.export_gpe_hap_refine_logs \
  --input "${TRACE_INPUT_WIN}" \
  --output-dir "$(to_win_path "${OUTPUT_DIR}/export")" \
  --write-source-jsonl

"${PYTHON_BIN}" -B -m user_simulator.evaluation.audit_real_rollout_bridge \
  --input-dir "$(to_win_path "${OUTPUT_DIR}/export")" \
  --output-dir "$(to_win_path "${OUTPUT_DIR}/audit")" \
  --fail-on-critical-error

"${PYTHON_BIN}" -B -m user_simulator.evaluation.materialize_cdpo_dataset \
  --input "$(to_win_path "${OUTPUT_DIR}/export/cdpo_pairs.jsonl")" \
  --output-dir "$(to_win_path "${OUTPUT_DIR}/cdpo_dataset")" \
  --seed 42 \
  --dev-ratio 0.1

bash "${ROOT_DIR}/scripts/validate_llamafactory_cdpo_dataset.sh" "${OUTPUT_DIR}/cdpo_dataset"

"${PYTHON_BIN}" - "${OUTPUT_DIR_WIN}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary = {
    "export": json.loads((output_dir / "export" / "export_metadata.json").read_text(encoding="utf-8")),
    "audit": json.loads((output_dir / "audit" / "audit.json").read_text(encoding="utf-8")),
    "dataset": json.loads((output_dir / "cdpo_dataset" / "manifest.json").read_text(encoding="utf-8")),
    "llamafactory_validation": (output_dir / "cdpo_dataset" / "llamafactory_validation.md").read_text(encoding="utf-8"),
}

md = [
    "# Real Rollout Bridge Pipeline Summary",
    "",
    f"- Export status: `{summary['export']['status']}`",
    f"- Export traces: `{summary['export']['trace_count']}`",
    f"- Bridge audit status: `{summary['audit']['status']}`",
    f"- Bridge critical errors: `{summary['audit']['critical_error_count']}`",
    f"- Dataset status: `{summary['dataset']['status']}`",
    f"- Train rows: `{summary['dataset']['splits']['train_count']}`",
    f"- Dev rows: `{summary['dataset']['splits']['dev_count']}`",
    f"- LLaMA-Factory validation: `{summary['llamafactory_validation'].splitlines()[2].split(':', 1)[-1].strip(' `') if summary['llamafactory_validation'].splitlines() else 'UNKNOWN'}`",
    "",
    "## Artifacts",
    f"- Export: `{output_dir / 'export'}`",
    f"- Audit: `{output_dir / 'audit'}`",
    f"- Dataset: `{output_dir / 'cdpo_dataset'}`",
]
Path(output_dir / "pipeline_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
PY
