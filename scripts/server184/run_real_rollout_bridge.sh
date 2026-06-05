#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$(command -v python3 || command -v python || true)"
TRACE_INPUT="${1:-}"
OUTPUT_DIR="${2:-}"

if [[ -z "${TRACE_INPUT}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: bash scripts/server184/run_real_rollout_bridge.sh <TRACE_DIR_OR_FILE> <OUTPUT_DIR>" >&2
  exit 1
fi

if [[ ! -e "${TRACE_INPUT}" ]]; then
  echo "BLOCKED_REAL_LOG_MISSING" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/audit" "${OUTPUT_DIR}/cdpo_dataset"
"${PYTHON_BIN:-python}" -B -m user_simulator.evaluation.export_gpe_hap_refine_logs \
  --input "${TRACE_INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --write-source-jsonl

"${PYTHON_BIN:-python}" -B -m user_simulator.evaluation.audit_real_rollout_bridge \
  --input-dir "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}/audit" \
  --fail-on-critical-error

"${PYTHON_BIN:-python}" -B -m user_simulator.evaluation.materialize_cdpo_dataset \
  --input "${OUTPUT_DIR}/cdpo_pairs.jsonl" \
  --output-dir "${OUTPUT_DIR}/cdpo_dataset" \
  --seed 42 \
  --dev-ratio 0.1

bash "${ROOT_DIR}/scripts/validate_llamafactory_cdpo_dataset.sh" "${OUTPUT_DIR}/cdpo_dataset"

"${PYTHON_BIN:-python}" - <<'PY' "${OUTPUT_DIR}"
from __future__ import annotations
import json
import sys
from pathlib import Path
output_dir = Path(sys.argv[1])
audit = json.loads((output_dir / "audit" / "audit.json").read_text(encoding="utf-8"))
manifest = json.loads((output_dir / "cdpo_dataset" / "manifest.json").read_text(encoding="utf-8"))
summary = [
    "# Server184 Real Rollout Bridge",
    "",
    f"- Audit status: `{audit['status']}`",
    f"- Source rows: `{audit['source_row_count']}`",
    f"- Branch rows: `{audit['branch_rollout_count']}`",
    f"- DPO pairs: `{audit['dpo_pair_count']}`",
    f"- CDPO pairs: `{audit['cdpo_pair_count']}`",
    f"- Train rows: `{manifest['splits']['train_count']}`",
    f"- Dev rows: `{manifest['splits']['dev_count']}`",
]
(output_dir / "pipeline_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
PY
