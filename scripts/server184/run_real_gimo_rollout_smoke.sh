#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$(command -v python3 || command -v python || true)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${GPE_HAP_OUTPUT_DIR:-${ROOT_DIR}/outputs/server184_real_rollout_smoke}"
RUN_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"
REFINE_DIR="${RUN_DIR}/refine_logs"
mkdir -p "${REFINE_DIR}"

if [[ -z "${GPE_HAP_DOMAIN:-}" ]]; then
  echo "BLOCKED_DATA_MISSING" >&2
  exit 1
fi
if [[ -z "${GPE_HAP_INPUT:-}" || ! -d "${GPE_HAP_INPUT}" ]]; then
  echo "BLOCKED_DATA_MISSING" >&2
  exit 1
fi
if [[ -z "${GIMO_INDEX_ROOT:-}" || ! -d "${GIMO_INDEX_ROOT}" ]]; then
  echo "BLOCKED_INDEX_MISSING" >&2
  exit 1
fi
if [[ -z "${GPE_HAP_CONFIG_PATH:-}" || ! -f "${ROOT_DIR}/${GPE_HAP_CONFIG_PATH}" && ! -f "${GPE_HAP_CONFIG_PATH}" ]]; then
  echo "BLOCKED_API_CONFIG" >&2
  exit 1
fi

CONFIG_PATH="${GPE_HAP_CONFIG_PATH}"
if [[ -f "${ROOT_DIR}/${GPE_HAP_CONFIG_PATH}" ]]; then
  CONFIG_PATH="${ROOT_DIR}/${GPE_HAP_CONFIG_PATH}"
fi

CMD=(
  "${PYTHON_BIN:-python}" "${ROOT_DIR}/GPE_HAP/rewrite_v3.py"
  --domain "${GPE_HAP_DOMAIN}"
  --config_path "${CONFIG_PATH}"
  --index_root "${GIMO_INDEX_ROOT}"
  --data_root "${GPE_HAP_INPUT}"
  --output_dir "${REFINE_DIR}"
  --task_limit "${SMOKE_SAMPLE_LIMIT:-3}"
  --sample_num "${SMOKE_SAMPLE_LIMIT:-3}"
  --max_workers "${GPE_HAP_MAX_WORKERS:-4}"
)

printf '%q ' "${CMD[@]}" > "${RUN_DIR}/command.sh"
printf '\n' >> "${RUN_DIR}/command.sh"
chmod +x "${RUN_DIR}/command.sh"

"${PYTHON_BIN:-python}" - <<'PY' "${RUN_DIR}/env_summary.json"
from __future__ import annotations
import json
import os
import sys
payload = {
    key: os.environ.get(key)
    for key in [
        "MODEL_NAME_OR_PATH",
        "LLAMAFACTORY_MODEL_NAME_OR_PATH",
        "GIMO_DATA_ROOT",
        "GIMO_INDEX_ROOT",
        "GPE_HAP_INPUT",
        "GPE_HAP_OUTPUT_DIR",
        "BRIDGE_OUTPUT_DIR",
        "VLLM_PORT",
        "CUDA_VISIBLE_DEVICES",
        "MAX_MODEL_LEN",
        "GPU_MEMORY_UTILIZATION",
        "SMOKE_SAMPLE_LIMIT",
        "GPE_HAP_DOMAIN",
        "GPE_HAP_CONFIG_PATH",
    ]
}
with open(sys.argv[1], "w", encoding="utf-8") as file:
    json.dump(payload, file, indent=2, ensure_ascii=False)
    file.write("\n")
PY

set +e
"${CMD[@]}" >"${RUN_DIR}/stdout.log" 2>"${RUN_DIR}/stderr.log"
EXIT_CODE=$?
set -e

"${PYTHON_BIN:-python}" - <<'PY' "${RUN_DIR}/run_metadata.json" "${EXIT_CODE}" "${REFINE_DIR}"
from __future__ import annotations
import json
import sys
from pathlib import Path
exit_code = int(sys.argv[2])
refine_dir = Path(sys.argv[3])
refine_logs = sorted(str(path) for path in refine_dir.rglob("*refine_log*.json*") if path.is_file())
status = "COMPLETED_REAL_GPU_ROLLOUT_SMOKE" if exit_code == 0 and refine_logs else "NOT_RUN"
if exit_code != 0 and not refine_logs:
    status = "BLOCKED_DEPENDENCY"
payload = {
    "status": status,
    "exit_code": exit_code,
    "refine_log_count": len(refine_logs),
    "refine_logs": refine_logs,
}
with open(sys.argv[1], "w", encoding="utf-8") as file:
    json.dump(payload, file, indent=2, ensure_ascii=False)
    file.write("\n")
PY

if [[ ${EXIT_CODE} -ne 0 ]]; then
  exit "${EXIT_CODE}"
fi
