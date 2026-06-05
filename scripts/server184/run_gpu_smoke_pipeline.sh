#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PIPELINE_ROOT="${ROOT_DIR}/outputs/server184_gpu_pipeline/${TIMESTAMP}"
REPORT_PATH="${PIPELINE_ROOT}/report.md"
mkdir -p "${PIPELINE_ROOT}"

cleanup() {
  bash "${ROOT_DIR}/scripts/server184/stop_vllm.sh" || true
}
trap cleanup EXIT

run_step() {
  local name="$1"
  shift
  echo "## ${name}" >> "${REPORT_PATH}"
  echo '```text' >> "${REPORT_PATH}"
  if "$@" >> "${REPORT_PATH}" 2>&1; then
    echo "PASS" >> "${REPORT_PATH}"
  else
    echo "FAIL" >> "${REPORT_PATH}"
    echo '```' >> "${REPORT_PATH}"
    return 1
  fi
  echo '```' >> "${REPORT_PATH}"
}

run_step check_env bash "${ROOT_DIR}/scripts/server184/check_env.sh"
run_step discover_resources bash "${ROOT_DIR}/scripts/server184/discover_resources.sh"
run_step serve_vllm_single bash "${ROOT_DIR}/scripts/server184/serve_vllm_single.sh"
run_step run_real_gimo_rollout_smoke bash "${ROOT_DIR}/scripts/server184/run_real_gimo_rollout_smoke.sh"

ROLLOUT_ROOT="${GPE_HAP_OUTPUT_DIR:-${ROOT_DIR}/outputs/server184_real_rollout_smoke}"
if [[ ! -d "${ROLLOUT_ROOT}" ]]; then
  echo "BLOCKED_REAL_LOG_MISSING" >> "${REPORT_PATH}"
  exit 1
fi

LATEST_ROLLOUT_DIR="$(find "${ROLLOUT_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
if [[ -z "${LATEST_ROLLOUT_DIR}" ]]; then
  echo "BLOCKED_REAL_LOG_MISSING" >> "${REPORT_PATH}"
  exit 1
fi

run_step run_real_rollout_bridge bash "${ROOT_DIR}/scripts/server184/run_real_rollout_bridge.sh" \
  "${LATEST_ROLLOUT_DIR}/refine_logs" \
  "${BRIDGE_OUTPUT_DIR:-${ROOT_DIR}/outputs/server184_real_rollout_bridge}/${TIMESTAMP}"
