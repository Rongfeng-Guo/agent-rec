#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${1:-${ROOT_DIR}/outputs/memory_baselines/${STAMP}}"
LOG_FILE="${OUT_DIR}/run.log"

mkdir -p "${OUT_DIR}"
{
  echo "[run_memory_baselines] root=${ROOT_DIR}"
  echo "[run_memory_baselines] output=${OUT_DIR}"
  cd "${ROOT_DIR}"
  python -B -m user_simulator.evaluation.run_memory_baselines \
    --modes none flat structured time_decay critiquescope \
    --scenario-set deterministic \
    --seeds 0 1 2 3 4 \
    --output-dir "${OUT_DIR}"
} 2>&1 | tee "${LOG_FILE}"
