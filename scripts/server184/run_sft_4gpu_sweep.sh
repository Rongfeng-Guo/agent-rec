#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" RUN_SEED="${gpu}" bash "${ROOT_DIR}/scripts/server184/run_sft_50step_smoke.sh"
done
