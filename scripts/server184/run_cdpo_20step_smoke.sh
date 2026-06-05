#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/server184/cdpo_smoke.yaml"
if [[ -z "${LLAMAFACTORY_MODEL_NAME_OR_PATH:-}" ]]; then
  echo "BLOCKED_MODEL_MISSING" >&2
  exit 1
fi
cd "${ROOT_DIR}/LLaMA-Factory"
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" llamafactory-cli train "${CONFIG_PATH}"
