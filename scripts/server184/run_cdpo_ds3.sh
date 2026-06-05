#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/server184/cdpo_smoke.yaml"
DS_CONFIG="${ROOT_DIR}/configs/server184/deepspeed_zero3.json"
cd "${ROOT_DIR}/LLaMA-Factory"
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  llamafactory-cli train "${CONFIG_PATH}" --deepspeed "${DS_CONFIG}"
