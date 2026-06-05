#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/server184_gpu_smoke}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
BASE_PORT="${VLLM_PORT:-8000}"
mkdir -p "${OUTPUT_DIR}/replicas"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
for index in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$index]}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export VLLM_PORT="$((BASE_PORT + index))"
  export OUTPUT_DIR="${OUTPUT_DIR}/replicas/gpu${gpu}"
  bash "${ROOT_DIR}/scripts/server184/serve_vllm_single.sh"
done
