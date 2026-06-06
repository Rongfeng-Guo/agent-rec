#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p outputs/server184_gimo/vllm
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/grf/migrate_runtime_bundle_20260512/root/models/Qwen2.5-3B-Instruct}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/grf/.conda/envs/gdpo/bin/python}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MODEL_ALIAS="${VLLM_MODEL_ALIAS:-qwen2.5-3b-instruct}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
if [ -f outputs/server184_gimo/vllm/vllm.pid ] && kill -0 "$(cat outputs/server184_gimo/vllm/vllm.pid)" 2>/dev/null; then
  echo VLLM_ALREADY_RUNNING
  exit 0
fi
CUDA_VISIBLE_DEVICES=0 nohup "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME_OR_PATH" \
  --served-model-name "$VLLM_MODEL_ALIAS" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --guided-decoding-backend lm-format-enforcer \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  > outputs/server184_gimo/vllm/vllm.log 2>&1 &
echo $! > outputs/server184_gimo/vllm/vllm.pid
