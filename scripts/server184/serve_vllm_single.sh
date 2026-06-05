#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/server184_gpu_smoke}"
PID_DIR="${OUTPUT_DIR}/pids"
PYTHON_BIN="$(command -v python3 || command -v python || true)"
mkdir -p "${OUTPUT_DIR}" "${PID_DIR}"

if ! command -v curl >/dev/null 2>&1; then
  echo "BLOCKED_DEPENDENCY: curl" >&2
  exit 1
fi

if [[ -z "${MODEL_NAME_OR_PATH:-}" ]]; then
  echo "BLOCKED_MODEL_MISSING" >&2
  exit 1
fi

if [[ ! -e "${MODEL_NAME_OR_PATH}" ]]; then
  echo "BLOCKED_MODEL_MISSING" >&2
  exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "BLOCKED_DEPENDENCY: python" >&2
  exit 1
fi

if "${PYTHON_BIN}" -c "import vllm" >/dev/null 2>&1; then
  VLLM_MODE="python-module"
else
  VLLM_MODE="missing"
fi

if [[ "${VLLM_MODE}" == "missing" ]] && ! command -v vllm >/dev/null 2>&1; then
  echo "BLOCKED_VLLM_NOT_INSTALLED" >&2
  exit 1
fi

PORT="${VLLM_PORT:-8000}"
LOG_PATH="${OUTPUT_DIR}/vllm_server.log"
HEALTH_PATH="${OUTPUT_DIR}/vllm_health.json"
CHAT_PATH="${OUTPUT_DIR}/vllm_chat_smoke.json"
PID_PATH="${PID_DIR}/vllm_single.pid"

if [[ "${VLLM_MODE}" == "python-module" ]]; then
  CMD=(
    "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
    --model "${MODEL_NAME_OR_PATH}"
    --host 127.0.0.1
    --port "${PORT}"
    --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN:-4096}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
  )
else
  CMD=(
    vllm serve "${MODEL_NAME_OR_PATH}"
    --host 127.0.0.1
    --port "${PORT}"
    --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN:-4096}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
  )
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" nohup "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_PATH}"

cleanup_on_failure() {
  if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}

for _ in $(seq 1 60); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" > "${HEALTH_PATH}"; then
    break
  fi
  sleep 2
done

if [[ ! -s "${HEALTH_PATH}" ]]; then
  cleanup_on_failure
  echo "BLOCKED_CUDA" >&2
  exit 1
fi

curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME_OR_PATH}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}],
    \"max_tokens\": 8,
    \"temperature\": 0
  }" > "${CHAT_PATH}" || {
  cleanup_on_failure
  echo "BLOCKED_TOKENIZER" >&2
  exit 1
}

echo "VLLM_READY"
