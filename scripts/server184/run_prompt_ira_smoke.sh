#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="outputs/server184_gimo/prompt_ira_smoke/$STAMP"
mkdir -p "$OUT"
DOMAIN="${DOMAIN:-Book}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MODEL_ALIAS="${VLLM_MODEL_ALIAS:-qwen2.5-3b-instruct}"
PYTHON_BIN="${PYTHON_BIN:-/home/grf/.conda/envs/gdpo/bin/python}"
CONFIG_JSON="$OUT/runtime_api_config.json"
cat > "$CONFIG_JSON" <<JSON
{
  "vllm": {
    "base_url": "$VLLM_BASE_URL",
    "api_key": "EMPTY",
    "model_path": "$VLLM_MODEL_ALIAS"
  }
}
JSON
cat > "$OUT/command.sh" <<SH
$PYTHON_BIN scripts/server184/prompt_ira_smoke_runner.py --domain "$DOMAIN" --config-path "$CONFIG_JSON" --format-path "configs/server184" --base-url "$VLLM_BASE_URL" --model-name "$VLLM_MODEL_ALIAS" --output-dir "$OUT"
SH
"$PYTHON_BIN" scripts/server184/prompt_ira_smoke_runner.py \
  --domain "$DOMAIN" \
  --config-path "$CONFIG_JSON" \
  --format-path "configs/server184" \
  --base-url "$VLLM_BASE_URL" \
  --model-name "$VLLM_MODEL_ALIAS" \
  --output-dir "$OUT" \
  > "$OUT/stdout.log" 2> "$OUT/stderr.log"
