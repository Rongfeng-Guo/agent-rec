#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p outputs/server184_gimo/vllm
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
VLLM_MODEL_ALIAS="${VLLM_MODEL_ALIAS:-qwen2.5-3b-instruct}"
curl -s "$VLLM_BASE_URL/models" > outputs/server184_gimo/vllm/models.json
python3 - <<PY
import json
from pathlib import Path
obj = json.loads(Path('outputs/server184_gimo/vllm/models.json').read_text(encoding='utf-8'))
if not obj.get('data'):
    raise SystemExit('empty models list')
PY
curl -s "$VLLM_BASE_URL/chat/completions" -H 'Content-Type: application/json' -d '{"model": "'"$VLLM_MODEL_ALIAS"'", "messages": [{"role": "user", "content": "Reply with exactly: VLLM_OK"}], "temperature": 0}' > outputs/server184_gimo/vllm/chat_smoke.json
