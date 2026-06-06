#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/grf/migrate_runtime_bundle_20260512/root/models/Qwen2.5-3B-Instruct}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/grf/.conda/envs/gdpo/bin/python}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
VLLM_MODEL_ALIAS="${VLLM_MODEL_ALIAS:-qwen2.5-3b-instruct}"
SMOKE_SAMPLE_LIMIT="${SMOKE_SAMPLE_LIMIT:-1}"
mkdir -p outputs/server184_gimo/env
python3 - <<PY > outputs/server184_gimo/env/env_report.json
import json, os, subprocess
report = {
  'repo_root': os.getcwd(),
  'gpu0_query': subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True),
  'model_exists': os.path.exists('${MODEL_NAME_OR_PATH}'),
  'model_path': '${MODEL_NAME_OR_PATH}',
  'vllm_python_exists': os.path.exists('${VLLM_PYTHON}'),
  'vllm_base_url': '${VLLM_BASE_URL}',
  'vllm_model_alias': '${VLLM_MODEL_ALIAS}',
  'smoke_sample_limit': '${SMOKE_SAMPLE_LIMIT}',
}
print(json.dumps(report, indent=2, ensure_ascii=False))
PY
cat > outputs/server184_gimo/env/env_report.md <<MD
# Server184 Environment Report

- repo_root: $ROOT
- model_path: $MODEL_NAME_OR_PATH
- model_exists: $(test -d "$MODEL_NAME_OR_PATH" && echo true || echo false)
- vllm_python: $VLLM_PYTHON
- vllm_base_url: $VLLM_BASE_URL
- vllm_model_alias: $VLLM_MODEL_ALIAS
- smoke_sample_limit: $SMOKE_SAMPLE_LIMIT
MD
