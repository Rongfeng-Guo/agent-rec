#!/usr/bin/env bash
set -euo pipefail
cd /home/grf/agent-rec
mkdir -p outputs/server184_gimo/gpe_hap_smoke/latest_real
/home/grf/.conda/envs/gdpo/bin/python GPE_HAP/rewrite_v3.py --mode ask,recommend --domain Book --input_root outputs/server184_gimo/prompt_ira_smoke/20260606_144253 --output_dir outputs/server184_gimo/gpe_hap_smoke/latest_real --base_url http://127.0.0.1:8000/v1 --api_key EMPTY --model_name qwen2.5-3b-instruct --mini_base_url http://127.0.0.1:8000/v1 --mini_api_key EMPTY --mini_model_name qwen2.5-3b-instruct --sample-limit 1 --task-limit 1
echo COMPLETED_GPE_HAP_REAL_SMOKE outputs/server184_gimo/gpe_hap_smoke/latest_real
