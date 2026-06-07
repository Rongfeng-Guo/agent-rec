#!/usr/bin/env bash
set -euo pipefail
cd /home/grf/agent-rec

python3 scripts/server184/build_server184_eval_index.py

printf "\n===== server184 eval index =====\n"
cat outputs/server184_gimo/index/index.md
