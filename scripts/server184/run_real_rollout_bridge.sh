#!/usr/bin/env bash
set -euo pipefail
cd /home/grf/agent-rec

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" scripts/server184/bridge_check.py "$@"

printf "\n===== bridge metadata =====\n"
cat outputs/server184_gimo/bridge/latest_real/bridge_metadata.json
printf "\n===== bridge report =====\n"
cat outputs/server184_gimo/bridge/latest_real/bridge_report.md
