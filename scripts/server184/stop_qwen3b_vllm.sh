#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PID_FILE="outputs/server184_gimo/vllm/vllm.pid"
if [ ! -f "$PID_FILE" ]; then
  echo NO_PID_FILE
  exit 0
fi
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  wait "$PID" 2>/dev/null || true
fi
rm -f "$PID_FILE"
