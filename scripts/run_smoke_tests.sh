#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python -B -m user_simulator.evaluation.drift_memory_eval
python -B -m user_simulator.evaluation.critique_scope_eval
python -B -m user_simulator.evaluation.run_memory_baselines \
  --modes none flat structured time_decay critiquescope \
  --scenario-set deterministic \
  --seeds 0 \
  --output-dir outputs/memory_baselines_smoke
pytest -q tests/test_critique_scope.py
