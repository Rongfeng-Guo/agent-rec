#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/outputs/gpe_hap_fixture_smoke"
FIXTURE_DIR="${OUTPUT_DIR}/fixtures"
PYTHON_BIN="$(command -v python.exe || command -v python)"
to_win_path() { wslpath -w "$1"; }
FIXTURE_DIR_WIN="$(to_win_path "${FIXTURE_DIR}")"
EXPORT_DIR_WIN="$(to_win_path "${OUTPUT_DIR}/export")"
AUDIOUT_DIR_WIN="$(to_win_path "${OUTPUT_DIR}/audit")"
DATASET_DIR="${OUTPUT_DIR}/cdpo_dataset_fixture_smoke"
DATASET_DIR_WIN="$(to_win_path "${DATASET_DIR}")"

mkdir -p "${FIXTURE_DIR}" "${OUTPUT_DIR}/export" "${OUTPUT_DIR}/audit" "${DATASET_DIR}"

"${PYTHON_BIN}" - "${FIXTURE_DIR_WIN}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

fixture_dir = Path(sys.argv[1])
fixture_dir.mkdir(parents=True, exist_ok=True)

recommend = {
    "task_type": "recommend",
    "input": "Recommend a better response for the scratchpad.",
    "original_response": "Recommend[old answer]",
    "ground_truth": "Recommend[new answer]",
    "potential_reward_output": "{\"reward\": 0.2}",
    "policy_improvement_output": "{\"refinement_output\": [\"Recommend[new answer]\"]}",
    "best_refinement": "Recommend[new answer]",
    "is_original_best": False,
    "sample_num": 2,
    "combined_log": {"task_type": "recommend", "source": "fixture"},
}

ask = {
    "task_type": "ask",
    "input": "Ask a clarifying question to reduce uncertainty.",
    "original_response": "What do you like?",
    "ground_truth": "Please ask about budget and cuisine.",
    "potential_reward_output": "{\"reward\": 0.1}",
    "policy_improvement_output": "{\"refinement_output\": [\"What budget and cuisine do you prefer?\"]}",
    "best_refinement": "What budget and cuisine do you prefer?",
    "is_original_best": False,
    "sample_num": 3,
    "combined_log": {"task_type": "ask", "source": "fixture"},
}

search = {
    "task_type": "search",
    "input": "Find a better query for the current search task.",
    "original_action": "search",
    "original_response": "cheap hiking boots",
    "original_query": "cheap hiking boots",
    "original_rank": 12,
    "ground_truth": "waterproof hiking boots",
    "refinements": ["add waterproof", "add trail grip"],
    "refined_queries": ["waterproof hiking boots"],
    "refined_ranks": [{"query": "waterproof hiking boots", "rank": 3}],
    "potential_reward_output": "{\"reward\": 0.3}",
    "policy_improvement_output": "{\"refinement_output\": [\"waterproof hiking boots\"]}",
    "best_refinement": "waterproof hiking boots",
    "is_original_best": False,
    "sample_num": 4,
    "combined_log": {"task_type": "search", "source": "fixture"},
}

(fixture_dir / "recommend_refine_log_sample1.json").write_text(json.dumps([recommend], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
(fixture_dir / "ask_refine_log_sample1.jsonl").write_text(json.dumps(ask, ensure_ascii=False) + "\n", encoding="utf-8")
(fixture_dir / "search_refine_log_sample1.json").write_text(json.dumps([search], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

"${PYTHON_BIN}" -B -m user_simulator.evaluation.export_gpe_hap_refine_logs \
  --input "${FIXTURE_DIR_WIN}" \
  --output-dir "${EXPORT_DIR_WIN}" \
  --write-source-jsonl

"${PYTHON_BIN}" -B -m user_simulator.evaluation.audit_real_rollout_bridge \
  --input-dir "${EXPORT_DIR_WIN}" \
  --output-dir "${AUDIOUT_DIR_WIN}" \
  --fail-on-critical-error

"${PYTHON_BIN}" -B -m user_simulator.evaluation.materialize_cdpo_dataset \
  --input "$(to_win_path "${OUTPUT_DIR}/export/cdpo_pairs.jsonl")" \
  --output-dir "${DATASET_DIR_WIN}" \
  --seed 42 \
  --dev-ratio 0.1

bash "${ROOT_DIR}/scripts/validate_llamafactory_cdpo_dataset.sh" "${DATASET_DIR}"

"${PYTHON_BIN}" - "$(to_win_path "${OUTPUT_DIR}")" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary = {
    "export": json.loads((output_dir / "export" / "export_metadata.json").read_text(encoding="utf-8")),
    "audit": json.loads((output_dir / "audit" / "audit.json").read_text(encoding="utf-8")),
    "dataset": json.loads((output_dir / "cdpo_dataset_fixture_smoke" / "manifest.json").read_text(encoding="utf-8")),
    "llamafactory_validation": (output_dir / "cdpo_dataset_fixture_smoke" / "llamafactory_validation.md").read_text(encoding="utf-8"),
}

md = [
    "# Fixture Rollout Bridge Smoke",
    "",
    f"- Export status: `{summary['export']['status']}`",
    f"- Export traces: `{summary['export']['trace_count']}`",
    f"- Bridge audit status: `{summary['audit']['status']}`",
    f"- Dataset status: `{summary['dataset']['status']}`",
    f"- Train rows: `{summary['dataset']['splits']['train_count']}`",
    f"- Dev rows: `{summary['dataset']['splits']['dev_count']}`",
    f"- LLaMA-Factory validation: `{summary['llamafactory_validation'].splitlines()[2].split(':', 1)[-1].strip(' `') if summary['llamafactory_validation'].splitlines() else 'UNKNOWN'}`",
]
Path(output_dir / "pipeline_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
PY
