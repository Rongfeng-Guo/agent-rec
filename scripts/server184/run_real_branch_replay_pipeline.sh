#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="/home/grf/.conda/envs/gdpo/bin/python"
PROMPT_IRA_DIR="${ROOT_DIR}/outputs/server184_gimo/prompt_ira_smoke/20260606_144253"
GPE_LOG_DIR="${ROOT_DIR}/outputs/server184_gimo/gpe_hap_smoke/latest_real"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/outputs/server184_gimo/real_branch_replay/${TS}"
SNAPSHOT_DIR="${OUT_DIR}/snapshots"
REPLAY_DIR="${OUT_DIR}/replay"
ADAPTER_DIR="${OUT_DIR}/adapter"
AUDIT_DIR="${OUT_DIR}/audit"
VALIDATION_FILE="${OUT_DIR}/cdpo_validation.json"
MANIFEST_FILE="${OUT_DIR}/cdpo_dataset_manifest.json"
DATASET_INFO_FILE="${OUT_DIR}/dataset_info.json"
TRAIN_FILE="${OUT_DIR}/train.jsonl"
DEV_FILE="${OUT_DIR}/dev.jsonl"

mkdir -p "${OUT_DIR}" "${SNAPSHOT_DIR}" "${REPLAY_DIR}" "${ADAPTER_DIR}" "${AUDIT_DIR}"

"${PYTHON_BIN}" -B -m user_simulator.evaluation.snapshot_prompt_ira_rollouts \
  --prompt-ira-dir "${PROMPT_IRA_DIR}" \
  --output-dir "${SNAPSHOT_DIR}" \
  --gpe-log-dir "${GPE_LOG_DIR}" \
  --max-episodes 1 \
  --max-snapshots 2

"${PYTHON_BIN}" -B -m user_simulator.evaluation.run_real_branch_replay \
  --snapshots "${SNAPSHOT_DIR}/replay_snapshots.jsonl" \
  --output-dir "${REPLAY_DIR}" \
  --utility-config "${ROOT_DIR}/configs/server184/real_branch_utility.yaml" \
  --horizon 3 \
  --max-snapshots 2

cp "${SNAPSHOT_DIR}/replay_snapshots.jsonl" "${REPLAY_DIR}/replay_snapshots.jsonl"
cp "${SNAPSHOT_DIR}/missing_fields.jsonl" "${REPLAY_DIR}/missing_fields.jsonl"
cp "${SNAPSHOT_DIR}/snapshot_audit.json" "${REPLAY_DIR}/snapshot_audit.json"
cp "${SNAPSHOT_DIR}/snapshot_audit.md" "${REPLAY_DIR}/snapshot_audit.md"
cp "${SNAPSHOT_DIR}/run_metadata.json" "${REPLAY_DIR}/snapshot_run_metadata.json"

"${PYTHON_BIN}" -B -m user_simulator.evaluation.critique_rollout_adapter \
  --input "${REPLAY_DIR}/branch_rollouts.jsonl" \
  --output-dir "${ADAPTER_DIR}"

"${PYTHON_BIN}" -B -m user_simulator.evaluation.validate_cdpo_pairs \
  --input "${ADAPTER_DIR}/cdpo_pairs.jsonl" \
  --output "${VALIDATION_FILE}"

"${PYTHON_BIN}" -B -m user_simulator.evaluation.build_cdpo_dataset_manifest \
  --input "${ADAPTER_DIR}/cdpo_pairs.jsonl" \
  --validation "${VALIDATION_FILE}" \
  --manifest-output "${MANIFEST_FILE}" \
  --dataset-info-output "${DATASET_INFO_FILE}" \
  --train-output "${TRAIN_FILE}" \
  --dev-output "${DEV_FILE}" \
  --dev-fraction 0.5

"${PYTHON_BIN}" -B -m user_simulator.evaluation.audit_real_branch_replay \
  --input-dir "${REPLAY_DIR}" \
  --output-dir "${AUDIT_DIR}" \
  --fail-on-critical-error

echo "Real branch replay pipeline complete: ${OUT_DIR}"
