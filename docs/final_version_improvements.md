# Final Version Improvements vs Original Repository

## Version Check

Checked branch state before adding this documentation-only summary:

- Branch: `codex/driftaware-structured-memory`
- Implementation HEAD: `d0be2f6d959f35955b1ddd93b8d71a7d3a80127c`
- Remote branch: `origin/codex/driftaware-structured-memory`
- Remote implementation HEAD:
  `d0be2f6d959f35955b1ddd93b8d71a7d3a80127c`
- Comparison base: `origin/main`

The local working tree was clean before this document was added, and the remote
feature branch matched the local implementation commit. This document is a
documentation-only follow-up summary on top of that implementation state.

## High-Level Delta

Relative to the original `origin/main` baseline, the implementation commit adds
the first complete bridge from GPE/HAP-style real refinement traces toward
CritiqueWorld branch-schema artifacts and GPU-ready training materialization.

The diff from `origin/main` to `HEAD` contains:

- `10` feature-branch commits.
- `81` changed files.
- Approximately `5253` inserted lines and `53` deleted lines.
- Removal of tracked Python bytecode cache files from the repository.

## Core Improvements

### 1. Real Rollout Trace Bridge

The original repository had the controlled CritiqueWorld and memory-evaluation
paths, but did not have a robust bridge for GPE/HAP real refine logs.

This branch extends `user_simulator/evaluation/critique_rollout_adapter.py` so
the adapter can ingest:

- JSONL trace files.
- JSON array trace files.
- Directory-level trace drops.
- Nested `log`, `trace`, `record`, and `entry` wrappers.
- `combined_log` aliases such as `original_output`, `query_text`, `sample_id`,
  and `policy_output`.
- `recommend`, `ask`, `search`, and unknown task types with generic fallback.

The bridge now preserves task type, source metadata, source references, branch
rows, DPO pairs, CDPO pairs, and LLaMA-Factory-ready pair metadata.

### 2. Export, Audit, and Dataset Materialization

This branch adds the supporting bridge tools:

- `user_simulator/evaluation/export_gpe_hap_refine_logs.py`
- `user_simulator/evaluation/audit_real_rollout_bridge.py`
- `user_simulator/evaluation/materialize_cdpo_dataset.py`
- `scripts/run_real_rollout_bridge.sh`
- `scripts/run_fixture_rollout_bridge_smoke.sh`
- `scripts/validate_llamafactory_cdpo_dataset.sh`

The new flow is:

```text
GPE/HAP refine log
-> exporter
-> adapter_input.jsonl
-> branch_rollouts.jsonl
-> dpo_pairs.jsonl
-> cdpo_pairs.jsonl
-> audit summaries
-> train/dev materialization
-> LLaMA-Factory dry-run validation
```

The audit reports source-row counts, converted rows, skipped rows, parse
errors, task-type distributions, branch-row counts, DPO/CDPO pair counts,
uplift quality, duplicate rows, missing metadata, missing critiques, missing
state snapshots, and train/dev leakage indicators.

### 3. Fixture Smoke Artifacts

The branch includes a small fixture smoke package under
`outputs/gpe_hap_fixture_smoke/`.

The fixture smoke demonstrates that:

- `recommend`, `ask`, and `search` traces are accepted.
- Each trace produces branch rows and preference pairs.
- CDPO train/dev materialization works.
- The LLaMA-Factory dry-run can validate schema shape and path hygiene.

The fixture smoke is explicitly labeled as `COMPLETED_FIXTURE_SMOKE`; it is not
claimed as a real GIMO rollout.

### 4. Server184 GPU Smoke Workflow

This branch adds a dedicated server184 workflow layer for the first real GPU
smoke on the target `4 x RTX 4090` server:

- `.env.server184.example`
- `scripts/server184/check_env.sh`
- `scripts/server184/discover_resources.sh`
- `scripts/server184/serve_vllm_single.sh`
- `scripts/server184/serve_vllm_replicas.sh`
- `scripts/server184/stop_vllm.sh`
- `scripts/server184/run_real_gimo_rollout_smoke.sh`
- `scripts/server184/run_real_rollout_bridge.sh`
- `scripts/server184/run_gpu_smoke_pipeline.sh`
- `scripts/server184/run_sft_50step_smoke.sh`
- `scripts/server184/run_sft_4gpu_sweep.sh`
- `scripts/server184/run_cdpo_20step_smoke.sh`
- `scripts/server184/run_cdpo_ds3.sh`
- `configs/server184/sft_smoke.yaml`
- `configs/server184/cdpo_smoke.yaml`
- `configs/server184/deepspeed_zero3.json`

The workflow records environment reports, resource inventories, `vllm` health,
chat smoke output, rollout stdout/stderr, generated refine logs, bridge audit
artifacts, train/dev materialization outputs, and blocker states.

The scripts intentionally report blockers such as:

- `NOT_ON_SERVER184`
- `CUDA_UNAVAILABLE`
- `GPU_COUNT_MISMATCH`
- `BLOCKED_MODEL_MISSING`
- `BLOCKED_DATA_MISSING`
- `BLOCKED_INDEX_MISSING`
- `BLOCKED_API_CONFIG`
- `BLOCKED_VLLM_NOT_INSTALLED`
- `BLOCKED_REAL_LOG_MISSING`

They do not fall back to fixture traces when real logs are missing.

### 5. GPE/HAP CLI Hardening

`GPE_HAP/rewrite_v3.py` now supports server-friendly configuration:

- `--data_root`
- `--output_root`
- `--output_dir`
- `--max_workers`
- `GPE_HAP_INPUT`
- `GPE_HAP_OUTPUT_DIR`
- OpenAI-compatible endpoint overrides through environment variables.

This removes the previous hardcoded `your/work/dir/...` dependency from the
real smoke path and lets the server184 scripts point the rollout to discovered
local data and output directories.

### 6. Documentation and Status Tracking

The branch updates the repository documentation to distinguish fixture smoke,
real rollout smoke, ready states, and blocked states.

Important docs include:

- `docs/server184_gpu_smoke.md`
- `docs/experiment_protocol.md`
- `docs/baseline_matrix.md`
- `docs/critique_world.md`
- `RESEARCH_STATUS.md`
- `Readme.md`

The docs make clear that the current local machine is not server184 and that a
real GPU rollout is still `NOT_RUN` until model, data, index, endpoint, and GPU
resources are available on the target server.

## Validation Evidence

The final local branch was validated with:

```bash
pytest -q
bash -n scripts/server184/check_env.sh scripts/server184/discover_resources.sh scripts/server184/serve_vllm_single.sh scripts/server184/serve_vllm_replicas.sh scripts/server184/stop_vllm.sh scripts/server184/run_real_gimo_rollout_smoke.sh scripts/server184/run_real_rollout_bridge.sh scripts/server184/run_gpu_smoke_pipeline.sh scripts/server184/run_sft_50step_smoke.sh scripts/server184/run_sft_4gpu_sweep.sh scripts/server184/run_cdpo_20step_smoke.sh scripts/server184/run_cdpo_ds3.sh
git diff --check
```

Observed results before this summary document was added:

- `pytest -q`: `83 passed`
- Server184 shell syntax check: passed
- `git diff --check`: no whitespace errors, only Windows LF-to-CRLF warnings
- Remote branch check: `origin/codex/driftaware-structured-memory` matched local
  commit `d0be2f6d959f35955b1ddd93b8d71a7d3a80127c`

## Remaining Work

The code and scripts are ready for the target server, but the real GPU smoke is
not complete until run on server184 with real resources.

Remaining external requirements:

- Access to server184.
- A usable 7B-8B instruct model path.
- GPE/HAP input data containing `recommend_data.json`, `ask_data.json`, and
  `search_data.json`.
- A real retrieval index, especially `faiss_index.bin`.
- OpenAI-compatible endpoint configuration or local `vllm`.
- LLaMA-Factory dependency stack for training smoke.

Once those are available, run:

```bash
source .env.server184.example
bash scripts/server184/check_env.sh
bash scripts/server184/discover_resources.sh
bash scripts/server184/run_gpu_smoke_pipeline.sh
```

The expected outcome is a timestamped real rollout smoke under
`outputs/server184_real_rollout_smoke/` and bridge materialization under
`outputs/server184_real_rollout_bridge/`.
