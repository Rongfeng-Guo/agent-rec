# Server184 GPU Smoke

## 1. Environment Check

```bash
source .env.server184.example
bash scripts/server184/check_env.sh
```

Outputs:

- `outputs/server184_env/env_report.md`
- `outputs/server184_env/env_report.json`

Status values:

- `SERVER184_GPU_READY`
- `NOT_ON_SERVER184`
- `CUDA_UNAVAILABLE`
- `GPU_COUNT_MISMATCH`

## 2. Resource Discovery

```bash
bash scripts/server184/discover_resources.sh
```

Outputs:

- `outputs/server184_env/resource_inventory.md`
- `outputs/server184_env/resource_inventory.json`

Status values:

- `MODEL_FOUND` / `MODEL_MISSING`
- `DATA_FOUND` / `DATA_MISSING`
- `INDEX_FOUND` / `INDEX_MISSING`
- `REAL_TRACE_FOUND` / `REAL_TRACE_MISSING`

## 3. Model Path Configuration

Set:

- `MODEL_NAME_OR_PATH`
- `LLAMAFACTORY_MODEL_NAME_OR_PATH`

Do not hardcode author-machine paths inside scripts. Use exported environment
variables or a copied `.env.server184`.

## 4. Data Path Configuration

Set:

- `GIMO_DATA_ROOT`
- `GIMO_INDEX_ROOT`
- `GPE_HAP_INPUT`
- `GPE_HAP_CONFIG_PATH`

`GPE_HAP_INPUT` must contain:

- `recommend_data.json`
- `ask_data.json`
- `search_data.json`

## 5. Single-Card vLLM Startup

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/server184/serve_vllm_single.sh
```

Outputs:

- `outputs/server184_gpu_smoke/vllm_server.log`
- `outputs/server184_gpu_smoke/vllm_health.json`
- `outputs/server184_gpu_smoke/vllm_chat_smoke.json`

Blockers:

- `BLOCKED_VLLM_NOT_INSTALLED`
- `BLOCKED_MODEL_MISSING`
- `BLOCKED_CUDA`
- `BLOCKED_TOKENIZER`

## 6. Real Rollout Smoke

```bash
bash scripts/server184/run_real_gimo_rollout_smoke.sh
```

Output root:

- `outputs/server184_real_rollout_smoke/YYYYMMDD_HHMMSS/`

Artifacts:

- `command.sh`
- `env_summary.json`
- `stdout.log`
- `stderr.log`
- `refine_logs/`
- `run_metadata.json`

This step must use a real configured endpoint or local `vllm`. It must not
reuse fixture traces.

## 7. Bridge Audit

```bash
bash scripts/server184/run_real_rollout_bridge.sh \
  outputs/server184_real_rollout_smoke/<TIMESTAMP>/refine_logs \
  outputs/server184_real_rollout_bridge/<TIMESTAMP>
```

Artifacts:

- `adapter_input.jsonl`
- `branch_rollouts.jsonl`
- `dpo_pairs.jsonl`
- `cdpo_pairs.jsonl`
- `audit/audit.json`
- `audit/audit.md`
- `audit/row_errors.jsonl`
- `audit/task_type_summary.csv`
- `audit/pair_quality_summary.csv`

## 8. Train/Dev Materialization

The bridge wrapper materializes:

- `cdpo_dataset/train.json`
- `cdpo_dataset/dev.json`
- `cdpo_dataset/dataset_info.json`
- `cdpo_dataset/manifest.json`
- `cdpo_dataset/audit.json`
- `cdpo_dataset/README.md`

## 9. LLaMA-Factory Dry-Run

```bash
bash scripts/validate_llamafactory_cdpo_dataset.sh \
  outputs/server184_real_rollout_bridge/<TIMESTAMP>/cdpo_dataset
```

Expected statuses:

- `READY_FOR_GPU_SFT_SMOKE`
- `READY_FOR_GPU_CDPO_SMOKE`
- `BLOCKED_MODEL_MISSING`
- `BLOCKED_DATASET_SCHEMA`
- `BLOCKED_LLAMAFACTORY_CONFIG`

## 10. SFT 50-Step Smoke

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/server184/run_sft_50step_smoke.sh
```

Intended shape:

- 8B instruct model
- LoRA
- 50 steps
- batch size 1

## 11. CDPO 20-Step Smoke

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/server184/run_cdpo_20step_smoke.sh
```

If OOM persists, escalate to:

```bash
bash scripts/server184/run_cdpo_ds3.sh
```

## 12. Four-GPU Allocation Strategy

- GPU 0: single-card `vllm` smoke or SFT smoke
- GPU 1-3: optional parallel `vllm` replicas or seed sweep
- Four-card rollout mode: `bash scripts/server184/serve_vllm_replicas.sh`
- Four-card training escalation: `configs/server184/deepspeed_zero3.json`

## 13. Common Blockers

- `NOT_ON_SERVER184`
- `BLOCKED_MODEL_MISSING`
- `BLOCKED_DATA_MISSING`
- `BLOCKED_INDEX_MISSING`
- `BLOCKED_API_CONFIG`
- `BLOCKED_VLLM_NOT_INSTALLED`
- `BLOCKED_OOM`
- `BLOCKED_DEPENDENCY`

## 14. Stop Services

```bash
bash scripts/server184/stop_vllm.sh
```

`run_gpu_smoke_pipeline.sh` also traps failures and stops `vllm` on exit.

## 15. Real vs Fixture

- `COMPLETED_FIXTURE_SMOKE`: only the checked-in fixture bridge path.
- `COMPLETED_REAL_GPU_ROLLOUT_SMOKE`: real model endpoint plus real refine logs.
- `READY_FOR_GPU_SMOKE`: scripts and validation are ready but real run not yet executed.
- `NOT_RUN`: no real rollout claimed.
