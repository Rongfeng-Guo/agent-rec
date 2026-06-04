# GIMO-MemoryLab

![Repository overview](pic/image.png)

GIMO-MemoryLab is a research and engineering repository for multi-turn
recommendation, memory-aware feedback modeling, and closed-loop evaluation.
The codebase provides executable pipelines for structured memory experiments,
scope-aware critique processing, controlled preference-pair construction, and
artifact-oriented evaluation outputs.

## What This Repository Contains

The current codebase includes the following components:

- `DriftAware-GIMO`: structured memory for positive, negative, hard, and soft
  preference tracking under interest drift.
- `CritiqueScope-GIMO`: fast/slow critique memory that distinguishes temporary
  feedback from durable user constraints.
- `CritiqueWorld`: a CPU-only, API-free closed-loop testbed for checking
  whether critique memory actually changes future recommendation slates.
- CDPO bridge tooling: controlled preference-pair export, validation, manifest
  generation, train/dev split materialization, and readable audit reports.

For the current implementation and experiment status, start with
[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md).

## Quick Start

Clone this repository and install the Python dependencies:

```bash
git clone https://github.com/Rongfeng-Guo/agent-rec.git
cd agent-rec
pip install -r requirements.txt
```

## Local Data and API Setup

### AILO environment assets

The AILO simulator path expects the embedding index assets used by the project
task pipeline.

1. Download the [index file](https://drive.google.com/file/d/1P6QkUrikHnwxNov0fUY3SxWQkl1qve0O/view?usp=drive_link).
2. Unzip the downloaded file into `user_simulator/embedding/`.

Additional simulator notes are available in [`user_simulator/readme.md`](user_simulator/readme.md).

### API configuration

Any path in this repository that calls an LLM uses an OpenAI-compatible API
interface.

1. Put your endpoint and key in `config/api_config.json`.
2. Closed-source models can be configured directly through that file.
3. Open-source models can be exposed through a local OpenAI-compatible server
   such as `vllm`.

The CritiqueWorld evaluation path does not require an API key.

## Main Workstreams

### 1. Baseline GIMO training path

This repository includes the SFT, GPE, HAP, and CDPO training entry points used
by the project workflow after datasets, model weights, and GPU resources are
configured.

SFT:

```bash
cd LLaMA-Factory
bash gimo/{dataset}/sft/sft.sh
```

GPE rollout:

```bash
cd GPE_HAP
python rewrite_v3.py --domain {dataset}
```

CDPO training:

```bash
cd LLaMA-Factory
bash gimo/{dataset}/gimo/adpo_v1_sample1.sh
```

### 2. DriftAware-GIMO

`StructuredMemory` adds explicit slots for positive preferences, negative
preferences, hard constraints, and soft preferences for preference-drift
analysis and memory-state inspection.

Example:

```python
env = UserAgentEnv(
    persona_path="user_simulator/task/Yelp_test.jsonl",
    user_id=0,
    item_id=0,
    config_path="config/api_config.json",
    format_path="config",
    domain="restaurant",
    model_type="openai",
    memory_mode="structured",
)
```

Run the offline benchmark:

```bash
python -m user_simulator.evaluation.drift_memory_eval
```

Protocol details are documented in [`docs/driftaware_gimo.md`](docs/driftaware_gimo.md).

### 3. CritiqueScope-GIMO

`CritiqueScopeMemory` models natural-language feedback as scope-aware memory
updates with separate handling for temporary and durable signals.

- Fast memory handles temporary fatigue, session context, and immediate
  diversity requests.
- Slow memory keeps durable constraints and preferences that are supported by
  persistent language or repeated evidence.

Example:

```python
env = UserAgentEnv(
    persona_path="user_simulator/task/Yelp_test.jsonl",
    user_id=0,
    item_id=0,
    config_path="config/api_config.json",
    format_path="config",
    domain="restaurant",
    model_type="openai",
    memory_mode="critiquescope",
)
```

Run the memory-level diagnostic benchmark:

```bash
python -B -m user_simulator.evaluation.critique_scope_eval
```

Build controlled preference pairs:

```bash
python -B -m user_simulator.evaluation.critique_uplift_pairs --output critique_pairs.jsonl
```

Normalize rollout-style critique scenarios and audit them before pair export:

```bash
python -B -m user_simulator.evaluation.critique_rollout_adapter \
  --output-dir outputs/rollout_adapter_smoke \
  --fail-on-audit-error
```

See [`docs/critiquescope_gimo.md`](docs/critiquescope_gimo.md) for the full
schema and protocol.

### 4. CritiqueWorld closed-loop evaluation

CritiqueWorld is a closed-loop evaluation environment for measuring how memory
interventions affect recommendation trajectories, branch rollouts, and
counterfactual preference-pair exports.

Recommended full pipeline, oracle parser:

```bash
python -B -m user_simulator.evaluation.run_closed_loop_pipeline \
  --modes none flat structured time_decay critiquescope \
  --scenarios all \
  --seeds 0 1 2 3 4 \
  --max-turns 12 \
  --top-k 5 \
  --parser-mode oracle \
  --output-dir outputs/closed_loop_oracle
```

Deterministic parser:

```bash
python -B -m user_simulator.evaluation.run_closed_loop_pipeline \
  --modes none flat structured time_decay critiquescope \
  --scenarios all \
  --seeds 0 1 2 \
  --max-turns 12 \
  --top-k 5 \
  --parser-mode deterministic \
  --output-dir outputs/closed_loop_deterministic
```

This pipeline runs the benchmark, validates `cdpo_pairs.jsonl`, materializes
`cdpo_train.jsonl` and `cdpo_dev.jsonl`, builds the dataset manifest, and
writes `closed_loop_report.md` plus `pipeline_metadata.json`.

Validity gate:

```bash
python -B -m user_simulator.evaluation.run_validity_gate \
  --modes none flat structured time_decay critiquescope \
  --scenarios all \
  --seeds 0 1 2 3 4 \
  --max-turns 12 \
  --top-k 5 \
  --output-dir outputs/validity_gate \
  --fail-on-critical-invariant
```

Pipeline with validity gate:

```bash
python -B -m user_simulator.evaluation.run_closed_loop_pipeline \
  --modes none flat structured time_decay critiquescope \
  --scenarios all \
  --seeds 0 1 2 3 4 \
  --max-turns 12 \
  --top-k 5 \
  --parser-mode oracle \
  --run-validity-gate \
  --fail-on-critical-invariant \
  --output-dir outputs/closed_loop_oracle
```

Interpretation:
the branch-level uplift and regret numbers are controlled counterfactual rollout
proxies intended for diagnostic evaluation.

More detail lives in [`docs/critique_world.md`](docs/critique_world.md) and
[`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Repository Outputs

The main generated artifacts currently tracked in this repository include:

- `outputs/memory_baselines`
- `outputs/memory_baselines_noisy`
- `outputs/rollout_adapter_smoke`
- `outputs/closed_loop_oracle`
- `outputs/closed_loop_deterministic`
- `outputs/validity_gate`

These folders contain JSONL trajectories, summary tables, validation files,
dataset manifests, train/dev split files, rollout-adapter audit logs, and
Markdown audit reports for the current controlled experiments.

## Current Position

Available and validated:

- controlled memory-level and closed-loop evaluation without calling paid APIs
- CDPO bridge export with validation and dataset manifests
- materialized train/dev split generation
- deterministic regression tests for CritiqueScope and CritiqueWorld

Requires external setup:

- full SFT and CDPO training
- real AILO rollout evaluation
- OpenAI-compatible parser mode
- GPU-backed model experiments

## Dependency Notes

This repository reuses parts of the GIMO project structure and related training
tooling. The following external components remain relevant to the current
implementation:

- GIMO training and simulator structure
- [ECPO](https://github.com/XueyangFeng/ECPO) for related evaluation ideas
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the training
  framework used by the original stack
