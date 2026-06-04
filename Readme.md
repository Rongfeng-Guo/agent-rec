# GIMO-MemoryLab

![Repository overview](pic/image.png)

This repository is our maintained research and engineering branch built on top
of the original GIMO codebase. The focus here is not to mirror the public
paper release page, but to extend the system with reproducible memory-aware
recommendation experiments, controlled critique handling, and closed-loop
evaluation artifacts that we can continue to evolve.

## What This Repository Contains

The current codebase combines the original multi-turn recommendation pipeline
with our own extensions for memory behavior analysis:

- `DriftAware-GIMO`: structured memory for positive, negative, hard, and soft
  preference tracking under interest drift.
- `CritiqueScope-GIMO`: fast/slow critique memory that distinguishes temporary
  feedback from durable user constraints.
- `CritiqueWorld`: a CPU-only, API-free closed-loop testbed for checking
  whether critique memory actually changes future recommendation slates.
- CDPO bridge tooling: controlled preference-pair export, validation, manifest
  generation, train/dev split materialization, and readable audit reports.

If you want the current state of the project at a glance, start with
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

The AILO simulator path still expects the embedding index assets referenced by
the original project structure.

1. Download the [index file](https://drive.google.com/file/d/1P6QkUrikHnwxNov0fUY3SxWQkl1qve0O/view?usp=drive_link).
2. Unzip the downloaded file into `user_simulator/embedding/`.

Additional simulator notes live in [`user_simulator/readme.md`](user_simulator/readme.md).

### API configuration

Any path in this repository that calls an LLM uses an OpenAI-compatible API
interface.

1. Put your endpoint and key in `config/api_config.json`.
2. Closed-source models can be configured directly through that file.
3. Open-source models can be exposed through a local OpenAI-compatible server
   such as `vllm`.

The new CritiqueWorld evaluation path does not require an API key.

## Main Workstreams

### 1. Baseline GIMO training path

The original training stack is still present for users who want to continue the
SFT, GPE, HAP, and CDPO workflow after configuring datasets, model weights, and
GPU resources.

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
preferences, hard constraints, and soft preferences so we can inspect how
memory behaves under preference drift.

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

`CritiqueScopeMemory` treats natural-language feedback as scope-aware memory
updates instead of promoting every complaint into a durable preference.

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

See [`docs/critiquescope_gimo.md`](docs/critiquescope_gimo.md) for the full
schema and protocol.

### 4. CritiqueWorld closed-loop evaluation

CritiqueWorld is our main addition for testing whether memory interventions
change actual recommendation trajectories rather than just memory state.

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

Important framing:
the branch-level uplift and regret numbers are controlled counterfactual rollout
proxies, not human-evaluation results and not full causal claims.

More detail lives in [`docs/critique_world.md`](docs/critique_world.md) and
[`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Repository Outputs

The main generated artifacts currently tracked in this repository include:

- `outputs/memory_baselines`
- `outputs/memory_baselines_noisy`
- `outputs/closed_loop_oracle`
- `outputs/closed_loop_deterministic`

Those folders contain JSONL trajectories, summary tables, validation files,
dataset manifests, train/dev split files, and Markdown audit reports for the
current controlled experiments.

## Current Position

What is already working:

- controlled memory-level and closed-loop evaluation without calling paid APIs
- CDPO bridge export with validation and dataset manifests
- materialized train/dev split generation
- deterministic regression tests for CritiqueScope and CritiqueWorld

What still depends on external setup:

- full SFT and CDPO training
- real AILO rollout evaluation
- OpenAI-compatible parser mode
- GPU-backed model experiments

## Acknowledgment

This repository started from the GIMO project structure and continues to reuse
parts of the surrounding training stack. The current branch, however, is aimed
at our own ongoing memory and evaluation improvements rather than serving as a
verbatim mirror of the original public release.

Relevant upstream components:

- GIMO training and simulator structure
- [ECPO](https://github.com/XueyangFeng/ECPO) for related evaluation ideas
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the training
  framework used by the original stack
