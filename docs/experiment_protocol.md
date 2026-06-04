# Experiment Protocol

## Hypothesis

User feedback in interactive recommendation is an exposure-conditioned critique
with object scope, temporal scope, horizon, and promotion conditions. Treating
every complaint as a durable preference causes memory contamination and
over-correction.

## Data

Tier A uses deterministic critique scenarios in
`user_simulator/evaluation/critique_scope_eval.py`. These scenarios require no
API key, GPU, or external data.

Tier A+ uses CritiqueWorld closed-loop scenarios in
`user_simulator/scenarios/closed_loop_scenarios.py`. These scenarios still
require no API key, GPU, or external data, but they evaluate slate generation,
user response, memory update, reranking, branch rollouts, and long-horizon
utility.

Tier B/C should reuse GIMO AILO task files under `user_simulator/task/` and the
original LLaMA-Factory scripts once model/data configuration is available.

## Baselines

- `none`: no memory update.
- `flat`: every critique is promoted to slow memory.
- `structured`: positive/negative/hard/soft structured memory without critique scope.
- `time_decay`: all critiques receive a uniform temporary horizon.
- `critiquescope`: fast/slow memory with semantic scope, horizon, promotion, and rollback.

## Metrics

- Immediate Instruction Satisfaction.
- Memory Contamination Rate.
- Over-Correction Rate.
- Over-Correction Regret.
- Promotion Precision.
- Promotion Recall.
- Rollback Accuracy.
- Drift Recovery Turns.
- Expired Constraint Violation Rate.
- Instruction Uplift.
- Over-Application Regret.
- Token Cost.
- CumulativeUtility.
- AverageSlateUtility.
- ClickRate.
- LeaveRate.
- AverageSessionLength.
- SlateDiversity.
- CategoryCoverage.
- InstructionUplift@1 and InstructionUplift@H.
- OverCorrectionRegret@1 and OverCorrectionRegret@H.
- ScopeClassificationAccuracy.
- Parser, memory-update, policy-application, and candidate-coverage error
  attribution.

`instruction_uplift` and `over_application_regret` are controlled
counterfactual rollout proxies, not full causal estimates.

## Random Seeds

The deterministic benchmark supports multiple seeds for runner consistency.
Current scenarios are deterministic, so metric means are expected to be stable
across seeds.

## Smoke Test

```bash
python -B -m user_simulator.evaluation.drift_memory_eval
python -B -m user_simulator.evaluation.critique_scope_eval
python -B -m user_simulator.evaluation.validate_critique_scenarios \
  --scenario-set deterministic \
  --output outputs/scenario_validation/deterministic.json
python -B -m user_simulator.evaluation.validate_critique_scenarios \
  --scenario-set noisy \
  --output outputs/scenario_validation/noisy.json
python -B -m user_simulator.evaluation.critique_parser \
  --backend deterministic \
  --output outputs/parser_smoke/parsed.jsonl
python -B -m user_simulator.evaluation.critique_rollout_adapter \
  --output-dir outputs/rollout_adapter_smoke
python -B -m user_simulator.evaluation.run_memory_baselines \
  --modes none flat structured time_decay critiquescope \
  --scenario-set deterministic \
  --seeds 0 1 2 3 4 \
  --output-dir outputs/memory_baselines
python -B -m user_simulator.evaluation.summarize_memory_baselines \
  --input outputs/memory_baselines/summary.csv \
  --output-dir outputs/memory_baselines/aggregate
python -B -m user_simulator.evaluation.run_memory_baselines \
  --modes none flat structured time_decay critiquescope \
  --scenario-set noisy \
  --seeds 0 1 2 \
  --output-dir outputs/memory_baselines_noisy
python -B -m user_simulator.evaluation.summarize_memory_baselines \
  --input outputs/memory_baselines_noisy/summary.csv \
  --output-dir outputs/memory_baselines_noisy/aggregate
pytest -q tests/test_critique_scope.py
pytest -q tests/test_critique_world.py
python -B -m user_simulator.evaluation.run_closed_loop_pipeline \
  --modes none flat structured time_decay critiquescope \
  --scenarios all \
  --seeds 0 1 2 3 4 \
  --max-turns 12 \
  --top-k 5 \
  --parser-mode oracle \
  --output-dir outputs/closed_loop_oracle
python -B -m user_simulator.evaluation.run_closed_loop_pipeline \
  --modes none flat structured time_decay critiquescope \
  --scenarios all \
  --seeds 0 1 2 \
  --max-turns 12 \
  --top-k 5 \
  --parser-mode deterministic \
  --output-dir outputs/closed_loop_deterministic
```

## Full Run

Full GIMO runs are pending model weights, API/model endpoint configuration, and
training data paths:

```bash
cd LLaMA-Factory
bash gimo/{dataset}/sft/sft.sh
bash gimo/{dataset}/gimo/adpo_v1_sample1.sh
```

Optional parser backend for an OpenAI-compatible endpoint:

```bash
python -B -m user_simulator.evaluation.critique_parser \
  --backend openai \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model "$OPENAI_MODEL" \
  --input utterances.txt \
  --output outputs/parser_runs/parsed.jsonl
```

Real rollout adapter input should be JSONL with:

```json
{
  "id": "scenario_id",
  "critique_type": "Temporary Fatigue",
  "utterance": "...",
  "critiques": [],
  "follow_value": [0.7, 0.8],
  "ignore_value": [0.2, 0.3],
  "over_apply_value": [0.6, 0.1],
  "post_expiry_items": []
}
```

Then run:

```bash
python -B -m user_simulator.evaluation.critique_rollout_adapter \
  --input real_rollouts.jsonl \
  --output-dir outputs/real_rollout_adapter
```

## Result Paths

The unified runner writes:

```text
outputs/memory_baselines/
  runs.jsonl
  summary.csv
  summary.json
  run_metadata.json
  README.md
  aggregate/
    method_summary.csv
    method_summary.json
    method_scenario_summary.csv
    method_scenario_summary.json
    method_summary.tex
outputs/memory_baselines_noisy/
  runs.jsonl
  summary.csv
  summary.json
  run_metadata.json
  README.md
  aggregate/
outputs/parser_smoke/
  parsed.jsonl
outputs/rollout_adapter_smoke/
  adapter_metadata.json
  critique_pairs.jsonl
  normalized_scenarios.jsonl
outputs/scenario_validation/
  deterministic.json
  noisy.json
outputs/closed_loop_oracle/
  trajectories.jsonl
  branch_rollouts.jsonl
  dpo_pairs.jsonl
  cdpo_pairs.jsonl
  cdpo_validation.json
  cdpo_dataset_manifest.json
  llamafactory_dataset_info_snippet.json
  cdpo_train.jsonl
  cdpo_dev.jsonl
  closed_loop_report.md
  summary.csv
  summary.json
  method_summary.csv
  method_scenario_summary.csv
  run_metadata.json
  pipeline_metadata.json
  tables.tex
outputs/closed_loop_deterministic/
  same schema as closed_loop_oracle
```

## Reproducibility

Every runner output includes command, timestamp, git commit, Python version,
platform, seeds, scenario set, memory modes, run mode, dataset/model labels, and
an environment summary.
