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
pytest -q tests/test_critique_scope.py
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
```

## Reproducibility

Every runner output includes command, timestamp, git commit, Python version,
platform, seeds, scenario set, memory modes, run mode, dataset/model labels, and
an environment summary.
