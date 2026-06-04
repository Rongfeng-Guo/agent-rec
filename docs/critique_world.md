# CritiqueWorld: Controlled Closed-Loop Testbed

## Why Memory-Level Benchmarks Are Not Enough

The earlier DriftAware and CritiqueScope diagnostics test whether a memory
module classifies and stores feedback correctly. That is useful, but incomplete:
a recommender can store the right memory and still fail when memory is applied
to an actual recommendation slate over multiple turns.

CritiqueWorld moves the test from memory-level diagnosis to slate-level,
trajectory-level, and long-horizon utility-level evaluation. Each turn produces
a slate, the user responds from a transparent latent state, memory is updated,
and later reranking is affected by that update.

## Positioning

CritiqueWorld is a controlled latent-state testbed. It is designed for
reproducible debugging and counterfactual comparison of memory policies. It
does not claim to simulate real human behavior, production traffic, or complete
causal effects.

The correct description for its branch metrics is:

```text
controlled counterfactual rollout proxy
```

## Latent User State

The world exposes a `LatentUserState` with:

- stable positive and negative preferences
- session-level contextual preferences
- genuine drift preferences
- item and category exposure counts
- clicked and skipped item history
- patience and active/leave state
- session id and turn index

Session reset clears contextual state while preserving stable preferences.

## Utility Function

Each item receives a transparent utility decomposition:

- `stable_match`
- `context_match`
- `drift_match`
- `negative_penalty`
- `fatigue_penalty`
- `novelty_bonus`
- `base_quality`
- `total`

The weights are configured in `CritiqueWorldConfig`. This makes failures
inspectable: a low score can be traced to fatigue, a negative constraint, missing
context, or candidate quality.

## Fatigue, Context, Drift, Patience, and Leave

Fatigue is induced by repeated category exposure. Context represents temporary
session needs, such as a family dinner. Drift represents real preference change,
such as moving from Windows to Mac. Patience decreases after low-utility slates;
when it falls below thresholds the user critiques or leaves.

## Memory-Aware Reranking

`user_simulator/policies/memory_rerank_policy.py` provides one base scorer and
five memory interventions:

- `none`
- `flat`
- `structured`
- `time_decay`
- `critiquescope`

All modes share the same underlying utility function. The modes differ only in
how memory affects reranking. This preserves the intended failure mode of flat
memory: temporary fatigue can become an over-applied persistent filter.

## Counterfactual Branching

At critique turns, the runner snapshots user state and memory state. It then
branches from the same snapshot:

- `follow`: apply the critique according to the current memory mode
- `ignore`: do not apply the critique
- `over_apply`: incorrectly upgrade temporary critiques into persistent
  constraints

Each branch continues for a fixed horizon and exports both branch trajectories
and DPO/CDPO-style preference pairs. The runner writes both the raw
counterfactual pair schema (`dpo_pairs.jsonl`) and a lightweight
LLaMA-Factory/DPO bridge schema (`cdpo_pairs.jsonl`) with `conversations`,
`chosen`, `rejected`, and `score_delta` fields. Pairs are emitted only when the
chosen `follow` branch has strictly higher rollout utility than the rejected
branch.

## Error Attribution

The runner reports a minimal error attribution surface:

- `parser_scope_error`
- `memory_update_error`
- `policy_application_error`
- `candidate_coverage_error`

`oracle` mode uses scenario ground truth. `deterministic` mode uses the local
cue-based parser. `openai_compatible` remains optional and blocked unless an API
endpoint is configured.

## Metrics

Closed-loop outputs include:

- CumulativeUtility
- AverageSlateUtility
- ClickRate
- LeaveRate
- AverageSessionLength
- SlateDiversity
- CategoryCoverage
- InstructionUplift@1 / @H
- OverCorrectionRegret@1 / @H
- ExpiredConstraintViolationRate
- DriftRecoveryTurns
- RollbackAccuracy
- MemoryContaminationRate
- PromotionPrecision / PromotionRecall
- ScopeClassificationAccuracy

## How to Run

Oracle parser:

```bash
python -B -m user_simulator.evaluation.run_closed_loop_benchmark \
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
python -B -m user_simulator.evaluation.run_closed_loop_benchmark \
  --modes none flat structured time_decay critiquescope \
  --scenarios all \
  --seeds 0 1 2 \
  --max-turns 12 \
  --top-k 5 \
  --parser-mode deterministic \
  --output-dir outputs/closed_loop_deterministic
```

## Output Files

Each run writes:

```text
trajectories.jsonl
branch_rollouts.jsonl
dpo_pairs.jsonl
cdpo_pairs.jsonl
summary.csv
summary.json
method_summary.csv
method_scenario_summary.csv
run_metadata.json
tables.tex
README.md
```

## Connecting Back to GIMO

CritiqueWorld can produce preference pairs for later DPO/CDPO-style training,
but it does not itself run SFT, GPE, HAP, or CDPO. The intended bridge is:

1. use CritiqueWorld to validate scope-aware memory behavior;
2. export `dpo_pairs.jsonl`;
3. inspect `cdpo_pairs.jsonl` as the first-pass GIMO/LLaMA-Factory data bridge;
4. run SFT/GPE/HAP/CDPO only after model weights, dataset paths, GPU resources,
   and API endpoints are configured.

## Limitations and Next Steps

CritiqueWorld is controlled and synthetic. It is useful for regression tests and
mechanistic diagnosis, not for final human-facing recommendation claims. The
next step is to feed real GIMO rollout logs into the same branch schema and
compare whether the controlled failure modes appear in actual simulator traces.
