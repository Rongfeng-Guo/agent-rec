# Research Status

## Current Branch

`codex/driftaware-structured-memory`

## Current HEAD

Run `git rev-parse HEAD` after the final status commit for the exact commit.

## COMPLETED

- `StructuredMemory` for positive, negative, hard, and soft preference slots.
- `CritiqueScopeMemory` with fast/slow memory, temporal scope, horizon decay,
  promotion conditions, session expiry, and behavioral rollback.
- Deterministic and noisy memory-level critique benchmarks.
- Critique parser with deterministic and optional OpenAI-compatible backends.
- Critique rollout adapter and counterfactual uplift preference-pair export.
- Unified memory baseline runner with CSV/JSON/JSONL/metadata output.
- Result aggregation and LaTeX export for memory-level diagnostics.
- `CritiqueWorld` controlled closed-loop world model:
  - latent user state
  - transparent utility decomposition
  - fatigue, context, drift, patience, and leave behavior
  - deterministic user response with explicit seeds
- Memory-aware reranking policy for:
  - `none`
  - `flat`
  - `structured`
  - `time_decay`
  - `critiquescope`
- Closed-loop scenario factories:
  - `temporary_fatigue`
  - `stable_dislike`
  - `diversity_request`
  - `session_context`
  - `genuine_drift`
  - `behavioral_rollback`
  - `mixed_multi_turn`
- Closed-loop benchmark runner with:
  - `trajectories.jsonl`
  - `branch_rollouts.jsonl`
  - `dpo_pairs.jsonl`
  - `summary.csv`
  - `summary.json`
  - `method_summary.csv`
  - `method_scenario_summary.csv`
  - `run_metadata.json`
  - `tables.tex`
  - per-run `README.md`
- Counterfactual branch rollout from the same critique snapshot:
  - `follow`
  - `ignore`
  - `over_apply`
- Oracle and deterministic parser modes for closed-loop evaluation.
- Minimal error attribution:
  - `parser_scope_error`
  - `memory_update_error`
  - `policy_application_error`
  - `candidate_coverage_error`
- Pytest regression coverage for CritiqueScope and CritiqueWorld.
- Documentation:
  - `docs/driftaware_gimo.md`
  - `docs/critiquescope_gimo.md`
  - `docs/critique_world.md`
  - `docs/experiment_protocol.md`
  - `docs/baseline_matrix.md`

## SMOKE_TEST_ONLY

These runs are controlled diagnostics, not full GIMO training, not human
evaluation, and not complete causal inference.

| Run | Status | Rows / Pairs | Output |
| --- | --- | ---: | --- |
| Memory baseline deterministic | SMOKE_TEST_ONLY | 150 rows | `outputs/memory_baselines` |
| Memory baseline noisy | SMOKE_TEST_ONLY | 75 rows | `outputs/memory_baselines_noisy` |
| CritiqueWorld oracle | SMOKE_TEST_ONLY | 175 summary rows; 1740 trajectory rows; 2850 branch rows; 355 DPO/CDPO pairs | `outputs/closed_loop_oracle` |
| CritiqueWorld deterministic parser | SMOKE_TEST_ONLY | 105 summary rows; 1044 trajectory rows; 1710 branch rows; 249 DPO/CDPO pairs | `outputs/closed_loop_deterministic` |

## Actual Closed-Loop Result Snapshot

Oracle parser method-level means:

| Method | N | CumulativeUtility | ClickRate | InstructionUplift@H | OverCorrectionRegret@H | ScopeAccuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| critiquescope | 35 | 11.003 | 0.710 | -0.076 | 0.099 | 1.000 |
| flat | 35 | 10.850 | 0.711 | -0.164 | 0.000 | 1.000 |
| none | 35 | 11.088 | 0.710 | 0.000 | 0.000 | 1.000 |
| structured | 35 | 11.020 | 0.703 | 0.009 | 0.125 | 1.000 |
| time_decay | 35 | 11.075 | 0.719 | 0.009 | 0.000 | 1.000 |

Deterministic parser method-level means:

| Method | N | CumulativeUtility | ClickRate | InstructionUplift@H | OverCorrectionRegret@H | ParserScopeError |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| critiquescope | 21 | 11.087 | 0.707 | -0.004 | 0.061 | 0.117 |
| flat | 21 | 10.980 | 0.723 | -0.064 | 0.000 | 0.119 |
| none | 21 | 11.087 | 0.707 | 0.000 | 0.000 | 0.117 |
| structured | 21 | 11.072 | 0.715 | 0.029 | 0.094 | 0.122 |
| time_decay | 21 | 11.095 | 0.719 | 0.022 | 0.000 | 0.116 |

Interpretation: these numbers are regression-test diagnostics for a controlled
latent-state environment. They should be used to identify failure modes and
branch-level preference-pair quality, not to claim real-world recommendation
effectiveness.

## NOT_RUN

- Full prompt-based IRA evaluation on original GIMO task files.
- Full SFT training.
- Full GPE/HAP pipeline.
- Full CDPO training.
- Real AILO simulator rollout connected to CritiqueWorld branch schema.
- Human evaluation.
- Full Recall/NDCG table for trained policies.

## BLOCKED_NO_GPU

- SFT and CDPO training require GPU resources and model weights.
- LLaMA-Factory training tests require the full training dependency stack.

## BLOCKED_NO_API_KEY

- OpenAI-compatible critique parser mode is intentionally not run in this
  branch because no API endpoint/key is configured.
- GPE/HAP scripts require an OpenAI-compatible endpoint and rollout inputs.

## BLOCKED_MISSING_DATA

- Original prompt-based IRA path references scripts/configuration that are not
  fully present in this checkout.
- Prebuilt embedding index, configured dataset paths, and model/data artifacts
  are needed for full GIMO evaluation and training.

## Commands Run In This Round

| Command | Status | Notes |
| --- | --- | --- |
| `git status --short` | PASS | Initial worktree was clean. |
| `git branch --show-current` | PASS | Branch: `codex/driftaware-structured-memory`. |
| `git log --oneline --decorate -n 15` | PASS | Verified current history. |
| `pytest -q tests/test_critique_world.py` | PASS | 12 CritiqueWorld tests passed. |
| `pytest -q` | PASS | 27 tests passed. |
| `python -B -m user_simulator.evaluation.run_closed_loop_benchmark --modes none flat structured time_decay critiquescope --scenarios all --seeds 0 1 2 3 4 --max-turns 12 --top-k 5 --parser-mode oracle --output-dir outputs\closed_loop_oracle` | PASS | 175 summary rows, 1740 trajectory rows, 2850 branch rows, 355 pairs. |
| `python -B -m user_simulator.evaluation.run_closed_loop_benchmark --modes none flat structured time_decay critiquescope --scenarios all --seeds 0 1 2 --max-turns 12 --top-k 5 --parser-mode deterministic --output-dir outputs\closed_loop_deterministic` | PASS | 105 summary rows, 1044 trajectory rows, 1710 branch rows, 249 pairs. |
| `python -m compileall user_simulator` | PASS | Bytecode side effects cleaned from the worktree. |
| `git diff --check` | PASS | Only Windows CRLF conversion warnings were reported. |

## Next Priorities

1. Connect real GIMO rollout logs to the CritiqueWorld branch schema.
2. Convert exported `dpo_pairs.jsonl` into the exact LLaMA-Factory/GIMO CDPO
   data format.
3. Add scenario coverage for noisy closed-loop ambiguity.
4. Run `openai_compatible` parser mode once API configuration is available.
5. Run full SFT/GPE/HAP/CDPO after GPU, model, and data paths are configured.

## Recommended Push Commands

Do not push automatically from this run. When ready:

```bash
git push -u origin codex/driftaware-structured-memory
git push -u agent-rec codex/driftaware-structured-memory
```
