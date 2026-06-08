# Research Status

## Current Server184 Status: H5-D Oracle Route Memory

Date: 2026-06-08

The active server-side research state is now the oracle route memory /
route-query-binding line, not the older DriftAware-GIMO branch snapshot below.
Detailed state is tracked in:

- `research-state.yaml`
- `research-log.md`
- `findings.md`
- `experiments/h5-candidate-level-source-reranker/README.md`
- `to_human/h5_fresh_confirmation_handoff_summary_20260608.md`
- `to_human/h5_github_update_candidate_20260608.md`

Current locked validation-selected H5-D policy:

- policy: `h5_pairwise_domain_routed_book_h100_game_h300`
- locked validation Recall@50: `0.07352941176470588` (`10/136`)
- Book: `3/65`
- Game: `7/71`
- claim boundary: validation-only, not a fresh blind-confirmation result

Current handoff gates:

- prep bundle:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16`
- prep bundle audit:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16_audit`
- readiness:
  `outputs/oracle_route_memory/h5_fresh_readiness_20260608_v16`
- handoff index validation:
  `outputs/oracle_route_memory/h5_handoff_index_validation_20260608_v16`

Gate status:

- bundle audit `status=ok`
- bundle audit `source_drift=[]`
- readiness `status=ok`
- readiness `bundle_audit_source_drift_count=0`
- handoff index validation `status=ok`

Next target: keep the locked H5-D manifest and v16 handoff gates unchanged while
waiting for a clearly fresh/unconsumed split. When that split exists, register
its manifest path/SHA-256, export candidate-level features with the locked query
sources/beam/depth, score with locked h100/h300 `model.pkl` files, apply the
locked domain route, and render a fresh-confirmation report with validation and
fresh metrics kept separate.

The sections below are earlier DriftAware-GIMO branch context retained for
historical reference.

## Current Branch

`codex/driftaware-structured-memory`

## Current HEAD

`d146e734931ad115c7bba142d0b51912c27cc921` at the start of the latest deterministic-parser alignment round.

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
  - `cdpo_pairs.jsonl`
  - `cdpo_validation.json`
  - `cdpo_dataset_manifest.json`
  - `llamafactory_dataset_info_snippet.json`
  - `cdpo_train.jsonl`
  - `cdpo_dev.jsonl`
  - `closed_loop_report.md`
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
- Lightweight LLaMA-Factory/DPO bridge rows with `conversations`, `chosen`,
  `rejected`, `score_delta`, and source metadata.
- CDPO bridge validator with strict positive `score_delta`, required fields,
  branch schema checks, duplicate-id rejection, and machine-readable validation
  summaries.
- CDPO dataset manifest builder with file hash, split ids, schema notes, and a
  LLaMA-Factory dataset-info snippet.
- Materialized CDPO train/dev JSONL split files with split hashes recorded in
  the manifest.
- Closed-loop report generator that audits output consistency and emits a
  readable Markdown result report.
- One-command closed-loop pipeline runner that chains benchmark execution, CDPO
  validation, manifest/split materialization, report generation, and
  `pipeline_metadata.json`.
- Critique lifecycle alignment for next-slate semantics:
  - critique write occurs before the next rerank
  - rerank applies at least once before decay
  - fast-memory decay occurs after effective application
  - `next_slate, horizon=1` affects exactly one subsequent slate
- Differential diversify reranking with:
  - `diversity_bonus`
  - `recent_exposure_penalty`
  - `intervention_score_delta`
  - per-item `rank_before` / `rank_after`
- CritiqueWorld validity gate with:
  - scenario-level invariant registry
  - lifecycle traces
  - score delta traces
  - invariant CSV/JSONL exports
  - fail-fast critical invariant mode
- Additional post-expiry metrics:
  - `DuringHorizonUtility`
  - `PostExpiryRecoveryUtility`
  - `PostExpirySuppressionRegret`
- API-free CPU GitHub Actions smoke workflow.
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
| CritiqueWorld oracle | SMOKE_TEST_ONLY | 175 summary rows; 2095 trajectory rows; 3375 branch rows; 65 strict-positive raw pairs; 65 CDPO bridge pairs; CDPO validation PASS; manifest built; pipeline validity gate PASS | `outputs/closed_loop_oracle` |
| CritiqueWorld deterministic parser | SMOKE_TEST_ONLY | 105 summary rows; 1257 trajectory rows; 2025 branch rows; 42 strict-positive raw pairs; 42 CDPO bridge pairs; CDPO validation PASS; manifest built; pipeline validity gate PASS | `outputs/closed_loop_deterministic` |
| CritiqueWorld validity gate | PASS | 100/100 invariants passed; 0 critical failures; oracle parser | `outputs/validity_gate` |

## Actual Closed-Loop Result Snapshot

Oracle parser method-level means:

| Method | N | CumulativeUtility | ClickRate | InstructionUplift@H | OverCorrectionRegret@H | ScopeAccuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| critiquescope | 35 | 21.475 | 0.998 | -0.364 | 0.114 | 1.000 |
| flat | 35 | 14.067 | 0.890 | -0.302 | 0.000 | 1.000 |
| none | 35 | 14.683 | 0.943 | 0.000 | 0.000 | 1.000 |
| structured | 35 | 14.130 | 0.847 | -0.252 | -0.119 | 1.000 |
| time_decay | 35 | 15.402 | 0.981 | 0.288 | 0.144 | 1.000 |

Deterministic parser method-level means:

| Method | N | CumulativeUtility | ClickRate | InstructionUplift@H | OverCorrectionRegret@H | ParserScopeError |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| critiquescope | 21 | 21.474 | 0.996 | -0.073 | 0.406 | 0.000 |
| flat | 21 | 14.072 | 0.897 | -0.306 | 0.000 | 0.000 |
| none | 21 | 14.691 | 0.960 | 0.000 | 0.000 | 0.000 |
| structured | 21 | 14.133 | 0.854 | -0.248 | -0.114 | 0.000 |
| time_decay | 21 | 15.401 | 0.976 | 0.286 | 0.142 | 0.000 |

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

## KNOWN_LIMITATIONS

- The deterministic parser is still a cue-based ruleset with narrower language
  coverage than the oracle or OpenAI-compatible parser paths.
- The oracle validity gate remains the highest-fidelity reference
  configuration for fail-fast invariant auditing.

## BLOCKED_MISSING_DATA

- Original prompt-based IRA path references scripts/configuration that are not
  fully present in this checkout.
- Prebuilt embedding index, configured dataset paths, and model/data artifacts
  are needed for full GIMO evaluation and training.

## Previous Pipeline Round

| Command | Status | Notes |
| --- | --- | --- |
| `git status --short` | PASS | Historical pipeline round worktree check. |
| `git branch --show-current` | PASS | Branch: `codex/driftaware-structured-memory`. |
| `git log --oneline --decorate -n 15` | PASS | Verified current history. |
| `pytest -q tests/test_critique_world.py` | PASS | 12 CritiqueWorld tests passed. |
| `pytest -q` | PASS | 27 tests passed. |
| `python -B -m user_simulator.evaluation.run_closed_loop_benchmark --modes none flat structured time_decay critiquescope --scenarios all --seeds 0 1 2 3 4 --max-turns 12 --top-k 5 --parser-mode oracle --output-dir outputs\closed_loop_oracle` | PASS | 175 summary rows, 1740 trajectory rows, 2850 branch rows, 80 strict-positive raw pairs, 80 CDPO bridge pairs. |
| `python -B -m user_simulator.evaluation.run_closed_loop_benchmark --modes none flat structured time_decay critiquescope --scenarios all --seeds 0 1 2 --max-turns 12 --top-k 5 --parser-mode deterministic --output-dir outputs\closed_loop_deterministic` | PASS | 105 summary rows, 1044 trajectory rows, 1710 branch rows, 27 strict-positive raw pairs, 27 CDPO bridge pairs. |
| `python -B -m user_simulator.evaluation.validate_cdpo_pairs --input outputs\closed_loop_oracle\cdpo_pairs.jsonl --output outputs\closed_loop_oracle\cdpo_validation.json` | PASS | 80 rows, min score delta 0.034, mean 0.147. |
| `python -B -m user_simulator.evaluation.validate_cdpo_pairs --input outputs\closed_loop_deterministic\cdpo_pairs.jsonl --output outputs\closed_loop_deterministic\cdpo_validation.json` | PASS | 27 rows, min score delta 0.066, mean 0.191. |
| `python -B -m user_simulator.evaluation.build_cdpo_dataset_manifest --input outputs\closed_loop_oracle\cdpo_pairs.jsonl --validation outputs\closed_loop_oracle\cdpo_validation.json --manifest-output outputs\closed_loop_oracle\cdpo_dataset_manifest.json --dataset-info-output outputs\closed_loop_oracle\llamafactory_dataset_info_snippet.json --train-output outputs\closed_loop_oracle\cdpo_train.jsonl --dev-output outputs\closed_loop_oracle\cdpo_dev.jsonl --dev-fraction 0.2` | PASS | 80 rows; train/dev files 64/16 written. |
| `python -B -m user_simulator.evaluation.build_cdpo_dataset_manifest --input outputs\closed_loop_deterministic\cdpo_pairs.jsonl --validation outputs\closed_loop_deterministic\cdpo_validation.json --manifest-output outputs\closed_loop_deterministic\cdpo_dataset_manifest.json --dataset-info-output outputs\closed_loop_deterministic\llamafactory_dataset_info_snippet.json --train-output outputs\closed_loop_deterministic\cdpo_train.jsonl --dev-output outputs\closed_loop_deterministic\cdpo_dev.jsonl --dev-fraction 0.2` | PASS | 27 rows; train/dev files 22/5 written. |
| `python -B -m user_simulator.evaluation.summarize_closed_loop_outputs --output-dir outputs\closed_loop_oracle --report-output outputs\closed_loop_oracle\closed_loop_report.md` | PASS | Report audit PASS. |
| `python -B -m user_simulator.evaluation.summarize_closed_loop_outputs --output-dir outputs\closed_loop_deterministic --report-output outputs\closed_loop_deterministic\closed_loop_report.md` | PASS | Report audit PASS. |
| `python -m compileall user_simulator` | PASS | Bytecode side effects cleaned from the worktree. |
| `git diff --check` | PASS | Only Windows CRLF conversion warnings were reported. |

## Latest Audit Round

| Command | Status | Notes |
| --- | --- | --- |
| `git status --short` | PASS | Worktree inspected before implementation. |
| `git branch --show-current` | PASS | Branch: `codex/driftaware-structured-memory`. |
| `git rev-parse HEAD` | PASS | Audit-round start SHA: `76491c69390e2ae549a33429b6539eb6b6be0624`. |
| `git log --oneline --decorate -n 20` | PASS | Recent local history inspected before code changes. |
| `pytest -q tests/test_critique_world.py tests/test_critique_scope.py` | PASS | 49 targeted tests passed after lifecycle/diversify fixes. |
| `pytest -q` | PASS | 51 tests passed after validity gate and CI additions. |
| `python -B -m user_simulator.evaluation.run_validity_gate --modes none flat structured time_decay critiquescope --scenarios all --seeds 0 1 --max-turns 12 --top-k 5 --output-dir outputs\\validity_gate_smoke --fail-on-critical-invariant` | PASS | Smoke gate passed after invariant and mechanism fixes. |
| `python -B -m user_simulator.evaluation.run_validity_gate --modes none flat structured time_decay critiquescope --scenarios all --seeds 0 1 2 3 4 --max-turns 12 --top-k 5 --output-dir outputs\\validity_gate --fail-on-critical-invariant` | PASS | Full gate passed with 100/100 invariants and 0 critical failures. |
| `python -m compileall user_simulator` | PASS | Compile check passed; generated bytecode requires cleanup before commit. |
| `git diff --check` | PASS | Only CRLF warnings; no whitespace errors. |
| `pytest -q tests\\test_critique_scope.py tests\\test_critique_world.py` | PASS | 53 targeted tests passed after deterministic parser normalization updates. |
| `python -B -m user_simulator.evaluation.run_validity_gate --modes none flat structured time_decay critiquescope --scenarios all --seeds 0 1 2 --max-turns 12 --top-k 5 --parser-mode deterministic --output-dir outputs\\validity_gate_deterministic --fail-on-critical-invariant` | PASS | Deterministic gate passed with 60/60 invariants and 0 critical failures. |
| `python -B -m user_simulator.evaluation.run_closed_loop_pipeline --modes none flat structured time_decay critiquescope --scenarios all --seeds 0 1 2 --max-turns 12 --top-k 5 --parser-mode deterministic --run-validity-gate --fail-on-critical-invariant --output-dir outputs\\closed_loop_deterministic` | PASS | Deterministic pipeline now passes fail-fast validity gate; 42 CDPO bridge pairs; train/dev 34/8. |

## Next Priorities

1. Connect real GIMO rollout logs to the CritiqueWorld branch schema.
2. Harden `cdpo_pairs.jsonl` into the exact dataset schema selected for the
   final LLaMA-Factory/GIMO CDPO training recipe.
3. Add scenario coverage for noisy closed-loop ambiguity.
4. Extend validity-gate invariants to real rollout adapter inputs.
5. Run `openai_compatible` parser mode once API configuration is available.
6. Run full SFT/GPE/HAP/CDPO after GPU, model, and data paths are configured.

## Recommended Push Commands

Do not push automatically from this run. When ready:

```bash
git push -u origin codex/driftaware-structured-memory
git push -u agent-rec codex/driftaware-structured-memory
```
