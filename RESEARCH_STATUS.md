# Research Status

## Current Branch

`codex/driftaware-structured-memory`

## Current HEAD

`5d2dd1c7b4c91c57ee73f5dd8fbc99fc09d3c3d4`

## Completed Commits

- `c1b59a8` Add drift-aware structured memory benchmark
- `bfd6be6` Add scope-aware critique memory
- `9207797` Add critique uplift preference pair builder
- `2d0bf65` infra: add reproducible memory baseline runner
- `5d2dd1c` docs: record research status and smoke results

## Completed Modules

- `StructuredMemory` for positive, negative, hard, and soft preference slots.
- `CritiqueScopeMemory` with fast/slow memory, temporal scope, horizon decay,
  promotion conditions, session expiry, and behavioral rollback.
- Compatibility import path: `user_simulator/state/critique_scope_memory.py`.
- Deterministic CritiqueScope benchmark covering:
  - Temporary Fatigue
  - Stable Dislike
  - Diversity Request
  - Session Context
  - Genuine Drift
  - Behavioral Rollback
- Counterfactual uplift preference-pair builder.
- Unified baseline runner for `none`, `flat`, `structured`, `time_decay`, and
  `critiquescope`.
- CSV/JSON/JSONL/metadata result output.
- Pytest regression suite for CritiqueScope.
- CPU/API-free config and shell scripts.
- Research docs:
  - `docs/critiquescope_gimo.md`
  - `docs/driftaware_gimo.md`
  - `docs/experiment_protocol.md`
  - `docs/baseline_matrix.md`

## Commands Run

| Command | Status | Notes |
| --- | --- | --- |
| `git status --short` | PASS | Initial worktree was clean. |
| `git branch --show-current` | PASS | Branch: `codex/driftaware-structured-memory`. |
| `git log --oneline --decorate -n 12` | PASS | Verified branch history. |
| `python --version` | PASS | Python 3.12.3. |
| `nvidia-smi` | PASS | RTX 4050 visible; not needed for Tier A. |
| `python -B -m user_simulator.evaluation.drift_memory_eval` | PASS | Deterministic DriftAware smoke result printed. |
| `python -B -m user_simulator.evaluation.critique_scope_eval` | PASS | Six deterministic critique scenarios evaluated. |
| `python -B -m user_simulator.evaluation.run_memory_baselines --modes none flat structured time_decay critiquescope --scenario-set deterministic --seeds 0 1 2 3 4 --output-dir outputs\memory_baselines` | PASS | 150 rows written. |
| `pytest -q` | FAIL then PASS | First failed by collecting LLaMA-Factory tests without `transformers`, `accelerate`, and `datasets`; added `pytest.ini`, then 10 tests passed. |
| `python -m compileall user_simulator` | PASS | Bytecode side effects cleaned/restored. |
| `git diff --check` | PASS | No whitespace errors. |

## Actual Results Summary

Result files:

```text
outputs/memory_baselines/
  README.md
  run_metadata.json
  runs.jsonl
  summary.csv
  summary.json
```

Run metadata:

- Git commit: `2d0bf654b4f4af607e1635b72a5cb73012d1f74a`
- Run mode: `SMOKE_TEST_ONLY`
- Dataset: `deterministic_critique_scenarios`
- Model: `none`
- Seeds: `0 1 2 3 4`
- Rows: `150`
- API key: `UNSET`

Representative deterministic findings:

- Temporary UFC fatigue:
  - `flat`: memory contamination `1.0`, over-correction regret `0.9`
  - `critiquescope`: memory contamination `0.0`, over-correction regret `0.0`
- Diversity request:
  - `flat`: memory contamination `1.0`
  - `critiquescope`: memory contamination `0.0`
- Behavioral rollback:
  - `flat`: rollback accuracy `0.0`, over-correction regret `0.9`
  - `critiquescope`: rollback accuracy `1.0`, over-correction regret `0.0`

These are controlled deterministic smoke-test values, not full GIMO or human
evaluation results.

## Not Yet Run

| Experiment | Status | Reason |
| --- | --- | --- |
| Prompt-based IRA | BLOCKED | README references `main.sh`, but this checkout does not contain it; API/config files are absent. |
| SFT | BLOCKED | Needs model weights, configured dataset paths, dependencies, and GPU training time. |
| GPE/HAP | BLOCKED | Needs OpenAI-compatible model endpoint and rollout/config inputs. |
| CDPO | BLOCKED | Needs model weights, preference data, LLaMA-Factory dependencies, and GPU. |
| Real AILO simulator rollout with CritiqueScope | PENDING | Requires deciding parser/model backend and integrating real recommender outputs. |

## Blockers

- `NO_API_KEY`: no closed-source LLM evaluation configured.
- `MISSING_DATA`: prebuilt embedding index and model/data config are not present.
- `MISSING_DEPS`: full LLaMA-Factory test/training stack lacks `transformers`,
  `accelerate`, `datasets`, and related dependencies.
- `NEEDS_REVIEW`: deterministic utility values are designed for diagnosis and
  should be reviewed before being used in paper tables.

## Next Priorities

1. Add an LLM critique parser that emits the CritiqueScope schema.
2. Add a real GIMO rollout adapter that records follow/ignore/over-apply branches.
3. Convert real rollouts into CDPO pairs with `critique_uplift_pairs.py`.
4. Add a small prompt-based IRA smoke runner once `main.sh` or equivalent entry is restored.
5. Add aggregate tables and LaTeX export for deterministic results.
6. Add noisy/ambiguous critique scenarios.
7. Add cross-simulator robustness evaluation.
8. Run full GIMO baselines after model/data/API configuration is available.

## Recommended Push Commands

Do not push automatically from this status file. When ready:

```bash
git push -u origin codex/driftaware-structured-memory
git push -u agent-rec codex/driftaware-structured-memory
```
