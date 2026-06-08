# Progress Report: H5 Candidate-Level Source Reranker

Date: 2026-06-08
Repo: `/home/grf/agent-rec`

## What Changed

- Added `scripts/oracle_route_memory/export_candidate_level_source_features.py`
  to export one row per candidate from the enlarged k500 pool.
- Added `scripts/oracle_route_memory/train_candidate_level_source_ranker.py` for
  a first pointwise candidate-level ranker.
- Added hard-negative sampling by source-local rank.
- Added pairwise linear RankNet-style training over positive/negative candidate
  pairs.
- Added a domain-routed output combiner for analyzer-compatible ranker outputs.
- Locked the validation-selected H5-D policy in
  `experiments/h5-candidate-level-source-reranker/locked_policy_manifest.json`.
- Added a single validation comparison table in
  `experiments/h5-candidate-level-source-reranker/validation_comparison.md`.
- Added `scripts/oracle_route_memory/validate_locked_policy_manifest.py` to
  recompute locked manifest metrics from selected output rows.
- Added focused tests for exporter feature rows, ranker feature vectors,
  grouping, sampling, and analyzer-compatible outputs.

## Exporter Output

Output:
`outputs/oracle_route_memory/h5_candidate_level_source_features_k500_20260608`

| split | samples | candidate rows | positives | pool hit | avg pool size | oracle src Hit@50 | avg oracle src rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 844 | 1888938 | 786 | 0.9313 | 2238.08 | 0.3389 | 220.27 |
| cold_like_val | 136 | 345969 | 125 | 0.9191 | 2543.89 | 0.0956 | 524.19 |

## Pointwise Ranker Results

| run | output | Recall@50 | avg rank-miss rank |
|---|---|---:|---:|
| random negatives | `h5_candidate_level_source_ranker_sgd_k500_n500_20260608` | 0.022059 | 1219.0738 |
| hard-negative mix | `h5_candidate_level_source_ranker_sgd_k500_n500_h100_20260608` | 0.029412 | 1292.5289 |

The hard-negative mix improves over random negatives, but neither pointwise
ranker beats the H3 selected baseline (`0.044118`).

## Pairwise Ranker Results

| run | output | Recall@50 | hits |
|---|---|---:|---:|
| pairwise h100/e20 | `h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h100_e20_20260608` | 0.044118 | 6/136 |
| pairwise h200/e30 | `h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h200_e30_20260608` | 0.066176 | 9/136 |
| pairwise h300/e30 | `h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h300_e30_20260608` | 0.051471 | 7/136 |
| domain-routed | `h5_pairwise_domain_routed_book_h100_game_h300_20260608` | 0.073529 | 10/136 |

Domain-routed policy:

- Book -> pairwise h100/e20: `3/65`, Recall@50 `0.046154`
- Game -> pairwise h300/e30: `7/71`, Recall@50 `0.098592`

Robustness checks:

- h100/e20 seed 42 and seed 7 both produce Recall@50 `0.044118`.
- h300/e30 seed 42 and seed 7 both produce Recall@50 `0.051471`.
- h200/e30 reaches Recall@50 `0.066176`, with Book `3/65` and Game `6/71`.
- locked manifest validation passes: overall `10/136`, Book `3/65`, Game `7/71`.

## Interpretation

H5 is now data-ready and has a validation-selected candidate policy that beats
the H3 validation baseline. The key change was moving from independent pointwise
classification to pairwise candidate ranking, then routing the stronger variant
by domain. The selected H5-D policy is locked for validation comparison only.

## Next Target

Keep the H5-D manifest unchanged and prepare exact commands/output paths for a
future fresh confirmation plan. Also add a concise paper-facing validation
summary that clearly separates validation selection from blind-confirmation
claims.

Claim boundary remains unchanged: H5 is validation-only and must not be treated
as a fresh blind-confirmation result.
