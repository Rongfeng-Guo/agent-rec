# H4: Enlarged-Pool Reranking

- Status: diagnostic complete; simple sample-level gate is negative
- Type: validation-only reranking/fusion experiment
- Claim status: not a model-performance claim until locked before a fresh blind confirmation

## Question

Given H3 candidate pools with high target entry, can a validation-only reranker
move pool-hit targets from deep candidate ranks into the top-50?

## Motivation

H3 selected `residual_predicted_route_p1_top4_zscore_k500_w0p5` with
`CandidatePoolHitRate=0.801471`, but `Recall@50=0.044118`. The updated rank
diagnostics show the selected policy has average pool-hit rank miss around
`917`, with median rank around `792`. This is now a ranking problem inside a
large candidate pool, not a route-hit or pool-entry problem.

## Required Inputs

- H3 selector rows with `candidate_pool_match_rank` diagnostics:
  `outputs/oracle_route_memory/validation_fusion_selector_h3_candidate_pool_depth_rankdiag_20260608/selector_rows.json`
- H3 rank diagnostics:
  `outputs/oracle_route_memory/route_query_binding_error_analysis_h3_candidate_pool_depth_rankdiag_20260608`
- H3 selected policy:
  `residual_predicted_route_p1_top4_zscore_k500_w0p5`

## Candidate Features

Start with features that can be computed before any fresh blind labels are read:

- route confidence / route beam rank;
- query source and effective query source;
- candidate source bucket and local route rank;
- original similarity score and route-score-adjusted score;
- candidate depth bucket from the H3 `k500` pool;
- agreement across learned/residual/mean/domain-adaptive lists.

## Validation Gate

The reranker must be selected only on protocol train-derived validation splits.
After selection, rerun the H2/H3 analyzer to verify that any `Recall@50` gain is
actually a rank improvement among pool hits, not a hidden change in route or pool
coverage.

## Completed Runs

### H4-A: LateBoundFusionRouter full-pool gate

Output:
`outputs/oracle_route_memory/h4_late_bound_fusion_router_k500_fullpool_sourcediag_20260608`

Analyzer:
`outputs/oracle_route_memory/route_query_binding_error_analysis_h4_late_bound_fusion_router_k500_fullpool_sourcediag_20260608`

Cold-like validation:

| metric | value |
|---|---:|
| Recall@50 | 0.029412 |
| CandidatePoolHitRate | 0.919118 |
| ConditionalRecall@50GivenPoolHit | 0.032000 |
| Book avg pool-hit rank-miss rank | 1186.0556 |
| Game avg pool-hit rank-miss rank | 1284.1343 |

The full-pool fix changed the earlier H4 interpretation. The first H4 gate run
was accidentally building its candidate union from each source's top-50 output.
After fixing it to use every source `score_map`, pool coverage rose to `0.9191`,
but Recall@50 remained `0.0294`.

### Source Oracle Diagnostic

The sourcediag run records each source's target rank before the learned sample
gate. Even an oracle that chooses the best single source per sample has limited
top-50 headroom:

| slice | oracle source Hit@50 | avg oracle source rank |
|---|---:|---:|
| Book | 0.0769 | 462.0893 |
| Game | 0.1127 | 574.5942 |
| All | 0.0956 | p50 393 |

Per-source cold-like target ranks:

| source | present | Hit@50 | avg rank | avg learned weight |
|---|---:|---:|---:|---:|
| learned | 112/136 | 0.0368 | 991.5714 | 0.000077 |
| residual | 109/136 | 0.0368 | 904.2569 | 0.000015 |
| mean | 98/136 | 0.0147 | 812.7449 | 0.000042 |
| prefix1_head | 95/136 | 0.0294 | 880.7684 | 0.999834 |

The learned gate collapses almost entirely onto `prefix1_head`, but the deeper
diagnostic shows that switching among the existing source-level rankings is also
insufficient. The target usually sits hundreds of positions down even in the
best single source.

## Interpretation

H4 rules out a simple sample-level weighted-source gate as the next useful
reranker. Enlarging the candidate union solves most pool-entry misses, but the
ranking signal is too weak at candidate level. The next experiment should expose
per-candidate source scores, source-local ranks, route scores, source presence,
and cross-source agreement, then train a candidate-level reranker.

## Next Target

Move to H5: candidate-level source/rank reranking. The first implementation
target is an exporter that materializes one row per candidate with query-source
scores, local ranks, route confidence, presence masks, and target labels from
protocol train-derived validation only.
