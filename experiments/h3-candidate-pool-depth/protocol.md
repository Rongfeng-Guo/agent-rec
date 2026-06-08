# H3: Candidate-Pool Depth and Query Calibration

- Status: planned
- Type: validation-only selector experiment
- Claim status: not a model-performance claim until locked before a fresh blind confirmation

## Question

Can candidate-pool hit improve inside high-route-hit prefix-1 buckets by varying
per-route candidate depth and query calibration, without reading fresh blind
labels?

## Motivation

H2 shows that cold-like high-route-hit policies are dominated by
`route_hit_pool_miss`. This means the next intervention should focus on target
entry into the candidate pool before treating rank-fusion improvements as the
main lever.

## Code Readiness

The explicit selector now keeps retrieval rows distinct by:

```text
(query_source, mode, route_score_weight, per_route_topk)
```

This matters because a depth-grid experiment is invalid if candidates with the
same query source and mode are silently merged across different `per_route_topk`
or `route_score_weight` values.

Validation coverage:

```text
PYTHONPATH=/home/grf/agent-rec /home/grf/.conda/envs/gdpo/bin/python3 -m pytest \
  tests/test_explicit_validation_fusion_selector.py \
  tests/test_route_query_binding_error_analysis.py \
  tests/test_late_bound_router.py -q
```

Observed result: `23 passed`.

## Candidate Grid

Start with validation-only policies over high-route-hit prefix-1 modes:

- query sources: `learned`, `residual`, `mean`, `domain_adaptive`
- modes: `predicted_route_p1_top4_zscore`, `domain_prior_p1_top4_zscore`
- `per_route_topk`: `50`, `100`, `200`, `500`
- `route_score_weight`: `0.0`, `0.5`, `1.0`

After each run, apply the H2 analyzer to measure whether `CandidatePoolHitRate`
improves before selecting any candidate for a fresh blind lock.

## Guardrails

- Use validation-only splits for selection.
- Do not reinterpret the consumed 2026-06-07 blind confirmation.
- Do not promote a candidate-grid result to a claim without a fresh pre-locked
  confirmation run.

## Next Target

Create the H3 explicit policy config, run it on cold-like validation, and compare
`CandidatePoolHitRate` and `route_hit_pool_miss` against the H2 baseline.


## First Run Result

Selector output:

```text
outputs/oracle_route_memory/validation_fusion_selector_h3_candidate_pool_depth_20260608
```

Error-slice output:

```text
outputs/oracle_route_memory/route_query_binding_error_analysis_h3_candidate_pool_depth_20260608
```

Selected policy:

```text
residual_predicted_route_p1_top4_zscore_k500_w0p5
```

Cold-like validation readout:

- `Recall@50`: `0.044118`
- `RouteHitRate`: `0.941176`
- `CandidatePoolHitRate`: `0.801471`
- `RouteHitPoolMissRate`: `0.139706`
- `PoolHitRankMissRate`: `0.757353`
- Dominant miss class: `pool_hit_rank_miss`

Interpretation: H3 strongly improves candidate-pool hit relative to the H2
baseline, but Recall@50 does not rise with the enlarged pool. The bottleneck has
moved from target entry into the pool to ranking within a much larger pool.

## Next Target

H4 should optimize ranking inside enlarged high-route-hit candidate pools. A good
next experiment is a validation-only reranking/fusion gate that uses source
scores, route confidence, bucket source, and candidate depth features after the
H3 `k500` retrieval step.


## Rank Diagnostic Rerun

After adding `candidate_pool_match_rank`, H3 was rerun into:

```text
outputs/oracle_route_memory/validation_fusion_selector_h3_candidate_pool_depth_rankdiag_20260608
outputs/oracle_route_memory/route_query_binding_error_analysis_h3_candidate_pool_depth_rankdiag_20260608
```

Selected policy remained `residual_predicted_route_p1_top4_zscore_k500_w0p5`.
For cold-like validation, pool-hit rank misses had:

- average `candidate_pool_match_rank`: `917.271845`
- median rank: `792`
- p75 rank: `1379`
- p90 rank: `1707`
- p95 rank: `1899`

This strengthens the H4 target: rank pool-hit targets much earlier inside the
H3 enlarged candidate pool.
