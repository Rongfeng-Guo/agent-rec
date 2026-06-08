# Progress Report: H3 Candidate-Pool Depth

Date: 2026-06-08
Repo: `/home/grf/agent-rec`

## What Changed

- Created `experiments/h3-candidate-pool-depth/policy_config.json` with 96
  validation-only policies.
- Fixed the explicit selector so policy-specific `route_score_weight` and
  `per_route_topk` are respected when building retrieval rows.
- Ran the H3 selector on the protocol-v3 train-derived validation splits.
- Ran the H2 analyzer on the H3 selector rows.

## Selected Policy

```text
residual_predicted_route_p1_top4_zscore_k500_w0p5
```

Cold-like validation:

| metric | value |
|---|---:|
| Recall@50 | 0.044118 |
| RouteHitRate | 0.941176 |
| CandidatePoolHitRate | 0.801471 |
| RouteHitPoolMissRate | 0.139706 |
| PoolHitRankMissRate | 0.757353 |

## Interpretation

H3 confirms that candidate-pool depth can largely fix the pool-entry bottleneck.
However, Recall@50 does not improve with the larger pool, so the active failure
mode is now ranking inside the enlarged candidate pool.

## Next Target

H4 should focus on reranking/fusion inside the H3 enlarged candidate pool. The
next useful validation-only experiment should use route confidence, query source,
bucket/source features, and candidate depth features to rank pool hits higher.

Claim boundary remains unchanged: H3 is validation-only and is not a fresh blind
confirmation result.


## Rank Diagnostic Rerun

A rank diagnostic rerun kept the same selected policy. Among cold-like
pool-hit rank misses, the target is very deep in the H3 pool:

- average rank: `917.271845`
- median rank: `792`
- p90 rank: `1707`

This makes H4 sharper: the next useful work is candidate-level feature export and
reranking inside the H3 `k500` pool.
