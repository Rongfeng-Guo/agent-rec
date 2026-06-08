# Progress Report: v3 Fusion Diagnostics

Date: 2026-06-08
Repo: `/home/grf/agent-rec`

## What Changed

I fixed a diagnostics bug in fusion row reporting:

```text
scripts/oracle_route_memory/eval_predicted_route.py
tests/test_late_bound_router.py
```

Fusion rows now propagate member-derived route diagnostics:

- `route_hit`
- `member_route_hit_count`
- `member_candidate_pool_hit_count`

This does not change the fusion ranking algorithm.

## Why It Matters

The previous route-source ablation report showed fusion policies with
`RouteHitRate = 0.0000`. That made it look like fusion route sources were
missing the true prefix-1 route entirely.

After the fix, the same ablation shows fusion RouteHitRate in the `0.9044` to
`0.9779` range while Recall@50 stays unchanged.

## Key Result

| policy | Recall@50 | corrected RouteHitRate | CandidatePoolHitRate |
|---|---:|---:|---:|
| fusion_domain_prior_history_vote_rrf | 0.044118 | 0.904412 | 0.044118 |
| fusion_domain_prior_predicted_rrf | 0.044118 | 0.963235 | 0.044118 |
| fusion_domain_prior_predicted_history_vote_rrf | 0.044118 | 0.977941 | 0.044118 |
| fusion_predicted_history_vote_rrf | 0.036765 | 0.977941 | 0.036765 |
| fusion_predicted_mean_rrf | 0.022059 | 0.941176 | 0.036765 |

Output directory:

```text
outputs/oracle_route_memory/validation_fusion_selector_v3_route_source_ablation_diagfix_20260608
```

## Updated Interpretation

The route-source fusion failure is not primarily a prefix-1 route-hit failure.
The corrected diagnostics show that the route is often covered. The failure is
downstream:

- the target item often does not enter the candidate pool, or
- it enters but the reranker/fusion strategy does not rank it effectively.

This pushes the next research step toward query binding, candidate depth, and
learned late-bound reranking rather than another route classifier-only tweak.

## Next Hypothesis

H2 should test whether candidate-pool hit and final Recall improve when the
policy explicitly optimizes retrieval inside high-hit prefix-1 buckets.

Candidate directions:

- increase per-route candidate depth and measure pool-vs-ranking loss;
- domain-specific query source calibration;
- learned late-bound fusion gate using route confidence, bucket size, query
  agreement, and source scores;
- oracle-assisted error slicing to separate route miss, pool miss, and ranking
  miss cases.

## Validation

Targeted tests passed:

```text
19 passed
```

No frozen protocol-v2 outputs were modified.
