# H1 Analysis: v3 Fusion Diagnostics

## Summary

H1 is supported. The old route-source ablation report under-reported fusion
route hits because fusion rows did not propagate member `route_hit` fields.

The code repair changed diagnostics only:

- Added `route_hit`
- Added `member_route_hit_count`
- Added `member_candidate_pool_hit_count`

The rerun output is:

```text
outputs/oracle_route_memory/validation_fusion_selector_v3_route_source_ablation_diagfix_20260608
```

## Validation

Targeted tests:

```text
tests/test_late_bound_router.py
tests/test_explicit_validation_fusion_selector.py
```

Result:

```text
19 passed
```

## Old vs New Cold-Like Fusion Summary

| policy | Recall@50 old | Recall@50 new | RouteHit old | RouteHit new | PoolHit old | PoolHit new |
|---|---:|---:|---:|---:|---:|---:|
| fusion_domain_prior_history_vote_rrf | 0.044118 | 0.044118 | 0.000000 | 0.904412 | 0.044118 | 0.044118 |
| fusion_domain_prior_predicted_history_vote_rrf | 0.044118 | 0.044118 | 0.000000 | 0.977941 | 0.044118 | 0.044118 |
| fusion_domain_prior_predicted_rrf | 0.044118 | 0.044118 | 0.000000 | 0.963235 | 0.044118 | 0.044118 |
| fusion_predicted_history_vote_rrf | 0.036765 | 0.036765 | 0.000000 | 0.977941 | 0.036765 | 0.036765 |
| fusion_predicted_mean_rrf | 0.022059 | 0.022059 | 0.000000 | 0.941176 | 0.036765 | 0.036765 |

## Interpretation

The prediction held:

- `Recall@50` did not change.
- `CandidatePoolHitRate` did not change.
- Fusion `RouteHitRate` changed from zero to high values.

This means the old report was not evidence that fusion route sources miss the
true prefix-1 route. Instead, the corrected report shows that route coverage is
often present, but the target item still fails to enter or rank highly in the
candidate pool.

## Consequence for Next Experiments

The next hypothesis should target query/candidate retrieval after route hit:

- More candidate depth per routed bucket.
- Query source calibration per domain.
- Learned late-bound fusion gate trained to choose source-specific scores based
  on route confidence, bucket size, and query agreement.
- Better reranking inside high-hit prefix-1 route buckets.

The top-conference path should not be framed as "route prediction alone solves
cold-start retrieval." The stronger story is likely about diagnosing and closing
the route/query-binding gap with locked validation and fresh blind confirmation.
