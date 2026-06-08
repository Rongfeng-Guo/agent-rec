# Progress Report: H2 Route/Query Binding Error Analysis

Date: 2026-06-08
Repo: `/home/grf/agent-rec`

## What Changed

- Added `scripts/oracle_route_memory/analyze_route_query_binding_errors.py`.
- Added `tests/test_route_query_binding_error_analysis.py`.
- Added package markers under `scripts/` so repository script imports are not
  shadowed by the conda `scripts` package during tests.
- Generated `outputs/oracle_route_memory/route_query_binding_error_analysis_h2_20260608`.

## Why It Matters

H1 showed that fusion policies often hit the true prefix-1 route after the
diagnostic repair. H2 asks what happens after that route hit: does the target
miss the candidate pool, or does it enter the pool and lose during ranking?

## Key Readout

| policy | split | n | Hit@50 | RouteHitRate | CandidatePoolHitRate | dominant miss |
|---|---|---:|---:|---:|---:|---|
| `fusion_domain_prior_predicted_history_vote_rrf` | cold-like | 136 | 0.044118 | 0.977941 | 0.044118 | `route_hit_pool_miss` |
| `fusion_domain_prior_predicted_rrf` | cold-like | 136 | 0.044118 | 0.963235 | 0.044118 | `route_hit_pool_miss` |
| `predicted_route_p1_top4` | cold-like | 136 | 0.036765 | 0.941176 | 0.154412 | `route_hit_pool_miss` |

Game-domain top-4 route policies often reach `RouteHitRate=1.0000`, but their
`CandidatePoolHitRate` is only `0.1831` or lower.

## Interpretation

The current failure is mainly candidate-pool/query binding inside already-covered
prefix-1 routes. Final rank fusion still matters, but many examples never reach
the stage where ranking can rescue them.

## Next Target

H3 should improve target entry into the per-route candidate pool before another
route-classifier-only round. Candidate directions:

- vary per-route candidate depth under validation-only selection;
- calibrate query source or query fusion inside high-hit prefix-1 buckets;
- measure `CandidatePoolHitRate` before treating ranking deltas as meaningful;
- lock any future policy on validation before a fresh blind confirmation.

Validation: `19 passed` for the H2 analyzer and late-bound router tests;
compile checks passed for the touched scripts.
