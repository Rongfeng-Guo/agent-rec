# Progress Report: H4 Late-Bound Fusion Router

Date: 2026-06-08
Repo: `/home/grf/agent-rec`

## What Changed

- Fixed H4 candidate union construction so the late-bound gate uses full source
  `score_map` candidates instead of each source's top-50 `ranked_ids`.
- Added H4 diagnostics for candidate-pool match rank, pool cutoff, gate source
  weights, route weight, per-source target ranks, and oracle source Hit@K.
- Extended the route/query-binding analyzer to report oracle source Hit@K and
  average oracle source rank.
- Added focused test coverage for full-pool candidate union and H4 diagnostic
  output fields.

## Main Result

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
| Book avg rank-miss rank | 1186.0556 |
| Game avg rank-miss rank | 1284.1343 |

## Source Diagnostic

The learned gate collapses to `prefix1_head`:

| source | present | Hit@50 | avg rank | avg learned weight |
|---|---:|---:|---:|---:|
| learned | 112/136 | 0.0368 | 991.5714 | 0.000077 |
| residual | 109/136 | 0.0368 | 904.2569 | 0.000015 |
| mean | 98/136 | 0.0147 | 812.7449 | 0.000042 |
| prefix1_head | 95/136 | 0.0294 | 880.7684 | 0.999834 |

Even an oracle best-single-source selector has limited top-50 headroom:

| metric | value |
|---|---:|
| oracle source Hit@50 | 0.0956 |
| oracle source median rank | 393 |
| oracle source p90 rank | 1112 |

## Interpretation

H4 is a useful negative result. Enlarging the pool fixes most pool-entry misses,
but a sample-level source gate cannot rank the target into top-50. The target is
usually hundreds of positions down even in the best single source.

## Next Target

Move to H5: candidate-level source/rank reranking. The immediate implementation
target is an exporter with one row per candidate, including per-source scores,
source-local ranks, source presence, route confidence, agreement features, and
train-derived target labels.

Claim boundary is unchanged: H4/H5 are validation-only diagnostics unless a
future policy is locked before a fresh blind confirmation.
