# H5 Validation Comparison

Date: 2026-06-08

Scope: protocol-v3 train-derived cold-like validation only. This table is for
policy selection and diagnostics; it is not a fresh blind-confirmation claim.

| run | hits | Recall@50 | PoolHit | avg rank-miss rank | output |
|---|---:|---:|---:|---:|---|
| H3 selected k500 | 6/136 | 0.044118 | 0.801471 | 917.2718 | `outputs/oracle_route_memory/validation_fusion_selector_h3_candidate_pool_depth_rankdiag_20260608/selector_rows.json` |
| H4 sample gate | 4/136 | 0.029412 | 0.919118 | 1240.3636 | `outputs/oracle_route_memory/h4_late_bound_fusion_router_k500_fullpool_sourcediag_20260608/cold_like_outputs.json` |
| H5 pointwise hard | 4/136 | 0.029412 | 0.919118 | 1292.5289 | `outputs/oracle_route_memory/h5_candidate_level_source_ranker_sgd_k500_n500_h100_20260608/cold_like_outputs.json` |
| H5 pairwise h100/e20 | 6/136 | 0.044118 | 0.919118 | 1207.7983 | `outputs/oracle_route_memory/h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h100_e20_20260608/cold_like_outputs.json` |
| H5 pairwise h200/e30 | 9/136 | 0.066176 | 0.919118 | 1261.6379 | `outputs/oracle_route_memory/h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h200_e30_20260608/cold_like_outputs.json` |
| H5 pairwise h300/e30 | 7/136 | 0.051471 | 0.919118 | 1254.2797 | `outputs/oracle_route_memory/h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h300_e30_20260608/cold_like_outputs.json` |
| H5 domain-routed Book h100 / Game h300 | 10/136 | 0.073529 | 0.919118 | 1271.7217 | `outputs/oracle_route_memory/h5_pairwise_domain_routed_book_h100_game_h300_20260608/cold_like_outputs.json` |

## Robustness Notes

Seed checks:

- h100/e20 seed 42 and seed 7 both produce Recall@50 `0.044118`.
- h300/e30 seed 42 and seed 7 both produce Recall@50 `0.051471`.

Nearby hard-negative checks:

- h100/e20 keeps Book hits (`3/65`) and balanced but lower Game hits (`3/71`).
- h200/e30 gives `9/136` overall with Book `3/65` and Game `6/71`.
- h300/e30 gives the strongest Game result (`7/71`) but zero Book hits.
- The current domain-routed policy combines the stable Book component from
  h100/e20 and the strongest Game component from h300/e30.

## Current Selection

Validation-selected candidate policy:
`h5_pairwise_domain_routed_book_h100_game_h300`

This policy should be treated as locked for validation comparison and robustness
tracking. It must not be reported as a blind result unless it is evaluated on a
fresh blind-confirmation split after locking.
