# H5 Paper-Facing Validation Summary

Date: 2026-06-08

## Claim Boundary

This is a validation-selected result on protocol-v3 train-derived cold-like
validation. It is not a fresh blind-confirmation result, and it must not be used
as a retroactive claim on the consumed 2026-06-07 blind-confirmation set.

## Locked Validation Candidate

Policy: `h5_pairwise_domain_routed_book_h100_game_h300`

Manifest:
`experiments/h5-candidate-level-source-reranker/locked_policy_manifest.json`

Manifest validator:
`scripts/oracle_route_memory/validate_locked_policy_manifest.py`

Fresh-confirmation checklist:
`experiments/h5-candidate-level-source-reranker/fresh_confirmation_checklist.md`

Validation output:
`outputs/oracle_route_memory/h5_pairwise_domain_routed_book_h100_game_h300_20260608`

Analyzer:
`outputs/oracle_route_memory/route_query_binding_error_analysis_h5_pairwise_domain_routed_book_h100_game_h300_20260608`

## Validation Metrics

| method | Recall@50 | hits | CandidatePoolHitRate |
|---|---:|---:|---:|
| H3 selected k500 | 0.044118 | 6/136 | 0.801471 |
| H4 sample gate | 0.029412 | 4/136 | 0.919118 |
| H5 pointwise hard | 0.029412 | 4/136 | 0.919118 |
| H5 pairwise h200/e30 | 0.066176 | 9/136 | 0.919118 |
| H5-D domain-routed | 0.073529 | 10/136 | 0.919118 |

Domain split for H5-D:

| domain | hits | Recall@50 |
|---|---:|---:|
| Book | 3/65 | 0.046154 |
| Game | 7/71 | 0.098592 |

## Interpretation

The useful change is candidate-level pairwise ranking. Pointwise classification
did not improve over H4, while pairwise training with hard negatives improved
validation Recall@50. Domain routing combines the more balanced h100/e20 Book
component with the stronger h300/e30 Game component.

## Next Step Before Any Claim

Keep the H5-D manifest locked and evaluate only on a fresh blind-confirmation
split. Do not tune the H5-D configuration after seeing any fresh blind labels.
Before running a fresh split, run the manifest validator and preserve its output
with the fresh-confirmation bundle. After locked-model scoring and domain
routing on the fresh split, use
`scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py` to publish a
gated report that keeps validation and fresh metrics separate.
