# H5: Candidate-Level Source Reranker

- Status: domain-routed pairwise validation policy found
- Type: validation-only candidate reranking experiment
- Claim status: not a model-performance claim until locked before a fresh blind confirmation

## Question

Can candidate-level source/rank features move enlarged-pool hits from rank
hundreds into the top-50, where a sample-level source gate failed?

## Motivation

H4 full-pool sourcediag reached high candidate-pool coverage but low final rank:

| metric | value |
|---|---:|
| CandidatePoolHitRate | 0.919118 |
| Recall@50 | 0.029412 |
| Oracle best-single-source Hit@50 | 0.0956 |
| Oracle source median rank | 393 |

The H4 gate collapsed to `prefix1_head` and could not recover most pool hits.
The oracle source diagnostic also shows that choosing one source per sample is
not enough. H5 should score each candidate with features from all sources.

## Required Inputs

- H4 sourcediag output:
  `outputs/oracle_route_memory/h4_late_bound_fusion_router_k500_fullpool_sourcediag_20260608`
- H4 analyzer output:
  `outputs/oracle_route_memory/route_query_binding_error_analysis_h4_late_bound_fusion_router_k500_fullpool_sourcediag_20260608`
- H3 selected policy baseline:
  `residual_predicted_route_p1_top4_zscore_k500_w0p5`
- Official protocol-v3 train-derived splits:
  `outputs/oracle_route_memory/official_protocol_v3_blind_confirmation/split_manifest.json`

## Candidate-Level Features

The exporter should produce one row per `(sample_id, candidate_id)` with:

- per-source normalized score for `learned`, `residual`, `mean`, and
  `prefix1_head`;
- per-source presence mask;
- per-source local rank within that source's retrieved pool;
- best, mean, max, and variance of available source scores;
- best source rank and number of sources containing the candidate;
- route score and route beam metadata;
- history length, route confidence, route entropy, bucket size, and source-list
  agreement;
- target label from protocol train-derived validation only.

## Validation Gate

Train or score only on protocol train-derived validation data. Compare against:

- H3 selected k500 policy;
- H4 full-pool sample-level gate;
- oracle best-single-source source rank diagnostic.

The primary validation metric is cold-like Recall@50. Secondary diagnostics are
CandidatePoolHitRate, ConditionalRecall@50GivenPoolHit, average pool-hit
rank-miss rank, and oracle-source gap.

## Completed Outputs

### Candidate-Level Exporter

Output:
`outputs/oracle_route_memory/h5_candidate_level_source_features_k500_20260608`

| split | samples | candidate rows | positives | pool hit | avg pool size | oracle src Hit@50 | avg oracle src rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 844 | 1888938 | 786 | 0.9313 | 2238.08 | 0.3389 | 220.27 |
| cold_like_val | 136 | 345969 | 125 | 0.9191 | 2543.89 | 0.0956 | 524.19 |

The exporter materializes one candidate row per sample/candidate with
per-source scores, per-source presence, source-local ranks, route score,
history/route confidence features, and target labels from train-derived splits.

### Pointwise SGD Rankers

Random negatives:
`outputs/oracle_route_memory/h5_candidate_level_source_ranker_sgd_k500_n500_20260608`

Hard-negative mix:
`outputs/oracle_route_memory/h5_candidate_level_source_ranker_sgd_k500_n500_h100_20260608`

| run | negatives | hard negatives | Recall@50 | avg rank-miss rank |
|---|---:|---:|---:|---:|
| SGD random | 393000 | 0 | 0.022059 | 1219.0738 |
| SGD hard mix | 393000 | 78600 | 0.029412 | 1292.5289 |

The hard-negative mix recovers the H4 sample-gate Recall@50 level but still does
not beat H3's selected `0.044118`. The pointwise objective is not enough for the
current sparse-positive, large-candidate setting.

### Pairwise Linear Rankers

Pairwise h100/e20:
`outputs/oracle_route_memory/h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h100_e20_20260608`

Pairwise h300/e30:
`outputs/oracle_route_memory/h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h300_e30_20260608`

| run | hard negatives | epochs | Recall@50 | hits | note |
|---|---:|---:|---:|---:|---|
| pairwise h100/e20 | 78600 | 20 | 0.044118 | 6/136 | matches H3 |
| pairwise h300/e30 | 235800 | 30 | 0.051471 | 7/136 | beats H3 overall, Game-heavy |

The h300/e30 variant improves overall Recall@50, but its hits are all in Game.
The h100/e20 variant is more balanced and supplies Book hits.

### Domain-Routed Pairwise Policy

Output:
`outputs/oracle_route_memory/h5_pairwise_domain_routed_book_h100_game_h300_20260608`

Analyzer:
`outputs/oracle_route_memory/route_query_binding_error_analysis_h5_pairwise_domain_routed_book_h100_game_h300_20260608`

Routing:

- Book -> pairwise h100/e20
- Game -> pairwise h300/e30

Cold-like validation:

| slice | n | hits | Recall@50 |
|---|---:|---:|---:|
| Book | 65 | 3 | 0.046154 |
| Game | 71 | 7 | 0.098592 |
| All | 136 | 10 | 0.073529 |

This is the strongest validation-only H5 result so far and exceeds the H3
selected validation baseline `0.044118`. It remains a train-derived validation
selection, not a fresh blind-confirmation claim.

Locked manifest:
`experiments/h5-candidate-level-source-reranker/locked_policy_manifest.json`

Validation comparison table:
`experiments/h5-candidate-level-source-reranker/validation_comparison.md`

Manifest validator:
`scripts/oracle_route_memory/validate_locked_policy_manifest.py`

Fresh-confirmation checklist:
`experiments/h5-candidate-level-source-reranker/fresh_confirmation_checklist.md`

Robustness checks completed:

- h100/e20 seed 42 and seed 7 both produce Recall@50 `0.044118`;
- h300/e30 seed 42 and seed 7 both produce Recall@50 `0.051471`;
- h200/e30 nearby hard-negative setting reaches Recall@50 `0.066176` with Book
  `3/65` and Game `6/71`;
- domain routing remains the strongest checked validation policy at `0.073529`.
- locked manifest validation passes and recomputes `10/136` hits from the
  selected output rows.

## Next Implementation Target

The H5-D validation policy is locked and the current handoff is packaged in:

- `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16`
- `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16_audit`
- `outputs/oracle_route_memory/h5_fresh_readiness_20260608_v16`

Next steps are only for a clearly fresh/unconsumed split:

- do not alter the locked domain route or component ranker parameters;
- register the fresh split path/SHA-256 and required fresh/unconsumed manifest
  fields before scoring;
- export fresh candidate-level rows to
  `outputs/oracle_route_memory/h5_fresh_candidate_level_source_features_k500_YYYYMMDD`
  with the locked query sources, prefix-1 beam, and per-route depth;
- score with locked h100/h300 `model.pkl` files, not retraining;
- keep validation metrics separate from future fresh-confirmation metrics;
- keep `scripts/oracle_route_memory/validate_h5_handoff_index.py` at
  `status=ok` after any handoff doc or bundle version change;
- render the final gated report with
  `scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py`;
- no consumed blind-confirmation labels.
