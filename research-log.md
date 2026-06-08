# Research Log

## 2026-06-08

- Reconfirmed protocol-v2 frozen result boundaries from existing bundles.
- Reconfirmed protocol-v3 blind-confirmation claim boundary: the locked
  validation-selected fusion result is claimable, while post-confirmation
  comparison rows are diagnostic only.
- Verified that the v3 bundle README renderer now preserves the claim boundary
  and reports blind-confirmation ALL metrics directly from generated rows.
- Ran targeted v3 / late-bound validation under
  `/home/grf/.conda/envs/gdpo/bin/python3`; result was `36 passed, 1 skipped`.
- Identified a suspicious diagnostic in
  `validation_fusion_selector_v3_route_source_ablation_20260608`: fusion
  policies reported `RouteHitRate = 0.0000` even when built from route-filtered
  member policies.
- Inspected `build_fusion_retrieval_row()` and found that fusion rows did not
  propagate member `route_hit`; selector summaries therefore treated every
  fusion row as a route miss.
- Started H1 (`v3-fusion-diagnostics`) to correct the diagnostic accounting
  before trusting route-source ablation interpretation.
- Implemented the H1 repair by adding member-derived `route_hit`,
  `member_route_hit_count`, and `member_candidate_pool_hit_count` to
  `build_fusion_retrieval_row()`.
- Added unit coverage in `tests/test_late_bound_router.py`; targeted tests
  passed with `19 passed`.
- Regenerated the ablation into
  `outputs/oracle_route_memory/validation_fusion_selector_v3_route_source_ablation_diagfix_20260608`.
- H1 result: fusion `Recall@50` and `CandidatePoolHitRate` were unchanged, while
  cold-like fusion `RouteHitRate` changed from `0.0000` to `0.9044-0.9779`.
  The earlier report was diagnostically wrong; the real bottleneck is now
  candidate retrieval/ranking after mostly successful prefix-1 route coverage.
- Started H2 (`route-query-binding-analysis`) to separate route misses,
  route-hit candidate-pool misses, and pool-hit ranking misses.
- Added `scripts/oracle_route_memory/analyze_route_query_binding_errors.py` plus
  `tests/test_route_query_binding_error_analysis.py`.
- Added `scripts/__init__.py` and `scripts/oracle_route_memory/__init__.py` so
  repo script imports win over the conda `scripts` package during tests.
- H2 first run wrote
  `outputs/oracle_route_memory/route_query_binding_error_analysis_h2_20260608`.
- H2 result: high-route-hit cold-like policies are dominated by
  `route_hit_pool_miss`, so H3 should target candidate-pool/query binding.
- H3 prep: fixed the explicit selector so policy-specific
  `route_score_weight` and `per_route_topk` are part of the retrieval cache key.
  Added coverage to prevent depth-grid candidates from being silently merged.
- Ran H3 candidate-pool depth grid with 96 validation-only policies. The selector
  chose `residual_predicted_route_p1_top4_zscore_k500_w0p5`: Recall@50 `0.044118`,
  RouteHitRate `0.941176`, CandidatePoolHitRate `0.801471`.
- H3 result: pool entry improved, but `pool_hit_rank_miss` became dominant; H4
  should target reranking inside enlarged high-route-hit pools.
- Added `candidate_pool_match_rank` diagnostics and reran H3 rankdiag. The H3
  selected policy stayed unchanged; pool-hit rank misses average rank `917.27`
  with median rank `792`, sharpening H4 toward candidate-level reranking.
- Implemented H4 full-pool candidate-union fix in
  `train_late_bound_fusion_router.py`: candidate unions now use complete source
  `score_map` entries instead of each source's top-50 `ranked_ids`.
- Added H4 output diagnostics for candidate-pool match rank, pool cutoff,
  source/route gate weights, per-source target ranks, and oracle source
  Hit@K. Extended the route/query-binding analyzer to summarize oracle source
  limits.
- Reran H4 full-pool sourcediag at
  `outputs/oracle_route_memory/h4_late_bound_fusion_router_k500_fullpool_sourcediag_20260608`.
  Cold-like Recall@50 stayed `0.029412`, while CandidatePoolHitRate rose to
  `0.919118`.
- H4 result: misses are still dominated by pool-hit rank misses. Average
  rank-miss rank is `1186.06` for Book and `1284.13` for Game. Oracle
  best-single-source Hit@50 is only `0.0956`, with median oracle source rank
  `393`.
- H4 conclusion: a sample-level source gate is not sufficient. The next target
  is H5, a candidate-level source/rank reranker with per-source local ranks,
  source presence, route confidence, and cross-source agreement features.
- Implemented H5 candidate-level feature exporter:
  `scripts/oracle_route_memory/export_candidate_level_source_features.py`.
  Output
  `outputs/oracle_route_memory/h5_candidate_level_source_features_k500_20260608`
  contains `1,888,938` train candidate rows and `345,969` cold-like candidate
  rows.
- H5 exporter summary: train has `786` positives with oracle source Hit@50
  `0.3389`; cold-like has `125` positives with oracle source Hit@50 `0.0956`
  and average oracle source rank `524.19`.
- Implemented first H5 pointwise ranker:
  `scripts/oracle_route_memory/train_candidate_level_source_ranker.py`, using
  `SGDClassifier(log_loss)` over exported candidate-level features.
- H5 pointwise random-negative run
  `h5_candidate_level_source_ranker_sgd_k500_n500_20260608` reached cold-like
  Recall@50 `0.022059`.
- Added hard-negative sampling by source-local rank. The hard-mix run
  `h5_candidate_level_source_ranker_sgd_k500_n500_h100_20260608` reached
  Recall@50 `0.029412`, still below H3 `0.044118`.
- H5 conclusion so far: candidate-level features are now exportable, but a
  pointwise classifier is too weak. The next target is pairwise/listwise
  per-sample ranking with hard negatives.
- Extended the H5 ranker with a pairwise linear RankNet-style objective
  (`softplus(pos-neg)`) while preserving analyzer-compatible outputs.
- H5 pairwise h100/e20
  `h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h100_e20_20260608`
  matched H3 with Recall@50 `0.044118`.
- H5 pairwise h300/e30
  `h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h300_e30_20260608`
  improved to Recall@50 `0.051471`, but the gain was Game-heavy.
- Added `combine_ranker_outputs_by_domain.py` and created a validation-only
  domain-routed H5 policy:
  `h5_pairwise_domain_routed_book_h100_game_h300_20260608`.
- H5 domain-routed result: Book uses h100/e20, Game uses h300/e30. Cold-like
  Recall@50 is `0.073529` (`10/136`), with Book `0.046154` and Game `0.098592`.
  This exceeds the H3 validation baseline but is still not a blind-confirmation
  claim.
- Locked the H5 domain-routed validation policy in
  `experiments/h5-candidate-level-source-reranker/locked_policy_manifest.json`.
- Ran robustness checks: h100/e20 seed 7 stayed at Recall@50 `0.044118`; h300/e30
  seed 7 stayed at `0.051471`; h200/e30 reached `0.066176` with Book `3/65` and
  Game `6/71`.
- Added `experiments/h5-candidate-level-source-reranker/validation_comparison.md`
  as the single H3/H4/H5 validation comparison table. Current selection remains
  the domain-routed H5 policy at Recall@50 `0.073529`.
- Added `validate_locked_policy_manifest.py` and validated the locked H5-D
  manifest against its selected output rows. The validator recomputed `10/136`
  hits, Recall@50 `0.073529`, Book `3/65`, and Game `7/71`.

- Added `scripts/oracle_route_memory/prepare_h5_fresh_confirmation_bundle.py`
  plus `tests/test_prepare_h5_fresh_confirmation_bundle.py`. The tool reruns the
  locked manifest validator, copies only validation-side docs/artifacts, writes
  SHA-256 hashes, and renders `bundle_manifest.json` plus `README.md` without
  reading fresh or consumed blind-confirmation labels.
- Current H5-D fresh-confirmation prep target is
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v11`.
  It should be treated as a validation-only handoff package; the next research
  action is to wait for a clearly fresh/unconsumed split, record its manifest
  path/hash, and only then run locked H5-D scoring.

- Added `scripts/oracle_route_memory/audit_h5_fresh_confirmation_bundle.py`
  plus `tests/test_audit_h5_fresh_confirmation_bundle.py`. The audit verifies
  bundle artifact hashes, source drift, saved validator output, and an optional
  locked-manifest validator rerun without reading blind-confirmation labels.
- The current H5-D handoff target is now the doc-synchronized v3 bundle
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v11`
  with audit output
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v11_audit`.
  The next gate is an audit status of `ok`, no `errors`, no `source_drift`, and
  rerun Recall@50 `0.07352941176470588` before any future fresh split scoring.
  The audit command now uses `--fail-on-source-drift` so source drift is a hard
  failure instead of a passive warning.

- Added `scripts/oracle_route_memory/register_h5_fresh_split.py` plus
  `tests/test_register_h5_fresh_split.py`. The registration guard records a
  future fresh split manifest path/SHA-256 only after the bundle audit passes,
  and rejects the consumed protocol-v3 split by path or SHA-256 before any H5-D
  scoring can proceed. The current handoff target is the v5 bundle/audit pair.

- Hardened `register_h5_fresh_split.py` so future registration requires at
  least one `--require-manifest-field KEY=VALUE` check. The tests now cover
  accepted fresh metadata, missing required fields, mismatched required fields,
  consumed split rejection, and audit drift rejection. The current handoff target
  is the v6 bundle/audit pair.

- Added `scripts/oracle_route_memory/score_candidate_level_source_ranker.py`
  plus `tests/test_score_candidate_level_source_ranker.py`. The scorer loads
  locked H5 component `model.pkl` files, including legacy `__main__` pickles,
  and scores candidate rows without retraining.
- Ran loaded-model validation replay: h100/e20 Recall@50 `0.044118`, h300/e30
  Recall@50 `0.051471`, and domain-routed Recall@50 `0.073529`. This confirms
  the selected H5-D validation result can be reconstructed from locked model
  artifacts and candidate rows without retraining. The current handoff target is
  the v7 bundle/audit pair.

- Added `scripts/oracle_route_memory/validate_h5_loaded_model_replay.py`
  plus `tests/test_validate_h5_loaded_model_replay.py`. The validator compares
  locked H5-D output rows with loaded-model replay rows on key sample fields and
  metrics.
- Ran real loaded-model replay validation at
  `outputs/oracle_route_memory/h5_loaded_model_replay_validation_20260608`:
  status `ok`, mismatch_count `0`, metric_errors `[]`, Recall@50 `0.073529`.
  The current handoff target is the v8 bundle/audit pair.

- Added `scripts/oracle_route_memory/check_h5_fresh_readiness.py` plus
  `tests/test_check_h5_fresh_readiness.py`. The readiness checker reruns the
  locked manifest validator, checks the prep-bundle audit, checks the
  loaded-model replay validation, and verifies h100/h300 `model.pkl` files.
- Ran real readiness at `outputs/oracle_route_memory/h5_fresh_readiness_20260608_v11`:
  status `ok`, source_drift_count `0`, loaded replay mismatch_count `0`, and
  h100/h300 model files present. The current handoff target is the v9
  bundle/audit pair.

- Added `to_human/h5_fresh_confirmation_handoff_summary_20260608.md` and
  included it in the prep-bundle artifact list. The current handoff target is
  the v11 bundle/audit/readiness trio.

- Added `experiments/h5-candidate-level-source-reranker/README.md` as the
  directory entry point and included it in prep-bundle artifacts. Updated
  `protocol.md` so its next target points to the current handoff gates. The
  current handoff target is the v11 bundle/audit/readiness trio.

- Added `scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py`
  plus `tests/test_render_h5_fresh_confirmation_report.py`. The report renderer
  consumes a future fresh split registration, readiness report, and domain-routed
  locked-model fresh outputs, then writes `fresh_confirmation_report.json` and
  `.md` with locked validation metrics separated from fresh confirmation
  metrics.
- Updated H5-D handoff docs so the next target is the v12
  bundle/audit/readiness trio plus the final report-rendering gate after any
  future fresh locked-model scoring.

- Added `experiments/h5-candidate-level-source-reranker/handoff_index.json`,
  `scripts/oracle_route_memory/validate_h5_handoff_index.py`, and
  `tests/test_validate_h5_handoff_index.py`. The index records the current v13
  handoff paths, required document mentions, the report renderer, and the next
  target; the validator checks artifact existence, audit/readiness status,
  locked manifest replay, and handoff doc consistency.
- Updated H5-D handoff docs so the next target is the v13
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v13`
  bundle/audit/readiness trio plus the handoff-index validation gate.

- Extended `experiments/h5-candidate-level-source-reranker/handoff_index.json`
  with `bundle_required_artifacts` and updated
  `scripts/oracle_route_memory/validate_h5_handoff_index.py` to check that the
  current prep-bundle manifest includes every required artifact. Added test
  coverage for missing bundle artifacts.
- Updated H5-D handoff docs so the next target is the v16
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16`
  bundle/audit/readiness trio, with handoff-index validation covering gate
  paths, document mentions, and prep-bundle artifact coverage.

- Added an explicit fresh candidate-level feature export template to
  `experiments/h5-candidate-level-source-reranker/repro_commands.md` and linked
  it from `fresh_confirmation_checklist.md`. The fresh scoring flow now records
  the exact locked query sources, prefix-1 beam, per-route depth, and fresh
  candidate output directory before loaded-model h100/h300 scoring. The current
  handoff target is the v16 bundle/audit/readiness/index-validation set.

## Open Discipline Notes

- The current H1 repair is an engineering-correctness experiment. It should not
  be reported as a new model improvement unless rerun metrics show a validated
  change in Recall@K from an independently locked policy.
- The consumed 2026-06-07 blind-confirmation set must remain closed for claims.
- Any future higher-performing policy must be locked on validation before a
  fresh blind-confirmation run.
