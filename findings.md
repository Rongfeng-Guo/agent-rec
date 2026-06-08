# Findings

## Current Understanding

The main bottleneck is route/query binding rather than candidate-memory capacity.
Protocol-v3 blind confirmation shows large oracle headroom:

- Selected validation-locked fusion blind ALL `Recall@50`: `0.084746`
- Strongest no-route metadata blind ALL `Recall@50`: `0.093220`
- Oracle prefix-1 blind ALL `Recall@50`: `0.127119`
- Oracle prefix-2 blind ALL `Recall@50`: `0.822034`

This means that simply adding route filtering is not sufficient. The route
prediction and query binding must work together, and post-confirmation
diagnostics must not be promoted to claims.

## Patterns and Insights

- Validation-selected route fusion can beat random matched-size buckets, but it
  currently does not beat the strongest no-route metadata baseline on the
  consumed v3 blind set.
- Diagnostic single-query rows can look stronger than the locked selected
  method, but they are not claimable unless locked before a fresh blind run.
- Route-source ablation reports need correct diagnostics before they can guide
  the next model change. A report showing fusion `RouteHitRate = 0.0000` for
  route-filtered fusion policies is likely measuring missing metadata rather
  than real route behavior.
- H1 confirmed that the fusion route-hit diagnostic was under-reported. After
  fixing member-derived diagnostics, fusion policies on cold-like validation
  show high route hit (`0.9044-0.9779`) with unchanged Recall@50. This changes
  the interpretation: the next bottleneck is not "fusion cannot find the right
  prefix-1 route"; it is "given mostly correct prefix-1 coverage, the query and
  ranking pipeline often fails to retrieve or rank the target item."
- H2 split the remaining misses into route miss, route-hit candidate-pool miss,
  and pool-hit ranking miss. On cold-like validation,
  `fusion_domain_prior_predicted_history_vote_rrf` has `RouteHitRate=0.977941`
  but only `CandidatePoolHitRate=0.044118`; `predicted_route_p1_top4` has
  `RouteHitRate=0.941176` and `CandidatePoolHitRate=0.154412`. The dominant
  miss class is `route_hit_pool_miss`, not `pool_hit_rank_miss`.
- This makes the next target more specific: improve target entry into the
  candidate pool inside already-covered prefix-1 routes before optimizing final
  rank fusion.
- H3 is now code-ready for a validation-only depth grid: explicit selector
  retrieval rows are keyed by query source, mode, route score weight, and
  per-route candidate depth.
- H3 depth grid selected `residual_predicted_route_p1_top4_zscore_k500_w0p5`
  with `CandidatePoolHitRate=0.801471` but `Recall@50=0.044118`. This shifts the
  active bottleneck from candidate-pool entry to ranking within an enlarged pool.
- H3 rankdiag shows pool-hit rank misses are deep: average rank `917.27`,
  median rank `792`, and p90 rank `1707`. H4 needs candidate-level features and
  reranking, not another pool-depth increase alone.
- H4 full-pool reranking fixed an implementation issue in the late-bound gate:
  candidate unions must use complete source `score_map` entries, not each
  source's top-50 list. After the fix, cold-like CandidatePoolHitRate reached
  `0.919118`, but Recall@50 stayed low at `0.029412`.
- H4 sourcediag shows the simple sample-level gate is the wrong abstraction. It
  collapses to `prefix1_head` with average learned weight `0.999834`, yet an
  oracle best-single-source selector only reaches Hit@50 `0.0956`. Targets are
  still deep even in the best source ranking: oracle source median rank `393`,
  Book avg oracle rank `462.09`, Game avg oracle rank `574.59`.
- The next useful direction is candidate-level reranking with per-source scores,
  source-local ranks, route confidence, source presence, and agreement features.
  Source-level weighting alone cannot recover most pool hits.
- H5 now has a candidate-level exporter. It produced `1,888,938` train rows and
  `345,969` cold-like rows from the k500 pool. Cold-like pool coverage remains
  high (`0.9191`), but the oracle best-source signal is weak (`0.0956` Hit@50,
  average oracle source rank `524.19`).
- The first candidate-level pointwise rankers are negative. Random-negative SGD
  reached Recall@50 `0.022059`; adding source-rank hard negatives reached
  `0.029412`. This does not beat H3 (`0.044118`) and suggests the next
  candidate-level attempt needs a pairwise/listwise ranking objective rather
  than independent candidate classification.
- Pairwise candidate-level ranking is materially better than pointwise. The
  h100/e20 pairwise linear ranker matched H3 at `0.044118`, and h300/e30 reached
  `0.051471`.
- H5 domain routing is the strongest validation-only result so far: Book routed
  to pairwise h100/e20 and Game routed to pairwise h300/e30 reaches Recall@50
  `0.073529` (`10/136`) on cold-like validation. This result must be treated as
  validation-selected only; it is a candidate for locking before a future fresh
  blind confirmation, not a retroactive blind claim.
- H5-D has now been locked as a validation-selected manifest. Seed checks were
  stable for both components, and the nearby h200/e30 hard-negative setting
  reached `0.066176`, supporting the broader conclusion that pairwise
  candidate-level ranking is better than pointwise ranking in this setup.
- H5-D now has a validation-only fresh-confirmation handoff at v16. The current
  prep bundle, audit, readiness, and handoff-index validation gates all pass:
  audit `source_drift=[]`, readiness `bundle_audit_source_drift_count=0`, and
  handoff-index validation covers `11` required bundle artifacts plus `8`
  document checks.

## Current H5-D Handoff State

- Locked validation policy:
  `h5_pairwise_domain_routed_book_h100_game_h300`
- Locked validation result:
  Recall@50 `0.07352941176470588` (`10/136`)
- Current prep bundle:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16`
- Current prep bundle audit:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16_audit`
- Current readiness:
  `outputs/oracle_route_memory/h5_fresh_readiness_20260608_v16`
- Current handoff index validation:
  `outputs/oracle_route_memory/h5_handoff_index_validation_20260608_v16`

Note: `outputs/oracle_route_memory/h5_handoff_index_validation_20260608_v15`
failed because `protocol.md` was missing the fresh candidate feature export
mention required by `handoff_index.json`. It is superseded by the v16 validation
listed above.

Next target: keep the locked H5-D manifest and v16 gates unchanged while waiting
for a clearly fresh/unconsumed split. When that split exists, register its
manifest path/SHA-256, score only with locked h100/h300 `model.pkl` files, apply
the locked domain route, and render a final fresh-confirmation report that keeps
validation and fresh metrics separate.

## Lessons and Constraints

- Do not retune or reinterpret consumed blind-confirmation outputs.
- Do not use post-confirmation comparison rows as paper claims.
- Before optimizing a new router, verify that selector diagnostics correctly
  decompose route misses, candidate-pool misses, and ranking misses.
- Default remote `python3` is not the right test environment; use
  `/home/grf/.conda/envs/gdpo/bin/python3` and account for the conda `scripts`
  package shadowing the repo namespace when needed.
- Do not interpret low Recall@50 with high RouteHitRate as a route-classifier
  failure. It may be a query/candidate-pool or rank-fusion failure.
- If any H5-D handoff doc or `handoff_index.json` changes, create a new
  versioned prep bundle/audit/readiness/index-validation set. Do not overwrite
  existing v16 evidence directories.

## Open Questions

- After fixing fusion route-hit accounting, do fusion policies actually have
  high route-hit but poor final ranking, or are they also route-missing?
  Answer so far: high route-hit is confirmed for cold-like validation fusion
  policies; the remaining issue is downstream retrieval/ranking.
- Is the next publishable direction a better route predictor, a better query
  generator, or a learned late-bound fusion gate that adapts per sample?
- Can a candidate-level reranker learn cross-source and route-aware features
  that move pool-hit targets from rank hundreds into top-50 without touching the
  consumed blind-confirmation set?
- Can the locked H5-D configuration survive a fresh confirmation split without
  the Book/Game domain route overfitting the small cold-like validation slice?
- Can we lock a future policy on validation that beats the no-route metadata
  baseline before running a new blind confirmation?
