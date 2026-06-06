# Predicted Route Experiment

This stage implements a lightweight `LateBoundRouter` that predicts hierarchical routes and a query embedding from user history only.

Pipeline:

1. Load metadata-derived item embeddings.
2. Load `PROXY_HIERARCHICAL_ROUTE` mappings.
3. Build warm training examples from the train split only.
4. Train a lightweight router with prefix-1 CE, prefix-2 conditional CE, and contrastive query binding.
5. Evaluate cold and warm retrieval with predicted routes and route beams.

Key rule: no target route, target metadata, or target item leakage is used at test time.

## 2026-06-06 Prefix-1 / Query Binding Diagnostic

Implemented a predicted-route eval update that separates prefix-1 routing from full prefix-2 beams. `predicted_route_p1` now uses `route1_log_probs` directly and retrieves inside prefix-1 memory. The eval script can also run `--query-source all`, which reports `learned`, `pooled`, `residual`, and `mean` query bindings in one run, plus per-domain retrieval and route diagnostics.

Main output directories:

- `outputs/oracle_route_memory/predicted_route_v0_eval_p1_residual_diag_20260606_213700`
- `outputs/oracle_route_memory/predicted_route_v0_eval_p1_top124_diag_20260606_213812`
- `outputs/oracle_route_memory/predicted_route_v0_eval_p1_top124_norouteweight_diag_20260606_213812`
- `outputs/oracle_route_memory/predicted_route_v0_eval_p1_top124_mean_diag_20260606_214138`
- `outputs/oracle_route_memory/predicted_route_v0_eval_p1_top124_mean_h5_diag_20260606_214230`

Cold route predictability from `route_diagnostics.csv`:

- prefix-1 top-1/top-2/top-4 accuracy: `0.1942 / 0.4290 / 0.7942`
- prefix-2 top-1/top-4/top-8 accuracy: `0.0116 / 0.0638 / 0.1594`
- Book prefix-1 top-1/top-2/top-4: `0.1990 / 0.3560 / 0.6387`
- Game prefix-1 top-1/top-2/top-4: `0.1883 / 0.5195 / 0.9870`

Cold Recall@50 from `summary.csv` remains below the first target threshold `0.029`:

- learned query, prefix-1 top-1/top-2/top-4: `0.0116 / 0.0116 / 0.0116`
- pooled query, prefix-1 top-1/top-2/top-4: `0.0058 / 0.0058 / 0.0058`
- residual query, prefix-1 top-1/top-2/top-4: `0.0058 / 0.0058 / 0.0058`
- mean query, prefix-1 top-1/top-2/top-4: `0.0058 / 0.0058 / 0.0058`

Conclusion: prefix-1 top-1 route prediction is too weak to recover the oracle prefix-1 upper bound. Prefix-1 top-4 has high candidate coverage, especially on Game, but multi-route merging still does not improve Recall@50, so the current bottleneck is not memory capacity. The next useful step is to improve query/rerank binding inside predicted prefix buckets, or train/evaluate a prefix-1-specialized router before expanding prefix-2 soft routing.

## 2026-06-06 Latest Prefix-1 Query-Source Validation

Ran the current predicted-route eval with true prefix-1 candidate generation from `route1_log_probs`, prefix-1 beams `1/2/4`, prefix-2 beams `1/4/8`, and query sources `learned`, `pooled`, `residual`, and `mean`.

Latest output directories:

- `outputs/oracle_route_memory/predicted_route_p1_querysource_eval_20260606_214257` (`route_score_weight=1.0`)
- `outputs/oracle_route_memory/predicted_route_p1_querysource_eval_norouteweight_20260606_214335` (`route_score_weight=0.0`)

Cold route prediction:

- prefix-1 top-1/top-2/top-4: `0.1942 / 0.4290 / 0.7942`
- prefix-2 top-1/top-4/top-8: `0.0116 / 0.0638 / 0.1594`
- Book prefix-1 top-1/top-2/top-4: `0.1990 / 0.3560 / 0.6387`
- Game prefix-1 top-1/top-2/top-4: `0.1883 / 0.5195 / 0.9870`

Cold Recall@50 remains below the first target threshold `0.029`:

- `route_score_weight=1.0`: best prefix-1 result is learned `predicted_route_p1/p1_top2/p1_top4 = 0.0116`; pooled/residual/mean stay at `0.0058`.
- `route_score_weight=0.0`: best overall result is residual `predicted_route_p2_top8 = 0.0145`; prefix-1 learned remains `0.0116`.
- Domain view for prefix-1 top-4: Book reaches at most `0.0209` with learned query at route score `1.0`; Game is `0.0` there and only `0.0130` with learned query at route score `0.0`.

Yelp is absent from current router eval rows because the usable eval samples with item ids and route/embedding coverage are only Book and Game: cold eval `345 = Book 191 + Game 154`; all eval `399 = Book 193 + Game 206`. The source audit records Yelp rows, but row errors are dominated by `Yelp missing_item_id_in_task`.

Conclusion: prefix-1 top-4 has strong route candidate coverage, especially for Game, but retrieval still does not cross `0.029`. The immediate bottleneck is query/rerank binding inside broad predicted prefix buckets, not memory capacity. Prefix-2 soft routing should wait until prefix-1 retrieval improves.

## 2026-06-06 Prefix-1 Bucket Rerank Update

Implemented multi-bucket merge/rerank controls in `scripts/oracle_route_memory/eval_predicted_route.py`:

- `--merge-strategies score zscore round_robin quota rrf`
- `--per-route-topk`
- `--query-source domain_adaptive`
- `--domain-query-source Book=residual Game=learned`

Main output directories:

- `outputs/oracle_route_memory/prefix1_merge_eval_20260606_230406`
- `outputs/oracle_route_memory/prefix1_merge_grid_20260606_230538`
- `outputs/oracle_route_memory/prefix1_route_source_merge_eval_20260606_230918`
- `outputs/oracle_route_memory/prefix1_domain_adaptive_merge_eval_20260606_232037`

Results:

- Predicted prefix-1 top-4 with learned query and `zscore` merge improves cold Recall@50 from `0.0116` to `0.0261`, but still misses the first target `0.029`.
- A small grid over route score weight and per-route top-k did not improve beyond `0.0261`.
- Domain/history prefix-1 route sources have higher candidate-route recall in some cases, but did not improve final Recall@50; best stayed at `0.0261` from predicted prefix-1 top-4 + learned + `zscore`.
- Diagnostic domain-adaptive query binding (`Book=residual`, `Game=learned`) crosses the first target: predicted prefix-1 top-4 + `zscore` cold Recall@50 = `0.0319`.
- The same domain-adaptive run has Book Recall@50 = `0.0314`, Game Recall@50 = `0.0325`, CandidatePoolHitRate = `0.0377`, CandidatePoolLossRate = `0.0058`, and warm Recall@50 = `0.0741`.

Interpretation:

The first target is reachable without oracle routes and without changing memory: prefix-1 top-4 has enough route coverage, and the new bucket-normalized merge reduces cross-bucket score-scale bias. The remaining gap is query binding. The `domain_adaptive` result should be treated as a diagnostic proof of direction, not a finalized general method, because the Book/Game query-source mapping was selected from the observed domain split. The next step should convert this into a learned or validation-selected query selector/head before expanding prefix-2 soft routing.

## 2026-06-06 Validation-Selected Query Selector

Implemented a validation-selected query-source selector:

- `scripts/oracle_route_memory/select_validation_query_source.py`
- `eval_predicted_route.py --domain-query-source-config`

The selector uses the same warm-train examples and the same deterministic `seed=42`, `val_ratio=0.1` split as router training. It selects each domain's query source on held-out train validation, then writes `domain_query_source_config.json` for eval. It does not use cold/test rows to choose Book/Game mappings.

Main output directories:

- selector without query head: `outputs/oracle_route_memory/validation_query_selector_20260606_234048`
- cold eval using that selector: `outputs/oracle_route_memory/validation_query_selector_eval_20260606_234148`
- selector/eval with existing `Prefix1QueryHead` candidates: `outputs/oracle_route_memory/validation_query_selector_with_head_20260606_234243`

Validation selection without `Prefix1QueryHead`:

- selected mapping: `Book=residual`, `Game=residual`, default `residual`
- validation Recall@50: ALL `0.0558`, Book `0.0526`, Game `0.0581`
- cold predicted prefix-1 top-4 + `zscore` Recall@50: `0.0174`
- cold Book/Game Recall@50 for that mode does not reproduce the prior diagnostic `Game=learned` gain.

Validation selection with existing `Prefix1QueryHead` candidates:

- using `prefix1_query_head_train_20260606_231155`: selected `Book=prefix1_head`, `Game=residual`, default `prefix1_head`; cold predicted prefix-1 top-4 + `zscore` Recall@50 = `0.0000`.
- using `prefix1_query_head_train_20260606_231925_e30_h127`: selected `Book=prefix1_head`, `Game=residual`, default `prefix1_head`; cold predicted prefix-1 top-4 + `zscore` Recall@50 = `0.0058`.

Interpretation:

The validation-selected selector is now reproducible, but it does not yet pass the first target on cold evaluation. It selects `Game=residual` from warm validation, while the earlier cold diagnostic needed `Game=learned` to cross `0.029`; this is a train-validation to cold-domain mismatch, not a memory bottleneck. Existing `Prefix1QueryHead` checkpoints improve warm validation metrics but do not transfer to predicted-route cold retrieval. Do not expand prefix-2 soft routing yet; the next useful step is a selector/head trained and validated against a cold-like objective, or a more robust query fusion that does not hard-switch by domain.
