# Oracle Route + Memory

## Current H5-D Status

Date: 2026-06-08

The active validation-selected policy is now the H5-D candidate-level source
reranker:

- policy: `h5_pairwise_domain_routed_book_h100_game_h300`
- locked validation Recall@50: `0.07352941176470588` (`10/136`)
- Book: `3/65`
- Game: `7/71`
- claim boundary: validation-only, not a fresh blind-confirmation result

Current handoff and reproduction entry points:

- `experiments/h5-candidate-level-source-reranker/README.md`
- `experiments/h5-candidate-level-source-reranker/repro_commands.md`
- `experiments/h5-candidate-level-source-reranker/handoff_index.json`
- `to_human/h5_fresh_confirmation_handoff_summary_20260608.md`
- `findings.md`

Current v16 gates:

- prep bundle:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16`
- bundle audit:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16_audit`
- readiness:
  `outputs/oracle_route_memory/h5_fresh_readiness_20260608_v16`
- handoff index validation:
  `outputs/oracle_route_memory/h5_handoff_index_validation_20260608_v16`

Next target: keep the locked H5-D manifest and v16 gates unchanged until a
clearly fresh/unconsumed split exists. Then register the split, export fresh
candidate-level features with the locked query sources/beam/depth, score with
locked h100/h300 `model.pkl` files, apply the locked domain route, and render a
fresh-confirmation report that separates validation and fresh metrics.

The sections below describe the original route-memory pivot and remain useful
background for the H5-D line.

## Idea

This branch tests a pivot away from full SID generation for cold-item recommendation.

Original route:

`History -> full SID -> item`

Hypothesis problem:

A cold item's full SID pattern may never appear during training, so the model is biased toward seen SID patterns.

New route:

`History -> route -> dynamic item memory -> item`

Where:

- `route` is the first `prefix_len` SID units.
- `dynamic item memory` is a retrieval bank over current catalog item embeddings.
- `oracle_route` uses the target item's true SID prefix as an upper bound.

The main question is:

If the route is correct, can metadata memory retrieve the cold target item?

## Why We Pivoted Back From CritiqueScope / GIMO

CritiqueScope / GIMO / inference-time reranking produced useful diagnostics and artifacts, but that line is frozen for now as a negative or inconclusive finding. This round does not continue GIMO tuning, CritiqueScope iteration, GenRecEdit editing, or new TIGER training. The focus is now a cleaner upper-bound study for route-conditioned memory binding.

## Modes

- `metadata`: no route restriction. The query is the mean embedding of the last `history_len` history items and retrieval is global.
- `oracle_route`: uses the target item's true SID prefix and only searches that route bucket.
- `predicted_route`: optional. Uses a provided predicted route file if available.

The important comparison is `metadata` versus `oracle_route`.

## Smoke Run

```bash
python3 run_oracle_route_memory_eval.py \
  --data_dir user_simulator \
  --item_embedding_path <ITEM_EMB_PATH> \
  --item_sid_path <ITEM_SID_PATH> \
  --mode all \
  --prefix_len 1 \
  --cold_only true \
  --max_eval_samples 1000
```

Prefix length 2:

```bash
python3 run_oracle_route_memory_eval.py \
  --data_dir user_simulator \
  --item_embedding_path <ITEM_EMB_PATH> \
  --item_sid_path <ITEM_SID_PATH> \
  --mode all \
  --prefix_len 2 \
  --cold_only true \
  --max_eval_samples 1000
```

You can also run both prefixes in one call:

```bash
python3 run_oracle_route_memory_eval.py \
  --data_dir user_simulator \
  --item_embedding_path <ITEM_EMB_PATH> \
  --item_sid_path <ITEM_SID_PATH> \
  --mode all \
  --prefix_len 1 2 \
  --cold_only true \
  --max_eval_samples 1000
```

## Full Eval

```bash
python3 run_oracle_route_memory_eval.py \
  --data_dir user_simulator \
  --item_embedding_path <ITEM_EMB_PATH> \
  --item_sid_path <ITEM_SID_PATH> \
  --mode all \
  --prefix_len 1 2 \
  --cold_only true \
  --history_len 5 \
  --topk 10 20 50
```

## How To Read Results

The script writes:

- `eval_results.json`
- `eval_results.md`
- `per_sample_results.jsonl`
- `memory_stats.json`
- `route_stats.csv`

Focus on the table with:

- `metadata`
- `oracle_route` with `prefix_len=1`
- `oracle_route` with `prefix_len=2`

## Success Signal

If `oracle_route` Recall@50 is clearly higher than `metadata` Recall@50, then route-conditioned memory binding has positive signal.

## Failure Signal

If `oracle_route` stays close to `metadata`, route conditioning gives limited gain. If `oracle_route` is still very low, item/user embeddings are likely too weak and the memory-binding upper bound is weak.

## Optional Predicted Route Reuse

If existing TIGER checkpoints or `predictions.jsonl` exports exist, they can later be wired into `--predicted_routes_path`. This code path is intentionally optional. Oracle-route evaluation must run without checkpoints as long as interactions, item embeddings, and item SID mappings are available.

## Next Step If Oracle Route Has Signal

Train a route predictor and a better query head, then test `predicted_route + memory`.

## Next Step If Oracle Route Has No Signal

Improve item/user representations first, because full SID generation is unlikely to be the main bottleneck.
