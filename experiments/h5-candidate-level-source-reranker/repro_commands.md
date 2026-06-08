# H5-D Repro Commands

Date: 2026-06-08

These commands reproduce the locked validation-selected H5-D policy candidate.
They use protocol-v3 train-derived validation data only. Do not run them against
the consumed blind-confirmation labels.

## 1. Export Candidate-Level Features

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/export_candidate_level_source_features.py \
  --data-dir user_simulator \
  --item-embedding-path outputs/oracle_route_memory/assets/metadata_embeddings/item_embeddings.npy \
  --item-sid-path outputs/oracle_route_memory/assets/proxy_routes_b16_d2/item_sid_mapping.json \
  --router-checkpoint-dir outputs/oracle_route_memory/predicted_route_v3_blind_confirmation_train_20260607 \
  --prefix1-query-head-checkpoint outputs/oracle_route_memory/prefix1_query_head_v3_blind_confirmation_train_20260607 \
  --protocol-manifest outputs/oracle_route_memory/official_protocol_v3_blind_confirmation/split_manifest.json \
  --query-sources learned residual mean prefix1_head \
  --prefix1-beam 4 \
  --per-route-topk 500 \
  --topk 50 \
  --output-dir outputs/oracle_route_memory/h5_candidate_level_source_features_k500_20260608
```

## 2. Train Book Component: h100/e20

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/train_candidate_level_source_ranker.py \
  --objective pairwise \
  --train-rows outputs/oracle_route_memory/h5_candidate_level_source_features_k500_20260608/train_candidate_rows.jsonl \
  --eval-rows outputs/oracle_route_memory/h5_candidate_level_source_features_k500_20260608/cold_like_candidate_rows.jsonl \
  --output-dir outputs/oracle_route_memory/h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h100_e20_20260608 \
  --topk 50 \
  --negatives-per-positive 500 \
  --hard-negatives-per-positive 100 \
  --pairwise-epochs 20 \
  --pairwise-batch-size 4096 \
  --pairwise-learning-rate 0.005 \
  --pairwise-weight-decay 0.0001 \
  --seed 42
```

## 3. Train Game Component: h300/e30

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/train_candidate_level_source_ranker.py \
  --objective pairwise \
  --train-rows outputs/oracle_route_memory/h5_candidate_level_source_features_k500_20260608/train_candidate_rows.jsonl \
  --eval-rows outputs/oracle_route_memory/h5_candidate_level_source_features_k500_20260608/cold_like_candidate_rows.jsonl \
  --output-dir outputs/oracle_route_memory/h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h300_e30_20260608 \
  --topk 50 \
  --negatives-per-positive 500 \
  --hard-negatives-per-positive 300 \
  --pairwise-epochs 30 \
  --pairwise-batch-size 4096 \
  --pairwise-learning-rate 0.005 \
  --pairwise-weight-decay 0.0001 \
  --seed 42
```

## 4. Combine by Domain

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/combine_ranker_outputs_by_domain.py \
  --source h100=outputs/oracle_route_memory/h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h100_e20_20260608/cold_like_outputs.json \
  --source h300=outputs/oracle_route_memory/h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h300_e30_20260608/cold_like_outputs.json \
  --domain-source Book=h100 \
  --domain-source Game=h300 \
  --default-source h300 \
  --output-dir outputs/oracle_route_memory/h5_pairwise_domain_routed_book_h100_game_h300_20260608 \
  --topk 50
```

## 5. Analyze

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/analyze_route_query_binding_errors.py \
  --selector-rows outputs/oracle_route_memory/h5_pairwise_domain_routed_book_h100_game_h300_20260608/cold_like_outputs.json \
  --output-dir outputs/oracle_route_memory/route_query_binding_error_analysis_h5_pairwise_domain_routed_book_h100_game_h300_20260608 \
  --top-k 50 \
  --group-by domain
```

## 6. Validate Locked Manifest

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/validate_locked_policy_manifest.py \
  --manifest experiments/h5-candidate-level-source-reranker/locked_policy_manifest.json \
  --repo-root . \
  --topk 50
```

Expected summary:

```json
{
  "status": "ok",
  "metric": {
    "sample_count": 136,
    "hits_at_50": 10,
    "Recall@50": 0.07352941176470588,
    "CandidatePoolHitRate": 0.9191176470588235
  }
}
```


## 7. Prepare Fresh-Confirmation Prep Bundle

This command validates the locked H5-D manifest again and copies only
validation-side documentation artifacts into a bundle. It does not read fresh or
consumed blind-confirmation labels.

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/prepare_h5_fresh_confirmation_bundle.py \
  --manifest experiments/h5-candidate-level-source-reranker/locked_policy_manifest.json \
  --repo-root . \
  --output-dir outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16 \
  --topk 50
```

Current bundle record:
`experiments/h5-candidate-level-source-reranker/fresh_confirmation_bundle.md`


## 8. Audit Fresh-Confirmation Prep Bundle

This command verifies the bundle hashes, checks source drift, and reruns the
locked manifest validator against the current repo outputs. It still uses only
validation-side artifacts.

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/audit_h5_fresh_confirmation_bundle.py \
  --bundle-dir outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16 \
  --repo-root . \
  --output-dir outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16_audit \
  --topk 50 \
  --rerun-validator \
  --fail-on-source-drift
```

Expected audit gate:

- `status=ok`
- no `errors`
- no `source_drift`
- rerun validator Recall@50 `0.07352941176470588`


## 9. Register Fresh Split Metadata

Run this only when a clearly fresh and unconsumed split manifest exists. The
registration step records the fresh split path and SHA-256, rejects the consumed
protocol-v3 split by path/hash, requires explicit fresh/unconsumed manifest
fields, and requires the current bundle audit to pass. Adjust the field names to
the actual fresh split schema if it differs from the placeholders below.

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/register_h5_fresh_split.py \
  --split-manifest path/to/fresh/split_manifest.json \
  --locked-policy-manifest experiments/h5-candidate-level-source-reranker/locked_policy_manifest.json \
  --bundle-audit outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16_audit \
  --repo-root . \
  --fresh-split-id fresh-confirmation-YYYYMMDD \
  --operator-note "fresh/unconsumed split confirmed before scoring" \
  --require-manifest-field fresh_status=fresh \
  --require-manifest-field consumed=false \
  --output-dir outputs/oracle_route_memory/h5_fresh_split_registration_YYYYMMDD
```

Expected registration gate:

- `status=ok`
- split manifest path is not the consumed protocol-v3 split path
- split manifest SHA-256 is not the consumed protocol-v3 split SHA-256
- current bundle audit has `status=ok`, no `errors`, and no `source_drift`
- required manifest fields prove the split is fresh/unconsumed


## 10. Score Registered Fresh Split with Locked Models

After registration succeeds and fresh candidate-level rows have been exported,
score them with the locked h100/h300 `model.pkl` files. Do not retrain after
fresh labels are available.

Export the fresh candidate-level rows with the same locked query sources,
prefix-1 beam, and per-route candidate depth used for validation. Replace only
the fresh split manifest path and the versioned output directory.

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/export_candidate_level_source_features.py \
  --data-dir user_simulator \
  --item-embedding-path outputs/oracle_route_memory/assets/metadata_embeddings/item_embeddings.npy \
  --item-sid-path outputs/oracle_route_memory/assets/proxy_routes_b16_d2/item_sid_mapping.json \
  --router-checkpoint-dir outputs/oracle_route_memory/predicted_route_v3_blind_confirmation_train_20260607 \
  --prefix1-query-head-checkpoint outputs/oracle_route_memory/prefix1_query_head_v3_blind_confirmation_train_20260607 \
  --protocol-manifest path/to/fresh/split_manifest.json \
  --query-sources learned residual mean prefix1_head \
  --prefix1-beam 4 \
  --per-route-topk 500 \
  --topk 50 \
  --output-dir outputs/oracle_route_memory/h5_fresh_candidate_level_source_features_k500_YYYYMMDD
```

Expected fresh export gate:

- query sources are exactly `learned residual mean prefix1_head`
- `prefix1_beam=4`
- `per_route_topk=500`
- no H5-D model, seed, hard-negative, route, or domain-routing parameter is
  changed after fresh labels are available

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/score_candidate_level_source_ranker.py \
  --model outputs/oracle_route_memory/h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h100_e20_20260608/model.pkl \
  --eval-rows outputs/oracle_route_memory/h5_fresh_candidate_level_source_features_k500_YYYYMMDD/cold_like_candidate_rows.jsonl \
  --output-dir outputs/oracle_route_memory/h5_fresh_loaded_score_h100_e20_YYYYMMDD \
  --topk 50

/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/score_candidate_level_source_ranker.py \
  --model outputs/oracle_route_memory/h5_candidate_level_source_ranker_pairwise_linear_k500_n500_h300_e30_20260608/model.pkl \
  --eval-rows outputs/oracle_route_memory/h5_fresh_candidate_level_source_features_k500_YYYYMMDD/cold_like_candidate_rows.jsonl \
  --output-dir outputs/oracle_route_memory/h5_fresh_loaded_score_h300_e30_YYYYMMDD \
  --topk 50

/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/combine_ranker_outputs_by_domain.py \
  --source h100=outputs/oracle_route_memory/h5_fresh_loaded_score_h100_e20_YYYYMMDD/scored_outputs.json \
  --source h300=outputs/oracle_route_memory/h5_fresh_loaded_score_h300_e30_YYYYMMDD/scored_outputs.json \
  --domain-source Book=h100 \
  --domain-source Game=h300 \
  --default-source h300 \
  --output-dir outputs/oracle_route_memory/h5_fresh_loaded_score_domain_routed_YYYYMMDD \
  --topk 50
```

Validation replay sanity check already completed:

- h100 loaded-model replay Recall@50 `0.04411764705882353`
- h300 loaded-model replay Recall@50 `0.051470588235294115`
- domain-routed loaded-model replay Recall@50 `0.07352941176470588`


## 11. Validate Loaded-Model Replay

This validation proves that loaded `model.pkl` scoring reconstructs the locked
H5-D validation output at the row/metric level without retraining.

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/validate_h5_loaded_model_replay.py \
  --locked-outputs outputs/oracle_route_memory/h5_pairwise_domain_routed_book_h100_game_h300_20260608/cold_like_outputs.json \
  --replay-outputs outputs/oracle_route_memory/h5_loaded_model_score_domain_routed_validation_replay_20260608/cold_like_outputs.json \
  --output-dir outputs/oracle_route_memory/h5_loaded_model_replay_validation_20260608 \
  --topk 50
```

Expected replay validation gate:

- `status=ok`
- `mismatch_count=0`
- `metric_errors=[]`
- Recall@50 `0.07352941176470588`


## 12. Check Fresh Readiness

This command summarizes the locked manifest validator, prep-bundle audit,
loaded-model replay validation, and locked component model file checks.

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/check_h5_fresh_readiness.py \
  --manifest experiments/h5-candidate-level-source-reranker/locked_policy_manifest.json \
  --bundle-audit outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16_audit/bundle_audit.json \
  --loaded-model-replay-validation outputs/oracle_route_memory/h5_loaded_model_replay_validation_20260608/loaded_model_replay_validation.json \
  --repo-root . \
  --output-dir outputs/oracle_route_memory/h5_fresh_readiness_20260608_v16 \
  --topk 50
```

Expected readiness gate:

- `status=ok`
- locked manifest validator Recall@50 `0.07352941176470588`
- bundle audit `status=ok` and source drift count `0`
- loaded-model replay validation `status=ok` and mismatch count `0`
- h100/h300 component `model.pkl` files exist


## 13. Validate Handoff Index

This command verifies that the structured handoff index points to existing
current artifacts, the audit/readiness gates are ok, and the key handoff docs
mention the same current paths.

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/validate_h5_handoff_index.py \
  --handoff-index experiments/h5-candidate-level-source-reranker/handoff_index.json \
  --repo-root . \
  --output-dir outputs/oracle_route_memory/h5_handoff_index_validation_20260608_v16 \
  --topk 50
```

Expected index gate:

- `status=ok`
- bundle audit source drift count `0`
- all required prep-bundle artifacts are included
- fresh readiness status `ok`
- all configured document checks are `ok`


## 14. Render Fresh-Confirmation Report

Run this only after a fresh split has passed registration/readiness, h100 and
h300 locked-model scoring, and locked domain routing. The report renderer does
not retrain and does not merge validation metrics with fresh metrics.

```bash
/home/grf/.conda/envs/gdpo/bin/python3 scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py \
  --locked-policy-manifest experiments/h5-candidate-level-source-reranker/locked_policy_manifest.json \
  --fresh-split-registration outputs/oracle_route_memory/h5_fresh_split_registration_YYYYMMDD/fresh_split_registration.json \
  --fresh-readiness outputs/oracle_route_memory/h5_fresh_readiness_20260608_v16/fresh_readiness.json \
  --fresh-domain-routed-outputs outputs/oracle_route_memory/h5_fresh_loaded_score_domain_routed_YYYYMMDD/cold_like_outputs.json \
  --repo-root . \
  --output-dir outputs/oracle_route_memory/h5_fresh_confirmation_report_YYYYMMDD \
  --topk 50
```

Expected report gate:

- `status=ok`
- registration status `ok`
- readiness status `ok`
- `fresh_confirmation_report.md` has separate locked-validation and fresh
  confirmation metric sections
- `fresh_domain_routed_outputs.sha256` records the exact scored rows
