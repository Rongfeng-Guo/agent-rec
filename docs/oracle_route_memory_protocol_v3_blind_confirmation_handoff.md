# Oracle Route Memory Protocol v3 Blind Confirmation Handoff

This handoff records the current item-level blind confirmation result for the oracle-route-memory line. It is separate from the protocol-v2 development result: v2 remains a development result, while v3 is the first blind-confirmation bundle in the current repo state.

## Artifacts

- Paper bundle: `outputs/oracle_route_memory/paper_ready_protocol_v3_blind_confirmation`
- Protocol split and lock: `outputs/oracle_route_memory/official_protocol_v3_blind_confirmation`
- Locked selected-policy eval: `outputs/oracle_route_memory/confirmation_eval_v3_blind_confirmation_20260607`
- Comparison-only eval: `outputs/oracle_route_memory/confirmation_eval_v3_blind_confirmation_comparison_20260607`
- Selector: `outputs/oracle_route_memory/validation_fusion_selector_v3_blind_confirmation_20260607`
- Builder: `scripts/oracle_route_memory/build_protocol_v3_confirmation_bundle.py`
- Future validation-only candidate preset: `v3_validation_rrf_candidate` in `scripts/oracle_route_memory/select_validation_fusion_policy_explicit.py`
- Validation-only dry run for that preset: `outputs/oracle_route_memory/validation_fusion_selector_v3_rrf_candidate_dryrun_20260607`
- Lock-builder dry run for that selector: `outputs/oracle_route_memory/confirmation_eval_lock_v3_rrf_candidate_dryrun_20260607`
- Route-source ablation report: `outputs/oracle_route_memory/ablation_reports/v3_route_source_ablation_20260608.md`

## Frozen Identifiers

- `split_hash`: `441f8741d2c1a0e229656dbda8d587dc62a3e4b388230914b69f457597c26e99`
- `config_hash`: `23333f630a251cf0b0d4fa779db276692727496a78903328ecbf78f72336f700`
- `lock_hash`: `3702cbff0c99f45a50cf53062f6b040bdf1ec3b642171c64d0c403e5204f8276`
- `official_comparison_hash`: `abd0178e5b3d333ac7d624730407da502a9849f94bafe96ca81d2e2c07004ba4`
- Bootstrap: `10000` repetitions, seed `20260607`

## Claim Boundary

Claimable selected method:

- `Predicted Route Validation-Selected Fusion`
- Locked before blind confirmation
- Selected only on `cold_like_validation`

Not claimable as the selected method:

- `Predicted Prefix-1 Top-1 Single Query`
- `Predicted Prefix-1 Top-4 Single Query`
- Any row from the comparison-only eval that was inspected after blind confirmation
- Oracle prefix rows, which are upper bounds

The comparison-only eval currently contains stronger blind rows, including `fusion_comparison_rrf` at `Recall@50 = 0.11864406779661017` and `domain_adaptive/domain_prior_p1_top4` at the same Recall@50. These are useful diagnostics, but they are not the locked selected claim for this blind run.

## Blind Confirmation ALL Readout

| method | level | n | Recall@50 | NDCG@50 | MRR |
|---|---|---:|---:|---:|---:|
| Popularity | blind_confirmation_result | 118 | 0.000000 | 0.000000 | 0.000433 |
| Metadata Global Mean Query | blind_confirmation_result | 118 | 0.093220 | 0.022836 | 0.007723 |
| Metadata Global Best Non-Route Query | blind_confirmation_result | 118 | 0.093220 | 0.027670 | 0.012389 |
| Dynamic Memory Without Route | blind_confirmation_result | 118 | 0.093220 | 0.022836 | 0.007723 |
| Random Matched-Size Bucket | blind_confirmation_result | 118 | 0.008475 | 0.001727 | 0.000292 |
| Predicted Prefix-1 Top-1 Single Query | diagnostic_result | 118 | 0.110169 | 0.034695 | 0.016565 |
| Predicted Prefix-1 Top-4 Single Query | diagnostic_result | 118 | 0.110169 | 0.034695 | 0.016565 |
| Predicted Route Validation-Selected Fusion | blind_confirmation_result | 118 | 0.084746 | 0.030022 | 0.015964 |
| Oracle Prefix-1 Route | oracle_upper_bound | 118 | 0.127119 | 0.033940 | 0.015038 |
| Oracle Prefix-2 Route | oracle_upper_bound | 118 | 0.822034 | 0.241413 | 0.105472 |

## Bootstrap Interpretation

Paired bootstrap on blind confirmation shows:

- Selected fusion vs metadata mean: delta `-0.008475`, 95% CI `[-0.067797, 0.050847]`, crosses zero.
- Selected fusion vs best non-route metadata: delta `-0.008475`, 95% CI `[-0.076271, 0.059322]`, crosses zero.
- Selected fusion vs random matched-size bucket: delta `0.076271`, 95% CI `[0.025424, 0.135593]`, does not cross zero.
- Selected fusion vs predicted prefix-1 top-4 single query: delta `-0.025424`, 95% CI `[-0.059322, 0.000000]`, crosses zero.

Current claim-safe interpretation:

- The locked selected fusion beats a random bucket with the same candidate-pool sizes.
- It does not beat the strongest no-route metadata baseline on this blind confirmation bundle.
- The oracle upper bounds remain much higher, so the bottleneck is still route/query binding rather than candidate-memory capacity.

## Verification Commands

Last verified through the `172.31.226.176` jump host on `172.31.233.184`:

```bash
source /home/grf/external_baselines/venv_rpg/bin/activate
python -m pytest tests/test_late_bound_router.py tests/test_closed_loop_pipeline_status.py tests/test_blind_confirmation_protocol.py tests/test_explicit_validation_fusion_selector.py tests/test_official_protocol.py tests/test_locked_predicted_route_eval.py tests/test_build_predicted_route_progress_report.py -q
python3 -m compileall genrec/models genrec/training scripts/oracle_route_memory user_simulator/evaluation
git diff --check
```

Observed result: `31 passed`, compileall passed, `git diff --check` passed, and no trailing whitespace was found in the touched v3 builder/test/README files.

## Next Research Step

Do not select a new policy on this consumed blind confirmation set. The next claimable improvement needs a policy pre-registered before a new blind confirmation run. The most promising diagnostic direction is to turn the comparison-only `fusion_comparison_rrf` / `domain_prior_p1_top4` behavior into a locked selector on validation data, then evaluate it on a fresh confirmation split.

The repo now has a validation-only preset for that next step:

```bash
python scripts/oracle_route_memory/select_validation_fusion_policy_explicit.py \
  --preset v3_validation_rrf_candidate \
  --protocol-manifest <fresh_protocol_dir>/split_manifest.json \
  --output-dir <fresh_selector_dir> \
  --data-dir user_simulator \
  --item-embedding-path outputs/oracle_route_memory/assets/metadata_embeddings/item_embeddings.npy \
  --item-sid-path outputs/oracle_route_memory/assets/proxy_routes_b16_d2/item_sid_mapping.json \
  --checkpoint-dir <router_checkpoint_dir> \
  --prefix1-query-head-checkpoint <prefix1_query_head_checkpoint_dir>
```

Use the resulting `fusion_config.json` to build a confirmation eval lock before reading any fresh blind-confirmation labels. Do not use this preset to relabel the already consumed `paper_ready_protocol_v3_blind_confirmation` result.

On the existing consumed v3 protocol, this preset was run only as a selector rehearsal. It selected `domain_adaptive_domain_prior_p1_top4` on `cold_like_validation`, not `fusion_comparison_rrf`:

| split | policy | Recall@50 | CandidatePoolHitRate | RouteHitRate |
|---|---|---:|---:|---:|
| cold_like_validation | `domain_adaptive_domain_prior_p1_top4` | 0.044118 | 0.154412 | 0.860294 |
| cold_like_validation | `domain_adaptive_predicted_route_p1_top4` | 0.036765 | 0.154412 | 0.941176 |
| cold_like_validation | `fusion_comparison_rrf` | 0.022059 | 0.036765 | 0.000000 |
| cold_like_validation | `mean_predicted_route_p1_top4` | 0.022059 | 0.125000 | 0.941176 |

Treat this dry run as a pipeline check only. It does not change the claim boundary for the already consumed blind-confirmation result.

The dry-run selector was also passed through the confirmation lock builder to verify the next pipeline stage:

- `confirmation_eval_consumed`: `false`
- Dry-run lock hash: `d7b02bf8e4b50ba16ff9b0bf201fd7bdaa7179c7a557925e778d5990c1a719b0`
- Fusion config hash: `7ce82e0c936886292ae48d124434ecc341f47af3f7110024229e2b0cb574a1ad`
- Selector rows hash: `bf96476fdc3325c912206196be9a74ed22b028c949783970a447b72a11570196`

This lock is a consumed-protocol rehearsal artifact. For a claimable result, create a fresh split, run the selector before blind labels are read, build a fresh lock, and only then run confirmation eval.

The lock builder now refuses to write into a non-empty output directory by default. This protects one-shot lock artifacts from accidental overwrite. Use `--force` only for an intentional non-claim rerun, and never to rewrite a claimable fresh blind lock after metrics have been read.

An additional validation-only route-source ablation was run on 2026-06-08. It selected `predicted_route_p1` on `cold_like_validation` with `Recall@50 = 0.051471`, while `fusion_predicted_mean_rrf` was best on `warm_validation` with `Recall@50 = 0.123288`. This reinforces that the next selector should optimize cold-like validation first and treat warm performance as a retention check.
