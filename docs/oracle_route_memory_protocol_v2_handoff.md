# Oracle Route Memory Protocol V2 Handoff

## Research Question

Can route-conditioned dynamic item memory improve cold-start item retrieval over global no-route metadata retrieval under the frozen protocol v2 bundle?

## Problem Framing

The weak formulation is `history -> full SID -> item`. It is brittle for cold items because full structured identifiers are sparse, exact, and hard to generate for items that have little or no interaction history. A model can learn frequent structural patterns, but a full-SID target still forces a long exact item-level decision where cold coverage is poor.

The current formulation is `history -> route -> dynamic item memory -> item`. The router predicts a coarse route, then retrieval is performed inside the route-conditioned item memory. This separates semantic routing from final item ranking and lets the method use metadata-backed candidates instead of requiring full-SID generation.

## Oracle Route Upper Bounds

Oracle prefix-1 and prefix-2 rows inject the target route prefix and measure how much headroom exists when routing is correct. These rows are diagnostic upper bounds. They are not claimable baselines because they use target-derived route information at evaluation time.

## Frozen Predicted Method

The claimable predicted-route method in the bundle is:

`Predicted Route Validation-Selected (Explicit Script OldV0)`

It uses the explicit selector preset `explicit_script_oldv0`, selected on train-derived validation evidence and then evaluated with the locked `fusion_config.json`. The selected configuration must keep `route_score_weight=0.0`; this is part of the frozen method because route score weighting was not the validated setting for the preserved result.

The frozen cold ALL result is:

`Recall@50 = 0.04927536231884058`

This value is a frozen development-test result. The current cold split has been touched during development history, so it should not be described as an untouched external holdout confirmation.

## Random Matched-Size Bucket

`Random Matched-Size Bucket` isolates the value of candidate-pool size reduction from semantic route quality. For each selected-policy row, it samples a uniform random candidate bucket with the same size as that row's selected-policy candidate pool. It uses a deterministic per-sample seed derived from `sha256(f"{seed}:{sample_id}")` and Python `random.Random(...).sample(...)`, matching the manual paper bundle.

Because `sample_id` is duplicated in the per-sample artifacts, the native builder processes selected rows in their original order and reads `target_item_id`, `candidate_pool_size`, and `match_rank` from each row. It does not build `dict[sample_id] = row`, which would silently overwrite duplicates and can corrupt row counts or bootstrap pairing.

## Reproduction

The native paper bundle can be rebuilt without `--reference-bundle-dir`:

```bash
cd /home/grf/agent-rec
source /home/grf/external_baselines/venv_rpg/bin/activate

python3 scripts/oracle_route_memory/build_protocol_v2_bundle.py \
  --output-dir outputs/oracle_route_memory/paper_ready_protocol_v2_native_20260607 \
  --protocol-dir outputs/oracle_route_memory/official_protocol_v2 \
  --selector-dir outputs/oracle_route_memory/validation_fusion_selector_explicit_script_oldv0_rerun_20260607 \
  --locked-eval-dir outputs/oracle_route_memory/validation_fusion_locked_cold_explicit_script_oldv0_rerun_20260607 \
  --eval-smoke-dir outputs/oracle_route_memory/eval_smoke_20260606_072300 \
  --progress-report-md outputs/oracle_route_memory/progress_report_20260607_explicit_script_oldv0/report.md
```

The bundle writes `official_comparison.csv`, `bootstrap_comparison.csv`, `latency_summary.csv`, route/candidate breakdowns, README, reproduction script, and run metadata from repo code. `--reference-bundle-dir` remains available only as an optional compatibility fallback.

## Claim Status

Claimable rows:

- `Predicted Route Validation-Selected (Explicit Script OldV0)`
- `Metadata Global Mean Query`
- `Popularity`
- `Random Matched-Size Bucket`

Diagnostic-only rows:

- `Oracle Prefix-1 Route`
- `Oracle Prefix-2 Route`
- Existing routed single-query references imported from checked summary artifacts

## Current Limits

- An untouched confirmation split is still missing; this should be the next step before making a stronger heldout claim.
- Yelp coverage is still incomplete and should be repaired before cross-domain generalization claims.
- External-dataset confirmation is still missing.
- A full-SID baseline under the exact same protocol is still unavailable and should not be fabricated from historical rows.
