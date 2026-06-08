# H5-D Fresh Confirmation Handoff Summary

Date: 2026-06-08

Scope: validation-only handoff for a future fresh/unconsumed confirmation split.
This is not a blind-confirmation result and must not be reported as a fresh
claim until evaluated on a clearly new split after the locked policy and gates
below remain unchanged.

## Current Handoff Artifacts

- Prep bundle:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16`
- Prep bundle audit:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16_audit`
- Fresh readiness:
  `outputs/oracle_route_memory/h5_fresh_readiness_20260608_v16`
- Locked policy manifest:
  `experiments/h5-candidate-level-source-reranker/locked_policy_manifest.json`
- Repro commands:
  `experiments/h5-candidate-level-source-reranker/repro_commands.md`

## Locked Validation Result

- Policy: `h5_pairwise_domain_routed_book_h100_game_h300`
- Selection split: protocol-v3 train-derived cold-like validation
- sample_count: `136`
- hits_at_50: `10`
- Recall@50: `0.07352941176470588`
- CandidatePoolHitRate: `0.9191176470588235`
- Book: `3/65`
- Game: `7/71`

## Gate Status

- Locked manifest validator: `ok`
- Prep bundle audit: `ok`, no `errors`, no `source_drift`
- Loaded-model replay validation: `ok`, mismatch_count `0`, metric_errors `[]`
- Fresh readiness check: `ok`
- h100/h300 component `model.pkl` files: present

## Fresh Split Requirements

Before scoring any future split:

1. Confirm the split is fresh and unconsumed.
2. Register it with `scripts/oracle_route_memory/register_h5_fresh_split.py`.
3. Require manifest fields that explicitly prove fresh/unconsumed status via
   `--require-manifest-field`.
4. Reject any split matching the consumed protocol-v3 split path or SHA-256.
5. Score candidate rows with locked h100/h300 `model.pkl` files via
   `scripts/oracle_route_memory/score_candidate_level_source_ranker.py`.
6. Do not retrain or change model, seed, hard-negative, route, or domain-routing
   parameters after fresh labels are available.
7. Keep `scripts/oracle_route_memory/validate_h5_handoff_index.py` at
   `status=ok` whenever the handoff bundle or docs change; this includes
   checking prep-bundle artifact coverage.
8. Render `fresh_confirmation_report.json` and `.md` with
   `scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py`; report
   fresh metrics separately from the locked validation metric.

## Claim Boundary

The H5-D validation result is useful for selection and preparation only. A
fresh claim requires a clearly new confirmation split and a report that separates
locked validation metrics from fresh confirmation metrics.
