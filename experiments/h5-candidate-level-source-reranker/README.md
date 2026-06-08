# H5-D Candidate-Level Source Reranker

Date: 2026-06-08

This directory contains the locked H5-D validation-selected policy and the
handoff material for a future fresh/unconsumed confirmation split.

## Current Handoff

- Prep bundle:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16`
- Prep bundle audit:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16_audit`
- Fresh readiness:
  `outputs/oracle_route_memory/h5_fresh_readiness_20260608_v16`
- Human summary:
  `to_human/h5_fresh_confirmation_handoff_summary_20260608.md`

## Key Files

- `locked_policy_manifest.json`: locked H5-D validation-selected policy.
- `handoff_index.json`: structured current handoff paths and required document
  mentions.
- `validation_comparison.md`: H3/H4/H5 validation comparison table.
- `repro_commands.md`: exact commands for validation replay, audit, readiness,
  future split registration, and locked-model scoring.
- `fresh_confirmation_checklist.md`: ordered future fresh-confirmation gates.
- `fresh_confirmation_bundle.md`: current prep-bundle and readiness record.
- `protocol.md`: original H5 experiment rationale and validation history.
- `scripts/oracle_route_memory/validate_h5_handoff_index.py`: consistency gate
  for the handoff index, current artifacts, prep-bundle artifact coverage, and
  document mentions.
- `scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py`: future
  fresh-result report renderer after registration, readiness, and locked-model
  scoring pass.

## Current Locked Validation Metric

- sample_count: `136`
- hits_at_50: `10`
- Recall@50: `0.07352941176470588`
- CandidatePoolHitRate: `0.9191176470588235`
- Book: `3/65`
- Game: `7/71`

## Claim Boundary

This is validation-only. Do not report H5-D as a fresh blind-confirmation claim
until a clearly fresh/unconsumed split is registered, scored with locked
`model.pkl` files without retraining, and reported separately from validation
metrics.

## Next Target

Current engineering target: keep the v16 handoff gates and handoff-index
validator clean, then use
`scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py` after a
clearly fresh/unconsumed split has passed registration, readiness, locked-model
scoring, and domain routing. The rendered report must keep locked validation
metrics separate from fresh confirmation metrics.
