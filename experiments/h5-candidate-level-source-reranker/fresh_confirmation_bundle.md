# H5-D Fresh Confirmation Prep Bundle

Date: 2026-06-08

Scope: validation-only preparation for a future fresh confirmation run. This is
not a blind-confirmation result and does not read or reinterpret the consumed
2026-06-07 protocol-v3 blind-confirmation labels.

## Current Bundle

- Bundle directory:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16`
- Bundle manifest:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16/bundle_manifest.json`
- Validator output:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16/validator_output.json`
- README:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16/README.md`
- Bundle audit:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16_audit/bundle_audit.json`
- Handoff summary:
  `to_human/h5_fresh_confirmation_handoff_summary_20260608.md`

## Locked Validation Check

Expected validator result for the locked H5-D manifest:

- sample_count: `136`
- hits_at_50: `10`
- Recall@50: `0.07352941176470588`
- CandidatePoolHitRate: `0.9191176470588235`
- Book: `3/65`
- Game: `7/71`

## Included Artifact Types

The prep bundle copies the experiment README, locked manifest, handoff index,
repro commands, validation comparison, fresh-confirmation checklist, bundle
record, paper-facing validation summary, handoff summary,
`research-state.yaml`, and `research-log.md`. Each copied file is recorded with a
SHA-256 hash in `bundle_manifest.json`.

## Audit Gate

The current audit command uses `--fail-on-source-drift`. It must report
`status=ok`, no `errors`, no `source_drift`, and
a rerun validator Recall@50 of `0.07352941176470588` before the bundle is used
as a handoff package. The handoff index validator also checks that
`bundle_manifest.json` includes every required artifact listed in
`handoff_index.json`.

## Loaded-Model Validation Replay

The locked h100/h300 `model.pkl` files have been replayed on the validation
candidate rows without retraining:

- h100 replay:
  `outputs/oracle_route_memory/h5_loaded_model_score_h100_e20_validation_replay_20260608`
- h300 replay:
  `outputs/oracle_route_memory/h5_loaded_model_score_h300_e30_validation_replay_20260608`
- domain-routed replay:
  `outputs/oracle_route_memory/h5_loaded_model_score_domain_routed_validation_replay_20260608`
- domain-routed replay Recall@50: `0.07352941176470588`

## Loaded-Model Replay Validation

The replay validator compares locked H5-D validation rows with loaded-model
replay rows on sample-level fields and metrics:

- validation report:
  `outputs/oracle_route_memory/h5_loaded_model_replay_validation_20260608`
- status: `ok`
- mismatch_count: `0`
- metric_errors: `[]`

## Fresh Readiness Check

The combined readiness check passed:

- readiness report:
  `outputs/oracle_route_memory/h5_fresh_readiness_20260608_v16`
- status: `ok`
- bundle_audit_source_drift_count: `0`
- loaded_model_replay_mismatch_count: `0`
- component model files: h100/h300 `model.pkl` present

## Next Target

Keep the H5-D manifest and v16 prep bundle unchanged until a clearly fresh and
unconsumed confirmation split exists. When that split exists, use
`scripts/oracle_route_memory/register_h5_fresh_split.py` to record its manifest
path and SHA-256 after the v16 audit passes. The registration command must use
`--require-manifest-field` checks that prove the split is explicitly marked
fresh/unconsumed. Only then score the fresh candidate rows with the locked
h100/h300 `model.pkl` files via
`scripts/oracle_route_memory/score_candidate_level_source_ranker.py`, without
changing model, seed, hard-negative, route, or domain-routing parameters after
seeing fresh labels. After domain routing, render the final gated report with
`scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py` and keep
validation metrics separate from fresh confirmation metrics. If any handoff doc
or bundle path changes, update `handoff_index.json` and rerun
`scripts/oracle_route_memory/validate_h5_handoff_index.py`.
