# H5-D Fresh Confirmation Checklist

Date: 2026-06-08

This checklist is for a future fresh blind-confirmation split. It must not be
run on the consumed 2026-06-07 protocol-v3 blind-confirmation labels.

## Preconditions

- H5-D locked manifest remains unchanged:
  `experiments/h5-candidate-level-source-reranker/locked_policy_manifest.json`
- Manifest validator passes on validation outputs before any fresh run.
- Current prep bundle: `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16`.
- Current prep bundle audit: `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16_audit`.
- Fresh readiness report passes: `outputs/oracle_route_memory/h5_fresh_readiness_20260608_v16`.
- Handoff index validation passes with `status=ok`.
- A new split manifest exists and is explicitly marked fresh/unconsumed.
- No model, route, hard-negative, seed, or domain-routing parameter is changed
  after seeing fresh labels.

## Execution Order

1. Create the validation-only prep bundle with
   `scripts/oracle_route_memory/prepare_h5_fresh_confirmation_bundle.py`; keep
   its copied locked manifest, validator output, and bundle manifest unchanged.
2. Audit the prep bundle before recording the fresh split; the audit must
   report `status=ok`, no `errors`, and no `source_drift`. Run the readiness
   checker and require `status=ok` before fresh scoring.
3. Register the fresh split metadata with
   `scripts/oracle_route_memory/register_h5_fresh_split.py`; it must reject the
   consumed protocol-v3 split by path/hash, verify explicit fresh/unconsumed
   fields via `--require-manifest-field`, and write the fresh split SHA-256.
4. Rebuild candidate-level features for the fresh split with
   `scripts/oracle_route_memory/export_candidate_level_source_features.py`,
   using the same query sources, route beam, and per-route topk from the locked
   manifest. Use the fresh export template in `repro_commands.md` and write a
   versioned `h5_fresh_candidate_level_source_features_k500_YYYYMMDD` output.
5. Score the fresh split with locked h100/h300 `model.pkl` files via
   `scripts/oracle_route_memory/score_candidate_level_source_ranker.py`; do not
   retrain after fresh labels are available. Before any fresh run, keep the
   validation loaded-model replay validator at `status=ok` with zero mismatches.
6. Apply the locked domain routing rule: Book -> h100, Game -> h300, default ->
   h300.
7. Run the route/query-binding analyzer grouped by domain.
8. Render a fresh-confirmation report with
   `scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py`; it must
   report `status=ok` and separate:
   - locked validation metrics;
   - fresh blind-confirmation metrics;
   - oracle diagnostics, if any, clearly marked as diagnostics.

## Stop Conditions

- Stop immediately if the fresh split is not clearly separate from consumed
  protocol-v3 blind-confirmation data.
- Stop if reproducing the locked validation manifest fails.
- Stop if any H5-D hyperparameter or routing rule is changed after fresh labels
  are available.

## Required Artifacts

- locked manifest copy;
- manifest validator JSON output;
- fresh split registration JSON/MD with manifest path, SHA-256, and required fresh/unconsumed field checks;
- fresh candidate rows, export summary, and loaded-model scored rows;
- domain-routed fresh outputs;
- loaded-model replay validation report;
- fresh readiness report;
- analyzer output;
- final `fresh_confirmation_report.json` and `.md` with claim boundary and
  separate validation/fresh metric sections.
