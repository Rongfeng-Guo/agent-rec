# H5-D GitHub Update Candidate

Date: 2026-06-08

Scope: local/server-side handoff for reviewing and staging the current
`agent-rec` changes before any GitHub update. This file is documentation only;
it is not a request to commit or push automatically.

## Repository State

- Repo: `/home/grf/agent-rec`
- Branch: `main`
- Current base commit observed before this note:
  `32c1931 Add oracle route blind confirmation protocol`
- Worktree status: dirty by design. Do not reset or revert unrelated tracked
  changes.

## Candidate Change Groups

- Root/current-status documentation:
  - `Readme.md`
  - `README_oracle_route_memory.md`
  - `RESEARCH_STATUS.md`
- H1/H2/H3/H4/H5 protocol and diagnostic docs under `experiments/` and
  `to_human/`.
- H5-D candidate-level source reranker scripts:
  - feature export;
  - pairwise ranker training;
  - domain-routed combiner;
  - locked manifest validator;
  - fresh prep bundle/audit/readiness/register/report gates;
  - handoff index validator.
- Focused test coverage for the new scripts and direct CLI smoke coverage for
  repro-command scripts without relying on `PYTHONPATH`.
- Research state/log docs that record the current hypothesis, validation-only
  claim boundary, and next target.

Tracked files already modified in the worktree before this note include:

- `scripts/oracle_route_memory/eval_predicted_route.py`
- `scripts/oracle_route_memory/select_validation_fusion_policy_explicit.py`
- `scripts/oracle_route_memory/train_late_bound_fusion_router.py`
- `tests/test_explicit_validation_fusion_selector.py`
- `tests/test_late_bound_router.py`

Keep these changes unless explicitly reviewed and rejected; they are part of the
current route/query-binding repair path.

## Current H5-D Gate State

- Locked policy:
  `h5_pairwise_domain_routed_book_h100_game_h300`
- Locked validation Recall@50:
  `0.07352941176470588` (`10/136`)
- Current prep bundle:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16`
- Current prep bundle audit:
  `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16_audit`
- Current readiness:
  `outputs/oracle_route_memory/h5_fresh_readiness_20260608_v16`
- Current handoff index validation:
  `outputs/oracle_route_memory/h5_handoff_index_validation_20260608_v16`

Primary review entry points:

- `Readme.md`
- `README_oracle_route_memory.md`
- `RESEARCH_STATUS.md`
- `experiments/h5-candidate-level-source-reranker/README.md`
- `experiments/h5-candidate-level-source-reranker/handoff_index.json`
- `experiments/h5-candidate-level-source-reranker/repro_commands.md`
- `to_human/h5_fresh_confirmation_handoff_summary_20260608.md`
- `to_human/h5_paper_facing_validation_summary_20260608.md`
- `to_human/h5_staging_file_manifest_20260608.md`
- `to_human/h5_pre_commit_review_note_20260608.md`

Gate summary:

- bundle audit `status=ok`
- bundle audit `source_drift=[]`
- readiness `status=ok`
- readiness `bundle_audit_source_drift_count=0`
- handoff index validation `status=ok`
- handoff index validation has `11` required bundle artifacts and `8` doc
  checks

Superseded artifact note:

- `outputs/oracle_route_memory/h5_handoff_index_validation_20260608_v15` failed
  because `protocol.md` was missing the fresh candidate feature export mention
  required by `handoff_index.json`.
- The current v16 handoff index validation fixes that mismatch and is the
  authoritative handoff gate.

## Verification Runs

The full related regression command is:

```bash
PYTHONPATH=/home/grf/agent-rec /home/grf/.conda/envs/gdpo/bin/python3 -m pytest \
  tests/test_check_h5_fresh_readiness.py \
  tests/test_validate_h5_loaded_model_replay.py \
  tests/test_candidate_level_source_feature_exporter.py \
  tests/test_candidate_level_source_ranker.py \
  tests/test_score_candidate_level_source_ranker.py \
  tests/test_combine_ranker_outputs_by_domain.py \
  tests/test_validate_locked_policy_manifest.py \
  tests/test_prepare_h5_fresh_confirmation_bundle.py \
  tests/test_audit_h5_fresh_confirmation_bundle.py \
  tests/test_register_h5_fresh_split.py \
  tests/test_late_bound_fusion_router_training.py \
  tests/test_late_bound_router.py \
  tests/test_route_query_binding_error_analysis.py \
  tests/test_explicit_validation_fusion_selector.py \
  tests/test_render_h5_fresh_confirmation_report.py \
  tests/test_validate_h5_handoff_index.py \
  tests/test_h5_handoff_cli_imports.py \
  -q
```

Current result after H5 repo-relative helper consolidation:
`87 passed`.

Earlier result after H5 shared repo-root path resolver consolidation:
`85 passed`.

Earlier result after H2/H4 shared handoff IO helper consolidation:
`84 passed`.

Earlier result after H3/H4 repo-root-relative input path handling:
`79 passed`.

Earlier result after H3/H4 cwd-independent output-dir handling:
`77 passed`.

Earlier result after H4 trainer and explicit selector handoff hardening:
`74 passed`.

Earlier result after H2 route/query analyzer hardening:
`71 passed`.

Earlier result after the prep-bundle builder JSON/README polish:
`70 passed`.

Also passed:

- `compileall` over `scripts/oracle_route_memory` and `tests`
- `git diff --check`
- focused trailing-whitespace check over touched H5-D scripts, tests, and docs

Earlier incremental verification after current-status README polish and H5-D
handoff gate/report hardening:

- `tests/test_validate_h5_handoff_index.py`: `4 passed`
- `tests/test_validate_h5_loaded_model_replay.py`: `2 passed`
- `tests/test_check_h5_fresh_readiness.py`: `3 passed`
- `tests/test_audit_h5_fresh_confirmation_bundle.py`: `4 passed`
- `tests/test_register_h5_fresh_split.py`: `5 passed`
- `tests/test_prepare_h5_fresh_confirmation_bundle.py`: `2 passed`
- `tests/test_candidate_level_source_feature_exporter.py`: `4 passed`
- `tests/test_candidate_level_source_ranker.py`: `6 passed`
- `tests/test_score_candidate_level_source_ranker.py`: `3 passed`
- `tests/test_combine_ranker_outputs_by_domain.py`: `3 passed`
- `tests/test_render_h5_fresh_confirmation_report.py`: `5 passed`
- `tests/test_validate_locked_policy_manifest.py`: `1 passed`
- full H5-D focused regression suite listed above: `70 passed`
- `git diff --check`: passed
- focused trailing-whitespace check over `Readme.md`,
  `README_oracle_route_memory.md`,
  `to_human/h5_github_update_candidate_20260608.md`,
  `scripts/oracle_route_memory/audit_h5_fresh_confirmation_bundle.py`,
  `scripts/oracle_route_memory/check_h5_fresh_readiness.py`,
  `scripts/oracle_route_memory/combine_ranker_outputs_by_domain.py`,
  `scripts/oracle_route_memory/export_candidate_level_source_features.py`,
  `scripts/oracle_route_memory/prepare_h5_fresh_confirmation_bundle.py`,
  `scripts/oracle_route_memory/register_h5_fresh_split.py`,
  `scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py`,
  `scripts/oracle_route_memory/score_candidate_level_source_ranker.py`,
  `scripts/oracle_route_memory/train_candidate_level_source_ranker.py`,
  `scripts/oracle_route_memory/validate_h5_loaded_model_replay.py`,
  `scripts/oracle_route_memory/validate_h5_handoff_index.py`,
  `scripts/oracle_route_memory/validate_locked_policy_manifest.py`,
  `tests/test_audit_h5_fresh_confirmation_bundle.py`,
  `tests/test_candidate_level_source_feature_exporter.py`,
  `tests/test_candidate_level_source_ranker.py`,
  `tests/test_check_h5_fresh_readiness.py`,
  `tests/test_combine_ranker_outputs_by_domain.py`,
  `tests/test_prepare_h5_fresh_confirmation_bundle.py`,
  `tests/test_register_h5_fresh_split.py`,
  `tests/test_render_h5_fresh_confirmation_report.py`,
  `tests/test_score_candidate_level_source_ranker.py`,
  `tests/test_validate_h5_loaded_model_replay.py`,
  `tests/test_validate_h5_handoff_index.py`, and
  `tests/test_validate_locked_policy_manifest.py`: passed
- temporary real-material validator run under `/tmp`: `status=ok`,
  `bundle_artifact_included_count=11/11`, `doc_check_ok_count=8/8`
- temporary real-material prep-bundle audit from `/tmp` with repo-root-relative
  `--bundle-dir`: `status=ok`, `artifact_check_count=11`,
  `source_drift_count=0`, `next_target` present in JSON, Gate Summary present
  in Markdown
- temporary real-material loaded-model replay validation under `/tmp`:
  `status=ok`, `mismatch_count=0`, `metric_errors=[]`,
  `next_target` present in JSON and Markdown
- temporary real-material readiness run under `/tmp`: `status=ok`,
  `component_model_count=2`, `missing_component_model_count=0`,
  `bundle_audit_source_drift_count=0`, `loaded_model_replay_mismatch_count=0`,
  model counts present in Markdown
- temporary real-material domain-routed combine under `/tmp`: Recall@50
  `0.07352941176470588`, `10/136` hits, `next_target` present in JSON and
  Markdown
- temporary real-material h100 locked-model score under `/tmp`: Recall@50
  `0.04411764705882353`, `sample_count=136`, `next_target` present in JSON and
  Markdown
- existing `h5_candidate_level_source_features_k500_20260608` summary re-render
  with the new exporter renderer: train rows `1888938`, cold-like rows
  `345969`, `next_target` present in rendered Markdown via fallback
- existing h100 pairwise ranker summary re-render with the new trainer
  renderer: Recall@50 `0.04411764705882353`, objective `pairwise`,
  `next_target` present in rendered Markdown via fallback
- real locked manifest validator run: `status=ok`, Recall@50
  `0.07352941176470588`, h100/h300 component `model.pkl` checks both
  `exists=true`, `claim_boundary` and `next_target` present

Code polish included in the current candidate:

- `scripts/oracle_route_memory/prepare_h5_fresh_confirmation_bundle.py` now
  writes top-level `artifact_count` into new bundle manifests and shows artifact
  count plus validator status in the generated bundle README.
- `scripts/oracle_route_memory/register_h5_fresh_split.py` now exposes
  `required_manifest_field_count`, `required_manifest_field_ok_count`,
  `consumed_manifest_path_match`, and `consumed_manifest_sha256_match` in JSON
  and Markdown so fresh split evidence can be reviewed without inferring these
  guard results from the error list.
- `scripts/oracle_route_memory/check_h5_fresh_readiness.py` now emits
  `component_model_count` and `missing_component_model_count` in JSON and
  Markdown so the final pre-fresh gate can be reviewed without manually
  counting component model checks.
- `scripts/oracle_route_memory/audit_h5_fresh_confirmation_bundle.py` now emits
  top-level `artifact_check_count`, `source_drift_count`, and `next_target`
  fields in JSON, and includes a Markdown Gate Summary for faster handoff
  review.
- `scripts/oracle_route_memory/validate_locked_policy_manifest.py` now includes
  a validation-only claim boundary, explicit next target, and non-failing
  component `model.pkl` visibility checks in its JSON output.
- `scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py` now
  rejects incomplete fresh split registration evidence, non-ok required
  manifest field checks, missing/nonzero readiness drift or replay mismatch
  counts, and renders split manifest path/SHA-256 in the final report inputs.
- `scripts/oracle_route_memory/train_candidate_level_source_ranker.py` now
  refuses to write into a non-empty output directory, renders missing average
  rank metrics as `n/a`, and records a `next_target` in new summary/report
  outputs. The report renderer also falls back to this guidance for older
  summaries.
- `scripts/oracle_route_memory/export_candidate_level_source_features.py` now
  refuses to write into a non-empty output directory and records a fresh-aware
  `next_target` in new summary/report outputs. The report renderer also has a
  fallback so older summaries can be re-rendered with the same handoff guidance.
- `scripts/oracle_route_memory/score_candidate_level_source_ranker.py` now
  records an explicit `next_target` in summary JSON and report Markdown so
  loaded-model scoring outputs point to replay validation or fresh domain
  routing without changing locked models.
- `scripts/oracle_route_memory/combine_ranker_outputs_by_domain.py` now refuses
  to write into a non-empty output directory, handles missing average-rank
  metrics as `n/a` in Markdown, and records an explicit `next_target` for both
  validation and future fresh domain-routed outputs.
- `scripts/oracle_route_memory/validate_h5_loaded_model_replay.py` now records
  an explicit `next_target` in JSON/Markdown so the replay gate has the same
  handoff guidance style as the other H5-D gates.
- `scripts/oracle_route_memory/validate_h5_handoff_index.py` now emits top-level
  bundle/document check counts for faster handoff review.
- `scripts/oracle_route_memory/check_h5_fresh_readiness.py` now resolves a
  relative `--output-dir` under `--repo-root`, matching the other H5-D handoff
  scripts and avoiding cwd-dependent output placement.
- `scripts/oracle_route_memory/audit_h5_fresh_confirmation_bundle.py` now
  resolves relative `--bundle-dir` and `--output-dir` under `--repo-root`, so
  prep-bundle audits can be run from outside the repository without writing
  evidence artifacts to the caller's cwd.

- `scripts/oracle_route_memory/analyze_route_query_binding_errors.py` now
  refuses to write into a non-empty output directory, writes an additive
  `analysis_manifest.json` with artifact names and `next_target`, and renders
  the same `Next Target` guidance in Markdown so H2 bottleneck reports have the
  same handoff hygiene style as the H5-D gates.

- `scripts/oracle_route_memory/train_late_bound_fusion_router.py` now refuses
  non-empty output directories, records a top-level `next_target` in
  `checkpoint_meta.json`, prints the same guidance in CLI JSON, renders a
  Markdown `Next Target`, and formats missing rank metrics as `n/a` instead of
  raising during report generation.
- `scripts/oracle_route_memory/select_validation_fusion_policy_explicit.py` now
  refuses non-empty output directories, writes an additive
  `selector_manifest.json` with artifact names and `next_target`, renders a
  Markdown `Next Target`, and keeps the selected `fusion_config.json` hash input
  unchanged by storing handoff guidance outside the policy config.

- `scripts/oracle_route_memory/analyze_route_query_binding_errors.py`,
  `scripts/oracle_route_memory/train_late_bound_fusion_router.py`, and
  `scripts/oracle_route_memory/select_validation_fusion_policy_explicit.py` now
  accept optional `--repo-root`; when it is provided, a relative `--output-dir`
  is resolved under that repo root instead of the caller's cwd. Existing command
  behavior is preserved when `--repo-root` is omitted.

- The same H2/H4/selector commands now resolve relative input paths under
  `--repo-root` when it is provided: analyzer `--selector-rows`; H4 trainer
  data/model/protocol/checkpoint paths; and explicit selector
  data/model/policy/protocol/checkpoint paths. Existing default behavior is
  preserved when `--repo-root` is omitted.

- `scripts/oracle_route_memory/handoff_io.py` now centralizes the shared
  H2/H4 `ensure_empty_output_dir`, `resolve_output_dir`, and `resolve_repo_path`
  behavior. The H2 analyzer, H4 trainer, and explicit selector import it with a
  direct-script fallback, reducing duplicated path/handoff code without changing
  CLI behavior.

- The shared `handoff_io.py` helper now uses output-path exceptions that satisfy
  both the newer `FileExistsError` checks and the older H5 `ValueError` checks,
  allowing the simple H5 output-dir-only cluster to share the same helper without
  changing existing tests or CLI refusal text expectations.
- `scripts/oracle_route_memory/export_candidate_level_source_features.py`,
  `scripts/oracle_route_memory/train_candidate_level_source_ranker.py`,
  `scripts/oracle_route_memory/score_candidate_level_source_ranker.py`,
  `scripts/oracle_route_memory/combine_ranker_outputs_by_domain.py`, and
  `scripts/oracle_route_memory/validate_h5_loaded_model_replay.py` now import
  the shared output-dir guard with direct-script fallback.

- `scripts/oracle_route_memory/check_h5_fresh_readiness.py`,
  `scripts/oracle_route_memory/validate_h5_handoff_index.py`, and
  `scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py` now also
  import the shared output-dir guard with direct-script fallback.

- `scripts/oracle_route_memory/prepare_h5_fresh_confirmation_bundle.py`,
  `scripts/oracle_route_memory/audit_h5_fresh_confirmation_bundle.py`, and
  `scripts/oracle_route_memory/register_h5_fresh_split.py` now import the shared
  output-dir guard with direct-script fallback. Local H5 `ensure_empty_output_dir`
  definitions have been removed; the only remaining implementation is in
  `scripts/oracle_route_memory/handoff_io.py`.

- `scripts/oracle_route_memory/handoff_io.py` now also exposes
  `resolve_path_under_repo_root`. `check_h5_fresh_readiness.py`,
  `validate_h5_handoff_index.py`, `render_h5_fresh_confirmation_report.py`, and
  `register_h5_fresh_split.py` import it as their `resolve_path` helper, removing
  another identical local repo-root path resolver.

- `scripts/oracle_route_memory/handoff_io.py` now also exposes
  `repo_relative_or_absolute` and `repo_relative_required`, preserving the two
  distinct H5 semantics for repo-relative display paths versus bundle-only paths
  that must stay inside `repo_root`.
- `prepare_h5_fresh_confirmation_bundle.py` imports `repo_relative_required`,
  while `validate_h5_handoff_index.py`, `render_h5_fresh_confirmation_report.py`,
  and `register_h5_fresh_split.py` import `repo_relative_or_absolute`.

- `prepare_h5_fresh_confirmation_bundle.py` now imports
  `resolve_path_under_repo_root` as its bundle artifact resolver, and
  `audit_h5_fresh_confirmation_bundle.py` imports shared `resolve_repo_path` as
  its optional repo path resolver. Grep now finds H5 path resolver definitions
  only in `scripts/oracle_route_memory/handoff_io.py`.

Latest docs-only staging manifest verification:

- `to_human/h5_staging_file_manifest_20260608.md` added as a review/staging
  boundary document for source, tests, docs, and generated evidence exclusions.
- No H5-D bundle-required docs or `handoff_index.json` changed in this pass, so
  no v17 evidence regeneration is required.
- `git diff --check`: passed after adding the manifest link.
- focused trailing-whitespace scan over the GitHub update candidate doc and the
  new staging manifest: passed.

Latest H2 route/query analysis hardening verification:

- `scripts/oracle_route_memory/analyze_route_query_binding_errors.py` now
  refuses non-empty output directories, writes an additive
  `analysis_manifest.json`, and renders an explicit Markdown `Next Target`.
- `tests/test_route_query_binding_error_analysis.py`: `4 passed`
- H5-D focused regression suite listed above, including the H2 analyzer test:
  `71 passed`
- `compileall` over the touched H2 analyzer script and test: passed
- `git diff --check`: passed
- focused trailing-whitespace scan over the touched H2 analyzer script, test,
  and staging docs: passed
- No H5-D bundle-required docs or `handoff_index.json` changed in this code
  polish pass, so no v17 evidence regeneration is required.

Latest H4 trainer and explicit selector handoff hardening verification:

- `scripts/oracle_route_memory/train_late_bound_fusion_router.py` now has
  non-empty output-directory protection, `next_target` in JSON/Markdown/CLI
  output, and `n/a` formatting for missing rank metrics.
- `scripts/oracle_route_memory/select_validation_fusion_policy_explicit.py` now
  has non-empty output-directory protection, an additive
  `selector_manifest.json`, and `next_target` in Markdown/CLI output without
  changing the selected policy config hash input.
- `tests/test_late_bound_fusion_router_training.py` plus
  `tests/test_explicit_validation_fusion_selector.py`: `9 passed`
- H5-D focused regression suite listed above: `74 passed`
- `compileall` over the touched H4/selector scripts and tests: passed
- `git diff --check`: passed
- focused trailing-whitespace scan over the touched H2/H4/selector scripts,
  tests, and staging docs: passed
- No H5-D bundle-required docs or `handoff_index.json` changed in this pass, so
  no v17 evidence regeneration is required.

Latest H3/H4 cwd-independent output-dir verification:

- H2 analyzer, H4 late-bound fusion trainer, and explicit validation selector
  now accept optional `--repo-root` for repo-root-relative output placement.
- Analyzer CLI smoke test from an outside cwd confirms relative output lands
  under the provided repo root, not under the caller's cwd.
- H4 trainer and explicit selector helper tests confirm relative, absolute, and
  omitted-`--repo-root` output-dir behavior.
- `tests/test_route_query_binding_error_analysis.py`,
  `tests/test_late_bound_fusion_router_training.py`, and
  `tests/test_explicit_validation_fusion_selector.py`: `16 passed`
- H5-D focused regression suite listed above: `77 passed`
- `compileall` over the touched H2/H4/selector scripts and tests: passed
- `git diff --check`: passed
- focused trailing-whitespace scan over the touched H2/H4/selector scripts,
  tests, and staging docs: passed
- No H5-D bundle-required docs or `handoff_index.json` changed in this pass, so
  no v17 evidence regeneration is required.

Latest H3/H4 repo-root-relative input path verification:

- H2 analyzer now resolves relative `--selector-rows` under `--repo-root`; its
  outside-cwd CLI smoke test now passes both input and output paths as repo-root
  relative values.
- H4 trainer and explicit selector now resolve relative data/model/protocol and
  optional checkpoint/policy input paths under `--repo-root` while leaving
  omitted-`--repo-root` behavior unchanged.
- `tests/test_route_query_binding_error_analysis.py`,
  `tests/test_late_bound_fusion_router_training.py`, and
  `tests/test_explicit_validation_fusion_selector.py`: `18 passed`
- H5-D focused regression suite listed above: `79 passed`
- `compileall` over the touched H2/H4/selector scripts and tests: passed
- `git diff --check`: passed
- focused trailing-whitespace scan over the touched H2/H4/selector scripts,
  tests, and staging docs: passed
- No H5-D bundle-required docs or `handoff_index.json` changed in this pass, so
  no v17 evidence regeneration is required.

Latest H2/H4 shared handoff IO helper verification:

- `scripts/oracle_route_memory/handoff_io.py` added with shared
  `ensure_empty_output_dir`, `resolve_output_dir`, and `resolve_repo_path`.
- H2 analyzer, H4 trainer, and explicit selector now import the shared helper
  with a direct-script fallback.
- `tests/test_handoff_io.py` plus the H2/H4/selector focused tests:
  `23 passed`
- H5-D focused regression suite listed above, now including
  `tests/test_handoff_io.py`: `84 passed`
- direct `--help` smoke from `/tmp` for the H2 analyzer, H4 trainer, and
  explicit selector scripts: passed
- `compileall` over the shared helper, touched scripts, and tests: passed
- `git diff --check`: passed
- focused trailing-whitespace scan over the touched helper/scripts/tests/docs:
  passed
- No H5-D bundle-required docs or `handoff_index.json` changed in this pass, so
  no v17 evidence regeneration is required.

Latest H5 output-dir-only shared helper verification:

- Simple H5 output-dir-only scripts now import the shared `handoff_io.py`
  output-dir guard: candidate feature export, candidate ranker training, locked
  model scoring, domain combine, and loaded-model replay validation.
- Shared helper output-path exceptions are compatible with both old H5
  `ValueError` tests and newer H2/H4 `FileExistsError` tests.
- `tests/test_handoff_io.py` plus the five H5 focused tests for this cluster:
  `23 passed`
- H5-D focused regression suite listed above: `84 passed`
- direct `--help` smoke from `/tmp` for the five refactored H5 scripts: passed
- `compileall` over the shared helper, touched H5 scripts, and tests: passed
- `git diff --check`: passed
- focused trailing-whitespace scan over the touched helper/scripts/tests/docs:
  passed
- No H5-D bundle-required docs or `handoff_index.json` changed in this pass, so
  no v17 evidence regeneration is required.

Latest H5 readiness/index/report gate shared helper verification:

- `check_h5_fresh_readiness.py`, `validate_h5_handoff_index.py`, and
  `render_h5_fresh_confirmation_report.py` now import the shared output-dir
  guard with direct-script fallback.
- `tests/test_handoff_io.py` plus readiness, handoff-index, and fresh-report
  focused tests: `17 passed`
- H5-D focused regression suite listed above: `84 passed`
- direct `--help` smoke from `/tmp` for the three refactored gate scripts:
  passed
- `compileall` over the shared helper, touched gate scripts, and tests: passed
- `git diff --check`: passed
- focused trailing-whitespace scan over the touched helper/scripts/tests/docs:
  passed
- No H5-D bundle-required docs or `handoff_index.json` changed in this pass, so
  no v17 evidence regeneration is required.

Latest H5 manifest-path gate shared helper verification:

- `prepare_h5_fresh_confirmation_bundle.py`,
  `audit_h5_fresh_confirmation_bundle.py`, and `register_h5_fresh_split.py` now
  import the shared output-dir guard with direct-script fallback.
- Local H5 `ensure_empty_output_dir` definitions have been removed; grep now
  finds only the shared implementation in `handoff_io.py`.
- `tests/test_handoff_io.py` plus prep-bundle, bundle-audit, and fresh-split
  registration focused tests: `16 passed`
- H5-D focused regression suite listed above: `84 passed`
- direct `--help` smoke from `/tmp` for the three refactored manifest-path gate
  scripts: passed
- `compileall` over the shared helper, touched gate scripts, and tests: passed
- `git diff --check`: passed
- focused trailing-whitespace scan over the touched helper/scripts/tests/docs:
  passed
- No H5-D bundle-required docs or `handoff_index.json` changed in this pass, so
  no v17 evidence regeneration is required.

Latest H5 shared repo-root path resolver verification:

- `handoff_io.py` now includes `resolve_path_under_repo_root` for the identical
  H5 `resolve_path(repo_root, value)` behavior.
- `check_h5_fresh_readiness.py`, `validate_h5_handoff_index.py`,
  `render_h5_fresh_confirmation_report.py`, and `register_h5_fresh_split.py`
  import the shared resolver as `resolve_path` with direct-script fallback.
- `tests/test_handoff_io.py` plus readiness, handoff-index, fresh-report, and
  fresh-split registration focused tests: `23 passed`
- H5-D focused regression suite listed above: `85 passed`
- direct `--help` smoke from `/tmp` for the four refactored resolver scripts:
  passed
- `compileall` over the shared helper, touched scripts, and tests: passed
- `git diff --check`: passed
- focused trailing-whitespace scan over the touched helper/scripts/tests/docs:
  passed
- No H5-D bundle-required docs or `handoff_index.json` changed in this pass, so
  no v17 evidence regeneration is required.

Latest H5 repo-relative helper verification:

- `handoff_io.py` now has separate helpers for the two H5 repo-relative
  semantics: `repo_relative_or_absolute` for display paths and
  `repo_relative_required` for prep-bundle artifacts that must stay inside
  `repo_root`.
- Prep-bundle now imports the required-inside-repo helper; handoff-index,
  fresh-report, and fresh-split registration import the display-path helper.
- `tests/test_handoff_io.py` plus prep-bundle, handoff-index, fresh-report, and
  fresh-split registration focused tests: `24 passed`
- H5-D focused regression suite listed above: `87 passed`
- direct `--help` smoke from `/tmp` for the four refactored repo-relative
  scripts: passed
- `compileall` over the shared helper, touched scripts, and tests: passed
- `git diff --check`: passed
- focused trailing-whitespace scan over the touched helper/scripts/tests/docs:
  passed
- No H5-D bundle-required docs or `handoff_index.json` changed in this pass, so
  no v17 evidence regeneration is required.

Latest final H5 path-helper consolidation verification:

- Prep-bundle now uses shared `resolve_path_under_repo_root` for required bundle
  artifact paths; bundle-audit now uses shared `resolve_repo_path` for optional
  repo-root-relative paths.
- Grep for local H5 path resolver definitions now returns only shared
  definitions in `scripts/oracle_route_memory/handoff_io.py`.
- `tests/test_handoff_io.py` plus prep-bundle and bundle-audit focused tests:
  `14 passed`
- H5-D focused regression suite listed above: `87 passed`
- direct `--help` smoke from `/tmp` for prep-bundle and bundle-audit scripts:
  passed
- `compileall` over the shared helper, touched scripts, and tests: passed
- `git diff --check`: passed
- focused trailing-whitespace scan over the touched helper/scripts/tests/docs:
  passed
- No H5-D bundle-required docs or `handoff_index.json` changed in this pass, so
  no v17 evidence regeneration is required.

Latest staging-readiness audit:

- Compared `git status --short --untracked-files=all` against
  `to_human/h5_staging_file_manifest_20260608.md`: `66` current status paths,
  `100` manifest path entries, `0` missing status paths.
- Confirmed shared helper and test entries are listed:
  `scripts/oracle_route_memory/handoff_io.py` and `tests/test_handoff_io.py`.
- Direct `--help` smoke from `/tmp` for the full touched handoff script set:
  `16` scripts checked, `0` failures.
- No generated `outputs/` paths were present in the staging candidate status.
- No H5-D bundle-required docs or `handoff_index.json` changed in this pass, so
  no v17 evidence regeneration is required.

Latest pre-commit review note update:

- `to_human/h5_pre_commit_review_note_20260608.md` added as a review aid for
  source/test/doc staging versus generated evidence exclusions.
- Post-note manifest coverage audit: `67` current status paths, `105` manifest
  path entries, `0` missing status paths.
- `git diff --check`: passed.
- focused trailing-whitespace scan over the pre-commit note and staging docs:
  passed.
- This is a docs-only update and does not require v17 evidence regeneration.

## Staging Notes

- Do not stage generated `outputs/` directories unless explicitly requested.
  They are evidence artifacts for reproduction and audit, not necessarily
  GitHub source changes.
- Stage source/docs/tests deliberately from `git status --short`; do not use
  destructive cleanup commands.
- Before committing, rerun the full verification command above plus:

```bash
git diff --check
```

- If any H5-D handoff doc or `handoff_index.json` changes, regenerate a new
  versioned prep bundle/audit/readiness/index validation directory instead of
  overwriting v16.

## Next Target

Keep the H5-D locked manifest and v16 handoff gates unchanged while waiting for
a clearly fresh/unconsumed split. For the next code-quality pass, keep staging
docs synchronized as files change, and if a commit is requested, rerun the full
focused regression plus `git diff --check` before staging only the source/test/docs
listed in the manifest. When a fresh split exists, register the split manifest
path/SHA-256, score only with locked h100/h300 `model.pkl` files, apply the
locked domain routing rule, and render a final fresh-confirmation report that
keeps validation and fresh metrics separate.
