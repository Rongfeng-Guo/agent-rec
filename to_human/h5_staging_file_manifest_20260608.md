# H5-D Staging File Manifest

Date: 2026-06-08

Scope: source, test, and documentation review manifest for the current
server184 `/home/grf/agent-rec` worktree before any GitHub update. This file is
not a commit request and does not include generated evidence directories.

## Purpose

The H5-D handoff now has enough moving parts that `git status --short` is too
coarse for review: `to_human/` is entirely untracked, several experiment docs
are untracked, and the source/test changes are mixed with generated caches. This
manifest records the intended staging surface so source changes, review docs,
and evidence artifacts remain separate.

## Candidate Source And Docs

These files are candidates for review and staging as the current H5-D source
bundle, assuming the reviewer accepts the validation-only claim boundary.

Root status and research records:

- `Readme.md`
- `README_oracle_route_memory.md`
- `RESEARCH_STATUS.md`
- `findings.md`
- `research-log.md`
- `research-state.yaml`

Experiment and protocol docs:

- `experiments/h2-route-query-binding-analysis/protocol.md`
- `experiments/h3-candidate-pool-depth/policy_config.json`
- `experiments/h3-candidate-pool-depth/protocol.md`
- `experiments/h4-enlarged-pool-rerank/protocol.md`
- `experiments/h5-candidate-level-source-reranker/README.md`
- `experiments/h5-candidate-level-source-reranker/fresh_confirmation_bundle.md`
- `experiments/h5-candidate-level-source-reranker/fresh_confirmation_checklist.md`
- `experiments/h5-candidate-level-source-reranker/handoff_index.json`
- `experiments/h5-candidate-level-source-reranker/locked_policy_manifest.json`
- `experiments/h5-candidate-level-source-reranker/protocol.md`
- `experiments/h5-candidate-level-source-reranker/repro_commands.md`
- `experiments/h5-candidate-level-source-reranker/validation_comparison.md`
- `experiments/v3-fusion-diagnostics/analysis.md`
- `experiments/v3-fusion-diagnostics/protocol.md`

Human-facing handoff and update docs:

- `to_human/h2_route_query_binding_error_report_20260608.md`
- `to_human/h3_candidate_pool_depth_report_20260608.md`
- `to_human/h4_late_bound_fusion_router_report_20260608.md`
- `to_human/h5_candidate_level_source_reranker_report_20260608.md`
- `to_human/h5_fresh_confirmation_handoff_summary_20260608.md`
- `to_human/h5_github_update_candidate_20260608.md`
- `to_human/h5_paper_facing_validation_summary_20260608.md`
- `to_human/h5_pre_commit_review_note_20260608.md`
- `to_human/h5_staging_file_manifest_20260608.md`
- `to_human/v3_fusion_diagnostics_report_20260608.md`

Source scripts:

- `scripts/__init__.py`
- `scripts/oracle_route_memory/__init__.py`
- `scripts/oracle_route_memory/analyze_route_query_binding_errors.py`
- `scripts/oracle_route_memory/handoff_io.py`
- `scripts/oracle_route_memory/audit_h5_fresh_confirmation_bundle.py`
- `scripts/oracle_route_memory/check_h5_fresh_readiness.py`
- `scripts/oracle_route_memory/combine_ranker_outputs_by_domain.py`
- `scripts/oracle_route_memory/eval_predicted_route.py`
- `scripts/oracle_route_memory/export_candidate_level_source_features.py`
- `scripts/oracle_route_memory/prepare_h5_fresh_confirmation_bundle.py`
- `scripts/oracle_route_memory/register_h5_fresh_split.py`
- `scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py`
- `scripts/oracle_route_memory/score_candidate_level_source_ranker.py`
- `scripts/oracle_route_memory/select_validation_fusion_policy_explicit.py`
- `scripts/oracle_route_memory/train_candidate_level_source_ranker.py`
- `scripts/oracle_route_memory/train_late_bound_fusion_router.py`
- `scripts/oracle_route_memory/validate_h5_handoff_index.py`
- `scripts/oracle_route_memory/validate_h5_loaded_model_replay.py`
- `scripts/oracle_route_memory/validate_locked_policy_manifest.py`

Focused tests:

- `tests/test_audit_h5_fresh_confirmation_bundle.py`
- `tests/test_candidate_level_source_feature_exporter.py`
- `tests/test_candidate_level_source_ranker.py`
- `tests/test_check_h5_fresh_readiness.py`
- `tests/test_combine_ranker_outputs_by_domain.py`
- `tests/test_explicit_validation_fusion_selector.py`
- `tests/test_handoff_io.py`
- `tests/test_h5_handoff_cli_imports.py`
- `tests/test_late_bound_fusion_router_training.py`
- `tests/test_late_bound_router.py`
- `tests/test_prepare_h5_fresh_confirmation_bundle.py`
- `tests/test_register_h5_fresh_split.py`
- `tests/test_render_h5_fresh_confirmation_report.py`
- `tests/test_route_query_binding_error_analysis.py`
- `tests/test_score_candidate_level_source_ranker.py`
- `tests/test_validate_h5_handoff_index.py`
- `tests/test_validate_h5_loaded_model_replay.py`
- `tests/test_validate_locked_policy_manifest.py`

## Already Modified Tracked Files

The following tracked files are modified in the current dirty worktree and are
part of the candidate review surface. They should not be reset without a direct
review decision.

- `README_oracle_route_memory.md`
- `RESEARCH_STATUS.md`
- `Readme.md`
- `scripts/oracle_route_memory/eval_predicted_route.py`
- `scripts/oracle_route_memory/select_validation_fusion_policy_explicit.py`
- `scripts/oracle_route_memory/train_late_bound_fusion_router.py`
- `tests/test_explicit_validation_fusion_selector.py`
- `tests/test_late_bound_router.py`

## Leave Unstaged By Default

Generated evidence and runtime artifacts should stay on server184 unless the
reviewer explicitly asks to publish them.

- `outputs/`
- any `__pycache__/` directory
- `.pytest_cache/`
- `*.pyc`
- temporary smoke-test outputs under `/tmp/`
- local virtual environments or conda environment files not already tracked
- `/home/grf/GenRecEdit-main`

Do not delete superseded evidence directories. In particular, keep the v15
handoff index validation failure evidence and the v16 authoritative handoff
evidence side by side.

## Evidence Boundary

The current authoritative H5-D evidence remains v16:

- prep bundle: `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16`
- prep bundle audit: `outputs/oracle_route_memory/h5_fresh_confirmation_prep_bundle_20260608_v16_audit`
- readiness: `outputs/oracle_route_memory/h5_fresh_readiness_20260608_v16`
- handoff index validation: `outputs/oracle_route_memory/h5_handoff_index_validation_20260608_v16`

If any bundle-required H5-D docs or
`experiments/h5-candidate-level-source-reranker/handoff_index.json` change,
generate a new versioned prep/audit/readiness/index-validation set rather than
editing or overwriting v16.

## Latest Verification

Earlier full focused regression recorded in
`to_human/h5_github_update_candidate_20260608.md`:

- H5-D focused pytest suite: `70 passed`
- `git diff --check`: passed
- focused trailing-whitespace checks over touched H5-D scripts, tests, and docs:
  passed
- real-material smoke checks under `/tmp`: status `ok` for handoff index,
  prep-bundle audit, loaded-model replay, readiness, domain-routed combine,
  h100 locked score, and locked manifest validation

This manifest was first added as a docs-only review boundary. After that, eleven
small code-quality passes were completed:

- H2 route/query analyzer now refuses non-empty output directories, writes
  `analysis_manifest.json`, and includes a Markdown `Next Target`.
- H4 late-bound fusion trainer now refuses non-empty output directories, records
  `next_target` in JSON/Markdown/CLI output, and renders missing rank metrics
  as `n/a`.
- Explicit validation selector now refuses non-empty output directories, writes
  `selector_manifest.json`, and records `next_target` outside the selected
  policy config hash input.
- H2 analyzer, H4 trainer, and explicit selector now accept optional
  `--repo-root`; relative `--output-dir` values and the relevant relative input
  paths resolve under that repo root when provided, while existing behavior is
  preserved when omitted.
- `scripts/oracle_route_memory/handoff_io.py` now centralizes shared handoff IO
  helpers. H2/H4 scripts and all current H5 scripts with output-dir guards import
  it with direct-script fallbacks.
- Refactored H5 output-dir-only cluster: candidate feature export, candidate
  ranker training, locked model scoring, domain combine, and loaded-model replay
  validation.
- Refactored H5 gate cluster: readiness, handoff-index validation, and fresh
  confirmation report rendering.
- Refactored H5 manifest-path gate cluster: prep-bundle, bundle-audit, and
  fresh-split registration.
- `resolve_path_under_repo_root`, `repo_relative_or_absolute`, and
  `repo_relative_required` added to `handoff_io.py`; matching H5 gate scripts now
  import these shared path helpers with direct-script fallbacks.
- Prep-bundle and bundle-audit now use shared final path resolvers; grep for H5
  local resolver definitions returns only shared definitions in `handoff_io.py`.
- `tests/test_handoff_io.py` plus prep/audit focused tests: `14 passed`
- full focused H5-D/H2/H4 regression listed in the GitHub update candidate doc:
  `87 passed`
- direct `--help` smoke from `/tmp` for prep-bundle and bundle-audit scripts:
  passed
- `compileall` over the touched helper, scripts, and tests: passed
- `git diff --check`: passed
- focused trailing-whitespace scan over the touched helper, scripts, tests, and
  staging docs: passed

Staging-readiness audit after the helper consolidation:

- Compared `git status --short --untracked-files=all` against this manifest:
  `66` current status paths, `100` manifest path entries, `0` missing status
  paths.
- Confirmed `scripts/oracle_route_memory/handoff_io.py` and
  `tests/test_handoff_io.py` are listed.
- Direct `--help` smoke from `/tmp` for the full touched handoff script set:
  `16` scripts checked, `0` failures.
- No generated `outputs/` paths were present in the staging candidate status.

Pre-commit review note:

- `to_human/h5_pre_commit_review_note_20260608.md` added as a source/test/doc
  versus generated-evidence review aid.
- Post-note manifest coverage audit: `67` current status paths, `105` manifest
  path entries, `0` missing status paths.
- `git diff --check`: passed.

Current follow-up verification from 2026-06-08:

- focused regression command listed in
  `to_human/h5_github_update_candidate_20260608.md`: `79 passed`
- `compileall` over `scripts/oracle_route_memory` and `tests`: passed
- `git diff --check`: passed
- older staging snapshots mention `87 passed`; treat the current explicit run
  above as the latest verification count.
- focused trailing-whitespace scan over the pre-commit note and staging docs:
  passed.
- The note is docs-only and does not require v17 evidence regeneration.

## Next Target

Keep the locked H5-D manifest and v16 handoff evidence unchanged. The next code
quality pass should keep the staging docs synchronized as files change, and if a
commit is requested, rerun the full focused regression plus `git diff --check`
before staging only the source/test/docs listed here.

When a genuinely fresh, unconsumed split is available, register its manifest
path and SHA-256, score only with the locked h100/h300 `model.pkl` files, apply
the locked domain-routing rule, and render a final fresh-confirmation report
that keeps validation metrics and fresh metrics separate.
