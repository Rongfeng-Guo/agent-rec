# H5-D Pre-Commit Review Note

Date: 2026-06-08

Scope: review aid for the current server184 `/home/grf/agent-rec` worktree. This
file is not a commit request and does not stage or publish any generated
evidence.

## Current Worktree Shape

Current `git status --short --untracked-files=all` review surface:

- `66` source/doc/test paths in status.
- `8` tracked modified files:
  - `README_oracle_route_memory.md`
  - `RESEARCH_STATUS.md`
  - `Readme.md`
  - `scripts/oracle_route_memory/eval_predicted_route.py`
  - `scripts/oracle_route_memory/select_validation_fusion_policy_explicit.py`
  - `scripts/oracle_route_memory/train_late_bound_fusion_router.py`
  - `tests/test_explicit_validation_fusion_selector.py`
  - `tests/test_late_bound_router.py`
- The remaining status entries are untracked source, tests, experiment docs,
  research records, and `to_human/` review docs.

## Source/Test/Doc Change Groups

Recommended review grouping before any commit:

- Current-status and research docs:
  - root README/status docs;
  - `findings.md`, `research-log.md`, and `research-state.yaml`;
  - experiment protocol docs under `experiments/`;
  - human-facing summaries under `to_human/`.
- H2/H3/H4 route and fusion diagnostics:
  - route/query binding analyzer;
  - explicit validation selector;
  - late-bound fusion router trainer;
  - focused H2/H4 selector/router tests.
- H5-D candidate-level source reranker and gates:
  - candidate feature export, pairwise ranker training, locked scoring, domain
    combine, loaded-model replay, locked manifest validation;
  - fresh prep bundle, bundle audit, readiness, split registration, report
    rendering, and handoff index validation;
  - focused H5-D tests and direct CLI import coverage.
- Shared handoff IO cleanup:
  - `scripts/oracle_route_memory/handoff_io.py` centralizes output-dir guards and
    repo-root/path helpers;
  - `tests/test_handoff_io.py` covers the shared helper behavior;
  - H2/H4/H5 scripts import the helper with direct-script fallbacks.

## Do Not Stage By Default

Generated evidence and runtime files should remain unstaged unless explicitly
requested:

- `outputs/`
- any `__pycache__/` directory
- `.pytest_cache/`
- `*.pyc`
- temporary smoke-test outputs under `/tmp/`
- local environments or machine-specific files
- `/home/grf/GenRecEdit-main`

Keep existing v15 and v16 evidence directories intact. Do not overwrite v16. If
bundle-required H5-D docs or `handoff_index.json` change, generate a new
versioned evidence set instead of editing v16.

## Latest Verification Snapshot

Latest verification recorded before this note:

- H5-D focused regression suite: `87 passed`
- direct `--help` smoke from `/tmp` for the full touched handoff script set:
  `16` scripts checked, `0` failures
- staging manifest coverage audit: `66` status paths, `100` manifest path
  entries, `0` missing status paths
- `git diff --check`: passed
- focused trailing-whitespace scans over touched source/test/docs: passed

This note is docs-only. Post-note checks run after adding it:

- manifest coverage audit: `67` current status paths, `107` manifest path
  entries, `0` missing status paths
- `git diff --check`: passed
- focused trailing-whitespace scan over this note and staging docs: passed

Current follow-up verification from 2026-06-08:

- focused regression command listed in
  `to_human/h5_github_update_candidate_20260608.md`: `79 passed`
- `compileall` over `scripts/oracle_route_memory` and `tests`: passed
- `git diff --check`: passed

Note: earlier review snapshots in the staging docs mention `87 passed`; the
current collected focused command result is `79 passed`, so use the explicit
command output rather than the older count when reviewing this handoff.

## Next Target

If preparing a commit, rerun the focused regression command from
`to_human/h5_github_update_candidate_20260608.md`, rerun `git diff --check`, and
stage source/test/docs deliberately from the manifest. Do not stage generated
`outputs/` evidence unless explicitly requested.
