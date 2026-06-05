# AGENTS.md

## Repository Overview

This repository is the working tree for `Rongfeng-Guo/agent-rec`.
It is a recommendation-agent research codebase centered on CritiqueScope-style memory, CritiqueWorld closed-loop evaluation, and CDPO dataset export.

The current local branch is `codex/driftaware-structured-memory`.
The latest local commit at the time of this handoff is `4f568f4` (`eval: add rollout-adapter audit summaries`).
The remote branch that matters to the user is `agent-rec/main`.

## Top-Level Directories

- `user_simulator/`: core implementation.
  - `state/`: memory state implementations, including CritiqueScope.
  - `policies/`: reranking logic and memory-aware intervention policy.
  - `worlds/`: CritiqueWorld closed-loop environment.
  - `scenarios/`: closed-loop scenario factories.
  - `evaluation/`: benchmark runners, rollout adapter, parser, validity gate, CDPO export.
- `tests/`: pytest regression coverage for CritiqueScope and CritiqueWorld.
- `docs/`: formal project documentation and experiment protocol.
- `outputs/`: committed smoke-test and audit artifacts.
- `LLaMA-Factory/`: upstream training framework dependency kept in-tree.
- `GPE_HAP/`: original supporting pipeline assets.
- `configs/`, `scripts/`, `pic/`: auxiliary project materials.

## Key Documents

- `Readme.md`: user-facing repository introduction. Keep tone formal and project-owned.
- `RESEARCH_STATUS.md`: current implementation status, smoke-test scope, and next priorities.
- `docs/experiment_protocol.md`: canonical runbook for deterministic/noisy/closed-loop evaluation.
- `docs/critiquescope_gimo.md`: focused description of the CritiqueScope integration path.
- `docs/critique_world.md`: CritiqueWorld mechanism notes.

## Important Output Paths

- `outputs/memory_baselines/`: deterministic baseline runs and summaries.
- `outputs/memory_baselines_noisy/`: noisy-scenario baseline runs.
- `outputs/closed_loop_oracle/`: oracle-parser closed-loop pipeline outputs.
- `outputs/closed_loop_deterministic/`: deterministic-parser closed-loop outputs.
- `outputs/validity_gate/`: invariant audit outputs.
- `outputs/rollout_adapter_smoke/`: rollout-adapter smoke artifacts, including:
  - `adapter_metadata.json`
  - `adapter_audit.jsonl`
  - `adapter_audit_summary.csv`
  - `adapter_failures.jsonl`
  - `adapter_report.md`

## What Was Just Finished

Recent local commits ahead of `agent-rec/main` are:

- `73c3e68` `eval: audit rollout-adapter inputs before pair export`
- `7f6a913` `docs: surface rollout-adapter audit in smoke paths`
- `4f568f4` `eval: add rollout-adapter audit summaries`

These changes added fail-fast rollout-input auditing, summary CSV/report generation, updated smoke documentation, and regression tests.

## Verified Commands

The following checks passed before this handoff:

- `pytest -q` -> `57 passed`
- `python -m compileall user_simulator` -> PASS
- `git diff --check` -> PASS
- `python -B -m user_simulator.evaluation.critique_rollout_adapter --output-dir outputs\rollout_adapter_smoke --fail-on-audit-error` -> PASS

## Current Git State

- Worktree was clean at handoff time.
- Local branch is ahead of `agent-rec/main` by 3 commits.
- Previous push attempts failed because GitHub HTTPS connections were reset (`Recv failure: Connection was reset`).
- This is a network issue, not a repo-state issue.

## Expectations From The User

- Push commits to GitHub promptly after meaningful progress.
- Treat this as the user's own improved repository, not as a paper-release mirror.
- Keep README and docs formal and project-oriented.
- When changes are complete, push the final version to `agent-rec/main`.

## What To Do Next

1. Retry pushing local HEAD to `agent-rec/main`:
   - `git push agent-rec HEAD:main`
2. Confirm GitHub shows `main` at the newest local commit.
3. Continue the next priority already listed in `RESEARCH_STATUS.md`:
   - connect real GIMO rollout logs to the CritiqueWorld / rollout-adapter branch schema
4. After that, harden the exported pair format for the final LLaMA-Factory/CDPO training recipe.
5. Keep updating `RESEARCH_STATUS.md`, `docs/experiment_protocol.md`, and committed output artifacts when evaluation behavior changes.

## Editing Notes For The Next Codex

- Prefer keeping changes tightly scoped to the existing evaluation and memory framework.
- Reuse current deterministic scenarios and smoke outputs unless a change truly requires regenerating them.
- If tests or compile steps create tracked `.pyc` noise, clean the worktree before committing.
- If push fails again, report it clearly and keep making local progress; do not lose momentum.
