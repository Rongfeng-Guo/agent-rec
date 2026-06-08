# H1: v3 Fusion Diagnostics

Status: protocol active
Type: engineering-correctness experiment
Claim status: not a model-performance claim

## Question

Are v3 route-source ablation fusion diagnostics under-reporting route-hit
behavior because fusion rows do not propagate member `route_hit` fields?

## Motivation

The existing route-source ablation report:

```text
outputs/oracle_route_memory/validation_fusion_selector_v3_route_source_ablation_20260608/report.md
```

shows fusion policies with `RouteHitRate = 0.0000`. This is suspicious because
those policies are built from route-filtered members such as predicted,
domain-prior, and history-vote prefix-1 policies. If the diagnostic is wrong,
the report can mislead the research loop into optimizing the wrong bottleneck.

## Change

Add member-derived diagnostics to `build_fusion_retrieval_row()`:

- `route_hit`: true if any member row reports `route_hit`.
- `member_route_hit_count`: number of member rows with `route_hit`.
- `member_candidate_pool_hit_count`: number of member rows with
  `candidate_pool_hit`.

This should not alter:

- `ranked_ids`
- `match_rank`
- `Recall@K`
- selected policy tie-breaking except through corrected diagnostic fields, if a
  downstream report uses them for interpretation.

## Prediction

After rerunning the ablation:

- Fusion `RouteHitRate` should no longer be systematically zero.
- Fusion `Recall@50` should remain unchanged relative to the same input rows and
  fusion ranking logic.
- The report should better distinguish route-binding errors from ranking errors.

## Measurements

Primary:

- `selector_summary.csv` / `selector_summary.json`
- Fusion rows' `RouteHitRate`
- Fusion rows' `CandidatePoolHitRate`
- Fusion rows' `Recall@50`

Secondary:

- `selector_rows.json` member diagnostic fields
- Tests covering fusion row construction

## Validation Gates

Run targeted tests:

```bash
PYTHONPATH=/home/grf/agent-rec /home/grf/.conda/envs/gdpo/bin/python3 -m pytest \
  /home/grf/agent-rec/tests/test_late_bound_router.py \
  /home/grf/agent-rec/tests/test_explicit_validation_fusion_selector.py
```

Run compile and whitespace checks:

```bash
/home/grf/.conda/envs/gdpo/bin/python3 -m compileall \
  /home/grf/agent-rec/genrec/training \
  /home/grf/agent-rec/scripts/oracle_route_memory

git -C /home/grf/agent-rec diff --check
```

## Interpretation Rules

- If only diagnostics change, this is a reporting-correctness result.
- If Recall changes unexpectedly, inspect fusion ranking and input rows before
  trusting the run.
- Any future high-performing policy discovered after this fix is exploratory
  until locked on validation before a fresh blind-confirmation evaluation.
