# CritiqueWorld Closed-Loop Report

This is a controlled counterfactual rollout proxy report. It is not human evaluation and not complete causal inference.

- Output directory: `outputs\closed_loop_deterministic`
- Parser mode: `deterministic`
- Git commit recorded by run: `d146e734931ad115c7bba142d0b51912c27cc921`
- Audit status: `PASS`

## Dataset Gate
| Field | Value |
| --- | --- |
| dataset | closed_loop_deterministic_cdpo |
| status | SMOKE_TEST_ONLY |
| validation | PASS |
| rows | 42 |
| train/dev | 34/8 |
| score_delta | 0.010 / 0.516 / 2.068 |
| sha256 | 396c41b459c70969ed2daebf77ca50749066ba7232c63f544b9416f753d8d546 |

## Output Counts
| Artifact | Rows |
| --- | --- |
| summary_rows | 105 |
| trajectory_rows | 1257 |
| branch_rows | 2025 |
| dpo_pairs | 42 |
| cdpo_pairs | 42 |
| cdpo_train | 34 |
| cdpo_dev | 8 |

## Method Summary
| Method | CumulativeUtility_mean | ClickRate_mean | InstructionUplift@H_mean | OverCorrectionRegret@H_mean | ScopeClassificationAccuracy_mean | parser_scope_error_mean | memory_update_error_mean | candidate_coverage_error_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| critiquescope | 21.474 | 0.996 | -0.073 | 0.406 | 1.000 | 0.000 | 0.000 | 0.000 |
| flat | 14.072 | 0.897 | -0.306 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| none | 14.691 | 0.960 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| structured | 14.133 | 0.854 | -0.248 | -0.114 | 1.000 | 0.000 | 0.000 | 0.000 |
| time_decay | 15.401 | 0.976 | 0.286 | 0.142 | 1.000 | 0.000 | 0.000 | 0.000 |

## Pair Distribution By Rejected Branch
| Rejected Branch | Rows |
| --- | --- |
| ignore | 22 |
| over_apply | 20 |

## Pair Distribution By Scenario
| Scenario | Rows |
| --- | --- |
| behavioral_rollback | 6 |
| diversity_request | 12 |
| genuine_drift | 1 |
| mixed_multi_turn | 10 |
| session_context | 3 |
| temporary_fatigue | 10 |

## Audit Errors
- none
