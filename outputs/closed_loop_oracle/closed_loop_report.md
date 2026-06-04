# CritiqueWorld Closed-Loop Report

This is a controlled counterfactual rollout proxy report. It is not human evaluation and not complete causal inference.

- Output directory: `outputs\closed_loop_oracle`
- Parser mode: `oracle`
- Git commit recorded by run: `76491c69390e2ae549a33429b6539eb6b6be0624`
- Audit status: `PASS`

## Dataset Gate
| Field | Value |
| --- | --- |
| dataset | closed_loop_oracle_cdpo |
| status | SMOKE_TEST_ONLY |
| validation | PASS |
| rows | 65 |
| train/dev | 52/13 |
| score_delta | 0.010 / 0.401 / 1.030 |
| sha256 | 0f44fc9f62deda0ef646c6175cebb3e27a27bfceac92acdfe68e0db916b2c30f |

## Output Counts
| Artifact | Rows |
| --- | --- |
| summary_rows | 175 |
| trajectory_rows | 2095 |
| branch_rows | 3375 |
| dpo_pairs | 65 |
| cdpo_pairs | 65 |
| cdpo_train | 52 |
| cdpo_dev | 13 |

## Method Summary
| Method | CumulativeUtility_mean | ClickRate_mean | InstructionUplift@H_mean | OverCorrectionRegret@H_mean | ScopeClassificationAccuracy_mean | parser_scope_error_mean | memory_update_error_mean | candidate_coverage_error_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| critiquescope | 21.475 | 0.998 | -0.364 | 0.114 | 1.000 | 0.000 | 0.000 | 0.000 |
| flat | 14.067 | 0.890 | -0.302 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| none | 14.683 | 0.943 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| structured | 14.130 | 0.847 | -0.252 | -0.119 | 1.000 | 0.000 | 0.000 | 0.000 |
| time_decay | 15.402 | 0.981 | 0.288 | 0.144 | 1.000 | 0.000 | 0.000 | 0.000 |

## Pair Distribution By Rejected Branch
| Rejected Branch | Rows |
| --- | --- |
| ignore | 36 |
| over_apply | 29 |

## Pair Distribution By Scenario
| Scenario | Rows |
| --- | --- |
| behavioral_rollback | 5 |
| diversity_request | 20 |
| genuine_drift | 1 |
| mixed_multi_turn | 17 |
| session_context | 5 |
| temporary_fatigue | 17 |

## Audit Errors
- none
