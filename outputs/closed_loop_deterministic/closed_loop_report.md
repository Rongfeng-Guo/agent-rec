# CritiqueWorld Closed-Loop Report

This is a controlled counterfactual rollout proxy report. It is not human evaluation and not complete causal inference.

- Output directory: `outputs\closed_loop_deterministic`
- Parser mode: `deterministic`
- Git commit recorded by run: `76491c69390e2ae549a33429b6539eb6b6be0624`
- Audit status: `PASS`

## Dataset Gate
| Field | Value |
| --- | --- |
| dataset | closed_loop_deterministic_cdpo |
| status | SMOKE_TEST_ONLY |
| validation | PASS |
| rows | 29 |
| train/dev | 23/6 |
| score_delta | 0.034 / 0.466 / 1.030 |
| sha256 | f6bd6f9e0d04853e886a93ef7fc965e463d01ac3679a279c9279c55e84a08748 |

## Output Counts
| Artifact | Rows |
| --- | --- |
| summary_rows | 105 |
| trajectory_rows | 1257 |
| branch_rows | 2025 |
| dpo_pairs | 29 |
| cdpo_pairs | 29 |
| cdpo_train | 23 |
| cdpo_dev | 6 |

## Method Summary
| Method | CumulativeUtility_mean | ClickRate_mean | InstructionUplift@H_mean | OverCorrectionRegret@H_mean | ScopeClassificationAccuracy_mean | parser_scope_error_mean | memory_update_error_mean | candidate_coverage_error_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| critiquescope | 21.474 | 0.996 | -0.412 | 0.067 | 0.905 | 0.095 | 0.000 | 0.060 |
| flat | 14.403 | 0.956 | -0.104 | 0.000 | 0.905 | 0.095 | 0.000 | 0.060 |
| none | 14.691 | 0.960 | 0.000 | 0.000 | 0.905 | 0.095 | 0.000 | 0.060 |
| structured | 14.192 | 0.865 | -0.135 | -0.031 | 0.904 | 0.096 | 0.000 | 0.060 |
| time_decay | 15.451 | 0.972 | 0.318 | 0.142 | 0.905 | 0.095 | 0.000 | 0.060 |

## Pair Distribution By Rejected Branch
| Rejected Branch | Rows |
| --- | --- |
| ignore | 21 |
| over_apply | 8 |

## Pair Distribution By Scenario
| Scenario | Rows |
| --- | --- |
| behavioral_rollback | 3 |
| diversity_request | 12 |
| mixed_multi_turn | 7 |
| temporary_fatigue | 7 |

## Audit Errors
- none
