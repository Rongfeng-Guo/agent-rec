# CritiqueWorld Closed-Loop Report

This is a controlled counterfactual rollout proxy report. It is not human evaluation and not complete causal inference.

- Output directory: `outputs\closed_loop_oracle`
- Parser mode: `oracle`
- Git commit recorded by run: `bdee4dfc0b0b7ed7d0b581efb4dc8cba28451b25`
- Audit status: `PASS`

## Dataset Gate
| Field | Value |
| --- | --- |
| dataset | closed_loop_oracle_cdpo |
| status | SMOKE_TEST_ONLY |
| validation | PASS |
| rows | 80 |
| train/dev | 64/16 |
| score_delta | 0.034 / 0.147 / 0.306 |
| sha256 | 66814677b59e039aeccaa7da91603ff2a497df43f03c08417a0165e2de464b4a |

## Output Counts
| Artifact | Rows |
| --- | --- |
| summary_rows | 175 |
| trajectory_rows | 1740 |
| branch_rows | 2850 |
| dpo_pairs | 80 |
| cdpo_pairs | 80 |
| cdpo_train | 64 |
| cdpo_dev | 16 |

## Method Summary
| Method | CumulativeUtility_mean | ClickRate_mean | InstructionUplift@H_mean | OverCorrectionRegret@H_mean | ScopeClassificationAccuracy_mean | parser_scope_error_mean | memory_update_error_mean | candidate_coverage_error_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| critiquescope | 11.003 | 0.710 | -0.076 | 0.099 | 1.000 | 0.000 | 0.016 | 0.000 |
| flat | 10.850 | 0.711 | -0.164 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| none | 11.088 | 0.710 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| structured | 11.020 | 0.703 | 0.009 | 0.125 | 1.000 | 0.000 | 0.000 | 0.000 |
| time_decay | 11.075 | 0.719 | 0.009 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Pair Distribution By Rejected Branch
| Rejected Branch | Rows |
| --- | --- |
| ignore | 30 |
| over_apply | 50 |

## Pair Distribution By Scenario
| Scenario | Rows |
| --- | --- |
| behavioral_rollback | 15 |
| mixed_multi_turn | 25 |
| session_context | 25 |
| temporary_fatigue | 15 |

## Audit Errors
- none
