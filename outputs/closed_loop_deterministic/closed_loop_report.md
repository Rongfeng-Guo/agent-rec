# CritiqueWorld Closed-Loop Report

This is a controlled counterfactual rollout proxy report. It is not human evaluation and not complete causal inference.

- Output directory: `outputs\closed_loop_deterministic`
- Parser mode: `deterministic`
- Git commit recorded by run: `5b037c7bdc03bc7ee56d5abf2f3ebdc67e337759`
- Audit status: `PASS`

## Dataset Gate
| Field | Value |
| --- | --- |
| dataset | closed_loop_deterministic_cdpo |
| status | SMOKE_TEST_ONLY |
| validation | PASS |
| rows | 27 |
| train/dev | 22/5 |
| score_delta | 0.066 / 0.191 / 0.306 |
| sha256 | 6e1091780d6350d2a6dda9ccc2e54b28525086923c954b0e1fc51a1060c9c509 |

## Output Counts
| Artifact | Rows |
| --- | --- |
| summary_rows | 105 |
| trajectory_rows | 1044 |
| branch_rows | 1710 |
| dpo_pairs | 27 |
| cdpo_pairs | 27 |

## Method Summary
| Method | CumulativeUtility_mean | ClickRate_mean | InstructionUplift@H_mean | OverCorrectionRegret@H_mean | ScopeClassificationAccuracy_mean | parser_scope_error_mean | memory_update_error_mean | candidate_coverage_error_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| critiquescope | 11.087 | 0.707 | -0.004 | 0.061 | 0.883 | 0.117 | 0.016 | 0.070 |
| flat | 10.980 | 0.723 | -0.064 | 0.000 | 0.881 | 0.119 | 0.000 | 0.070 |
| none | 11.087 | 0.707 | 0.000 | 0.000 | 0.883 | 0.117 | 0.000 | 0.070 |
| structured | 11.072 | 0.715 | 0.029 | 0.094 | 0.878 | 0.122 | 0.000 | 0.070 |
| time_decay | 11.095 | 0.719 | 0.022 | 0.000 | 0.884 | 0.116 | 0.000 | 0.070 |

## Pair Distribution By Rejected Branch
| Rejected Branch | Rows |
| --- | --- |
| ignore | 9 |
| over_apply | 18 |

## Pair Distribution By Scenario
| Scenario | Rows |
| --- | --- |
| behavioral_rollback | 9 |
| mixed_multi_turn | 9 |
| temporary_fatigue | 9 |

## Audit Errors
- none
