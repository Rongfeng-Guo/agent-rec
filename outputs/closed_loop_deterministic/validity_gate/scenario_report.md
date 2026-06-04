# CritiqueWorld Validity Gate

- Invariants passed: `36/60`
- Critical failures: `24/60`

## Scenario Summary
- `behavioral_rollback`: 6/6 passed
- `diversity_request`: 9/9 passed
- `genuine_drift`: 9/9 passed
- `mixed_multi_turn`: 0/6 passed
- `session_context`: 9/9 passed
- `stable_dislike`: 3/9 passed
- `temporary_fatigue`: 0/12 passed

## Critical Failures
- `temporary_fatigue` / `critiquescope` / seed `0`: `follow_outperforms_over_apply`
- `temporary_fatigue` / `critiquescope` / seed `0`: `suppression_expires_after_horizon`
- `temporary_fatigue` / `flat` / seed `0`: `flat_retains_longer_suppression_than_critiquescope`
- `temporary_fatigue` / `critiquescope` / seed `0`: `post_expiry_target_eligible_again`
- `temporary_fatigue` / `critiquescope` / seed `1`: `follow_outperforms_over_apply`
- `temporary_fatigue` / `critiquescope` / seed `1`: `suppression_expires_after_horizon`
- `temporary_fatigue` / `flat` / seed `1`: `flat_retains_longer_suppression_than_critiquescope`
- `temporary_fatigue` / `critiquescope` / seed `1`: `post_expiry_target_eligible_again`
- `temporary_fatigue` / `critiquescope` / seed `2`: `follow_outperforms_over_apply`
- `temporary_fatigue` / `critiquescope` / seed `2`: `suppression_expires_after_horizon`
- `temporary_fatigue` / `flat` / seed `2`: `flat_retains_longer_suppression_than_critiquescope`
- `temporary_fatigue` / `critiquescope` / seed `2`: `post_expiry_target_eligible_again`
- `stable_dislike` / `critiquescope` / seed `0`: `persistent_filter_enters_slow_memory`
- `stable_dislike` / `critiquescope` / seed `0`: `filter_survives_session_reset`
- `stable_dislike` / `critiquescope` / seed `1`: `persistent_filter_enters_slow_memory`
- `stable_dislike` / `critiquescope` / seed `1`: `filter_survives_session_reset`
- `stable_dislike` / `critiquescope` / seed `2`: `persistent_filter_enters_slow_memory`
- `stable_dislike` / `critiquescope` / seed `2`: `filter_survives_session_reset`
- `mixed_multi_turn` / `critiquescope` / seed `0`: `temporary_fatigue_recovers_before_later_drift`
- `mixed_multi_turn` / `critiquescope` / seed `0`: `later_drift_promotes_mac`
- `mixed_multi_turn` / `critiquescope` / seed `1`: `temporary_fatigue_recovers_before_later_drift`
- `mixed_multi_turn` / `critiquescope` / seed `1`: `later_drift_promotes_mac`
- `mixed_multi_turn` / `critiquescope` / seed `2`: `temporary_fatigue_recovers_before_later_drift`
- `mixed_multi_turn` / `critiquescope` / seed `2`: `later_drift_promotes_mac`
