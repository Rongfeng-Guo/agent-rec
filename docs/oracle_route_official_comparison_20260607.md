# Oracle Route Official Comparison (2026-06-07)

This note fixes the official comparison protocol for the current oracle-route-memory line.

## Scope

We only compare methods that share all of the following:

- same dataset split: `user_simulator/test`
- same warm/cold partition
- same retrieval metrics: `Recall@10/20/50`, `NDCG@10/20/50`, `MRR@10/20/50`
- same item-memory candidate universe

Anything outside this protocol, including closed-loop GIMO / CritiqueScope utility outputs, is out of scope for this table.

## Official Artifact

- Table output: `outputs/oracle_route_memory/official_comparison_20260607/comparison.md`
- Machine-readable rows: `outputs/oracle_route_memory/official_comparison_20260607/comparison.csv`
- Generator: `scripts/oracle_route_memory/build_official_comparison_table.py`

## Main Readout

Claimable rows in the main table are currently:

- `Metadata Global`
- `Predicted Route Validation-Selected`

Non-claim diagnostic rows are currently:

- `Oracle Route P1`
- `Oracle Route P2`
- `Predicted Route Diagnostic Fusion LR P1T4`

Key outcome:

- Best claimable cold row is `Predicted Route Validation-Selected` with `Recall@50 = 0.0116`.
- Best diagnostic cold predicted-route row is `Predicted Route Diagnostic Fusion LR P1T4` with `Recall@50 = 0.0319`, but it is not validation-selected and therefore cannot be used as the official test claim.
- Oracle upper bounds remain much higher (`Oracle Route P1 cold Recall@50 = 0.0754`, `Oracle Route P2 cold Recall@50 = 0.7449`), so the bottleneck is still predicted-route/query binding, not memory capacity.

## What We Can And Cannot Claim

What we can claim now:

- Predicted-route memory is better than the no-route metadata baseline on cold retrieval under the official validation-selected protocol (`0.0116` vs `0.0058` Recall@50).
- The route-conditioned memory direction still has positive headroom because oracle-route upper bounds are far above the current predicted-route result.

What we cannot claim now:

- We cannot claim a stable official result above the `0.029` target threshold.
- We cannot claim to beat the earlier diagnostic `0.0319` with a validation-selected method.
- We cannot claim to beat the original full-SID / generative paper method, because no same-split same-metric artifact for that method was found in the current repo state.

## Next Step For A Complete Paper-Level Comparison

To complete the missing comparison against the original paper method, we need one of:

- a checked-in full-SID / generative eval artifact already run on `user_simulator/test` with the same retrieval metrics, or
- a runnable script/checkpoint pair that can produce those rows now under the same protocol.

Until that exists, the official comparison table should be treated as the authoritative same-metric comparison among currently available retrieval artifacts only.
