# GIMO Real CDPO Bridge

This bridge connects real GIMO replay rollouts to the existing CDPO validation and dataset manifest tooling.

## Flow

1. Export replay snapshots from a real Prompt IRA trace.
2. Run real branch replay for `follow`, `ignore`, and `over_apply`.
3. Adapt the branch rows into replay pairs and CDPO pairs.
4. Validate the positive-uplift CDPO pairs.
5. Build the dataset manifest and split files.

## Important constraint

Only positive-uplift pairs are exported into `cdpo_pairs.jsonl`. Zero and negative pairs remain in `replay_pairs.jsonl` and `dpo_pairs.jsonl` for analysis.

## Synthetic safety

The existing CritiqueWorld path stays in place and still uses the synthetic provenance labels. Real replay outputs use `REAL_TRACE` and `REAL_USER_SIM_REPLAY` instead.
