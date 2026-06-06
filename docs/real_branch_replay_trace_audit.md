# Real Branch Replay Trace Audit

The audit step checks whether the replay export is internally consistent before we hand it to the bridge tooling.

## Checks

- snapshot count is non-zero
- each branch has provenance and utility
- all three branches exist
- branch trajectories are not identical across the whole group
- replay pairs carry provenance
- uplift statistics are tracked for positive, zero, and negative outcomes

## Output files

- `audit.json`
- `audit.md`
- `task_type_summary.csv`
- `branch_summary.csv`
- `uplift_summary.csv`
- `pair_quality_summary.csv`
- `row_errors.jsonl`

## Failure mode

If a critical error is present, the audit command exits non-zero when `--fail-on-critical-error` is set.
