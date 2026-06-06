# Real Branch Utility

The real branch utility is a transparent, hand-auditable scorer for replay branches.

## Inputs

It consumes a branch row with:

- `snapshot`
- `trajectory`
- `branch_type`
- per-step satisfaction and parser status fields

## What it scores

The utility combines:

- task success
- satisfaction signal
- constraint satisfaction
- continuation signal
- recommendation relevance
- extra turn cost
- repetition penalty
- tool failure
- parse failure

## Configuration

Weights live in `configs/server184/real_branch_utility.yaml`.

## Notes

This utility is for controlled real replay analysis only. It is not a substitute for human evaluation, and it is not the synthetic CritiqueWorld scorer.
