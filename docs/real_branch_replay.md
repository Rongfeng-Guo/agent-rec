# Real Branch Replay

This pipeline turns a real Prompt IRA trace into replay-ready snapshots, runs real branch rollouts, and bridges them into CDPO pairs.

## Provenance

The pipeline keeps the following labels distinct:

- `REAL_TRACE` for exported snapshots from the real prompt trace
- `REAL_USER_SIM_REPLAY` for branch rollout rows and replay pairs
- `CONTROLLED_SIMULATOR_REPLAY_PROXY` for the policy layer that rewrites the branch action
- `SYNTHETIC_CRITIQUEWORLD` for the existing synthetic benchmark path

## Main entrypoint

Run the server-184 pipeline from the repo root:

```bash
bash scripts/server184/run_real_branch_replay_pipeline.sh
```

It writes outputs under `outputs/server184_gimo/real_branch_replay/<timestamp>/` and keeps the synthetic regression path untouched.

## Output layout

- `snapshots/` contains `replay_snapshots.jsonl`, `snapshot_audit.json`, and related trace files
- `replay/` contains `branch_rollouts.jsonl`, `replay_pairs.jsonl`, and replay failures
- `adapter/` contains `cdpo_pairs.jsonl` and the bridge metadata
- `audit/` contains the replay audit summary and CSVs

## Guardrails

The pipeline does not promote synthetic CritiqueWorld pairs into real replay outputs. CDPO output is only built from positive-uplift real replay pairs.
