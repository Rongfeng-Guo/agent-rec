# CDPO Dataset Materialization

- Input: `C:\Users\grfpa\Documents\Codex\2026-06-04\rongfeng-guo-gimo-https-github-com\GIMO\outputs\gpe_hap_fixture_smoke\export\cdpo_pairs.jsonl`
- Output: `C:\Users\grfpa\Documents\Codex\2026-06-04\rongfeng-guo-gimo-https-github-com\GIMO\outputs\gpe_hap_fixture_smoke\cdpo_dataset_fixture_smoke`
- Status: `COMPLETED_FIXTURE_SMOKE`
- Rows: `6`
- Train: `4`
- Dev: `2`
- Train/dev overlap: `0`
- Git commit: `a6d84a6bf74f751164b55d28d4ac809eb77011e1`

## Notes
- Splits are grouped by `source_ref` to avoid leakage.
- `state_snapshot_hash` is preserved on each row.
- This materialization is intended for bridge validation and smoke runs.
