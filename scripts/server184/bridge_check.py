from pathlib import Path
import json
refine_dir = Path(\ /home/grf/agent-rec/outputs/server184_gimo/gpe_hap_smoke/latest_real\)
out_dir.mkdir(parents=True, exist_ok=True)
out_dir = Path(\ /home/grf/agent-rec/outputs/server184_gimo/bridge/latest_real\)
logs = sorted(refine_dir.glob(\ *refine_log*.json\))
rows = json.loads(logs[-1].read_text(encoding=\ utf-8\)) if logs else []
missing = sorted({k for row in rows for k in [\ follow_value\,\ignore_value\,\over_apply_value\] if k not in row}) if rows else [\follow_value\,\ignore_value\,\over_apply_value\]
payload = {\ status\: \BLOCKED_REAL_BRANCH_VALUES_MISSING\, \refine_log\: str(logs[-1]) if logs else None, \row_count\: len(rows), \missing_fields\: missing}
(out_dir / \ bridge_metadata.json\).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + \\\n\, encoding=\utf-8\)
print(json.dumps(payload, ensure_ascii=False, indent=2))
raise SystemExit(2 if missing else 0)
