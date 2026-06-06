import json
from pathlib import Path


REFINE_DIR = Path("/home/grf/agent-rec/outputs/server184_gimo/gpe_hap_smoke/latest_real")
OUT_DIR = Path("/home/grf/agent-rec/outputs/server184_gimo/bridge/latest_real")
REQUIRED_FIELDS = ("follow_value", "ignore_value", "over_apply_value")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logs = sorted(REFINE_DIR.glob("*refine_log*.json"))
    rows = json.loads(logs[-1].read_text(encoding="utf-8")) if logs else []
    missing = (
        sorted({field for row in rows for field in REQUIRED_FIELDS if field not in row})
        if rows
        else list(REQUIRED_FIELDS)
    )
    payload = {
        "status": "OK" if not missing else "BLOCKED_REAL_BRANCH_VALUES_MISSING",
        "refine_log": str(logs[-1]) if logs else None,
        "row_count": len(rows),
        "missing_fields": missing,
    }
    (OUT_DIR / "bridge_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(2 if missing else 0)


if __name__ == "__main__":
    main()
