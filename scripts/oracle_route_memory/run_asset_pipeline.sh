#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${1:-user_simulator}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/outputs/oracle_route_memory/assets/pipeline_${STAMP}"
DISCOVER_DIR="$ROOT_DIR/outputs/oracle_route_memory/assets"
EMB_DIR="$ROOT_DIR/outputs/oracle_route_memory/assets/metadata_embeddings"
ROUTE_DIR="$ROOT_DIR/outputs/oracle_route_memory/assets/proxy_routes_b16_d2"
AUDIT_DIR="$ROOT_DIR/outputs/oracle_route_memory/assets/proxy_asset_audit"

mkdir -p "$OUT_DIR"

python "$ROOT_DIR/scripts/oracle_route_memory/audit_source_data.py"   --data-dir "$DATA_DIR"   --output-dir "$ROOT_DIR/outputs/oracle_route_memory/assets/source_data_audit"

python "$ROOT_DIR/scripts/oracle_route_memory/discover_reusable_assets.py"   --data-dir "$DATA_DIR"   --output-dir "$DISCOVER_DIR"

python "$ROOT_DIR/scripts/oracle_route_memory/export_metadata_embeddings.py"   --data-dir "$DATA_DIR"   --output-dir "$EMB_DIR"   --batch-size 128   --device auto   --normalize true   --resume

python "$ROOT_DIR/scripts/oracle_route_memory/build_proxy_semantic_routes.py"   --item-embedding-path "$EMB_DIR/item_embeddings.npy"   --item-id-path "$EMB_DIR/item_ids.json"   --output-dir "$ROUTE_DIR"   --branching-factor 16   --depth 2   --seed 42   --backend auto

python "$ROOT_DIR/scripts/oracle_route_memory/audit_memory_assets.py"   --data-dir "$DATA_DIR"   --item-embedding-path "$EMB_DIR/item_embeddings.npy"   --item-sid-path "$ROUTE_DIR/item_sid_mapping.json"   --output-dir "$AUDIT_DIR"

ROOT_DIR_ENV="$ROOT_DIR" OUT_DIR_ENV="$OUT_DIR" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR_ENV"])
out_dir = Path(os.environ["OUT_DIR_ENV"])
discover = json.loads((root / "outputs/oracle_route_memory/assets/discovered_assets.json").read_text(encoding="utf-8"))
emb = json.loads((root / "outputs/oracle_route_memory/assets/metadata_embeddings/manifest.json").read_text(encoding="utf-8"))
route = json.loads((root / "outputs/oracle_route_memory/assets/proxy_routes_b16_d2/manifest.json").read_text(encoding="utf-8"))
audit = json.loads((root / "outputs/oracle_route_memory/assets/proxy_asset_audit/audit.json").read_text(encoding="utf-8"))

selected = {
    "embedding_status": "EXPORTED_METADATA_EMBEDDING",
    "route_status": "GENERATED_PROXY_HIERARCHICAL_ROUTE",
    "embedding_manifest": emb,
    "route_manifest": route,
    "audit": audit,
    "discover_summary": {
        "REUSABLE_EMBEDDING_FOUND": discover["REUSABLE_EMBEDDING_FOUND"],
        "REUSABLE_SID_FOUND": discover["REUSABLE_SID_FOUND"],
    },
}
stage_exit_codes = {
    "audit_source_data": 0,
    "discover_reusable_assets": 0,
    "export_metadata_embeddings": 0,
    "build_proxy_semantic_routes": 0,
    "audit_memory_assets": 0,
}

(out_dir / "selected_assets.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "
", encoding="utf-8")
(out_dir / "stage_exit_codes.json").write_text(json.dumps(stage_exit_codes, indent=2, ensure_ascii=False) + "
", encoding="utf-8")
pipeline_summary = {
    "selected_assets": selected,
    "stage_exit_codes": stage_exit_codes,
}
(out_dir / "pipeline_summary.json").write_text(json.dumps(pipeline_summary, indent=2, ensure_ascii=False) + "
", encoding="utf-8")
(out_dir / "pipeline_summary.md").write_text(
    "
".join([
        "# Asset Pipeline Summary",
        "",
        "- Embedding status: `EXPORTED_METADATA_EMBEDDING`",
        "- Route status: `GENERATED_PROXY_HIERARCHICAL_ROUTE`",
        f"- Reusable embedding discovered: `{discover['REUSABLE_EMBEDDING_FOUND']}`",
        f"- Reusable SID discovered: `{discover['REUSABLE_SID_FOUND']}`",
        f"- Coverage ratio: `{audit['coverage_ratio']:.4f}`",
        f"- Route depth distribution: `{audit['route_depth_distribution']}`",
    ]) + "
",
    encoding="utf-8",
)
PY
