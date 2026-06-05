#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/server184_env}"
PYTHON_BIN="$(command -v python3 || command -v python || true)"
mkdir -p "${OUTPUT_DIR}"

MODEL_ROOTS_DEFAULT="/data:/mnt:/home:/root:/workspace:/models:/share:/home/share"
MODEL_ROOTS="${MODEL_SCAN_ROOTS:-${MODEL_ROOTS_DEFAULT}}"
DATA_ROOTS_DEFAULT="${ROOT_DIR}:/data:/mnt:/home/share"
DATA_ROOTS="${DATA_SCAN_ROOTS:-${DATA_ROOTS_DEFAULT}}"

"${PYTHON_BIN:-python}" - <<'PY' "${ROOT_DIR}" "${OUTPUT_DIR}" "${MODEL_ROOTS}" "${DATA_ROOTS}"
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

root_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
model_roots = [Path(p) for p in sys.argv[3].split(":") if p]
data_roots = [Path(p) for p in sys.argv[4].split(":") if p]

model_hits = []
for root in model_roots:
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        if path.name not in {"config.json", "tokenizer_config.json"} and path.suffix != ".safetensors":
            continue
        text = str(path).lower()
        if not any(token in text for token in ["llama", "qwen", "instruct", "toolalpaca", "model"]):
            continue
        model_dir = path.parent
        size_bytes = sum(f.stat().st_size for f in model_dir.glob("*.safetensors") if f.is_file())
        model_hits.append(
            {
                "model_name": model_dir.name,
                "model_path": str(model_dir),
                "tokenizer_path": str(model_dir / "tokenizer_config.json") if (model_dir / "tokenizer_config.json").exists() else None,
                "weight_files": [str(f) for f in sorted(model_dir.glob("*.safetensors"))[:20]],
                "estimated_size_gb": round(size_bytes / 1024**3, 2),
                "usable_for_vllm": any("instruct" in model_dir.name.lower() or "qwen" in model_dir.name.lower() or "llama" in model_dir.name.lower() for _ in [0]),
                "usable_for_llamafactory": True,
            }
        )
model_hits = sorted({item["model_path"]: item for item in model_hits}.values(), key=lambda item: item["model_path"])[:200]

data_hits = []
for root in data_roots:
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        lowered = path.name.lower()
        if any(token in lowered for token in ["amazon", "yelp", "persona", "task", "refine_log", "faiss_index", "metadata.json"]):
            data_hits.append(str(path))
data_hits = data_hits[:500]

real_trace_hits = [
    path
    for path in data_hits
    if "refine_log" in Path(path).name.lower() and "fixture" not in path.lower()
]

status = {
    "model_status": "MODEL_FOUND" if model_hits else "MODEL_MISSING",
    "data_status": "DATA_FOUND" if any("recommend_data.json" in path or "ask_data.json" in path or "search_data.json" in path for path in data_hits) else "DATA_MISSING",
    "index_status": "INDEX_FOUND" if any(Path(path).name.lower() == "faiss_index.bin" for path in data_hits) else "INDEX_MISSING",
    "real_trace_status": "REAL_TRACE_FOUND" if real_trace_hits else "REAL_TRACE_MISSING",
}

payload = {
    "status": status,
    "scan_roots": {
        "model_roots": [str(p) for p in model_roots],
        "data_roots": [str(p) for p in data_roots],
    },
    "models": model_hits,
    "data_and_index_hits": data_hits,
    "real_trace_hits": real_trace_hits,
}
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "resource_inventory.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
md = [
    "# Server184 Resource Inventory",
    "",
    f"- Model status: `{status['model_status']}`",
    f"- Data status: `{status['data_status']}`",
    f"- Index status: `{status['index_status']}`",
    f"- Real trace status: `{status['real_trace_status']}`",
    f"- Model candidates: `{len(model_hits)}`",
    f"- Data/index hits: `{len(data_hits)}`",
]
if model_hits:
    md.append("")
    md.append("## Model Candidates")
    for item in model_hits[:20]:
        md.append(f"- `{item['model_name']}` at `{item['model_path']}` ({item['estimated_size_gb']} GB)")
if data_hits:
    md.append("")
    md.append("## Data and Index Hits")
    for item in data_hits[:50]:
        md.append(f"- `{item}`")
(output_dir / "resource_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps(status, ensure_ascii=False))
PY
