#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/server184_env}"
HOST_PATTERN="${SERVER184_HOST_PATTERN:-184}"
PYTHON_BIN="$(command -v python3 || command -v python || true)"
mkdir -p "${OUTPUT_DIR}"

HOSTNAME_VALUE="$(hostname 2>/dev/null || echo unknown)"
NVIDIA_SMI_OUTPUT="$(nvidia-smi 2>&1 || true)"
TOPO_OUTPUT="$(nvidia-smi topo -m 2>&1 || true)"
FREE_OUTPUT="$(free -h 2>&1 || true)"
DF_OUTPUT="$(df -h 2>&1 || true)"
PYTHON_VERSION="$("${PYTHON_BIN:-python}" --version 2>&1 || true)"

TORCH_JSON="$("${PYTHON_BIN:-python}" - <<'PY'
import json
try:
    import torch
    payload = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_count": int(torch.cuda.device_count()),
        "gpus": [],
    }
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        payload["gpus"].append(
            {
                "index": i,
                "name": p.name,
                "total_memory_gb": round(p.total_memory / 1024**3, 2),
            }
        )
except Exception as exc:
    payload = {"torch_error": str(exc), "cuda_available": False, "gpu_count": 0, "gpus": []}
print(json.dumps(payload, ensure_ascii=False))
PY
)"

"${PYTHON_BIN:-python}" - <<'PY' "${OUTPUT_DIR}" "${HOSTNAME_VALUE}" "${HOST_PATTERN}" "${NVIDIA_SMI_OUTPUT}" "${TOPO_OUTPUT}" "${FREE_OUTPUT}" "${DF_OUTPUT}" "${PYTHON_BIN}" "${PYTHON_VERSION}" "${TORCH_JSON}"
from __future__ import annotations
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
hostname_value = sys.argv[2]
host_pattern = sys.argv[3]
nvidia_smi = sys.argv[4]
topo = sys.argv[5]
free_out = sys.argv[6]
df_out = sys.argv[7]
python_bin = sys.argv[8]
python_version = sys.argv[9]
torch_payload = json.loads(sys.argv[10])

is_server184 = host_pattern in hostname_value
gpu_count = int(torch_payload.get("gpu_count", 0) or 0)
cuda_available = bool(torch_payload.get("cuda_available"))

if not is_server184:
    status = "NOT_ON_SERVER184"
elif not cuda_available:
    status = "CUDA_UNAVAILABLE"
elif gpu_count != 4:
    status = "GPU_COUNT_MISMATCH"
else:
    status = "SERVER184_GPU_READY"

payload = {
    "status": status,
    "hostname": hostname_value,
    "server184_host_pattern": host_pattern,
    "python_bin": python_bin,
    "python_version": python_version,
    "nvidia_smi": nvidia_smi,
    "nvidia_smi_topo": topo,
    "free_h": free_out,
    "df_h": df_out,
    "torch": torch_payload,
}
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "env_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
md = [
    "# Server184 Environment Report",
    "",
    f"- Status: `{status}`",
    f"- Hostname: `{hostname_value}`",
    f"- Python: `{python_version}`",
    f"- Python path: `{python_bin or 'missing'}`",
    f"- CUDA available: `{cuda_available}`",
    f"- GPU count: `{gpu_count}`",
]
for gpu in torch_payload.get("gpus", []):
    md.append(f"- GPU {gpu['index']}: `{gpu['name']}` (`{gpu['total_memory_gb']} GB`)")
if torch_payload.get("torch_error"):
    md.append(f"- Torch error: `{torch_payload['torch_error']}`")
md.extend(
    [
        "",
        "## Raw Checks",
        "### nvidia-smi",
        "```text",
        nvidia_smi.strip() or "unavailable",
        "```",
        "### nvidia-smi topo -m",
        "```text",
        topo.strip() or "unavailable",
        "```",
        "### free -h",
        "```text",
        free_out.strip() or "unavailable",
        "```",
        "### df -h",
        "```text",
        df_out.strip() or "unavailable",
        "```",
    ]
)
(output_dir / "env_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(status)
PY
