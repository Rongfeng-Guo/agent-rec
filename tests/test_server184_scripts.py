from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER184 = ROOT / "scripts" / "server184"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_server184_shell_scripts_exist():
    expected = {
        "check_env.sh",
        "discover_resources.sh",
        "serve_vllm_single.sh",
        "serve_vllm_replicas.sh",
        "stop_vllm.sh",
        "run_real_gimo_rollout_smoke.sh",
        "run_real_rollout_bridge.sh",
        "run_gpu_smoke_pipeline.sh",
        "run_sft_50step_smoke.sh",
        "run_sft_4gpu_sweep.sh",
        "run_cdpo_20step_smoke.sh",
        "run_cdpo_ds3.sh",
    }
    assert expected <= {path.name for path in SERVER184.glob("*.sh")}


def test_server184_shell_scripts_use_strict_mode():
    for path in SERVER184.glob("*.sh"):
        text = read(path)
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")


def test_server184_scripts_do_not_hardcode_author_paths():
    banned = [
        "C:\\Users\\grfpa",
        "/Users/grfpa",
        "your/work/dir",
    ]
    for path in SERVER184.glob("*.sh"):
        text = read(path)
        for needle in banned:
            assert needle not in text, f"{needle} found in {path.name}"


def test_env_example_contains_required_variables():
    text = read(ROOT / ".env.server184.example")
    for name in [
        "MODEL_NAME_OR_PATH",
        "LLAMAFACTORY_MODEL_NAME_OR_PATH",
        "GIMO_DATA_ROOT",
        "GIMO_INDEX_ROOT",
        "GPE_HAP_INPUT",
        "GPE_HAP_OUTPUT_DIR",
        "BRIDGE_OUTPUT_DIR",
        "VLLM_PORT",
        "CUDA_VISIBLE_DEVICES",
        "MAX_MODEL_LEN",
        "GPU_MEMORY_UTILIZATION",
        "SMOKE_SAMPLE_LIMIT",
    ]:
        assert f"export {name}=" in text


def test_rollout_smoke_uses_timestamped_output():
    text = read(SERVER184 / "run_real_gimo_rollout_smoke.sh")
    assert 'TIMESTAMP="$(date +%Y%m%d_%H%M%S)"' in text
    assert 'RUN_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"' in text


def test_real_bridge_does_not_fallback_to_fixture():
    text = read(SERVER184 / "run_real_rollout_bridge.sh")
    assert "BLOCKED_REAL_LOG_MISSING" in text
    assert "fixture" not in text.lower()


def test_gpu_pipeline_stops_vllm_on_failure():
    text = read(SERVER184 / "run_gpu_smoke_pipeline.sh")
    assert 'trap cleanup EXIT' in text
    assert 'bash "${ROOT_DIR}/scripts/server184/stop_vllm.sh"' in text
    assert 'ROLLOUT_ROOT="${GPE_HAP_OUTPUT_DIR:-${ROOT_DIR}/outputs/server184_real_rollout_smoke}"' in text
    assert 'BLOCKED_REAL_LOG_MISSING' in text


def test_resource_discovery_filters_hidden_paths_and_uses_specific_index_marker():
    text = read(SERVER184 / "discover_resources.sh")
    assert 'any(part.startswith(".") for part in path.parts)' in text
    assert 'Path(path).name.lower() == "faiss_index.bin"' in text
    assert '.git/index' not in text


def test_vllm_single_prefers_python_bin_fallback():
    text = read(SERVER184 / "serve_vllm_single.sh")
    assert 'PYTHON_BIN="$(command -v python3 || command -v python || true)"' in text
    assert '"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server' in text
