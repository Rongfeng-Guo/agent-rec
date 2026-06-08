from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_PATTERN = re.compile(r"scripts/oracle_route_memory/[A-Za-z0-9_]+\.py")
H5_HANDOFF_SCRIPTS = [
    "scripts/oracle_route_memory/analyze_route_query_binding_errors.py",
    "scripts/oracle_route_memory/audit_h5_fresh_confirmation_bundle.py",
    "scripts/oracle_route_memory/check_h5_fresh_readiness.py",
    "scripts/oracle_route_memory/combine_ranker_outputs_by_domain.py",
    "scripts/oracle_route_memory/export_candidate_level_source_features.py",
    "scripts/oracle_route_memory/prepare_h5_fresh_confirmation_bundle.py",
    "scripts/oracle_route_memory/register_h5_fresh_split.py",
    "scripts/oracle_route_memory/render_h5_fresh_confirmation_report.py",
    "scripts/oracle_route_memory/score_candidate_level_source_ranker.py",
    "scripts/oracle_route_memory/train_candidate_level_source_ranker.py",
    "scripts/oracle_route_memory/validate_h5_handoff_index.py",
    "scripts/oracle_route_memory/validate_h5_loaded_model_replay.py",
    "scripts/oracle_route_memory/validate_locked_policy_manifest.py",
]


def repro_command_scripts(repo_root: Path) -> list[str]:
    repro_path = repo_root / "experiments" / "h5-candidate-level-source-reranker" / "repro_commands.md"
    return sorted(set(SCRIPT_PATTERN.findall(repro_path.read_text(encoding="utf-8"))))


def test_h5_handoff_cli_smoke_matches_repro_commands() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert repro_command_scripts(repo_root) == H5_HANDOFF_SCRIPTS


def test_h5_handoff_scripts_show_help_without_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for script in H5_HANDOFF_SCRIPTS:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=repo_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, f"{script} failed: {result.stderr}"
        assert "usage:" in result.stdout
