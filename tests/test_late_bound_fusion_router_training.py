from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "train_late_bound_fusion_router.py"
    spec = importlib.util.spec_from_file_location("train_late_bound_fusion_router", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_union_uses_full_source_score_maps() -> None:
    module = load_module()
    source_outputs = {
        "learned": {
            "ranked_ids": ["a", "b"],
            "score_map": {"a": 1.0, "b": 0.9, "target": 0.1},
        },
        "residual": {
            "ranked_ids": ["c"],
            "score_map": {"c": 1.0},
        },
    }

    union = module.candidate_union_from_source_outputs(source_outputs, ["learned", "residual"])

    assert union == ["a", "b", "c", "target"]


def test_evaluate_gate_reports_candidate_pool_rank_diagnostics() -> None:
    module = load_module()

    class StaticGate:
        def eval(self) -> None:
            return None

        def __call__(self, sample_features, source_scores, route_scores):
            del sample_features, source_scores, route_scores
            logits = torch.tensor([[0.1, 0.4, 0.2, 0.3]], dtype=torch.float32)
            weights = torch.tensor([[0.2, 0.3, 0.1, 0.4]], dtype=torch.float32)
            return logits, weights

    rows = [
        {
            "sample_id": "row-1",
            "domain": "Book",
            "target_item_id": "target",
            "candidate_ids": ["a", "target", "b", "c"],
            "query_sources": ["learned", "residual", "prefix1_head"],
            "source_scores": np.asarray(
                [
                    [0.9, 0.0, 0.0],
                    [0.2, 0.8, 0.1],
                    [0.1, 0.5, 0.7],
                    [0.0, 0.4, 0.6],
                ],
                dtype=np.float32,
            ),
            "source_presence": np.ones((4, 3), dtype=np.bool_),
            "route_scores": np.zeros(4, dtype=np.float32),
            "sample_features": np.zeros(5, dtype=np.float32),
            "target_index": 1,
            "route_hit": True,
            "candidate_pool_hit": True,
            "candidate_pool_size": 4,
            "num_route_candidates": 2,
            "member_route_hit_count": 3,
            "member_candidate_pool_hit_count": 2,
        }
    ]

    metrics, outputs = module.evaluate_gate(StaticGate(), rows, device="cpu", topk=2)

    assert metrics["Recall@50"] == 1.0
    assert metrics["AvgCandidatePoolMatchRank"] == 1.0
    assert metrics["OracleSourceHit@2Rate"] == 1.0
    assert metrics["AvgOracleSourceMatchRank"] == 1.0
    assert outputs[0]["match_rank"] == 1
    assert outputs[0]["candidate_pool_match_rank"] == 1
    assert outputs[0]["candidate_pool_rank_cutoff"] == 4
    assert outputs[0]["candidate_pool_size"] == 4
    assert outputs[0]["num_route_candidates"] == 2
    assert outputs[0]["source_target_ranks"] == {
        "learned": 2,
        "residual": 1,
        "prefix1_head": 3,
    }
    assert outputs[0]["oracle_source_match_rank"] == 1
    assert outputs[0]["oracle_source_hit_at_topk"] is True
    np.testing.assert_allclose(outputs[0]["source_weights"], [0.2, 0.3, 0.1])
    np.testing.assert_allclose(outputs[0]["route_weight"], 0.4)


def test_output_dir_guard_refuses_non_empty_directory(tmp_path) -> None:
    module = load_module()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        module.ensure_empty_output_dir(output_dir)

    assert (output_dir / "old.txt").read_text(encoding="utf-8") == "keep"


def test_format_metric_renders_missing_values_as_na() -> None:
    module = load_module()

    assert module.format_metric(None) == "n/a"
    assert module.format_metric(0.12345) == "0.1235"


def test_resolve_output_dir_uses_repo_root_for_relative_paths(tmp_path) -> None:
    module = load_module()
    repo_root = tmp_path / "repo"
    absolute = tmp_path / "absolute"

    assert module.resolve_output_dir("outputs/run", repo_root) == repo_root / "outputs" / "run"
    assert module.resolve_output_dir(absolute, repo_root) == absolute
    assert module.resolve_output_dir("outputs/run") == Path("outputs/run")


def test_resolve_repo_path_uses_repo_root_for_relative_inputs(tmp_path) -> None:
    module = load_module()
    repo_root = tmp_path / "repo"
    absolute = tmp_path / "input.json"

    assert module.resolve_repo_path("data/file.json", repo_root) == repo_root / "data" / "file.json"
    assert module.resolve_repo_path(absolute, repo_root) == absolute
    assert module.resolve_repo_path("data/file.json") == Path("data/file.json")
    assert module.resolve_repo_path(None, repo_root) is None
