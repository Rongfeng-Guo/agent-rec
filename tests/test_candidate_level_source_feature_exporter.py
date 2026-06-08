from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "export_candidate_level_source_features.py"
    spec = importlib.util.spec_from_file_location("export_candidate_level_source_features", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compute_source_local_ranks_respects_presence_mask() -> None:
    module = load_module()
    source_scores = np.asarray(
        [
            [0.9, 0.0],
            [0.2, 0.8],
            [0.1, 0.5],
        ],
        dtype=np.float32,
    )
    source_presence = np.asarray(
        [
            [True, False],
            [True, True],
            [False, True],
        ],
        dtype=np.bool_,
    )

    ranks = module.compute_source_local_ranks(source_scores, source_presence)

    assert ranks == [
        [1, None],
        [2, 1],
        [None, 2],
    ]


def test_candidate_feature_rows_include_source_rank_and_aggregate_features() -> None:
    module = load_module()
    gate_row = {
        "sample_id": "sample-1",
        "domain": "Book",
        "target_item_id": "target",
        "candidate_ids": ["a", "target", "b"],
        "query_sources": ["learned", "residual"],
        "source_scores": np.asarray(
            [
                [0.9, 0.0],
                [0.2, 0.8],
                [0.1, 0.5],
            ],
            dtype=np.float32,
        ),
        "source_presence": np.asarray(
            [
                [True, False],
                [True, True],
                [True, True],
            ],
            dtype=np.bool_,
        ),
        "route_scores": np.asarray([0.3, 0.7, 0.1], dtype=np.float32),
        "sample_features": np.asarray([0.4, 0.8, 0.2, 0.1, 0.6], dtype=np.float32),
        "target_index": 1,
        "route_hit": True,
        "candidate_pool_hit": True,
        "candidate_pool_size": 3,
        "num_route_candidates": 2,
        "member_route_hit_count": 2,
        "member_candidate_pool_hit_count": 2,
    }

    rows = module.candidate_feature_rows_from_gate_row(gate_row, split="cold_like_val")
    target_row = rows[1]

    assert len(rows) == 3
    assert target_row["label"] == 1
    assert target_row["learned_rank"] == 2
    assert target_row["residual_rank"] == 1
    assert target_row["best_source"] == "residual"
    assert target_row["best_source_rank"] == 1
    assert target_row["source_presence_count"] == 2
    np.testing.assert_allclose(target_row["route_score"], 0.7)
    np.testing.assert_allclose(target_row["history_len_norm"], 0.4)


def test_ensure_empty_output_dir_rejects_non_empty_dir(tmp_path) -> None:
    module = load_module()
    output_dir = tmp_path / "export"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("do not overwrite\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists and is not empty"):
        module.ensure_empty_output_dir(output_dir)


def test_render_report_includes_next_target() -> None:
    module = load_module()
    summary = {
        "query_sources": ["learned", "residual"],
        "fusion_config": {"prefix1_beam": 4, "per_route_topk": 500},
        "files": {
            "train_candidate_rows": "train.jsonl",
            "cold_like_candidate_rows": "cold.jsonl",
            "sample_groups": "groups.json",
        },
        "splits": [
            {
                "split": "cold_like_val",
                "sample_count": 1,
                "candidate_row_count": 3,
                "positive_row_count": 1,
                "candidate_pool_hit_rate": 1.0,
                "avg_candidate_pool_size": 3.0,
                "oracle_source_hit_at_50_rate": 1.0,
                "avg_oracle_source_match_rank": None,
            }
        ],
        "next_target": module.NEXT_TARGET,
    }

    rendered = module.render_report(summary, protocol_hash="abc123")

    assert "## Next Target" in rendered
    assert "locked h100/h300 model.pkl files" in rendered
