from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "analyze_route_query_binding_errors.py"
    spec = importlib.util.spec_from_file_location("analyze_route_query_binding_errors", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_rows() -> list[dict]:
    base = {
        "split": "cold_like_validation",
        "policy_name": "fusion_a",
        "domain": "Book",
        "candidate_pool_size": 50,
        "num_route_candidates": 4,
    }
    return [
        {
            **base,
            "sample_id": "hit",
            "route_hit": True,
            "candidate_pool_hit": True,
            "match_rank": 3,
            "member_route_hit_count": 2,
            "member_candidate_pool_hit_count": 2,
            "candidate_pool_match_rank": 3,
            "oracle_source_hit_at_topk": True,
            "oracle_source_match_rank": 2,
        },
        {
            **base,
            "sample_id": "route-miss",
            "route_hit": False,
            "candidate_pool_hit": False,
            "match_rank": None,
            "member_route_hit_count": 0,
            "member_candidate_pool_hit_count": 0,
            "oracle_source_hit_at_topk": False,
            "oracle_source_match_rank": None,
        },
        {
            **base,
            "sample_id": "pool-miss",
            "route_hit": True,
            "candidate_pool_hit": False,
            "match_rank": None,
            "member_route_hit_count": 1,
            "member_candidate_pool_hit_count": 0,
            "oracle_source_hit_at_topk": False,
            "oracle_source_match_rank": None,
        },
        {
            **base,
            "sample_id": "rank-miss",
            "route_hit": True,
            "candidate_pool_hit": True,
            "match_rank": 87,
            "member_route_hit_count": 1,
            "member_candidate_pool_hit_count": 1,
            "candidate_pool_match_rank": 87,
            "oracle_source_hit_at_topk": True,
            "oracle_source_match_rank": 45,
        }
    ]


def test_classify_error_separates_route_pool_and_rank_misses():
    module = load_module()
    rows = sample_rows()

    assert [module.classify_error(row, 50) for row in rows] == [
        "hit_at_k",
        "route_miss_or_unreported",
        "route_hit_pool_miss",
        "pool_hit_rank_miss",
    ]


def test_build_summary_reports_conditional_and_dominant_miss_rates():
    module = load_module()
    summary = module.build_summary(sample_rows(), ["split", "policy_name", "domain"], 50)

    assert len(summary) == 1
    row = summary[0]
    assert row["sample_count"] == 4
    assert row["Hit@50Count"] == 1
    assert row["RouteHitCount"] == 3
    assert row["CandidatePoolHitCount"] == 2
    assert row["RouteMissOrUnreportedCount"] == 1
    assert row["RouteHitPoolMissCount"] == 1
    assert row["PoolHitRankMissCount"] == 1
    assert row["PoolHitGivenRouteHit"] == 2 / 3
    assert row["Hit@50GivenPoolHit"] == 1 / 2
    assert row["DominantMissClass"] == "pool_hit_rank_miss"
    assert row["AvgCandidatePoolMatchRank"] == 45.0
    assert row["AvgPoolHitRankMissMatchRank"] == 87.0
    assert row["OracleSourceHit@50Count"] == 2
    assert row["OracleSourceHit@50Rate"] == 1 / 2
    assert row["AvgOracleSourceMatchRank"] == 23.5


def test_report_writers_emit_csv_json_and_markdown(tmp_path):
    module = load_module()
    rows_path = tmp_path / "selector_rows.json"
    out_dir = tmp_path / "analysis"
    rows_path.write_text(json.dumps(sample_rows()), encoding="utf-8")

    argv = sys.argv
    sys.argv = ["analyze", "--selector-rows", str(rows_path), "--output-dir", str(out_dir), "--top-k", "50"]
    try:
        module.main()
    finally:
        sys.argv = argv

    assert (out_dir / "error_summary.csv").exists()
    assert (out_dir / "error_summary.json").exists()
    assert (out_dir / "error_summary.md").exists()
    manifest = json.loads((out_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    markdown = (out_dir / "error_summary.md").read_text(encoding="utf-8")
    assert manifest["artifacts"]["markdown"] == "error_summary.md"
    assert manifest["next_target"] == module.NEXT_TARGET
    assert "## Next Target" in markdown
    assert module.NEXT_TARGET in markdown


def test_refuses_non_empty_output_dir(tmp_path):
    module = load_module()
    rows_path = tmp_path / "selector_rows.json"
    out_dir = tmp_path / "analysis"
    out_dir.mkdir()
    (out_dir / "old.txt").write_text("keep", encoding="utf-8")
    rows_path.write_text(json.dumps(sample_rows()), encoding="utf-8")

    argv = sys.argv
    sys.argv = ["analyze", "--selector-rows", str(rows_path), "--output-dir", str(out_dir)]
    try:
        with pytest.raises(FileExistsError, match="non-empty output directory"):
            module.main()
    finally:
        sys.argv = argv

    assert (out_dir / "old.txt").read_text(encoding="utf-8") == "keep"


def test_repo_root_relative_output_dir_is_resolved_from_outside_cwd(tmp_path, monkeypatch):
    module = load_module()
    repo_root = tmp_path / "repo"
    outside_cwd = tmp_path / "outside"
    repo_root.mkdir()
    outside_cwd.mkdir()
    rows_path = repo_root / "selector_rows.json"
    rows_path.write_text(json.dumps(sample_rows()), encoding="utf-8")
    monkeypatch.chdir(outside_cwd)

    argv = sys.argv
    sys.argv = [
        "analyze",
        "--selector-rows",
        "selector_rows.json",
        "--repo-root",
        str(repo_root),
        "--output-dir",
        "outputs/analysis",
    ]
    try:
        module.main()
    finally:
        sys.argv = argv

    assert (repo_root / "outputs" / "analysis" / "analysis_manifest.json").exists()
    assert not (outside_cwd / "outputs").exists()
