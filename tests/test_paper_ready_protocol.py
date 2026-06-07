from __future__ import annotations

import importlib.util
import json
import math
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from genrec.training.paper_ready_protocol import (
    bootstrap_paired_delta,
    build_random_matched_size_bucket_rows,
    filter_selected_policy_rows,
    read_csv_rows,
    read_jsonl,
    summarize_selected_policy_rows,
)


def load_bundle_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "build_protocol_v2_bundle.py"
    spec = importlib.util.spec_from_file_location("build_protocol_v2_bundle", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_reference_bundle_mode_copies_gold_tables_and_writes_repro_metadata(tmp_path: Path) -> None:
    module = load_bundle_module()

    protocol_dir = tmp_path / "protocol"
    selector_dir = tmp_path / "selector"
    locked_eval_dir = tmp_path / "locked"
    eval_smoke_dir = tmp_path / "eval_smoke"
    reference_bundle_dir = tmp_path / "reference_bundle"
    comparison_dir = tmp_path / "comparison"
    output_dir = tmp_path / "out"

    _write(protocol_dir / "split_manifest.json", "{}\n")
    _write(protocol_dir / "leakage_audit.json", "{}\n")
    _write(protocol_dir / "leakage_audit.md", "audit\n")
    _write(selector_dir / "fusion_config.json", json.dumps({"locked": True}) + "\n")
    _write(
        locked_eval_dir / "results.jsonl",
        json.dumps(
            {
                "query_source": "selected_policy",
                "mode": "validation_selected",
                "subset": "cold",
                "domain": "Book",
                "sample_id": "s1",
                "match_rank": 7,
                "candidate_pool_hit": True,
                "candidate_pool_size": 90,
                "latency_ms": 1.25,
                "route_candidates": [],
                "true_route": "1|1",
            }
        )
        + "\n",
    )
    _write(
        eval_smoke_dir / "per_sample_results.jsonl",
        json.dumps({"mode": "metadata", "sample_id": "s1", "domain": "Book", "match_rank": None}) + "\n",
    )
    _write(
        comparison_dir / "comparison.csv",
        "method_key,display_name,family,claimable,subset,domain,sample_count,Recall@10,Recall@20,Recall@50,NDCG@10,NDCG@20,NDCG@50,MRR@50,notes\n"
        "predicted_route_validation_selected,Predicted Route Validation-Selected (Explicit Script OldV0),predicted_route,True,cold,Book,1,0.0,0.0,0.0,0.1,0.2,0.3,0.4,ref row\n",
    )
    _write(tmp_path / "progress_report.md", "progress\n")

    _write(reference_bundle_dir / "official_comparison.csv", "gold,table\n1,2\n")
    _write(reference_bundle_dir / "official_comparison.md", "official md\n")
    _write(reference_bundle_dir / "bootstrap_comparison.csv", "bootstrap,table\n3,4\n")
    _write(reference_bundle_dir / "bootstrap_comparison.json", "[]\n")
    _write(reference_bundle_dir / "bootstrap_report.md", "bootstrap md\n")
    _write(reference_bundle_dir / "latency_summary.csv", "latency,table\n5,6\n")
    _write(reference_bundle_dir / "method_config.json", '{"copied": true}\n')
    _write(reference_bundle_dir / "README.md", "reference readme\n")

    module.build_eval_samples = lambda data_dir, split, cold_only: [
        {"sample_id": "s1", "cold": True, "domain": "Book"}
    ]
    module._git_commit = lambda repo_root: "deadbeef"

    old_argv = sys.argv
    try:
        sys.argv = [
            "build_protocol_v2_bundle.py",
            "--output-dir",
            str(output_dir),
            "--protocol-dir",
            str(protocol_dir),
            "--selector-dir",
            str(selector_dir),
            "--locked-eval-dir",
            str(locked_eval_dir),
            "--eval-smoke-dir",
            str(eval_smoke_dir),
            "--official-comparison-csv",
            str(comparison_dir / "comparison.csv"),
            "--progress-report-md",
            str(tmp_path / "progress_report.md"),
            "--reference-bundle-dir",
            str(reference_bundle_dir),
            "--selector-command",
            "selector_cmd --cuda:0",
            "--locked-eval-command",
            "locked_eval_cmd --cuda:0",
        ]
        module.main()
    finally:
        sys.argv = old_argv

    assert (output_dir / "official_comparison.csv").read_text(encoding="utf-8") == "gold,table\n1,2\n"
    assert (output_dir / "bootstrap_comparison.csv").read_text(encoding="utf-8") == "bootstrap,table\n3,4\n"
    assert (output_dir / "latency_summary.csv").read_text(encoding="utf-8") == "latency,table\n5,6\n"
    assert json.loads((output_dir / "method_config.json").read_text(encoding="utf-8")) == {"copied": True}
    assert (output_dir / "README.md").read_text(encoding="utf-8") == "reference readme\n"

    reproduce = (output_dir / "reproduce.sh").read_text(encoding="utf-8")
    assert "selector_cmd --cuda:0" in reproduce
    assert "locked_eval_cmd --cuda:0" in reproduce
    assert "--reference-bundle-dir" in reproduce

    run_metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert run_metadata["git_commit"] == "deadbeef"
    assert run_metadata["reference_bundle_dir"] == str(reference_bundle_dir.resolve())
    assert run_metadata["claimable_cold_recall50"] == 1.0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    _write(path, "".join(json.dumps(row) + "\n" for row in rows))


def _patch_catalog_loaders(module) -> None:
    module.load_item_embeddings = lambda data_dir, item_embedding_path: {
        "item_a": [0.0],
        "item_b": [0.0],
        "item_c": [0.0],
        "item_d": [0.0],
    }
    module.load_route_mapping = lambda item_sid_path: {}
    module.build_training_examples = lambda data_dir, item_embeddings, route_mapping, max_history: [
        SimpleNamespace(target_item_id="item_a"),
        SimpleNamespace(target_item_id="item_a"),
        SimpleNamespace(target_item_id="item_d"),
    ]


def test_non_reference_mode_builds_complete_native_bundle_with_duplicate_sample_ids(tmp_path: Path) -> None:
    module = load_bundle_module()

    protocol_dir = tmp_path / "protocol"
    selector_dir = tmp_path / "selector"
    locked_eval_dir = tmp_path / "locked"
    eval_smoke_dir = tmp_path / "eval_smoke"
    output_dir = tmp_path / "out"

    _write(protocol_dir / "split_manifest.json", "{}\n")
    _write(protocol_dir / "leakage_audit.json", "{}\n")
    _write(protocol_dir / "leakage_audit.md", "audit\n")
    _write(selector_dir / "fusion_config.json", '{"fallback": true}\n')

    selected_rows = [
        {
            "query_source": "selected_policy",
            "mode": "validation_selected",
            "subset": "cold",
            "domain": "Book",
            "sample_id": "dup",
            "target_item_id": "item_a",
            "match_rank": 1,
            "candidate_pool_hit": True,
            "candidate_pool_size": 2,
            "latency_ms": 1.0,
            "route_candidates": [],
            "true_route": "1|1",
        },
        {
            "query_source": "selected_policy",
            "mode": "validation_selected",
            "subset": "cold",
            "domain": "Book",
            "sample_id": "dup",
            "target_item_id": "item_b",
            "match_rank": None,
            "candidate_pool_hit": False,
            "candidate_pool_size": 3,
            "latency_ms": 2.0,
            "route_candidates": [],
            "true_route": "1|2",
        },
        {
            "query_source": "selected_policy",
            "mode": "validation_selected",
            "subset": "cold",
            "domain": "Game",
            "sample_id": "s2",
            "target_item_id": "item_c",
            "match_rank": 51,
            "candidate_pool_hit": True,
            "candidate_pool_size": 1,
            "latency_ms": 3.0,
            "route_candidates": [],
            "true_route": "2|1",
        },
    ]
    _write_jsonl(locked_eval_dir / "results.jsonl", selected_rows)

    eval_rows = []
    for row, metadata_rank, p1_rank, p2_rank in zip(selected_rows, [None, 3, None], [10, None, 20], [1, 2, 3]):
        eval_rows.append(
            {
                "query_source": "selected_policy",
                "mode": "metadata",
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "target": row["target_item_id"],
                "candidate_count": 4,
                "match_rank": metadata_rank,
                "latency_ms": 0.5,
            }
        )
        eval_rows.append(
            {
                "mode": "oracle_route",
                "prefix_len": 1,
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "target": row["target_item_id"],
                "candidate_count": 4,
                "match_rank": p1_rank,
                "latency_ms": 0.25,
            }
        )
        eval_rows.append(
            {
                "mode": "oracle_route",
                "prefix_len": 2,
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "target": row["target_item_id"],
                "candidate_count": 4,
                "match_rank": p2_rank,
                "latency_ms": 0.125,
            }
        )
    _write_jsonl(eval_smoke_dir / "per_sample_results.jsonl", eval_rows)
    _write(tmp_path / "progress_report.md", "progress\n")

    _patch_catalog_loaders(module)
    module._git_commit = lambda repo_root: "deadbeef"

    old_argv = sys.argv
    try:
        sys.argv = [
            "build_protocol_v2_bundle.py",
            "--output-dir",
            str(output_dir),
            "--protocol-dir",
            str(protocol_dir),
            "--selector-dir",
            str(selector_dir),
            "--locked-eval-dir",
            str(locked_eval_dir),
            "--eval-smoke-dir",
            str(eval_smoke_dir),
            "--progress-report-md",
            str(tmp_path / "progress_report.md"),
            "--bootstrap-reps",
            "20",
            "--data-dir",
            str(tmp_path / "data"),
            "--item-embedding-path",
            str(tmp_path / "item_embeddings.npy"),
            "--item-sid-path",
            str(tmp_path / "item_sid_mapping.json"),
        ]
        module.main()
    finally:
        sys.argv = old_argv

    expected_files = {
        "README.md",
        "reproduce.sh",
        "method_config.json",
        "fusion_config.json",
        "split_manifest.json",
        "leakage_audit.json",
        "official_comparison.csv",
        "official_comparison.md",
        "bootstrap_comparison.csv",
        "bootstrap_report.md",
        "route_error_breakdown.csv",
        "candidate_pool_breakdown.csv",
        "latency_summary.csv",
        "run_metadata.json",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})
    assert json.loads((output_dir / "method_config.json").read_text(encoding="utf-8")) == {"fallback": True}

    official_rows = read_csv_rows(output_dir / "official_comparison.csv")
    selected_all = next(
        row
        for row in official_rows
        if row["method_key"] == "predicted_route_validation_selected" and row["subset"] == "cold" and row["domain"] == "ALL"
    )
    assert int(selected_all["sample_count"]) == 3
    assert float(selected_all["Recall@50"]) == pytest.approx(1.0 / 3.0)
    assert float(selected_all["NDCG@50"]) == pytest.approx(1.0 / 3.0)
    assert float(selected_all["MRR"]) == pytest.approx(1.0 / 3.0)
    assert float(selected_all["AverageCandidatePoolSize"]) == pytest.approx(2.0)

    random_all = next(
        row
        for row in official_rows
        if row["method_key"] == "random_matched_size_bucket" and row["subset"] == "cold" and row["domain"] == "ALL"
    )
    assert int(random_all["sample_count"]) == 3

    bootstrap_rows = read_csv_rows(output_dir / "bootstrap_comparison.csv")
    metadata_bootstrap = next(row for row in bootstrap_rows if row["baseline_method_key"] == "metadata_global_mean_query")
    assert int(metadata_bootstrap["sample_count"]) == 3
    assert float(metadata_bootstrap["paired_delta_mean"]) == pytest.approx(0.0)
    assert float(metadata_bootstrap["win_rate"]) == pytest.approx(1.0 / 3.0)

    run_metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert run_metadata["copied_reference_tables"] is False
    assert run_metadata["claimable_cold_recall50"] == pytest.approx(1.0 / 3.0)


def test_duplicate_sample_id_summary_and_bootstrap_keep_all_pairs() -> None:
    selected_rows = [
        {"sample_id": "dup", "subset": "cold", "domain": "Book", "match_rank": 1, "candidate_pool_hit": True, "candidate_pool_size": 2},
        {"sample_id": "dup", "subset": "cold", "domain": "Book", "match_rank": None, "candidate_pool_hit": False, "candidate_pool_size": 3},
        {"sample_id": "dup", "subset": "cold", "domain": "Book", "match_rank": 60, "candidate_pool_hit": True, "candidate_pool_size": 4},
    ]
    baseline_rows = [
        {"sample_id": "dup", "subset": "cold", "domain": "Book", "match_rank": None},
        {"sample_id": "dup", "subset": "cold", "domain": "Book", "match_rank": 2},
        {"sample_id": "dup", "subset": "cold", "domain": "Book", "match_rank": 60},
    ]

    summary = summarize_selected_policy_rows(selected_rows)
    cold_all = next(row for row in summary if row["subset"] == "cold" and row["domain"] == "ALL")
    assert cold_all["sample_count"] == 3
    assert cold_all["Recall@50"] == pytest.approx(1.0 / 3.0)
    assert cold_all["CandidatePoolHitRate@50"] == pytest.approx(2.0 / 3.0)

    bootstrap = bootstrap_paired_delta(selected_rows, baseline_rows, reps=20, seed=7)
    assert bootstrap["sample_count"] == 3
    assert bootstrap["paired_delta_mean"] == pytest.approx(0.0)
    assert bootstrap["win_rate"] == pytest.approx(1.0 / 3.0)


def test_random_matched_size_bucket_rows_are_reproducible_and_order_preserving() -> None:
    selected_rows = [
        {"sample_id": "dup", "target_item_id": "item_a", "subset": "cold", "domain": "Book", "candidate_pool_size": 2},
        {"sample_id": "dup", "target_item_id": "item_b", "subset": "cold", "domain": "Book", "candidate_pool_size": 3},
        {"sample_id": "s2", "target_item_id": "item_c", "subset": "warm", "domain": "Game", "candidate_pool_size": 10},
    ]
    catalog = ["item_a", "item_b", "item_c", "item_d"]

    first = build_random_matched_size_bucket_rows(selected_rows, catalog, seed=42)
    second = build_random_matched_size_bucket_rows(selected_rows, catalog, seed=42)

    assert first == second
    assert [row["sample_id"] for row in first] == ["dup", "dup", "s2"]
    assert [row["candidate_pool_size"] for row in first] == [2, 3, 4]
    assert len(first) == len(selected_rows)


def test_frozen_selected_validation_result_does_not_regress_when_artifact_exists() -> None:
    results_path = Path("outputs/oracle_route_memory/validation_fusion_locked_cold_explicit_script_oldv0_rerun_20260607/results.jsonl")
    if not results_path.exists():
        pytest.skip("Frozen rerun artifact is not available in this checkout.")

    selected_rows = filter_selected_policy_rows(read_jsonl(results_path))
    cold_rows = [row for row in selected_rows if row.get("subset") == "cold"]
    summary = summarize_selected_policy_rows(selected_rows)
    selected_cold_all = next(row for row in summary if row["subset"] == "cold" and row["domain"] == "ALL")

    assert len(selected_rows) == 399
    assert len(cold_rows) == 345
    assert selected_cold_all["Recall@50"] == pytest.approx(0.04927536231884058)
