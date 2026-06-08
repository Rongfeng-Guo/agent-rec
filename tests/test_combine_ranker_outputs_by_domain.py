from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "oracle_route_memory" / "combine_ranker_outputs_by_domain.py"
    spec = importlib.util.spec_from_file_location("combine_ranker_outputs_by_domain", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(sample_id: str, domain: str, rank: int | None) -> dict:
    return {
        "sample_id": sample_id,
        "domain": domain,
        "target_item_id": f"target-{sample_id}",
        "match_rank": rank,
        "route_hit": True,
        "candidate_pool_hit": rank is not None,
        "candidate_pool_match_rank": rank,
    }


def test_combine_rows_by_domain_selects_mapped_source() -> None:
    module = load_module()
    sources = {
        "book_model": [row("s1", "Book", 3), row("s2", "Game", 80)],
        "game_model": [row("s1", "Book", 90), row("s2", "Game", 4)],
    }

    combined = module.combine_rows_by_domain(
        sources,
        {"Book": "book_model", "Game": "game_model"},
        default_source="book_model",
    )
    metric = module.summarize_eval_rows(combined, topk=50)

    assert [row["selected_source"] for row in combined] == ["book_model", "game_model"]
    assert [row["match_rank"] for row in combined] == [3, 4]
    assert metric["Recall@50"] == 1.0


def test_render_report_includes_next_target_and_handles_missing_rank_metrics() -> None:
    module = load_module()
    sources = {
        "book_model": [row("s1", "Book", None)],
        "game_model": [row("s1", "Book", None)],
    }
    combined = module.combine_rows_by_domain(
        sources,
        {"Book": "book_model"},
        default_source="book_model",
    )
    summary = {
        "metric": module.summarize_eval_rows(combined, topk=50),
        "source_files": {"book_model": "book.json", "game_model": "game.json"},
        "next_target": module.NEXT_TARGET,
    }

    rendered = module.render_report(summary, {"Book": "book_model"}, summary["source_files"], topk=50)

    assert "## Next Target" in rendered
    assert "fresh confirmation report" in rendered
    assert "- AvgCandidatePoolMatchRank: `n/a`" in rendered


def test_main_rejects_non_empty_output_dir(tmp_path, monkeypatch) -> None:
    module = load_module()
    book_rows = [row("s1", "Book", 3)]
    book_path = tmp_path / "book.json"
    book_path.write_text(json.dumps(book_rows), encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("do not overwrite\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "combine_ranker_outputs_by_domain.py",
            "--source",
            f"book_model={book_path}",
            "--domain-source",
            "Book=book_model",
            "--default-source",
            "book_model",
            "--output-dir",
            str(output_dir),
            "--topk",
            "50",
        ],
    )

    with pytest.raises(ValueError, match="already exists and is not empty"):
        module.main()
