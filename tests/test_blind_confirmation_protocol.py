import json
from pathlib import Path

import pytest

from scripts.oracle_route_memory.build_confirmation_eval_lock import (
    ensure_lock_output_dir,
    render_markdown as render_lock_markdown,
)
from scripts.oracle_route_memory.build_protocol_v3_confirmation_bundle import render_readme_markdown
from genrec.training.official_protocol import (
    PROTOCOL_V3_NAME,
    V3_BLIND_CONFIRMATION,
    V3_COLD_LIKE_VALIDATION,
    V3_SELECTION_TRAIN,
    V3_WARM_VALIDATION,
    assert_leakage_audit_passed,
    build_blind_confirmation_protocol_manifest,
    build_confirmation_eval_lock_payload,
    build_leakage_audit,
    hash_file,
    mark_confirmation_eval_consumed,
    protocol_split_examples,
    validate_confirmation_eval_lock,
)
from genrec.training.router_dataset import RouterExample


def _example(sample_id: str, history: list[str], target: str, domain: str) -> RouterExample:
    return RouterExample(
        sample_id=sample_id,
        domain=domain,
        user_id=f"user-{sample_id}",
        history_item_ids=history,
        target_item_id=target,
        route_prefix1=1 if domain == "Book" else 2,
        route_prefix2=1,
        cold=False,
    )


def _examples() -> list[RouterExample]:
    rows: list[RouterExample] = []
    for domain, prefix in [("Book", "b"), ("Game", "g")]:
        for idx in range(1, 21):
            rows.append(_example(f"{domain}:{idx}", [f"{prefix}h{idx}", f"{prefix}p{idx}"], f"{prefix}t{idx}", domain))
    return rows


def test_blind_confirmation_split_is_item_level_and_deterministic() -> None:
    examples = _examples()
    manifest = build_blind_confirmation_protocol_manifest(
        examples,
        seed=20260607,
        cold_like_item_ratio=0.20,
        confirmation_item_ratio=0.20,
        warm_val_ratio=0.10,
        max_val_examples=100,
        max_confirmation_examples=100,
    )
    again = build_blind_confirmation_protocol_manifest(
        examples,
        seed=20260607,
        cold_like_item_ratio=0.20,
        confirmation_item_ratio=0.20,
        warm_val_ratio=0.10,
        max_val_examples=100,
        max_confirmation_examples=100,
    )

    assert manifest["protocol_name"] == PROTOCOL_V3_NAME
    assert manifest["split_hash"] == again["split_hash"]
    assert set(manifest["confirmation_item_ids"]).isdisjoint(manifest["cold_like_validation_item_ids"])

    confirmation_items = set(manifest["confirmation_item_ids"])
    cold_like_items = set(manifest["cold_like_validation_item_ids"])
    train_examples = protocol_split_examples(examples, manifest, V3_SELECTION_TRAIN)
    warm_examples = protocol_split_examples(examples, manifest, V3_WARM_VALIDATION)
    cold_like_examples = protocol_split_examples(examples, manifest, V3_COLD_LIKE_VALIDATION)
    confirmation_examples = protocol_split_examples(examples, manifest, V3_BLIND_CONFIRMATION)

    for split_examples in [train_examples, warm_examples, cold_like_examples]:
        assert all(row.target_item_id not in confirmation_items for row in split_examples)
        assert all(confirmation_items.isdisjoint(row.history_item_ids) for row in split_examples)

    assert all(row.target_item_id in confirmation_items for row in confirmation_examples)
    assert all(confirmation_items.isdisjoint(row.history_item_ids) for row in confirmation_examples)
    assert all(cold_like_items.isdisjoint(row.history_item_ids) for row in confirmation_examples)

    stats = manifest["stats"]["domain_stratification"]
    assert stats["Book"]["confirmation_item_count"] > 0
    assert stats["Game"]["confirmation_item_count"] > 0
    assert stats["Book"]["confirmation_sample_count"] > 0
    assert stats["Game"]["confirmation_sample_count"] > 0

    audit = build_leakage_audit(examples, manifest, hard_negative_items=manifest["hard_negative_item_ids"])
    assert audit["all_passed"] is True


def test_blind_confirmation_leakage_fail_fast_on_train_target_and_history() -> None:
    examples = _examples()
    manifest = build_blind_confirmation_protocol_manifest(
        examples,
        seed=20260607,
        cold_like_item_ratio=0.20,
        confirmation_item_ratio=0.20,
        warm_val_ratio=0.10,
    )
    confirmation_item = manifest["confirmation_item_ids"][0]
    train_sample_id = manifest["splits"][V3_SELECTION_TRAIN][0]
    train_example = next(row for row in examples if row.sample_id == train_sample_id)
    train_example.target_item_id = confirmation_item

    audit = build_leakage_audit(examples, manifest)
    assert audit["passed"]["confirmation_items_not_in_selection_train_targets"] is False
    with pytest.raises(ValueError):
        assert_leakage_audit_passed(audit)

    train_example.target_item_id = "safe-target"
    train_example.history_item_ids.append(confirmation_item)
    audit = build_leakage_audit(examples, manifest)
    assert audit["passed"]["confirmation_items_not_in_selection_train_histories"] is False


def test_blind_confirmation_leakage_fail_fast_on_selector_and_hard_negatives() -> None:
    examples = _examples()
    manifest = build_blind_confirmation_protocol_manifest(
        examples,
        seed=20260607,
        cold_like_item_ratio=0.20,
        confirmation_item_ratio=0.20,
        warm_val_ratio=0.10,
    )
    confirmation_item = manifest["confirmation_item_ids"][0]
    confirmation_sample = manifest["splits"][V3_BLIND_CONFIRMATION][0]

    audit = build_leakage_audit(examples, manifest, hard_negative_items=[confirmation_item])
    assert audit["passed"]["confirmation_items_not_in_hard_negative_items"] is False

    audit = build_leakage_audit(examples, manifest, selector_seen_items=[confirmation_item])
    assert audit["passed"]["confirmation_items_not_in_selector_seen_items"] is False

    audit = build_leakage_audit(examples, manifest, selector_seen_sample_ids=[confirmation_sample])
    assert audit["passed"]["confirmation_sample_ids_not_in_selector_seen_sample_ids"] is False
    with pytest.raises(ValueError):
        assert_leakage_audit_passed(audit)


def test_confirmation_eval_lock_blocks_missing_mismatch_and_repeat(tmp_path: Path) -> None:
    split_manifest = tmp_path / "split_manifest.json"
    leakage_audit = tmp_path / "leakage_audit.json"
    router_dir = tmp_path / "router"
    query_dir = tmp_path / "query"
    fusion_config = tmp_path / "fusion_config.json"
    selector_rows = tmp_path / "selector_rows.csv"
    for path in [router_dir, query_dir]:
        path.mkdir()
        (path / "checkpoint_meta.json").write_text('{"ok": true}\n', encoding="utf-8")
    split_manifest.write_text('{"split": 1}\n', encoding="utf-8")
    leakage_audit.write_text('{"all_passed": true}\n', encoding="utf-8")
    fusion_config.write_text('{"config": true}\n', encoding="utf-8")
    selector_rows.write_text("policy,metric\np,1\n", encoding="utf-8")

    payload = build_confirmation_eval_lock_payload(
        split_manifest_path=split_manifest,
        leakage_audit_path=leakage_audit,
        router_checkpoint_dir=router_dir,
        query_head_checkpoint_dir=query_dir,
        fusion_config_path=fusion_config,
        selector_rows_path=selector_rows,
        git_commit="abc123",
    )
    lock_path = tmp_path / "confirmation_eval_lock.json"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_confirmation_eval_lock(lock_path, split_manifest_path=split_manifest)["split_manifest_hash"] == hash_file(split_manifest)
    split_manifest.write_text('{"split": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError):
        validate_confirmation_eval_lock(lock_path, split_manifest_path=split_manifest)
    split_manifest.write_text('{"split": 1}\n', encoding="utf-8")

    mark_confirmation_eval_consumed(lock_path, output_dir=tmp_path / "eval")
    with pytest.raises(RuntimeError):
        validate_confirmation_eval_lock(lock_path, split_manifest_path=split_manifest)
    with pytest.raises(ValueError):
        validate_confirmation_eval_lock(lock_path, split_manifest_path=split_manifest, allow_rerun=True)
    assert validate_confirmation_eval_lock(
        lock_path,
        split_manifest_path=split_manifest,
        allow_rerun=True,
        rerun_reason="debugging non-metric I/O failure",
    )["confirmation_eval_consumed"] is True


def test_confirmation_eval_lock_markdown_marks_unconsumed_and_immutable() -> None:
    markdown = render_lock_markdown(
        {
            "split_manifest_hash": "split",
            "leakage_audit_hash": "leakage",
            "router_checkpoint_hash": "router",
            "query_head_checkpoint_hash": "query",
            "fusion_config_hash": "fusion",
            "selector_rows_hash": "selector",
            "lock_hash": "lock",
        }
    )

    assert "`confirmation_eval_consumed`: `false`" in markdown
    assert "Do not tune checkpoints, split, selector candidates, or fusion config after this file is created." in markdown
    assert "- Lock hash: `lock`" in markdown


def test_confirmation_eval_lock_output_dir_requires_explicit_force(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty-lock"
    ensure_lock_output_dir(empty_dir)
    assert empty_dir.is_dir()

    existing_dir = tmp_path / "existing-lock"
    existing_dir.mkdir()
    (existing_dir / "confirmation_eval_lock.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ensure_lock_output_dir(existing_dir)

    ensure_lock_output_dir(existing_dir, force=True)


def test_protocol_v3_readme_renderer_preserves_claim_boundary() -> None:
    def row(method_key: str, label: str, level: str, recall: float) -> dict[str, str]:
        return {
            "method_key": method_key,
            "display_name": label,
            "result_level": level,
            "subset": "blind_confirmation",
            "domain": "ALL",
            "sample_count": "118",
            "Recall@50": str(recall),
            "NDCG@50": "0.2",
            "MRR": "0.1",
        }

    rows = [
        row("metadata_global_mean_query", "Metadata Global Mean Query", "blind_confirmation_result", 0.09322),
        row(
            "metadata_global_best_non_route_query",
            "Metadata Global Best Non-Route Query",
            "blind_confirmation_result",
            0.09322,
        ),
        row("dynamic_memory_without_route", "Dynamic Memory Without Route", "blind_confirmation_result", 0.09322),
        row("random_matched_size_bucket", "Random Matched-Size Bucket", "blind_confirmation_result", 0.008475),
        row(
            "predicted_route_validation_selected",
            "Predicted Route Validation-Selected Fusion",
            "blind_confirmation_result",
            0.084746,
        ),
        row("predicted_prefix1_top1_single_query", "Predicted Prefix-1 Top-1 Single Query", "diagnostic_result", 0.110169),
        row("predicted_prefix1_top4_single_query", "Predicted Prefix-1 Top-4 Single Query", "diagnostic_result", 0.110169),
        row("oracle_prefix1_route", "Oracle Prefix-1 Route", "oracle_upper_bound", 0.127119),
        row("oracle_prefix2_route", "Oracle Prefix-2 Route", "oracle_upper_bound", 0.822034),
    ]

    markdown = render_readme_markdown(rows)

    assert "Claimable result: `Predicted Route Validation-Selected Fusion`" in markdown
    assert "Selected-policy sample count: `118`." in markdown
    assert "| Predicted Prefix-1 Top-1 Single Query | diagnostic_result | 0.110169 |" in markdown
    assert "does not beat the strongest no-route metadata baseline" in markdown
