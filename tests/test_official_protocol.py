from genrec.training.official_protocol import (
    build_leakage_audit,
    build_official_protocol_manifest,
    filter_training_item_embeddings,
    protocol_split_examples,
)
from genrec.training.router_dataset import RouterExample


def _example(sample_id: str, history: list[str], target: str, domain: str = "Book") -> RouterExample:
    return RouterExample(
        sample_id=sample_id,
        domain=domain,
        user_id="u1",
        history_item_ids=history,
        target_item_id=target,
        route_prefix1=1,
        route_prefix2=1,
        cold=False,
    )


def test_official_protocol_removes_heldout_from_train_visible_items() -> None:
    examples = [
        _example("s1", ["a"], "b"),
        _example("s2", ["b"], "c"),
        _example("s3", ["c"], "d"),
        _example("s4", ["e"], "f"),
        _example("s5", ["f"], "g"),
        _example("s6", ["g"], "h"),
    ]

    manifest = build_official_protocol_manifest(
        examples,
        cold_like_item_ratio=0.25,
        warm_val_ratio=0.25,
        max_val_examples=10,
        seed=7,
    )
    audit = build_leakage_audit(examples, manifest)

    assert audit["all_passed"] is True
    train_examples = protocol_split_examples(examples, manifest, "train")
    heldout_items = set(manifest["heldout_item_ids"])
    assert train_examples
    assert all(example.target_item_id not in heldout_items for example in train_examples)
    assert set(manifest["split_item_ids"]["train_visible_items"]).isdisjoint(heldout_items)


def test_filter_training_item_embeddings_excludes_heldout_items() -> None:
    manifest = {
        "protocol_name": "cold_like_val_v1",
        "heldout_item_ids": ["b", "d"],
        "splits": {"train": [], "warm_val": [], "cold_like_val": []},
    }
    filtered = filter_training_item_embeddings({"a": 1, "b": 2, "c": 3, "d": 4}, manifest)
    assert filtered == {"a": 1, "c": 3}


def test_protocol_splits_are_disjoint_and_cold_targets_stay_out_of_train() -> None:
    examples = [
        _example("s1", ["a"], "b", domain="Book"),
        _example("s2", ["c"], "d", domain="Book"),
        _example("s3", ["e"], "f", domain="Book"),
        _example("s4", ["g"], "h", domain="Game"),
        _example("s5", ["i"], "j", domain="Game"),
        _example("s6", ["k"], "l", domain="Game"),
        _example("s7", ["m"], "n", domain="Game"),
    ]
    manifest = build_official_protocol_manifest(
        examples,
        cold_like_item_ratio=0.28,
        warm_val_ratio=0.25,
        max_val_examples=10,
        seed=11,
    )

    train_ids = set(manifest["splits"]["train"])
    warm_ids = set(manifest["splits"]["warm_val"])
    cold_ids = set(manifest["splits"]["cold_like_val"])
    assert train_ids.isdisjoint(warm_ids)
    assert train_ids.isdisjoint(cold_ids)
    assert warm_ids.isdisjoint(cold_ids)

    heldout_items = set(manifest["heldout_item_ids"])
    train_targets = set(manifest["split_target_item_ids"]["train"])
    warm_targets = set(manifest["split_target_item_ids"]["warm_val"])
    cold_targets = set(manifest["split_target_item_ids"]["cold_like_val"])
    assert train_targets.isdisjoint(heldout_items)
    assert warm_targets.isdisjoint(heldout_items)
    assert cold_targets.issubset(heldout_items)
