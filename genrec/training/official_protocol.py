from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import yaml  # type: ignore

from .router_dataset import RouterExample

PROTOCOL_NAME = "cold_like_val_v1"
PROTOCOL_V3_NAME = "official_protocol_v3_blind_confirmation"

V3_SELECTION_TRAIN = "selection_train"
V3_COLD_LIKE_VALIDATION = "cold_like_validation"
V3_WARM_VALIDATION = "warm_validation"
V3_BLIND_CONFIRMATION = "blind_confirmation"

_V3_SPLIT_ALIASES = {
    "train": V3_SELECTION_TRAIN,
    "warm_val": V3_WARM_VALIDATION,
    "cold_like_val": V3_COLD_LIKE_VALIDATION,
    "cold": V3_BLIND_CONFIRMATION,
    "confirmation": V3_BLIND_CONFIRMATION,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_config_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _hash_values(values: Iterable[Any]) -> str:
    return compute_config_hash([str(value) for value in sorted(values)])


def _example_contains_items(example: RouterExample, blocked_items: set[str]) -> bool:
    if str(example.target_item_id) in blocked_items:
        return True
    return any(str(item_id) in blocked_items for item_id in example.history_item_ids)


def _example_history_overlaps(example: RouterExample, blocked_items: set[str]) -> bool:
    return any(str(item_id) in blocked_items for item_id in example.history_item_ids)


def _examples_by_ids(examples: Sequence[RouterExample], sample_ids: Iterable[str]) -> list[RouterExample]:
    lookup = set(sample_ids)
    return [example for example in examples if example.sample_id in lookup]


def _domain_counts(examples: Sequence[RouterExample]) -> Dict[str, int]:
    return dict(sorted(Counter(str(example.domain) for example in examples).items()))


def _unique_items(examples: Sequence[RouterExample]) -> list[str]:
    values = set()
    for example in examples:
        values.add(str(example.target_item_id))
        values.update(str(item_id) for item_id in example.history_item_ids)
    return sorted(values)


def _unique_targets(examples: Sequence[RouterExample]) -> list[str]:
    return sorted({str(example.target_item_id) for example in examples})


def _split_hash(manifest: Mapping[str, Any]) -> str:
    payload = {
        "protocol_name": manifest.get("protocol_name"),
        "seed": manifest.get("seed"),
        "split_config": manifest.get("split_config", {}),
        "splits": manifest.get("splits", {}),
        "confirmation_item_ids": manifest.get("confirmation_item_ids", []),
        "cold_like_validation_item_ids": manifest.get("cold_like_validation_item_ids", []),
    }
    return compute_config_hash(payload)


def _choose_domain_stratified_items(
    examples: Sequence[RouterExample],
    *,
    blocked_items: set[str],
    ratio: float,
    rng: random.Random,
) -> tuple[set[str], dict[str, dict[str, int]]]:
    by_domain: dict[str, set[str]] = {}
    for example in examples:
        target = str(example.target_item_id)
        if target in blocked_items:
            continue
        by_domain.setdefault(str(example.domain), set()).add(target)

    chosen: set[str] = set()
    stats: dict[str, dict[str, int]] = {}
    for domain in sorted(by_domain):
        eligible = sorted(by_domain[domain])
        count = int(round(len(eligible) * float(ratio)))
        if eligible and ratio > 0.0:
            count = max(1, count)
        count = min(count, len(eligible))
        shuffled = list(eligible)
        rng.shuffle(shuffled)
        selected = set(shuffled[:count])
        chosen.update(selected)
        stats[domain] = {
            "eligible_item_count": len(eligible),
            "selected_item_count": len(selected),
        }
    return chosen, stats


def _non_empty_split(name: str, examples: Sequence[RouterExample]) -> None:
    if not examples:
        raise ValueError(f"Split {name!r} is empty. Adjust the fixed seed/ratio only before any confirmation eval.")


def build_official_protocol_manifest(
    examples: Sequence[RouterExample],
    cold_like_item_ratio: float,
    warm_val_ratio: float,
    max_val_examples: int,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if not examples:
        raise ValueError("Cannot build an official protocol manifest from an empty example list.")

    rng = random.Random(seed)
    target_items = sorted({str(example.target_item_id) for example in examples})
    holdout_count = max(1, int(round(len(target_items) * float(cold_like_item_ratio))))
    rng.shuffle(target_items)
    heldout_items = set(target_items[:holdout_count])

    contaminated_examples = [example for example in examples if _example_history_overlaps(example, heldout_items)]
    contaminated_ids = {example.sample_id for example in contaminated_examples}
    clean_examples = [example for example in examples if example.sample_id not in contaminated_ids]
    cold_like_candidates = [example for example in clean_examples if str(example.target_item_id) in heldout_items]
    train_visible_candidates = [example for example in clean_examples if str(example.target_item_id) not in heldout_items]

    if not cold_like_candidates:
        raise ValueError("The sampled holdout items produced no clean cold-like validation examples. Adjust the seed or ratio.")
    if len(train_visible_candidates) < 2:
        raise ValueError("Not enough clean train-visible examples remain after holdout filtering.")

    rng.shuffle(cold_like_candidates)
    rng.shuffle(train_visible_candidates)

    warm_val_size = max(1, int(round(len(train_visible_candidates) * float(warm_val_ratio))))
    warm_val_size = min(warm_val_size, max(len(train_visible_candidates) - 1, 1))
    warm_val_examples = list(train_visible_candidates[:warm_val_size])
    train_examples = list(train_visible_candidates[warm_val_size:])
    if not train_examples:
        train_examples = warm_val_examples[-1:]
        warm_val_examples = warm_val_examples[:-1]
    if not warm_val_examples:
        warm_val_examples = train_examples[-1:]
        train_examples = train_examples[:-1]
    if not train_examples or not warm_val_examples:
        raise ValueError("Failed to create non-empty train and warm_val splits for the official protocol.")

    if max_val_examples > 0:
        cold_like_candidates = cold_like_candidates[:max_val_examples]
        warm_val_examples = warm_val_examples[:max_val_examples]

    protocol_core = {
        "protocol_name": PROTOCOL_NAME,
        "seed": int(seed),
        "cold_like_item_ratio": float(cold_like_item_ratio),
        "warm_val_ratio": float(warm_val_ratio),
        "max_val_examples": int(max_val_examples),
        "heldout_item_ids": sorted(heldout_items),
        "splits": {
            "train": [example.sample_id for example in train_examples],
            "warm_val": [example.sample_id for example in warm_val_examples],
            "cold_like_val": [example.sample_id for example in cold_like_candidates],
        },
        "split_item_ids": {
            "train_visible_items": _unique_items(train_examples),
            "warm_val_visible_items": _unique_items(warm_val_examples),
            "cold_like_visible_items": _unique_items(cold_like_candidates),
            "cold_like_target_items": _unique_targets(cold_like_candidates),
        },
        "split_target_item_ids": {
            "train": _unique_targets(train_examples),
            "warm_val": _unique_targets(warm_val_examples),
            "cold_like_val": _unique_targets(cold_like_candidates),
        },
        "stats": {
            "num_total_examples": len(examples),
            "num_target_items": len(target_items),
            "num_heldout_items": len(heldout_items),
            "num_contaminated_examples_removed": len(contaminated_examples),
            "num_train_examples": len(train_examples),
            "num_warm_val_examples": len(warm_val_examples),
            "num_cold_like_val_examples": len(cold_like_candidates),
            "domain_counts": {
                "train": _domain_counts(train_examples),
                "warm_val": _domain_counts(warm_val_examples),
                "cold_like_val": _domain_counts(cold_like_candidates),
            },
        },
        "metadata": dict(metadata or {}),
    }
    protocol_core["config_hash"] = compute_config_hash(protocol_core)
    return protocol_core


def build_blind_confirmation_protocol_manifest(
    examples: Sequence[RouterExample],
    *,
    cold_like_item_ratio: float = 0.12,
    confirmation_item_ratio: float = 0.10,
    warm_val_ratio: float = 0.08,
    max_val_examples: int = 1200,
    max_confirmation_examples: int = 0,
    seed: int = 20260607,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if not examples:
        raise ValueError("Cannot build a blind-confirmation protocol manifest from an empty example list.")

    rng = random.Random(seed)
    confirmation_items, confirmation_item_stats = _choose_domain_stratified_items(
        examples,
        blocked_items=set(),
        ratio=confirmation_item_ratio,
        rng=rng,
    )
    cold_like_items, cold_like_item_stats = _choose_domain_stratified_items(
        examples,
        blocked_items=confirmation_items,
        ratio=cold_like_item_ratio,
        rng=rng,
    )
    if not confirmation_items:
        raise ValueError("No confirmation items were selected. Increase confirmation_item_ratio before any blind eval.")
    if not cold_like_items:
        raise ValueError("No cold-like validation items were selected. Increase cold_like_item_ratio before any blind eval.")

    training_excluded_items = confirmation_items | cold_like_items
    clean_history_examples = [
        example for example in examples if not _example_history_overlaps(example, training_excluded_items)
    ]
    confirmation_candidates = [
        example for example in clean_history_examples if str(example.target_item_id) in confirmation_items
    ]
    cold_like_candidates = [
        example for example in clean_history_examples if str(example.target_item_id) in cold_like_items
    ]
    train_visible_candidates = [
        example for example in clean_history_examples if str(example.target_item_id) not in training_excluded_items
    ]

    _non_empty_split(V3_BLIND_CONFIRMATION, confirmation_candidates)
    _non_empty_split(V3_COLD_LIKE_VALIDATION, cold_like_candidates)
    if len(train_visible_candidates) < 2:
        raise ValueError("Not enough clean selection-train examples remain after v3 holdout filtering.")

    rng.shuffle(confirmation_candidates)
    rng.shuffle(cold_like_candidates)
    rng.shuffle(train_visible_candidates)

    warm_val_size = max(1, int(round(len(train_visible_candidates) * float(warm_val_ratio))))
    warm_val_size = min(warm_val_size, max(len(train_visible_candidates) - 1, 1))
    warm_validation_examples = list(train_visible_candidates[:warm_val_size])
    selection_train_examples = list(train_visible_candidates[warm_val_size:])
    if not selection_train_examples:
        selection_train_examples = warm_validation_examples[-1:]
        warm_validation_examples = warm_validation_examples[:-1]
    if not warm_validation_examples:
        warm_validation_examples = selection_train_examples[-1:]
        selection_train_examples = selection_train_examples[:-1]

    if max_val_examples > 0:
        cold_like_candidates = cold_like_candidates[:max_val_examples]
        warm_validation_examples = warm_validation_examples[:max_val_examples]
    if max_confirmation_examples > 0:
        confirmation_candidates = confirmation_candidates[:max_confirmation_examples]

    _non_empty_split(V3_SELECTION_TRAIN, selection_train_examples)
    _non_empty_split(V3_WARM_VALIDATION, warm_validation_examples)
    _non_empty_split(V3_COLD_LIKE_VALIDATION, cold_like_candidates)
    _non_empty_split(V3_BLIND_CONFIRMATION, confirmation_candidates)

    splits = {
        V3_SELECTION_TRAIN: [example.sample_id for example in selection_train_examples],
        V3_WARM_VALIDATION: [example.sample_id for example in warm_validation_examples],
        V3_COLD_LIKE_VALIDATION: [example.sample_id for example in cold_like_candidates],
        V3_BLIND_CONFIRMATION: [example.sample_id for example in confirmation_candidates],
    }
    split_examples = {
        V3_SELECTION_TRAIN: selection_train_examples,
        V3_WARM_VALIDATION: warm_validation_examples,
        V3_COLD_LIKE_VALIDATION: cold_like_candidates,
        V3_BLIND_CONFIRMATION: confirmation_candidates,
    }
    split_target_item_ids = {name: _unique_targets(values) for name, values in split_examples.items()}
    split_item_ids = {f"{name}_visible_items": _unique_items(values) for name, values in split_examples.items()}

    domains = sorted({str(example.domain) for example in examples})
    domain_stats: dict[str, dict[str, int]] = {}
    for domain in domains:
        confirmation_domain_examples = [example for example in confirmation_candidates if str(example.domain) == domain]
        cold_domain_examples = [example for example in cold_like_candidates if str(example.domain) == domain]
        selection_domain_examples = [example for example in selection_train_examples if str(example.domain) == domain]
        warm_domain_examples = [example for example in warm_validation_examples if str(example.domain) == domain]
        domain_stats[domain] = {
            "eligible_item_count": confirmation_item_stats.get(domain, {}).get("eligible_item_count", 0),
            "confirmation_item_count": confirmation_item_stats.get(domain, {}).get("selected_item_count", 0),
            "confirmation_sample_count": len(confirmation_domain_examples),
            "selection_train_item_count": len(_unique_targets(selection_domain_examples)),
            "selection_train_sample_count": len(selection_domain_examples),
            "cold_like_validation_item_count": cold_like_item_stats.get(domain, {}).get("selected_item_count", 0),
            "cold_like_validation_sample_count": len(cold_domain_examples),
            "warm_validation_item_count": len(_unique_targets(warm_domain_examples)),
            "warm_validation_sample_count": len(warm_domain_examples),
        }

    contaminated_history_ids = [
        example.sample_id for example in examples if _example_history_overlaps(example, training_excluded_items)
    ]
    protocol_core = {
        "protocol_name": PROTOCOL_V3_NAME,
        "split_schema_version": 3,
        "seed": int(seed),
        "split_config": {
            "cold_like_item_ratio": float(cold_like_item_ratio),
            "confirmation_item_ratio": float(confirmation_item_ratio),
            "warm_val_ratio": float(warm_val_ratio),
            "max_val_examples": int(max_val_examples),
            "max_confirmation_examples": int(max_confirmation_examples),
        },
        "cold_like_item_ratio": float(cold_like_item_ratio),
        "confirmation_item_ratio": float(confirmation_item_ratio),
        "warm_val_ratio": float(warm_val_ratio),
        "max_val_examples": int(max_val_examples),
        "max_confirmation_examples": int(max_confirmation_examples),
        "confirmation_item_ids": sorted(confirmation_items),
        "cold_like_validation_item_ids": sorted(cold_like_items),
        "training_excluded_item_ids": sorted(training_excluded_items),
        "heldout_item_ids": sorted(training_excluded_items),
        "splits": splits,
        "split_aliases": dict(_V3_SPLIT_ALIASES),
        "split_item_ids": split_item_ids,
        "split_target_item_ids": split_target_item_ids,
        "hard_negative_item_ids": _unique_items(selection_train_examples),
        "selector_seen_item_ids": sorted(set(_unique_items(cold_like_candidates)) | set(_unique_items(warm_validation_examples))),
        "selector_seen_sample_ids": sorted(splits[V3_COLD_LIKE_VALIDATION] + splits[V3_WARM_VALIDATION]),
        "stats": {
            "num_total_examples": len(examples),
            "num_target_items": len({str(example.target_item_id) for example in examples}),
            "num_confirmation_items": len(confirmation_items),
            "num_cold_like_validation_items": len(cold_like_items),
            "num_training_excluded_items": len(training_excluded_items),
            "num_contaminated_history_examples_removed": len(contaminated_history_ids),
            "num_selection_train_examples": len(selection_train_examples),
            "num_warm_validation_examples": len(warm_validation_examples),
            "num_cold_like_validation_examples": len(cold_like_candidates),
            "num_blind_confirmation_examples": len(confirmation_candidates),
            "domain_counts": {name: _domain_counts(values) for name, values in split_examples.items()},
            "domain_stratification": domain_stats,
        },
        "metadata": dict(metadata or {}),
    }
    protocol_core["split_hash"] = _split_hash(protocol_core)
    protocol_core["item_id_hash"] = _hash_values(
        set(protocol_core["confirmation_item_ids"]) | set(protocol_core["cold_like_validation_item_ids"])
    )
    protocol_core["sample_id_hash"] = _hash_values(sample_id for sample_ids in splits.values() for sample_id in sample_ids)
    protocol_core["config_hash"] = compute_config_hash(protocol_core)
    return protocol_core


def resolve_protocol_split_name(manifest: Mapping[str, Any], split_name: str) -> str:
    splits = manifest.get("splits", {})
    if split_name in splits:
        return split_name
    if str(manifest.get("protocol_name")) == PROTOCOL_V3_NAME:
        alias = _V3_SPLIT_ALIASES.get(split_name)
        if alias in splits:
            return alias
    raise KeyError(f"Split {split_name!r} not found in protocol manifest.")


def default_train_split_name(manifest: Mapping[str, Any]) -> str:
    return V3_SELECTION_TRAIN if str(manifest.get("protocol_name")) == PROTOCOL_V3_NAME else "train"


def default_validation_split_name(manifest: Mapping[str, Any]) -> str:
    return V3_WARM_VALIDATION if str(manifest.get("protocol_name")) == PROTOCOL_V3_NAME else "warm_val"


def default_cold_like_validation_split_name(manifest: Mapping[str, Any]) -> str:
    return V3_COLD_LIKE_VALIDATION if str(manifest.get("protocol_name")) == PROTOCOL_V3_NAME else "cold_like_val"


def protocol_split_examples(examples: Sequence[RouterExample], manifest: Mapping[str, Any], split_name: str) -> list[RouterExample]:
    resolved = resolve_protocol_split_name(manifest, split_name)
    return _examples_by_ids(examples, manifest.get("splits", {})[resolved])


def filter_training_item_embeddings(item_embeddings: Mapping[str, Any], manifest: Mapping[str, Any]) -> Dict[str, Any]:
    heldout_items = set(str(item_id) for item_id in manifest.get("heldout_item_ids", []))
    return {str(item_id): value for item_id, value in item_embeddings.items() if str(item_id) not in heldout_items}


def _items_in_history(examples: Sequence[RouterExample]) -> set[str]:
    return {str(item_id) for example in examples for item_id in example.history_item_ids}


def _items_in_targets(examples: Sequence[RouterExample]) -> set[str]:
    return {str(example.target_item_id) for example in examples}


def _audit_intersection(a: Iterable[Any], b: Iterable[Any]) -> list[str]:
    return sorted(set(str(x) for x in a) & set(str(y) for y in b))


def _build_v1_leakage_audit(examples: Sequence[RouterExample], manifest: Mapping[str, Any]) -> Dict[str, Any]:
    heldout_items = set(str(item_id) for item_id in manifest.get("heldout_item_ids", []))
    train_examples = protocol_split_examples(examples, manifest, "train")
    warm_val_examples = protocol_split_examples(examples, manifest, "warm_val")
    cold_like_examples = protocol_split_examples(examples, manifest, "cold_like_val")

    train_visible_items = set(manifest.get("split_item_ids", {}).get("train_visible_items", []))
    warm_val_visible_items = set(manifest.get("split_item_ids", {}).get("warm_val_visible_items", []))
    cold_like_target_items = set(manifest.get("split_target_item_ids", {}).get("cold_like_val", []))

    train_target_items = {str(example.target_item_id) for example in train_examples}
    warm_val_target_items = {str(example.target_item_id) for example in warm_val_examples}
    contaminated_training = [example.sample_id for example in train_examples if _example_contains_items(example, heldout_items)]
    contaminated_warm_val = [example.sample_id for example in warm_val_examples if _example_contains_items(example, heldout_items)]
    contaminated_cold_like_targets = [
        example.sample_id
        for example in cold_like_examples
        if str(example.target_item_id) not in heldout_items or _example_history_overlaps(example, heldout_items)
    ]

    checks = {
        "heldout_not_in_train_targets": sorted(heldout_items & train_target_items),
        "heldout_not_in_warm_val_targets": sorted(heldout_items & warm_val_target_items),
        "heldout_not_in_train_visible_items": sorted(heldout_items & train_visible_items),
        "heldout_not_in_warm_val_visible_items": sorted(heldout_items & warm_val_visible_items),
        "cold_like_targets_equal_heldout_subset": sorted(cold_like_target_items - heldout_items),
        "train_examples_without_heldout_history": contaminated_training,
        "warm_val_examples_without_heldout_history": contaminated_warm_val,
        "cold_like_examples_are_clean_targets": contaminated_cold_like_targets,
    }
    passed = {key: len(value) == 0 for key, value in checks.items()}
    return {
        "protocol_name": str(manifest.get("protocol_name", PROTOCOL_NAME)),
        "config_hash": str(manifest.get("config_hash", "")),
        "summary": {
            "num_examples": len(examples),
            "num_train_examples": len(train_examples),
            "num_warm_val_examples": len(warm_val_examples),
            "num_cold_like_val_examples": len(cold_like_examples),
            "num_heldout_items": len(heldout_items),
        },
        "checks": checks,
        "passed": passed,
        "all_passed": all(passed.values()),
    }


def build_leakage_audit(
    examples: Sequence[RouterExample],
    manifest: Mapping[str, Any],
    *,
    hard_negative_items: Iterable[Any] | None = None,
    selector_seen_items: Iterable[Any] | None = None,
    selector_seen_sample_ids: Iterable[Any] | None = None,
) -> Dict[str, Any]:
    if str(manifest.get("protocol_name", PROTOCOL_NAME)) != PROTOCOL_V3_NAME:
        return _build_v1_leakage_audit(examples, manifest)

    confirmation_items = set(str(item_id) for item_id in manifest.get("confirmation_item_ids", []))
    cold_like_items = set(str(item_id) for item_id in manifest.get("cold_like_validation_item_ids", []))
    selection_train = protocol_split_examples(examples, manifest, V3_SELECTION_TRAIN)
    cold_like_validation = protocol_split_examples(examples, manifest, V3_COLD_LIKE_VALIDATION)
    warm_validation = protocol_split_examples(examples, manifest, V3_WARM_VALIDATION)
    blind_confirmation = protocol_split_examples(examples, manifest, V3_BLIND_CONFIRMATION)

    if hard_negative_items is None:
        hard_negative_items = manifest.get("hard_negative_item_ids", [])
    if selector_seen_items is None:
        selector_seen_items = manifest.get("selector_seen_item_ids", [])
    if selector_seen_sample_ids is None:
        selector_seen_sample_ids = manifest.get("selector_seen_sample_ids", [])

    confirmation_sample_ids = [example.sample_id for example in blind_confirmation]
    checks = {
        "confirmation_items_not_in_selection_train_targets": _audit_intersection(
            confirmation_items,
            _items_in_targets(selection_train),
        ),
        "confirmation_items_not_in_selection_train_histories": _audit_intersection(
            confirmation_items,
            _items_in_history(selection_train),
        ),
        "confirmation_items_not_in_cold_like_validation_targets": _audit_intersection(
            confirmation_items,
            _items_in_targets(cold_like_validation),
        ),
        "confirmation_items_not_in_cold_like_validation_histories": _audit_intersection(
            confirmation_items,
            _items_in_history(cold_like_validation),
        ),
        "confirmation_items_not_in_warm_validation_targets": _audit_intersection(
            confirmation_items,
            _items_in_targets(warm_validation),
        ),
        "confirmation_items_not_in_warm_validation_histories": _audit_intersection(
            confirmation_items,
            _items_in_history(warm_validation),
        ),
        "confirmation_items_not_in_hard_negative_items": _audit_intersection(
            confirmation_items,
            hard_negative_items,
        ),
        "confirmation_items_not_in_selector_seen_items": _audit_intersection(
            confirmation_items,
            selector_seen_items,
        ),
        "confirmation_sample_ids_not_in_selector_seen_sample_ids": _audit_intersection(
            confirmation_sample_ids,
            selector_seen_sample_ids,
        ),
        "cold_like_validation_items_not_in_confirmation_items": _audit_intersection(
            cold_like_items,
            confirmation_items,
        ),
        "cold_like_items_not_in_selection_train_targets": _audit_intersection(
            cold_like_items,
            _items_in_targets(selection_train),
        ),
        "cold_like_items_not_in_selection_train_histories": _audit_intersection(
            cold_like_items,
            _items_in_history(selection_train),
        ),
        "blind_confirmation_targets_are_confirmation_items": sorted(
            _items_in_targets(blind_confirmation) - confirmation_items
        ),
        "blind_confirmation_histories_do_not_contain_training_excluded_items": [
            example.sample_id
            for example in blind_confirmation
            if _example_history_overlaps(example, confirmation_items | cold_like_items)
        ],
    }
    passed = {key: len(value) == 0 for key, value in checks.items()}
    split_sample_ids = [sample_id for sample_ids in manifest.get("splits", {}).values() for sample_id in sample_ids]
    return {
        "protocol_name": PROTOCOL_V3_NAME,
        "config_hash": str(manifest.get("config_hash", "")),
        "split_hash": str(manifest.get("split_hash", _split_hash(manifest))),
        "item_id_hash": str(manifest.get("item_id_hash", _hash_values(confirmation_items | cold_like_items))),
        "sample_id_hash": str(manifest.get("sample_id_hash", _hash_values(split_sample_ids))),
        "created_at": _utc_now(),
        "git_commit": manifest.get("metadata", {}).get("git_commit"),
        "hostname": manifest.get("metadata", {}).get("hostname"),
        "split_seed": manifest.get("seed"),
        "split_config": manifest.get("split_config", {}),
        "summary": {
            "num_examples": len(examples),
            "num_selection_train_examples": len(selection_train),
            "num_warm_validation_examples": len(warm_validation),
            "num_cold_like_validation_examples": len(cold_like_validation),
            "num_blind_confirmation_examples": len(blind_confirmation),
            "num_confirmation_items": len(confirmation_items),
            "num_cold_like_validation_items": len(cold_like_items),
        },
        "checks": checks,
        "passed": passed,
        "all_passed": all(passed.values()),
    }


def assert_leakage_audit_passed(audit: Mapping[str, Any]) -> None:
    if not bool(audit.get("all_passed")):
        failed = [key for key, passed in audit.get("passed", {}).items() if not passed]
        raise ValueError(f"Leakage audit failed; refusing to continue: {failed}")


def render_split_manifest_markdown(manifest: Mapping[str, Any]) -> str:
    stats = manifest.get("stats", {})
    lines = [
        "# Official Protocol Split Manifest",
        "",
        f"- Protocol: `{manifest.get('protocol_name', PROTOCOL_NAME)}`",
        f"- Config hash: `{manifest.get('config_hash', '')}`",
        f"- Split hash: `{manifest.get('split_hash', '')}`",
        f"- Seed: `{manifest.get('seed')}`",
        "",
        "## Split Sizes",
        "",
    ]
    for name, sample_ids in manifest.get("splits", {}).items():
        lines.append(f"- `{name}`: `{len(sample_ids)}` samples")
    if manifest.get("protocol_name") == PROTOCOL_V3_NAME:
        lines.extend(
            [
                "",
                "## Item Holdouts",
                "",
                f"- Confirmation items: `{len(manifest.get('confirmation_item_ids', []))}`",
                f"- Cold-like validation items: `{len(manifest.get('cold_like_validation_item_ids', []))}`",
                f"- Training-excluded items: `{len(manifest.get('training_excluded_item_ids', []))}`",
                "",
                "## Domain Stratification",
                "",
                "| domain | eligible items | confirmation items | confirmation samples | selection-train items | cold-like val items | cold-like val samples |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for domain, row in sorted(stats.get("domain_stratification", {}).items()):
            lines.append(
                f"| {domain} | {row.get('eligible_item_count', 0)} | {row.get('confirmation_item_count', 0)} | "
                f"{row.get('confirmation_sample_count', 0)} | {row.get('selection_train_item_count', 0)} | "
                f"{row.get('cold_like_validation_item_count', 0)} | {row.get('cold_like_validation_sample_count', 0)} |"
            )
    return "\n".join(lines) + "\n"


def render_leakage_audit_markdown(manifest: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    stats = manifest.get("stats", {})
    lines = [
        "# Leakage Audit",
        "",
        f"- Protocol: `{manifest.get('protocol_name', PROTOCOL_NAME)}`",
        f"- Config hash: `{manifest.get('config_hash', '')}`",
        f"- Split hash: `{manifest.get('split_hash', '')}`",
        f"- All checks passed: `{'PASS' if audit.get('all_passed') else 'FAIL'}`",
        "",
        "## Summary",
        "",
    ]
    if manifest.get("protocol_name") == PROTOCOL_V3_NAME:
        lines.extend(
            [
                f"- Selection-train / warm / cold-like / confirmation examples: "
                f"`{stats.get('num_selection_train_examples', 0)}` / "
                f"`{stats.get('num_warm_validation_examples', 0)}` / "
                f"`{stats.get('num_cold_like_validation_examples', 0)}` / "
                f"`{stats.get('num_blind_confirmation_examples', 0)}`",
                f"- Confirmation items: `{stats.get('num_confirmation_items', 0)}`",
                f"- Cold-like validation items: `{stats.get('num_cold_like_validation_items', 0)}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Heldout items: `{stats.get('num_heldout_items', 0)}`",
                f"- Train / warm_val / cold_like_val examples: `{stats.get('num_train_examples', 0)}` / `{stats.get('num_warm_val_examples', 0)}` / `{stats.get('num_cold_like_val_examples', 0)}`",
                f"- Contaminated examples removed before splitting: `{stats.get('num_contaminated_examples_removed', 0)}`",
            ]
        )
    lines.extend(["", "## Checks", ""])
    for key, passed in audit.get("passed", {}).items():
        lines.append(f"- `{key}`: `{'PASS' if passed else 'FAIL'}`")
        offenders = audit.get("checks", {}).get(key, [])
        if offenders:
            lines.append(f"  Offenders: `{offenders[:10]}`")
    lines.extend(
        [
            "",
            "## Leakage Rules",
            "",
            "- Confirmation items are removed from router/query/fusion training targets.",
            "- Confirmation items are removed from selection, cold-like validation, and warm validation histories.",
            "- Training hard-negative item pools must exclude confirmation items.",
            "- Confirmation items may re-enter only at blind confirmation retrieval time through dynamic catalog memory.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_protocol_bundle(output_dir: str | Path, manifest: Mapping[str, Any], audit: Mapping[str, Any]) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_manifest.json"
    manifest_md_path = out_dir / "split_manifest.md"
    audit_json_path = out_dir / "leakage_audit.json"
    audit_md_path = out_dir / "leakage_audit.md"
    yaml_name = "official_protocol_v3.yaml" if manifest.get("protocol_name") == PROTOCOL_V3_NAME else "official_protocol_v1.yaml"
    protocol_yaml_path = out_dir / yaml_name

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_md_path.write_text(render_split_manifest_markdown(manifest), encoding="utf-8")
    audit_json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit_md_path.write_text(render_leakage_audit_markdown(manifest, audit), encoding="utf-8")

    yaml_payload = {
        "protocol_name": manifest.get("protocol_name", PROTOCOL_NAME),
        "config_hash": manifest.get("config_hash", ""),
        "split_hash": manifest.get("split_hash", ""),
        "artifacts": {
            "split_manifest": manifest_path.name,
            "split_manifest_md": manifest_md_path.name,
            "leakage_audit_json": audit_json_path.name,
            "leakage_audit_md": audit_md_path.name,
        },
        "split_parameters": {
            "seed": manifest.get("seed"),
            "cold_like_item_ratio": manifest.get("cold_like_item_ratio"),
            "confirmation_item_ratio": manifest.get("confirmation_item_ratio"),
            "warm_val_ratio": manifest.get("warm_val_ratio"),
            "max_val_examples": manifest.get("max_val_examples"),
            "max_confirmation_examples": manifest.get("max_confirmation_examples"),
        },
    }
    protocol_yaml_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "manifest_md_path": str(manifest_md_path),
        "audit_json_path": str(audit_json_path),
        "audit_md_path": str(audit_md_path),
        "protocol_yaml_path": str(protocol_yaml_path),
    }


def load_protocol_manifest(path: str | Path) -> Dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_name = str(payload.get("protocol_name", ""))
    if protocol_name not in {PROTOCOL_NAME, PROTOCOL_V3_NAME}:
        raise ValueError(f"Unsupported protocol_name in manifest: {payload.get('protocol_name')!r}")
    return payload


def hash_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_confirmation_eval_lock_payload(
    *,
    split_manifest_path: str | Path,
    leakage_audit_path: str | Path,
    router_checkpoint_dir: str | Path,
    query_head_checkpoint_dir: str | Path,
    fusion_config_path: str | Path,
    selector_rows_path: str | Path,
    git_commit: str | None,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    router_meta = Path(router_checkpoint_dir) / "checkpoint_meta.json"
    query_head_meta = Path(query_head_checkpoint_dir) / "checkpoint_meta.json"
    payload = {
        "split_manifest_hash": hash_file(split_manifest_path),
        "leakage_audit_hash": hash_file(leakage_audit_path),
        "router_checkpoint_hash": hash_file(router_meta),
        "query_head_checkpoint_hash": hash_file(query_head_meta),
        "fusion_config_hash": hash_file(fusion_config_path),
        "selector_rows_hash": hash_file(selector_rows_path),
        "split_manifest_path": str(Path(split_manifest_path).resolve()),
        "leakage_audit_path": str(Path(leakage_audit_path).resolve()),
        "router_checkpoint_dir": str(Path(router_checkpoint_dir).resolve()),
        "query_head_checkpoint_dir": str(Path(query_head_checkpoint_dir).resolve()),
        "fusion_config_path": str(Path(fusion_config_path).resolve()),
        "selector_rows_path": str(Path(selector_rows_path).resolve()),
        "git_commit": git_commit,
        "created_at": _utc_now(),
        "confirmation_eval_consumed": False,
        "metadata": dict(metadata or {}),
    }
    payload["lock_hash"] = compute_config_hash(payload)
    return payload


def validate_confirmation_eval_lock(
    lock_path: str | Path,
    *,
    split_manifest_path: str | Path | None = None,
    allow_rerun: bool = False,
    rerun_reason: str | None = None,
) -> Dict[str, Any]:
    path = Path(lock_path)
    if not path.exists():
        raise FileNotFoundError(f"Confirmation eval lock does not exist: {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if bool(lock.get("confirmation_eval_consumed")) and not allow_rerun:
        raise RuntimeError("Confirmation eval has already been consumed; pass explicit rerun controls to override.")
    if allow_rerun and not rerun_reason:
        raise ValueError("--rerun-reason is required when --allow-confirmation-rerun is used.")
    if split_manifest_path is not None:
        current_hash = hash_file(split_manifest_path)
        if current_hash != lock.get("split_manifest_hash"):
            raise RuntimeError("Confirmation lock split_manifest_hash does not match the current split manifest.")
    return lock


def mark_confirmation_eval_consumed(
    lock_path: str | Path,
    *,
    output_dir: str | Path,
    allow_rerun: bool = False,
    rerun_reason: str | None = None,
) -> Dict[str, Any]:
    path = Path(lock_path)
    lock = json.loads(path.read_text(encoding="utf-8"))
    already_consumed = bool(lock.get("confirmation_eval_consumed"))
    if already_consumed and not allow_rerun:
        raise RuntimeError("Confirmation eval lock was already consumed.")
    if allow_rerun:
        lock.setdefault("reruns", []).append(
            {
                "reason": rerun_reason,
                "rerun_at": _utc_now(),
                "output_dir": str(Path(output_dir).resolve()),
            }
        )
        lock["last_rerun_output_dir"] = str(Path(output_dir).resolve())
        lock["last_rerun_completed_at"] = _utc_now()
        path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return lock
    lock["confirmation_eval_consumed"] = True
    lock["confirmation_eval_completed_at"] = _utc_now()
    lock["confirmation_eval_output_dir"] = str(Path(output_dir).resolve())
    path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return lock
