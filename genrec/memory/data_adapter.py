"""Data adapters for oracle-route retrieval experiments.

The current hypothesis is not to train a new model first. Instead we test:

    History -> route -> dynamic item memory -> item

These helpers keep the evaluation path flexible across slightly different local
layouts. They prefer explicit user-provided paths, then fall back to common
project locations, and raise clear errors if required artifacts are missing.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ITEM_ID_KEYS = ("item_id", "ItemID", "ParentASIN", "BusinessID")
ITEM_NAME_KEYS = ("label", "ItemName", "BusinessName", "title", "name")
EMBEDDING_KEYS = ("embedding", "emb", "vector", "item_embedding")
SID_KEYS = ("sid", "SID", "semantic_id", "semantic_ids", "tokens", "token_ids", "item_sid")
PREDICTION_FILE_HINTS = ("prediction", "predictions", "routes")
TOKENIZER_FILE_HINTS = ("tokenizer", "semantic", "sid")


def _candidate_roots(data_dir: str | Path) -> List[Path]:
    root = Path(data_dir).expanduser().resolve()
    roots = [root]
    if (root / "task").exists() or (root / "raw_data").exists():
        return roots
    for child in (root / "user_simulator", root.parent / "user_simulator"):
        if child.exists():
            roots.append(child.resolve())
    return list(dict.fromkeys(roots))


def _find_split_files(data_dir: str | Path, split: str) -> List[Path]:
    matches: List[Path] = []
    for root in _candidate_roots(data_dir):
        for folder in (root / "task", root):
            if not folder.exists():
                continue
            for pattern in (f"*_{split}.jsonl", f"{split}.jsonl"):
                matches.extend(sorted(folder.glob(pattern)))
    unique = []
    seen = set()
    for path in matches:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _extract_domain(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return path.parent.name or "unknown"


def _extract_item_id(item: Mapping[str, Any]) -> Optional[str]:
    for key in ITEM_ID_KEYS:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _extract_item_ids(record: Mapping[str, Any]) -> List[str]:
    items = record.get("Items") or record.get("ReviewList") or record.get("items") or []
    if isinstance(items, dict):
        items = [items]
    item_ids = []
    for item in items:
        if isinstance(item, Mapping):
            item_id = _extract_item_id(item)
            if item_id is not None:
                item_ids.append(item_id)
        elif item not in (None, ""):
            item_ids.append(str(item))
    if not item_ids:
        for key in ITEM_ID_KEYS:
            if key in record and record[key] not in (None, ""):
                item_ids.append(str(record[key]))
                break
    return item_ids


def _load_json_or_jsonl(path: Path) -> Any:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_sid_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [str(part) for part in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        for sep in (",", "|", "/"):
            if sep in text:
                return [part.strip() for part in text.split(sep) if part.strip()]
        if " " in text:
            return [part for part in text.split() if part]
        return text
    return str(value)


def _load_mapping_records(path: Path) -> Dict[str, Any]:
    data = _load_json_or_jsonl(path)
    if isinstance(data, Mapping):
        return {str(key): _normalize_sid_value(value) for key, value in data.items()}
    mapping: Dict[str, Any] = {}
    for row in data:
        if not isinstance(row, Mapping):
            continue
        item_id = _extract_item_id(row)
        if item_id is None:
            continue
        sid_value = None
        for key in SID_KEYS:
            if key in row and row[key] not in (None, ""):
                sid_value = row[key]
                break
        if sid_value is not None:
            mapping[item_id] = _normalize_sid_value(sid_value)
    return mapping


def _load_pickle_mapping(path: Path) -> Dict[str, Any]:
    with path.open("rb") as file:
        data = pickle.load(file)
    if isinstance(data, Mapping):
        if "item_ids" in data and "sids" in data:
            return {str(item_id): _normalize_sid_value(sid) for item_id, sid in zip(data["item_ids"], data["sids"])}
        return {str(key): _normalize_sid_value(value) for key, value in data.items()}
    raise ValueError(f"Unsupported SID pickle structure in {path}.")


def _load_embedding_sidecar_ids(path: Path) -> List[str]:
    candidates = [
        path.with_suffix(".item_ids.json"),
        path.with_name(f"{path.stem}_item_ids.json"),
        path.parent / "item_ids.json",
        path.parent / "metadata.json",
        path.parent / "metadata.jsonl",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        data = _load_json_or_jsonl(candidate)
        if isinstance(data, list):
            ids = []
            for row in data:
                if isinstance(row, Mapping):
                    item_id = _extract_item_id(row)
                    if item_id is not None:
                        ids.append(item_id)
                elif row not in (None, ""):
                    ids.append(str(row))
            if ids:
                return ids
        elif isinstance(data, Mapping):
            if "item_ids" in data:
                return [str(item_id) for item_id in data["item_ids"]]
            ids = [str(key) for key in data.keys()]
            if ids:
                return ids
    raise FileNotFoundError(
        f"{path} is a raw .npy embedding matrix without an item-id sidecar. Provide a sidecar such as "
        f"{path.with_name(path.stem + '_item_ids.json').name} or pass a .pkl/.jsonl file that contains both ids and embeddings."
    )


def _load_embeddings_from_path(path: Path) -> Dict[str, np.ndarray]:
    if path.suffix == ".npy":
        matrix = np.load(path, allow_pickle=True)
        if matrix.dtype == object and matrix.shape == ():
            matrix = matrix.item()
        if isinstance(matrix, Mapping):
            return {str(key): np.asarray(value, dtype=np.float32) for key, value in matrix.items()}
        item_ids = _load_embedding_sidecar_ids(path)
        matrix = np.asarray(matrix, dtype=np.float32)
        if len(item_ids) != len(matrix):
            raise ValueError(f"Embedding matrix rows do not match item ids in sidecar for {path}.")
        return {item_id: matrix[idx].astype(np.float32) for idx, item_id in enumerate(item_ids)}
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        item_ids = [str(item_id) for item_id in data["item_ids"]]
        matrix = np.asarray(data["embeddings"], dtype=np.float32)
        return {item_id: matrix[idx] for idx, item_id in enumerate(item_ids)}
    if path.suffix == ".pkl":
        with path.open("rb") as file:
            data = pickle.load(file)
        if isinstance(data, Mapping):
            if "item_ids" in data and "embeddings" in data:
                return {
                    str(item_id): np.asarray(embedding, dtype=np.float32)
                    for item_id, embedding in zip(data["item_ids"], data["embeddings"])
                }
            return {str(key): np.asarray(value, dtype=np.float32) for key, value in data.items()}
        raise ValueError(f"Unsupported embedding pickle structure in {path}.")
    if path.suffix in {".json", ".jsonl"}:
        data = _load_json_or_jsonl(path)
        rows = data if isinstance(data, list) else [data]
        result = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            item_id = _extract_item_id(row)
            if item_id is None:
                continue
            for key in EMBEDDING_KEYS:
                if key in row and row[key] is not None:
                    result[item_id] = np.asarray(row[key], dtype=np.float32)
                    break
        return result
    raise ValueError(f"Unsupported embedding file format: {path}.")


def inspect_available_artifacts(data_dir: str | Path) -> Dict[str, List[str]]:
    roots = _candidate_roots(data_dir)
    split_files = []
    sid_candidates = []
    embedding_candidates = []
    prediction_candidates = []
    tokenizer_candidates = []
    metadata_candidates = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lowered = path.name.lower()
            if path.suffix == ".jsonl" and any(split in lowered for split in ("train", "valid", "test")):
                split_files.append(str(path))
            if any(hint in lowered for hint in TOKENIZER_FILE_HINTS):
                tokenizer_candidates.append(str(path))
            if "sid" in lowered:
                sid_candidates.append(str(path))
            if "embed" in lowered or path.name in {"embeddings.npy", "item_embeddings.npy"}:
                embedding_candidates.append(str(path))
            if any(hint in lowered for hint in PREDICTION_FILE_HINTS):
                prediction_candidates.append(str(path))
            if "processed_data" in lowered or "items" in lowered:
                metadata_candidates.append(str(path))
    return {
        "split_files": sorted(set(split_files))[:50],
        "sid_candidates": sorted(set(sid_candidates))[:50],
        "embedding_candidates": sorted(set(embedding_candidates))[:50],
        "prediction_candidates": sorted(set(prediction_candidates))[:50],
        "tokenizer_candidates": sorted(set(tokenizer_candidates))[:50],
        "metadata_candidates": sorted(set(metadata_candidates))[:50],
    }


def load_interactions(data_dir: str | Path, split: str) -> List[Dict[str, Any]]:
    files = _find_split_files(data_dir, split)
    if not files:
        raise FileNotFoundError(
            f"Could not locate a '{split}' interaction split under {data_dir}. "
            f"Expected files such as '*_{split}.jsonl' in task/ or the data root."
        )
    records: List[Dict[str, Any]] = []
    for path in files:
        domain = _extract_domain(path)
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object rows in {path}:{line_number}.")
                row = dict(row)
                row["__domain__"] = domain
                row["__source_path__"] = str(path)
                records.append(row)
    return records


def load_train_item_set(data_dir: str | Path) -> set[str]:
    train_records = load_interactions(data_dir, "train")
    return {item_id for record in train_records for item_id in _extract_item_ids(record)}


def _auto_find_sid_file(data_dir: str | Path) -> Optional[Path]:
    for root in _candidate_roots(data_dir):
        candidates = []
        for pattern in ("*sid*.json", "*sid*.jsonl", "*sid*.pkl", "*semantic*.json", "*semantic*.jsonl"):
            candidates.extend(sorted(root.rglob(pattern)))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    return None


def load_item_sids(data_dir: str | Path, item_sid_path: Optional[str | Path] = None) -> Dict[str, Any]:
    path = Path(item_sid_path).expanduser().resolve() if item_sid_path else _auto_find_sid_file(data_dir)
    if path is None or not path.exists():
        artifacts = inspect_available_artifacts(data_dir)
        raise FileNotFoundError(
            "Could not locate an item SID mapping automatically. "
            "Provide --item_sid_path pointing to a .json/.jsonl/.pkl mapping from item_id to SID or semantic tokens. "
            f"Nearby tokenizer/SID candidates: {artifacts['sid_candidates'][:5] + artifacts['tokenizer_candidates'][:5]}"
        )
    if path.suffix in {".json", ".jsonl"}:
        mapping = _load_mapping_records(path)
    elif path.suffix == ".pkl":
        mapping = _load_pickle_mapping(path)
    else:
        raise ValueError(f"Unsupported SID file format: {path}.")
    if not mapping:
        raise ValueError(f"Loaded SID file {path} but did not find any item_id -> SID entries.")
    return mapping


def _auto_find_embedding_file(data_dir: str | Path) -> Optional[Path]:
    for root in _candidate_roots(data_dir):
        candidates = []
        for pattern in ("embeddings.npy", "item_embeddings.npy", "*embed*.pkl", "*embed*.jsonl", "*.npz"):
            candidates.extend(sorted(root.rglob(pattern)))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    return None


def load_item_embeddings(data_dir: str | Path, item_embedding_path: Optional[str | Path] = None) -> Dict[str, np.ndarray]:
    path = Path(item_embedding_path).expanduser().resolve() if item_embedding_path else _auto_find_embedding_file(data_dir)
    if path is None or not path.exists():
        artifacts = inspect_available_artifacts(data_dir)
        raise FileNotFoundError(
            "Could not locate item embeddings automatically. "
            "Provide --item_embedding_path pointing to a .npy/.npz/.pkl/.jsonl file containing item embeddings. "
            f"Nearby embedding candidates: {artifacts['embedding_candidates'][:10]}"
        )
    embeddings = _load_embeddings_from_path(path)
    if not embeddings:
        raise ValueError(f"Loaded embedding file {path} but did not find any item_id -> embedding entries.")
    return embeddings


def load_item_metadata(data_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    roots = _candidate_roots(data_dir)
    domains = {record["__domain__"] for record in load_interactions(data_dir, "train") + load_interactions(data_dir, "test")}
    preferred_files: List[Path] = []
    for root in roots:
        raw_root = root / "raw_data"
        if not raw_root.exists():
            continue
        for domain in sorted(domains):
            for name in ("processed_data_with_summaries.jsonl", "processed_data.jsonl"):
                candidate = raw_root / domain / name
                if candidate.exists():
                    preferred_files.append(candidate)
    for path in preferred_files:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                review_list = row.get("ReviewList") or []
                if isinstance(review_list, dict):
                    review_list = [review_list]
                for item in review_list:
                    if not isinstance(item, Mapping):
                        continue
                    item_id = _extract_item_id(item)
                    if item_id is None or item_id in metadata:
                        continue
                    label = None
                    for key in ITEM_NAME_KEYS:
                        if key in item and item[key] not in (None, ""):
                            label = str(item[key])
                            break
                    metadata[item_id] = {
                        "label": label or item_id,
                        "metadata": dict(item),
                    }
    return metadata


def build_eval_samples(data_dir: str | Path, split: str = "test", cold_only: bool = True) -> List[Dict[str, Any]]:
    train_records = load_interactions(data_dir, "train")
    eval_records = load_interactions(data_dir, split)
    train_item_set = load_train_item_set(data_dir)

    history_by_user: Dict[Tuple[str, str], List[str]] = {}
    for record in train_records:
        user_id = str(record.get("UserID") or record.get("user_id") or record.get("uid"))
        history_by_user[(record["__domain__"], user_id)] = _extract_item_ids(record)

    samples: List[Dict[str, Any]] = []
    for record in eval_records:
        user_id = str(record.get("UserID") or record.get("user_id") or record.get("uid"))
        domain = record["__domain__"]
        history = history_by_user.get((domain, user_id), [])
        targets = _extract_item_ids(record)
        for offset, target in enumerate(targets):
            is_cold = target not in train_item_set
            if cold_only and not is_cold:
                continue
            samples.append(
                {
                    "sample_id": f"{domain}:{user_id}:{offset}",
                    "domain": domain,
                    "user_id": user_id,
                    "history": history,
                    "target": target,
                    "cold": is_cold,
                    "source_path": record["__source_path__"],
                }
            )
    if not samples:
        raise ValueError(
            f"No evaluation samples were built for split={split!r}, cold_only={cold_only}. "
            "Check the split files and whether the requested cold-item filter is too strict."
        )
    return samples
