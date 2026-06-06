from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from genrec.memory.data_adapter import build_eval_samples, load_interactions


@dataclass
class RouterExample:
    sample_id: str
    domain: str
    user_id: str
    history_item_ids: List[str]
    target_item_id: str
    route_prefix1: int
    route_prefix2: int
    cold: bool


@dataclass
class RouteVocab:
    prefix1_values: List[int]
    prefix2_values: Dict[int, List[int]]

    @property
    def num_prefix1(self) -> int:
        return len(self.prefix1_values)

    @property
    def num_prefix2(self) -> int:
        return max((len(values) for values in self.prefix2_values.values()), default=0)

    def encode(self, route: Sequence[Any]) -> Tuple[int, int]:
        prefix1 = int(route[0])
        prefix2 = int(route[1])
        return self.prefix1_values.index(prefix1), self.prefix2_values[prefix1].index(prefix2)

    def decode_prefix1(self, index: int) -> int:
        return int(self.prefix1_values[index])

    def decode_prefix2(self, prefix1_index: int, prefix2_index: int) -> Tuple[int, int]:
        prefix1_value = self.decode_prefix1(prefix1_index)
        prefix2_value = int(self.prefix2_values[prefix1_value][prefix2_index])
        return prefix1_value, prefix2_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prefix1_values": self.prefix1_values,
            "prefix2_values": {str(key): value for key, value in self.prefix2_values.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouteVocab":
        return cls(
            prefix1_values=[int(value) for value in data["prefix1_values"]],
            prefix2_values={int(key): [int(value) for value in values] for key, values in data["prefix2_values"].items()},
        )


def load_route_mapping(path: str | Path) -> Dict[str, Tuple[int, int]]:
    route_path = Path(path)
    if route_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in route_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mapping = {str(row["item_id"]): tuple(int(part) for part in row["sid"]) for row in rows}
    else:
        payload = json.loads(route_path.read_text(encoding="utf-8"))
        mapping = {str(item_id): tuple(int(part) for part in sid) for item_id, sid in payload.items()}
    return mapping


def build_route_vocab(route_mapping: Mapping[str, Sequence[Any]]) -> RouteVocab:
    prefix1_values = sorted({int(route[0]) for route in route_mapping.values()})
    prefix2_values: Dict[int, List[int]] = {}
    for prefix1 in prefix1_values:
        prefix2_values[prefix1] = sorted({int(route[1]) for route in route_mapping.values() if int(route[0]) == prefix1})
    return RouteVocab(prefix1_values=prefix1_values, prefix2_values=prefix2_values)


def _item_ids_from_record(record: Mapping[str, Any]) -> List[str]:
    items = record.get("Items") or record.get("ReviewList") or []
    if isinstance(items, dict):
        items = [items]
    values: List[str] = []
    for item in items:
        if isinstance(item, Mapping):
            for key in ("item_id", "ItemID", "ParentASIN", "BusinessID"):
                if item.get(key) not in (None, ""):
                    values.append(str(item[key]))
                    break
    return values


def build_training_examples(
    data_dir: str | Path,
    item_embeddings: Mapping[str, np.ndarray],
    route_mapping: Mapping[str, Tuple[int, int]],
    max_history: int = 10,
) -> List[RouterExample]:
    sequences: Dict[Tuple[str, str], List[str]] = {}
    for record in load_interactions(data_dir, "train"):
        domain = record["__domain__"]
        user_id = str(record.get("UserID") or record.get("user_id") or record.get("uid"))
        key = (domain, user_id)
        sequences.setdefault(key, [])
        for item_id in _item_ids_from_record(record):
            if item_id in item_embeddings and item_id in route_mapping:
                sequences[key].append(item_id)

    examples: List[RouterExample] = []
    for (domain, user_id), sequence in sequences.items():
        for idx in range(1, len(sequence)):
            history = sequence[max(0, idx - max_history):idx]
            target = sequence[idx]
            if not history:
                continue
            prefix1, prefix2 = route_mapping[target]
            examples.append(
                RouterExample(
                    sample_id=f"train:{domain}:{user_id}:{idx}",
                    domain=domain,
                    user_id=user_id,
                    history_item_ids=history,
                    target_item_id=target,
                    route_prefix1=int(prefix1),
                    route_prefix2=int(prefix2),
                    cold=False,
                )
            )
    if not examples:
        raise ValueError("No router training examples were created from the warm train split.")
    return examples


def build_eval_router_samples(
    data_dir: str | Path,
    route_mapping: Mapping[str, Tuple[int, int]],
    cold_only: bool,
    item_embeddings: Mapping[str, np.ndarray],
    max_history: int = 10,
) -> List[RouterExample]:
    samples = build_eval_samples(data_dir, split="test", cold_only=cold_only)
    examples: List[RouterExample] = []
    for sample in samples:
        target = str(sample["target"])
        if target not in route_mapping or target not in item_embeddings:
            continue
        history = [item_id for item_id in sample["history"] if item_id in item_embeddings]
        if not history:
            continue
        prefix1, prefix2 = route_mapping[target]
        examples.append(
            RouterExample(
                sample_id=str(sample["sample_id"]),
                domain=str(sample["domain"]),
                user_id=str(sample["user_id"]),
                history_item_ids=history[-max_history:],
                target_item_id=target,
                route_prefix1=int(prefix1),
                route_prefix2=int(prefix2),
                cold=bool(sample["cold"]),
            )
        )
    if not examples:
        raise ValueError("No router evaluation examples were created.")
    return examples


class RouterDataset(Dataset):
    def __init__(self, examples: Sequence[RouterExample], item_embeddings: Mapping[str, np.ndarray], route_vocab: RouteVocab) -> None:
        self.examples = list(examples)
        self.item_embeddings = item_embeddings
        self.route_vocab = route_vocab
        self.embedding_dim = len(next(iter(item_embeddings.values())))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        history = np.stack([np.asarray(self.item_embeddings[item_id], dtype=np.float32) for item_id in example.history_item_ids], axis=0)
        target_embedding = np.asarray(self.item_embeddings[example.target_item_id], dtype=np.float32)
        route1_idx, route2_idx = self.route_vocab.encode((example.route_prefix1, example.route_prefix2))
        return {
            "sample_id": example.sample_id,
            "domain": example.domain,
            "user_id": example.user_id,
            "history_item_ids": example.history_item_ids,
            "target_item_id": example.target_item_id,
            "history_embs": history,
            "target_embedding": target_embedding,
            "route1_idx": route1_idx,
            "route2_idx": route2_idx,
            "cold": example.cold,
        }

    def collate_fn(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(row["history_embs"]) for row in batch)
        history_embs = torch.zeros(len(batch), max_len, self.embedding_dim, dtype=torch.float32)
        history_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        target_embeddings = torch.zeros(len(batch), self.embedding_dim, dtype=torch.float32)
        route1_idx = torch.zeros(len(batch), dtype=torch.long)
        route2_idx = torch.zeros(len(batch), dtype=torch.long)
        cold = torch.zeros(len(batch), dtype=torch.bool)
        for idx, row in enumerate(batch):
            current = torch.from_numpy(row["history_embs"])
            history_embs[idx, -len(current):] = current
            history_mask[idx, -len(current):] = True
            target_embeddings[idx] = torch.from_numpy(row["target_embedding"])
            route1_idx[idx] = int(row["route1_idx"])
            route2_idx[idx] = int(row["route2_idx"])
            cold[idx] = bool(row["cold"])
        return {
            "sample_id": [row["sample_id"] for row in batch],
            "domain": [row["domain"] for row in batch],
            "user_id": [row["user_id"] for row in batch],
            "history_item_ids": [row["history_item_ids"] for row in batch],
            "target_item_id": [row["target_item_id"] for row in batch],
            "history_embs": history_embs,
            "history_mask": history_mask,
            "target_embedding": target_embeddings,
            "route1_idx": route1_idx,
            "route2_idx": route2_idx,
            "cold": cold,
        }
