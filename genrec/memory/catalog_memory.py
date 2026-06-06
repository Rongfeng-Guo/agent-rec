"""Catalog memory for route-conditioned item retrieval.

This module supports the current hypothesis shift from:

    History -> full SID -> item

to:

    History -> route -> dynamic item memory -> item

The `CatalogMemory` class stores item embeddings together with route assignments
and supports either global retrieval or route-conditioned retrieval. If a route
bucket is missing, retrieval falls back to the global catalog and increments
`route_miss_count` so the caller can audit how often routing failed.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

Route = Tuple[str, ...]
_GLOBAL_ROUTE_KEY: Route = ("__global__",)


def _to_float32_matrix(item_embs: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(item_embs, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {matrix.shape!r}.")
    return matrix


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _normalize_route(route: Optional[Sequence[Any] | str]) -> Optional[Route]:
    if route is None:
        return None
    if isinstance(route, tuple):
        parts = list(route)
    elif isinstance(route, list):
        parts = list(route)
    elif isinstance(route, str):
        text = route.strip()
        if not text:
            return tuple()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if re.search(r"[\s,|/]+", text):
            parts = [part for part in re.split(r"[\s,|/]+", text) if part]
        else:
            parts = [text]
    else:
        parts = [route]
    return tuple(str(part) for part in parts if str(part) != "")


class CatalogMemory:
    """A CPU-friendly catalog memory with optional FAISS acceleration."""

    def __init__(self, normalize: bool = True, prefer_faiss: bool = True) -> None:
        self.normalize = normalize
        self.prefer_faiss = prefer_faiss
        self.use_faiss = bool(prefer_faiss and faiss is not None)
        self.item_ids: List[str] = []
        self.labels: List[Optional[str]] = []
        self.metadata: List[Optional[Mapping[str, Any]]] = []
        self.routes: List[Optional[Route]] = []
        self._embeddings = np.zeros((0, 0), dtype=np.float32)
        self._route_to_indices: Dict[Route, List[int]] = {}
        self._index_cache: Dict[Route, Any] = {}
        self.route_miss_count = 0

    @property
    def num_items(self) -> int:
        return len(self.item_ids)

    @property
    def embedding_dim(self) -> int:
        return 0 if self._embeddings.size == 0 else int(self._embeddings.shape[1])

    @property
    def backend(self) -> str:
        return "faiss" if self.use_faiss else "numpy"

    def add_items(
        self,
        item_ids: Sequence[str],
        item_embs: Sequence[Sequence[float]] | np.ndarray,
        routes: Optional[Sequence[Optional[Sequence[Any] | str]]] = None,
        labels: Optional[Sequence[Optional[str]]] = None,
        metadata: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
    ) -> None:
        ids = [str(item_id) for item_id in item_ids]
        if not ids:
            return

        embs = _to_float32_matrix(item_embs)
        if len(ids) != len(embs):
            raise ValueError("item_ids and item_embs must have the same length.")
        if self._embeddings.size and embs.shape[1] != self._embeddings.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embeddings.shape[1]}, got {embs.shape[1]}."
            )
        if self.normalize:
            embs = _l2_normalize(embs)

        route_values = list(routes) if routes is not None else [None] * len(ids)
        label_values = list(labels) if labels is not None else [None] * len(ids)
        metadata_values = list(metadata) if metadata is not None else [None] * len(ids)
        if len(route_values) != len(ids):
            raise ValueError("routes must match the number of items.")
        if len(label_values) != len(ids):
            raise ValueError("labels must match the number of items.")
        if len(metadata_values) != len(ids):
            raise ValueError("metadata must match the number of items.")

        start_index = len(self.item_ids)
        self.item_ids.extend(ids)
        self.labels.extend(label_values)
        self.metadata.extend(metadata_values)

        normalized_routes: List[Optional[Route]] = []
        for offset, route in enumerate(route_values):
            normalized = _normalize_route(route)
            normalized_routes.append(normalized)
            if normalized is not None:
                self._route_to_indices.setdefault(normalized, []).append(start_index + offset)
        self.routes.extend(normalized_routes)

        if self._embeddings.size == 0:
            self._embeddings = embs.astype(np.float32, copy=False)
        else:
            self._embeddings = np.vstack([self._embeddings, embs]).astype(np.float32, copy=False)
        self._index_cache.clear()

    def _build_index(self, matrix: np.ndarray):
        if not self.use_faiss:
            return None
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(np.ascontiguousarray(matrix, dtype=np.float32))
        return index

    def _candidate_indices(self, route: Optional[Sequence[Any] | str]) -> Tuple[Route, np.ndarray]:
        if route is None:
            return _GLOBAL_ROUTE_KEY, np.arange(self.num_items, dtype=np.int64)
        normalized = _normalize_route(route)
        if normalized is None:
            return _GLOBAL_ROUTE_KEY, np.arange(self.num_items, dtype=np.int64)
        candidate_list = self._route_to_indices.get(normalized)
        if not candidate_list:
            self.route_miss_count += 1
            return _GLOBAL_ROUTE_KEY, np.arange(self.num_items, dtype=np.int64)
        return normalized, np.asarray(candidate_list, dtype=np.int64)

    def _search_numpy(self, query: np.ndarray, candidate_indices: np.ndarray, topk: int) -> List[Dict[str, Any]]:
        matrix = self._embeddings[candidate_indices]
        scores = matrix @ query[0]
        order = np.argsort(-scores)[:topk]
        results = []
        for rank, local_index in enumerate(order, start=1):
            global_index = int(candidate_indices[int(local_index)])
            results.append(
                {
                    "item_id": self.item_ids[global_index],
                    "score": float(scores[int(local_index)]),
                    "rank": rank,
                    "route": self.routes[global_index],
                    "label": self.labels[global_index],
                }
            )
        return results

    def _search_faiss(self, query: np.ndarray, cache_key: Route, candidate_indices: np.ndarray, topk: int) -> List[Dict[str, Any]]:
        index = self._index_cache.get(cache_key)
        if index is None:
            index = self._build_index(self._embeddings[candidate_indices])
            self._index_cache[cache_key] = index
        scores, local_indices = index.search(np.ascontiguousarray(query, dtype=np.float32), topk)
        results = []
        for rank, (score, local_index) in enumerate(zip(scores[0], local_indices[0]), start=1):
            if local_index < 0:
                continue
            global_index = int(candidate_indices[int(local_index)])
            results.append(
                {
                    "item_id": self.item_ids[global_index],
                    "score": float(score),
                    "rank": rank,
                    "route": self.routes[global_index],
                    "label": self.labels[global_index],
                }
            )
        return results

    def search(
        self,
        query: Sequence[float] | np.ndarray,
        route: Optional[Sequence[Any] | str] = None,
        topk: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search the catalog globally or within a route bucket."""
        if self.num_items == 0:
            return []

        query_vec = _to_float32_matrix(query)
        if query_vec.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self.embedding_dim}, got {query_vec.shape[1]}."
            )
        if self.normalize:
            query_vec = _l2_normalize(query_vec)

        cache_key, candidate_indices = self._candidate_indices(route)
        if len(candidate_indices) == 0:
            return []
        topk = min(int(topk), len(candidate_indices))
        if topk <= 0:
            return []
        if self.use_faiss:
            return self._search_faiss(query_vec, cache_key, candidate_indices, topk)
        return self._search_numpy(query_vec, candidate_indices, topk)

    def route_stats(self) -> List[Dict[str, Any]]:
        stats = []
        for route, indices in sorted(self._route_to_indices.items(), key=lambda item: (-len(item[1]), item[0])):
            stats.append({"route": route, "size": len(indices)})
        return stats

    def has_route(self, route: Optional[Sequence[Any] | str]) -> bool:
        normalized = _normalize_route(route)
        return normalized is not None and normalized in self._route_to_indices

    def candidate_count(self, route: Optional[Sequence[Any] | str] = None) -> int:
        if route is None:
            return self.num_items
        normalized = _normalize_route(route)
        if normalized is None:
            return self.num_items
        candidate_list = self._route_to_indices.get(normalized)
        if not candidate_list:
            return self.num_items
        return len(candidate_list)

    def save(self, path: str | Path) -> None:
        payload = {
            "normalize": self.normalize,
            "prefer_faiss": self.prefer_faiss,
            "item_ids": self.item_ids,
            "labels": self.labels,
            "metadata": self.metadata,
            "routes": self.routes,
            "embeddings": self._embeddings,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path) -> "CatalogMemory":
        with Path(path).open("rb") as file:
            payload = pickle.load(file)
        memory = cls(
            normalize=bool(payload.get("normalize", True)),
            prefer_faiss=bool(payload.get("prefer_faiss", True)),
        )
        memory.add_items(
            item_ids=payload["item_ids"],
            item_embs=payload["embeddings"],
            routes=payload.get("routes"),
            labels=payload.get("labels"),
            metadata=payload.get("metadata"),
        )
        return memory
