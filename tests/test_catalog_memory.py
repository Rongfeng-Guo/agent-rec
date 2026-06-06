import numpy as np

from genrec.memory.catalog_memory import CatalogMemory


ITEM_IDS = ["a", "b", "c"]
EMBS = np.array(
    [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ],
    dtype=np.float32,
)
ROUTES = [("r1",), ("r1",), ("r2",)]


def build_memory(prefer_faiss=True):
    memory = CatalogMemory(normalize=True, prefer_faiss=prefer_faiss)
    memory.add_items(ITEM_IDS, EMBS, routes=ROUTES, labels=["A", "B", "C"])
    return memory


def test_global_search_after_add_items():
    memory = build_memory(prefer_faiss=False)
    results = memory.search(np.array([1.0, 0.0], dtype=np.float32), topk=2)
    assert results[0]["item_id"] == "a"
    assert len(results) == 2


def test_route_conditioned_search_filters_bucket():
    memory = build_memory(prefer_faiss=False)
    results = memory.search(np.array([0.0, 1.0], dtype=np.float32), route=("r2",), topk=5)
    assert [row["item_id"] for row in results] == ["c"]


def test_route_missing_falls_back_to_global():
    memory = build_memory(prefer_faiss=False)
    results = memory.search(np.array([1.0, 0.0], dtype=np.float32), route=("missing",), topk=1)
    assert results[0]["item_id"] == "a"
    assert memory.route_miss_count == 1


def test_save_and_load_round_trip(tmp_path):
    memory = build_memory(prefer_faiss=False)
    save_path = tmp_path / "catalog_memory.pkl"
    memory.save(save_path)
    loaded = CatalogMemory.load(save_path)
    original = memory.search(np.array([1.0, 0.0], dtype=np.float32), route=("r1",), topk=2)
    restored = loaded.search(np.array([1.0, 0.0], dtype=np.float32), route=("r1",), topk=2)
    assert original == restored


def test_numpy_fallback_without_faiss():
    memory = build_memory(prefer_faiss=False)
    assert memory.backend == "numpy"
    results = memory.search(np.array([1.0, 0.0], dtype=np.float32), topk=3)
    assert len(results) == 3
