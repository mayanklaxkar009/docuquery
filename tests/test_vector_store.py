import numpy as np
import pytest

from rag.chunker import Chunk
from rag.vector_store import HAS_FAISS, VectorStore

BACKENDS = [False] + ([True] if HAS_FAISS else [])


def _chunks(n):
    return [Chunk(i, f"text {i}", "doc.pdf", i + 1) for i in range(n)]


@pytest.mark.parametrize("use_faiss", BACKENDS)
def test_search_returns_nearest_by_cosine(use_faiss):
    store = VectorStore(use_faiss=use_faiss)
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], dtype=np.float32)
    store.add(_chunks(3), vectors)
    results = store.search(np.array([1, 0, 0]), k=2)
    assert [r.chunk.chunk_id for r in results] == [0, 2]
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[0].score >= results[1].score


@pytest.mark.parametrize("use_faiss", BACKENDS)
def test_k_larger_than_index_and_empty_store(use_faiss):
    store = VectorStore(use_faiss=use_faiss)
    assert store.search(np.ones(3)) == []
    store.add(_chunks(2), np.eye(2, 3))
    assert len(store.search(np.ones(3), k=10)) == 2


def test_dimension_mismatch_is_rejected():
    store = VectorStore(use_faiss=False)
    store.add(_chunks(1), np.ones((1, 3)))
    with pytest.raises(ValueError):
        store.add(_chunks(1), np.ones((1, 4)))


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        VectorStore(use_faiss=False).add(_chunks(2), np.ones((1, 3)))


def test_clear_resets_everything():
    store = VectorStore(use_faiss=False)
    store.add(_chunks(2), np.eye(2, 3))
    store.clear()
    assert len(store) == 0 and store.dim is None and store.sources() == []


@pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
def test_faiss_and_numpy_agree():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(50, 16)).astype(np.float32)
    query = rng.normal(size=16)
    a, b = VectorStore(use_faiss=True), VectorStore(use_faiss=False)
    a.add(_chunks(50), vectors)
    b.add(_chunks(50), vectors)
    ra, rb = a.search(query, 5), b.search(query, 5)
    assert [r.chunk.chunk_id for r in ra] == [r.chunk.chunk_id for r in rb]
    assert np.allclose([r.score for r in ra], [r.score for r in rb], atol=1e-5)
