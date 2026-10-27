"""In-memory vector index. Uses FAISS (exact inner-product search) when it is
installed and falls back to NumPy, which gives identical results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag.chunker import Chunk
from rag.embeddings import l2_normalise

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except ImportError:  # pragma: no cover - depends on environment
    faiss = None
    HAS_FAISS = False


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity, -1..1


class VectorStore:
    def __init__(self, use_faiss: bool | None = None):
        self.use_faiss = HAS_FAISS if use_faiss is None else (use_faiss and HAS_FAISS)
        self._chunks: list[Chunk] = []
        self._index = None  # faiss.IndexFlatIP
        self._matrix: np.ndarray | None = None  # numpy backend
        self.dim: int | None = None

    @property
    def backend(self) -> str:
        return "faiss" if self.use_faiss else "numpy"

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        vectors = l2_normalise(vectors)
        if self.dim is None:
            self.dim = vectors.shape[1]
            if self.use_faiss:
                self._index = faiss.IndexFlatIP(self.dim)
        elif vectors.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim}-d vectors, got {vectors.shape[1]}-d")

        if self.use_faiss:
            self._index.add(vectors)
        else:
            self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[SearchResult]:
        if not self._chunks:
            return []
        query = l2_normalise(np.asarray(query_vector).reshape(1, -1))
        k = min(k, len(self._chunks))
        if self.use_faiss:
            scores, ids = self._index.search(query, k)
            pairs = zip(ids[0].tolist(), scores[0].tolist())
        else:
            sims = self._matrix @ query[0]
            top = np.argsort(-sims, kind="stable")[:k]
            pairs = ((int(i), float(sims[i])) for i in top)
        return [SearchResult(self._chunks[i], float(s)) for i, s in pairs if i >= 0]

    def clear(self) -> None:
        self.__init__(use_faiss=self.use_faiss)

    def sources(self) -> list[str]:
        return sorted({c.source for c in self._chunks})
