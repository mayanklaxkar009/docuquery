"""Text embeddings. Any object with `embed(list[str]) -> np.ndarray` works,
which lets the tests swap in a tiny deterministic embedder."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class SentenceTransformerEmbedder:
    """Runs a sentence-transformers model locally on CPU. Vectors are
    L2-normalised, so inner product equals cosine similarity."""

    def __init__(self, model_name: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)
