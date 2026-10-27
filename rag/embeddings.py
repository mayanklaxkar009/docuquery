"""Text embeddings. Any object with `embed(list[str]) -> np.ndarray` works,
which lets the tests swap in a tiny deterministic embedder.

`HFInferenceEmbedder` (the default) calls the Hugging Face Inference API, so the
app needs no PyTorch and fits in a small free web-service instance.
With no embedder configured the pipeline uses BM25 keyword retrieval instead.
`SentenceTransformerEmbedder` runs the model locally if you install
sentence-transformers yourself (set EMBEDDING_BACKEND=local).
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _to_matrix(raw, n_texts: int) -> np.ndarray:
    """Normalise the API's output to shape (n_texts, dim). Sentence-transformer
    models return pooled vectors; plain transformer models return one vector
    per token, which we mean-pool."""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 3:
        arr = arr.mean(axis=1)
    if arr.ndim != 2 or arr.shape[0] != n_texts:
        raise ValueError(f"unexpected embedding shape {arr.shape} for {n_texts} texts")
    return arr


class HFInferenceEmbedder:
    """Embeds text with a sentence-transformers model hosted on the Hugging Face
    Inference API. Requests are batched and retried, and vectors are
    L2-normalised so inner product equals cosine similarity."""

    def __init__(self, model_name: str, token: str, batch_size: int = 32, retries: int = 3, client=None):
        if client is None:
            from huggingface_hub import InferenceClient

            client = InferenceClient(model=model_name, token=token, timeout=60)
        self.client = client
        self.model_name = model_name
        self.batch_size = batch_size
        self.retries = retries
        self.dim: int | None = None

    def _call(self, batch: list[str]) -> np.ndarray:
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                try:
                    return _to_matrix(self.client.feature_extraction(batch), len(batch))
                except (TypeError, ValueError):
                    if len(batch) == 1:
                        raise
                    # Some providers only accept one string per request.
                    return np.vstack([_to_matrix(self.client.feature_extraction(t), 1) for t in batch])
            except Exception as exc:  # noqa: BLE001 - network/model-loading errors are retried
                last_exc = exc
                if attempt < self.retries - 1:
                    wait = 2**attempt
                    log.warning("Embedding request failed (%s); retrying in %ss", exc, wait)
                    time.sleep(wait)
        raise RuntimeError(f"embedding API failed: {last_exc}") from last_exc

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 384), dtype=np.float32)
        parts = [self._call(texts[i : i + self.batch_size]) for i in range(0, len(texts), self.batch_size)]
        vectors = l2_normalise(np.vstack(parts))
        self.dim = vectors.shape[1]
        return vectors



class SentenceTransformerEmbedder:
    """Runs a sentence-transformers model locally on CPU (optional; needs
    `pip install sentence-transformers`)."""

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


def make_embedder(model_name: str, token: str | None, backend: str = "bm25") -> Embedder | None:
    """Pick the embedder for this deployment, or None for BM25 retrieval.
    For the API backend a test call runs at start-up; if it fails (no token,
    no credits, network) the app uses BM25 instead of crashing."""
    if backend == "local":
        return SentenceTransformerEmbedder(model_name)
    if backend != "api":
        return None
    if not token:
        log.warning("EMBEDDING_BACKEND=api but HF_TOKEN is not set: using BM25")
        return None
    embedder = HFInferenceEmbedder(model_name, token)
    try:
        embedder.embed(["warm-up"])
        return embedder
    except Exception as exc:  # noqa: BLE001
        log.warning("Embedding API unavailable (%s): using BM25", exc)
        return None
