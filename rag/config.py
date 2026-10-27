"""Runtime settings, read from environment variables so the same code runs
locally and on the host (Render, Hugging Face Spaces), where secrets become env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _default_llm_model() -> str:
    if os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY"):
        return "openai/gpt-oss-120b"  # on Groq's free tier
    return "Qwen/Qwen2.5-7B-Instruct"  # via Hugging Face Inference Providers


@dataclass(frozen=True)
class Settings:
    # Embedding model: small, fast on CPU, 384-dimensional vectors.
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    # Retrieval: "bm25" (default, free, no model), "api" (dense embeddings via the
    # Hugging Face Inference API) or "local" (sentence-transformers on this machine).
    embedding_backend: str = field(default_factory=lambda: os.environ.get("EMBEDDING_BACKEND", "bm25"))
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN") or None)

    # LLM through any OpenAI-compatible API. Default: Groq's free tier.
    llm_api_key: str | None = field(
        default_factory=lambda: os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY") or None
    )
    llm_base_url: str = field(
        default_factory=lambda: os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    )
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", _default_llm_model()))

    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 150))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 4))
    # Cosine similarity below this means "not in the documents".
    min_score: float = field(default_factory=lambda: _float("MIN_SCORE", 0.25))
    # BM25 scores are normalised to 0..1 (share of the best possible match).
    bm25_min_score: float = field(default_factory=lambda: _float("BM25_MIN_SCORE", 0.08))
    max_new_tokens: int = field(default_factory=lambda: _int("MAX_NEW_TOKENS", 1024))

    # Guard rails for a small free instance.
    max_files: int = field(default_factory=lambda: _int("MAX_FILES", 5))
    max_pages_per_file: int = field(default_factory=lambda: _int("MAX_PAGES", 200))
