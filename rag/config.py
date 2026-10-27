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


@dataclass(frozen=True)
class Settings:
    # Embedding model: small, fast on CPU, 384-dimensional vectors.
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    # "api" (Hugging Face Inference API, default) or "local" (sentence-transformers).
    embedding_backend: str = field(default_factory=lambda: os.environ.get("EMBEDDING_BACKEND", "api"))
    # Open-weight chat model served through Hugging Face Inference Providers.
    llm_model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    )
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN") or None)

    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 150))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 4))
    # Cosine similarity below this means "not in the documents".
    min_score: float = field(default_factory=lambda: _float("MIN_SCORE", 0.25))
    max_new_tokens: int = field(default_factory=lambda: _int("MAX_NEW_TOKENS", 512))

    # Guard rails for a small free instance.
    max_files: int = field(default_factory=lambda: _int("MAX_FILES", 5))
    max_pages_per_file: int = field(default_factory=lambda: _int("MAX_PAGES", 200))
