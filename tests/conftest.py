"""Test doubles: a deterministic bag-of-words embedder and a fake LLM, so the
whole pipeline can be tested in seconds with no model downloads."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import Settings  # noqa: E402

_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Maps each word to a fixed dimension via a stable hash. Texts that share
    words get similar vectors, which is enough to test retrieval logic."""

    def __init__(self, dim: int = 4096):  # large enough to avoid hash collisions
        self.dim = dim
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                if len(word) < 3:
                    continue
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
        return out


class FakeLLM:
    name = "fake-llm"

    def __init__(self, reply: str = "Fake answer [1].", fail: bool = False):
        self.reply = reply
        self.fail = fail
        self.last_messages: list[dict] | None = None

    def generate(self, messages):
        self.last_messages = messages
        if self.fail:
            raise RuntimeError("simulated API outage")
        return self.reply


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def settings():
    return Settings(hf_token=None, llm_api_key=None, chunk_size=300, chunk_overlap=60, top_k=3, min_score=0.1)


@pytest.fixture
def sample_pdf():
    path = Path(__file__).resolve().parent.parent / "samples" / "rag_primer.pdf"
    if not path.exists():
        from scripts.make_sample_pdf import build

        path.parent.mkdir(exist_ok=True)
        build(path)
    return path
