"""BM25 keyword retrieval. Runs locally with no model, API key or GPU, so the
app keeps working on the smallest free hosting tiers.

Scores are normalised to 0..1 by dividing by the best score the query could
possibly get, which makes a single relevance threshold work across documents.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from rag.chunker import Chunk
from rag.vector_store import SearchResult

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a about after all also am an and any are as at be because been but by can could did do does each "
    "for from had has have he her his how i if in into is it its me more most my no not now of on only "
    "or other our out over she should so some such than that the their them then there these they this "
    "those through to too under up us very was we were what when where which while who whom why will "
    "with would you your".split()
)


def stem(word: str) -> str:
    """Tiny suffix stripper so 'chunks', 'chunked' and 'chunking' match."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


class BM25Index:
    backend = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.clear()

    def clear(self) -> None:
        self._chunks: list[Chunk] = []
        self._tfs: list[Counter] = []
        self._lengths: list[int] = []
        self._df: Counter = Counter()

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors=None) -> None:  # vectors unused; same API as VectorStore
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            tf = Counter(tokens)
            self._chunks.append(chunk)
            self._tfs.append(tf)
            self._lengths.append(len(tokens))
            self._df.update(tf.keys())

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        return math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))

    def search(self, query: str, k: int) -> list[SearchResult]:
        terms = list(dict.fromkeys(tokenize(query)))  # unique, keep order
        if not terms or not self._chunks or k <= 0:
            return []
        idf = {t: self._idf(t) for t in terms}
        # Best achievable score: every query term present with saturated frequency.
        max_score = sum(idf.values()) * (self.k1 + 1)
        avg_len = sum(self._lengths) / len(self._lengths) or 1.0
        scored = []
        for i, tf in enumerate(self._tfs):
            norm = self.k1 * (1 - self.b + self.b * self._lengths[i] / avg_len)
            score = sum(idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if tf[t])
            if score > 0:
                scored.append((score / max_score, i))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [SearchResult(self._chunks[i], float(s)) for s, i in scored[:k]]
