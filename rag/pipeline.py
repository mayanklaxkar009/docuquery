"""Ties the pieces together: PDFs -> chunks -> vectors -> retrieval -> answer."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunker import chunk_pages
from rag.config import Settings
from rag.bm25 import BM25Index
from rag.embeddings import Embedder, make_embedder
from rag.llm import LLM, ExtractiveAnswerer, build_messages
from rag.loader import load_pdf
from rag.vector_store import SearchResult, VectorStore

log = logging.getLogger(__name__)


@dataclass
class IndexReport:
    files: list[str] = field(default_factory=list)
    pages: int = 0
    chunks: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    sources: list[SearchResult]
    mode: str  # "llm", "extractive" or "no_match"
    note: str = ""


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        store: VectorStore | BM25Index | None = None,
        retrieval: str | None = None,
    ):
        """retrieval: "dense" or "bm25". By default it is "dense" when an embedder
        is passed in or configured (EMBEDDING_BACKEND), otherwise "bm25"."""
        self.settings = settings or Settings()
        if embedder is None and retrieval != "bm25":
            embedder = make_embedder(
                self.settings.embedding_model, self.settings.hf_token, self.settings.embedding_backend
            )
        self._embedder = embedder
        self.retrieval = retrieval or ("dense" if embedder is not None else "bm25")
        if self.retrieval == "dense" and embedder is None:
            raise ValueError("dense retrieval needs an embedder")
        self.store = store or (VectorStore() if self.retrieval == "dense" else BM25Index())
        self.extractive = ExtractiveAnswerer()
        self.llm = llm if llm is not None else self._default_llm()

    @property
    def embedder(self) -> Embedder | None:
        return self._embedder

    @property
    def min_score(self) -> float:
        return self.settings.min_score if self.retrieval == "dense" else self.settings.bm25_min_score

    def _default_llm(self) -> LLM | None:
        s = self.settings
        if s.llm_api_key:
            from rag.llm import OpenAICompatibleLLM

            return OpenAICompatibleLLM(s.llm_model, s.llm_api_key, s.llm_base_url, s.max_new_tokens)
        if s.hf_token:
            from rag.llm import HFInferenceLLM

            return HFInferenceLLM(s.llm_model, s.hf_token, s.max_new_tokens)
        log.warning("No LLM_API_KEY/GROQ_API_KEY or HF_TOKEN: answers will be extractive only")
        return None

    # -- indexing ------------------------------------------------------------
    def index_files(self, paths: list[str | Path], reset: bool = True) -> IndexReport:
        start = time.perf_counter()
        report = IndexReport()
        if reset:
            self.store.clear()

        paths = [Path(p) for p in paths]
        if len(paths) > self.settings.max_files:
            report.warnings.append(f"Only the first {self.settings.max_files} files were indexed.")
            paths = paths[: self.settings.max_files]

        all_chunks = []
        for path in paths:
            if path.suffix.lower() != ".pdf":
                report.warnings.append(f"{path.name}: skipped, not a PDF.")
                continue
            try:
                pages = load_pdf(path, max_pages=self.settings.max_pages_per_file)
            except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
                report.warnings.append(f"{path.name}: could not be read ({exc}).")
                continue
            if not pages:
                report.warnings.append(f"{path.name}: no text found (it may be a scanned image PDF).")
                continue
            chunks = chunk_pages(
                pages, self.settings.chunk_size, self.settings.chunk_overlap, start_id=len(self.store) + len(all_chunks)
            )
            all_chunks.extend(chunks)
            report.files.append(path.name)
            report.pages += len(pages)

        if all_chunks:
            if self.retrieval == "dense":
                self.store.add(all_chunks, self._embedder.embed([c.text for c in all_chunks]))
            else:
                self.store.add(all_chunks)
        report.chunks = len(all_chunks)
        report.seconds = time.perf_counter() - start
        return report

    # -- querying ------------------------------------------------------------
    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        k = k or self.settings.top_k
        if self.retrieval == "bm25":
            return self.store.search(question, k)
        return self.store.search(self._embedder.embed([question])[0], k)

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        question = question.strip()
        if not question:
            return Answer("Please type a question.", [], "no_match")
        if len(self.store) == 0:
            return Answer("Upload and index at least one PDF first.", [], "no_match")

        results = self.retrieve(question)
        relevant = [r for r in results if r.score >= self.min_score]
        if not relevant:
            return Answer(
                "I could not find anything relevant to that question in the uploaded documents.",
                [],
                "no_match",
            )

        if self.llm is not None:
            try:
                text = self.llm.generate(build_messages(question, relevant, history))
                return Answer(text, relevant, "llm")
            except Exception as exc:  # noqa: BLE001 - any API failure falls back
                log.warning("LLM call failed, using extractive fallback: %s", exc)
                note = "The language model was unavailable, so this answer quotes the documents directly."
                return Answer(self.extractive.answer(question, relevant), relevant, "extractive", note)

        note = "No LLM API key is configured, so this answer quotes the documents directly."
        return Answer(self.extractive.answer(question, relevant), relevant, "extractive", note)
