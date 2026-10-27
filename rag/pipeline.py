"""Ties the pieces together: PDFs -> chunks -> vectors -> retrieval -> answer."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunker import chunk_pages
from rag.config import Settings
from rag.embeddings import Embedder
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
        store: VectorStore | None = None,
    ):
        self.settings = settings or Settings()
        self._embedder = embedder
        self.store = store or VectorStore()
        self.extractive = ExtractiveAnswerer()
        self.llm = llm if llm is not None else self._default_llm()

    # -- lazy construction so importing the module is cheap ---------------
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            from rag.embeddings import make_embedder

            self._embedder = make_embedder(
                self.settings.embedding_model, self.settings.hf_token, self.settings.embedding_backend
            )
        return self._embedder

    def _default_llm(self) -> LLM | None:
        if not self.settings.hf_token:
            log.warning("HF_TOKEN not set: answers will be extractive only")
            return None
        from rag.llm import HFInferenceLLM

        return HFInferenceLLM(self.settings.llm_model, self.settings.hf_token, self.settings.max_new_tokens)

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
            vectors = self.embedder.embed([c.text for c in all_chunks])
            self.store.add(all_chunks, vectors)
        report.chunks = len(all_chunks)
        report.seconds = time.perf_counter() - start
        return report

    # -- querying ------------------------------------------------------------
    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, k or self.settings.top_k)

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        question = question.strip()
        if not question:
            return Answer("Please type a question.", [], "no_match")
        if len(self.store) == 0:
            return Answer("Upload and index at least one PDF first.", [], "no_match")

        results = self.retrieve(question)
        min_score = getattr(self.embedder, "min_score", self.settings.min_score)
        relevant = [r for r in results if r.score >= min_score]
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

        note = "No HF_TOKEN is configured, so this answer quotes the documents directly."
        return Answer(self.extractive.answer(question, relevant), relevant, "extractive", note)
