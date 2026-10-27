from pathlib import Path

from rag.llm import ExtractiveAnswerer, build_messages
from rag.pipeline import RAGPipeline
from tests.conftest import FakeLLM


def _pipeline(settings, embedder, llm=None):
    return RAGPipeline(settings, embedder=embedder, llm=llm)


def test_index_sample_pdf(settings, embedder, sample_pdf):
    p = _pipeline(settings, embedder)
    report = p.index_files([sample_pdf])
    assert report.files == ["rag_primer.pdf"]
    assert report.pages == 3
    assert report.chunks == len(p.store) > 3
    assert embedder.calls == 1  # all chunks embedded in one batch


def test_retrieval_finds_the_right_page(settings, embedder, sample_pdf):
    p = _pipeline(settings, embedder)
    p.index_files([sample_pdf])
    top = p.retrieve("What is hit rate and mean reciprocal rank used for in evaluation?")[0]
    assert top.chunk.page_number == 3
    assert "hit rate" in top.chunk.text


def test_llm_receives_numbered_context_and_answer_has_sources(settings, embedder, sample_pdf):
    llm = FakeLLM("RAG grounds answers in documents [1].")
    p = _pipeline(settings, embedder, llm)
    p.index_files([sample_pdf])
    answer = p.ask("What problems does retrieval-augmented generation solve?")
    assert answer.mode == "llm"
    assert answer.text == "RAG grounds answers in documents [1]."
    assert answer.sources
    prompt = llm.last_messages[-1]["content"]
    assert "[1] (source: rag_primer.pdf, page" in prompt
    assert llm.last_messages[0]["role"] == "system"


def test_falls_back_to_extractive_when_llm_fails(settings, embedder, sample_pdf):
    p = _pipeline(settings, embedder, FakeLLM(fail=True))
    p.index_files([sample_pdf])
    answer = p.ask("What does FAISS do with embeddings?")
    assert answer.mode == "extractive"
    assert "unavailable" in answer.note
    assert "[" in answer.text  # keeps citation markers


def test_extractive_mode_without_token(settings, embedder, sample_pdf):
    p = _pipeline(settings, embedder)
    assert p.llm is None
    p.index_files([sample_pdf])
    answer = p.ask("What is chunk overlap for?")
    assert answer.mode == "extractive" and "HF_TOKEN" in answer.note


def test_unrelated_question_is_refused(settings, embedder, sample_pdf):
    p = _pipeline(settings, embedder, FakeLLM())
    p.index_files([sample_pdf])
    answer = p.ask("Zebra giraffe saxophone quantum")
    assert answer.mode == "no_match"


def test_ask_before_indexing_and_empty_question(settings, embedder):
    p = _pipeline(settings, embedder)
    assert "Upload" in p.ask("anything").text
    p.store.add([], embedder.embed([]))
    assert p.ask("   ").mode == "no_match"


def test_bad_files_are_reported_not_fatal(settings, embedder, sample_pdf, tmp_path: Path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not really a pdf")
    notes = tmp_path / "notes.txt"
    notes.write_text("hello")
    report = _pipeline(settings, embedder).index_files([broken, notes, sample_pdf])
    assert report.files == ["rag_primer.pdf"]
    assert len(report.warnings) == 2


def test_reindexing_replaces_previous_documents(settings, embedder, sample_pdf):
    p = _pipeline(settings, embedder)
    p.index_files([sample_pdf])
    first = len(p.store)
    p.index_files([sample_pdf])
    assert len(p.store) == first


def test_history_is_passed_to_llm(settings, embedder, sample_pdf):
    llm = FakeLLM()
    p = _pipeline(settings, embedder, llm)
    p.index_files([sample_pdf])
    history = [{"role": "user", "content": "earlier q"}, {"role": "assistant", "content": "earlier a"}]
    p.ask("What is top-k?", history=history)
    assert [m["role"] for m in llm.last_messages] == ["system", "user", "assistant", "user"]


def test_build_messages_ignores_malformed_history():
    msgs = build_messages("q", [], history=[{"role": "tool", "content": "x"}, {"role": "user"}])
    assert len(msgs) == 2


def test_extractive_answerer_without_overlap():
    assert "could not find" in ExtractiveAnswerer().answer("anything", [])
