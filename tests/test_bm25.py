from rag.bm25 import BM25Index, stem, tokenize
from rag.chunker import Chunk
from rag.config import Settings
from rag.pipeline import RAGPipeline


def _chunks():
    texts = [
        "FAISS searches millions of vectors quickly using inner product.",
        "Chunks overlap so that a sentence split across chunks is not lost.",
        "Hit rate measures how often the answer page is retrieved.",
    ]
    return [Chunk(i, t, "doc.pdf", i + 1) for i, t in enumerate(texts)]


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("What are the chunks overlapping?") == ["chunk", "overlapp"]
    assert stem("chunking") == stem("chunked") == stem("chunks") == "chunk"


def test_search_ranks_matching_chunk_first_with_normalised_scores():
    index = BM25Index()
    index.add(_chunks())
    results = index.search("Why do chunks overlap?", k=3)
    assert results[0].chunk.chunk_id == 1
    assert all(0 < r.score <= 1 for r in results)
    assert index.search("capital of France", k=3) == []
    assert index.search("the and of", k=3) == []  # only stopwords


def test_clear_and_empty_index():
    index = BM25Index()
    assert index.search("faiss", k=2) == []
    index.add(_chunks())
    assert len(index) == 3
    index.clear()
    assert len(index) == 0


def test_pipeline_uses_bm25_without_embedder(sample_pdf):
    p = RAGPipeline(Settings(hf_token=None, llm_api_key=None, embedding_backend="bm25"))
    assert p.retrieval == "bm25" and p.embedder is None
    p.index_files([sample_pdf])
    answer = p.ask("What is hit rate?")
    assert answer.mode == "extractive" and answer.sources
    assert any(r.chunk.page_number == 3 for r in answer.sources)
    assert p.ask("Who won the 1998 world cup?").mode == "no_match"
