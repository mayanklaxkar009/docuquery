import pytest

from rag.chunker import chunk_pages, chunk_text, split_sentences
from rag.loader import Page, clean_text, load_pdf


def test_clean_text_joins_wrapped_lines_and_hyphenation():
    raw = "Retrieval-aug-\nmented generation\nworks   well.\n\n\n\nNew paragraph."
    assert clean_text(raw) == "Retrieval-augmented generation works well.\n\nNew paragraph."


def test_load_pdf_returns_numbered_pages(sample_pdf):
    pages = load_pdf(sample_pdf)
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert all(p.source == "rag_primer.pdf" for p in pages)
    assert "Retrieval-augmented generation" in pages[0].text


def test_load_pdf_respects_max_pages(sample_pdf):
    assert len(load_pdf(sample_pdf, max_pages=1)) == 1


def test_split_sentences():
    assert split_sentences("One. Two? Three!\n\nFour") == ["One.", "Two?", "Three!", "Four"]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # The last sentence of each chunk is repeated at the start of the next.
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = a.split(". ")[-1]
        assert b.startswith(last_sentence.rstrip("."))


def test_long_unpunctuated_text_is_hard_split():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 500


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_settings(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("Some text.", size, overlap)


def test_chunk_pages_keeps_page_metadata_and_unique_ids():
    pages = [Page("a.pdf", 1, "Alpha. " * 80), Page("a.pdf", 2, "Beta. " * 80)]
    chunks = chunk_pages(pages, chunk_size=150, overlap=20, start_id=10)
    assert chunks[0].chunk_id == 10
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.page_number for c in chunks} == {1, 2}
    assert all("Beta" not in c.text for c in chunks if c.page_number == 1)
