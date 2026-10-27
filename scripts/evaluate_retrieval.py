"""Measure retrieval quality on the sample document with the real embedding model.

For each test question we know which page holds the answer. We report:
  hit rate @k  - share of questions whose answer page appears in the top-k chunks
  MRR          - mean reciprocal rank of the first chunk from the answer page

Run: python scripts/evaluate_retrieval.py   (downloads the embedding model once)
Try other settings: CHUNK_SIZE=500 TOP_K=3 python scripts/evaluate_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import Settings  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from scripts.make_sample_pdf import build  # noqa: E402

# (question, page that contains the answer)
QUESTIONS = [
    ("What does RAG stand for?", 1),
    ("Why do language models hallucinate less with retrieval?", 1),
    ("How can a model answer questions about a private contract?", 1),
    ("Why do chunks overlap?", 1),
    ("What is an embedding?", 1),
    ("Which library can search millions of vectors quickly?", 1),
    ("How is the user's question compared with the stored chunks?", 2),
    ("How does the system admit it does not know the answer?", 2),
    ("How are sources shown to the reader?", 2),
    ("What chunk size is a good starting point?", 2),
    ("How many chunks should be retrieved?", 2),
    ("What is hit rate?", 3),
    ("What does mean reciprocal rank reward?", 3),
    ("How is faithfulness defined?", 3),
    ("Why can't scanned PDFs be indexed directly?", 3),
    ("What happens to tables during text extraction?", 3),
]


def main() -> None:
    pdf = ROOT / "samples" / "rag_primer.pdf"
    if not pdf.exists():
        pdf.parent.mkdir(exist_ok=True)
        build(pdf)

    settings = Settings(hf_token=None)
    pipeline = RAGPipeline(settings)
    report = pipeline.index_files([pdf])
    print(f"Indexed {report.chunks} chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap})\n")

    k = settings.top_k
    hits, reciprocal_ranks = 0, []
    for question, page in QUESTIONS:
        results = pipeline.retrieve(question, k=k)
        pages = [r.chunk.page_number for r in results]
        rank = pages.index(page) + 1 if page in pages else None
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        mark = f"rank {rank}" if rank else "MISS"
        print(f"{mark:>7}  top score {results[0].score:.2f}  {question}")

    print(f"\nHit rate @{k}: {hits / len(QUESTIONS):.0%}")
    print(f"MRR:         {sum(reciprocal_ranks) / len(QUESTIONS):.2f}")


if __name__ == "__main__":
    main()
