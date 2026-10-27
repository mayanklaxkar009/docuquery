"""Generate samples/rag_primer.pdf, a short original document used by the
"Try the sample" button. Run: python scripts/make_sample_pdf.py"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

SECTIONS = [
    (
        "1. What retrieval-augmented generation is",
        [
            "Retrieval-augmented generation, usually shortened to RAG, is a way of answering questions "
            "with a language model while grounding the answer in a specific collection of documents. "
            "Instead of relying only on what the model memorised during training, the system first "
            "searches the documents for passages related to the question and then gives those passages "
            "to the model as context.",
            "RAG solves two common problems with language models. The first is hallucination, where a "
            "model produces fluent but unsupported statements. The second is stale or missing knowledge: "
            "a model cannot know the contents of a private contract, a new policy or yesterday's report. "
            "Because the answer is built from retrieved passages, RAG can cite its sources, and the "
            "document collection can be updated without retraining the model.",
        ],
    ),
    (
        "2. The indexing stage",
        [
            "Before any question is asked, the documents are prepared. Text is extracted from each file "
            "and split into chunks of a few hundred words. Chunks overlap slightly so that a sentence "
            "near a boundary is not lost. Each chunk keeps a note of the file and page it came from, "
            "which later allows the answer to cite an exact page.",
            "Every chunk is then converted into an embedding, a vector of numbers that captures its "
            "meaning. Texts with similar meanings produce vectors that point in similar directions. "
            "The embeddings are stored in a vector index such as FAISS, which can find the nearest "
            "vectors to a query among millions of entries in milliseconds.",
        ],
    ),
    (
        "3. The question-answering stage",
        [
            "When a user asks a question, the question is embedded with the same model used for the "
            "chunks. The vector index returns the chunks whose embeddings are most similar, typically "
            "measured with cosine similarity. A threshold filters out weak matches so the system can "
            "admit that the documents do not contain an answer.",
            "The selected chunks are numbered and placed into a prompt together with the question. "
            "The language model is instructed to answer only from those passages and to cite them by "
            "number. The application then shows the answer alongside the source passages and their page "
            "numbers, so the reader can check every claim.",
        ],
    ),
    (
        "4. Choosing chunk size and top-k",
        [
            "Chunk size is a trade-off. Small chunks give precise matches but may cut an explanation in "
            "half, while large chunks carry more context but dilute the embedding and waste prompt space. "
            "A size of roughly 500 to 1,000 characters with an overlap of 100 to 200 characters is a "
            "reasonable starting point for prose documents.",
            "The number of chunks retrieved, called top-k, is a similar trade-off. Retrieving three to "
            "five chunks usually gives the model enough evidence without burying the relevant passage "
            "among irrelevant ones. Both settings should be tuned by testing real questions against the "
            "documents and checking whether the correct passage is retrieved.",
        ],
    ),
    (
        "5. Evaluating a RAG system",
        [
            "A RAG system has two parts that can fail independently, so each should be measured. "
            "Retrieval quality is measured with metrics such as hit rate, the share of test questions "
            "for which the correct passage appears in the top-k results, and mean reciprocal rank, which "
            "rewards placing the correct passage near the top.",
            "Generation quality is measured by faithfulness, whether every statement in the answer is "
            "supported by the retrieved passages, and by answer relevance, whether the answer actually "
            "addresses the question. A common practical approach is to write a small set of questions "
            "with known answers and re-run it after every change to the system.",
        ],
    ),
    (
        "6. Limitations",
        [
            "RAG is only as good as its retrieval. If the relevant passage is not retrieved, the model "
            "cannot use it. Scanned PDFs without a text layer need optical character recognition before "
            "they can be indexed. Tables and charts often lose their structure during text extraction, "
            "and questions that require combining facts spread across many pages remain difficult.",
        ],
    ),
]


def build(path: Path) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, title="A Short Primer on Retrieval-Augmented Generation")
    story = [Paragraph("A Short Primer on Retrieval-Augmented Generation", styles["Title"]), Spacer(1, 12)]
    for i, (heading, paragraphs) in enumerate(SECTIONS):
        if i in (2, 4):
            story.append(PageBreak())
        story.append(Paragraph(heading, styles["Heading2"]))
        for p in paragraphs:
            story.append(Paragraph(p, styles["BodyText"]))
            story.append(Spacer(1, 8))
    doc.build(story)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "samples" / "rag_primer.pdf"
    out.parent.mkdir(exist_ok=True)
    build(out)
    print(f"wrote {out}")
