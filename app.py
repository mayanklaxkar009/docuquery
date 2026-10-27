"""Gradio front end for DocuQuery. Runs locally (`python app.py`) and on
Hugging Face Spaces, which starts this file automatically."""

from __future__ import annotations

import html
import logging
from pathlib import Path

import gradio as gr

try:  # Hugging Face ZeroGPU hardware expects at least one @spaces.GPU function.
    import spaces

    @spaces.GPU
    def _zerogpu_probe():  # never called; the app runs on CPU
        return None
except ImportError:
    pass

from rag.config import Settings
from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SETTINGS = Settings()
SAMPLE_PDF = Path(__file__).parent / "samples" / "rag_primer.pdf"
if not SAMPLE_PDF.exists():  # generated, not committed: Spaces rejects plain-git binaries
    from scripts.make_sample_pdf import build

    SAMPLE_PDF.parent.mkdir(exist_ok=True)
    build(SAMPLE_PDF)

# The embedding model and LLM client are loaded once and shared by every
# visitor; each browser session gets its own private document index.
_shared = RAGPipeline(SETTINGS)
_ = _shared.embedder  # warm the model at start-up, not on the first click


def _new_session_pipeline() -> RAGPipeline:
    return RAGPipeline(SETTINGS, embedder=_shared.embedder, llm=_shared.llm)


def _format_sources(answer) -> str:
    if not answer.sources:
        return ""
    lines = ["### Sources"]
    for n, r in enumerate(answer.sources, start=1):
        snippet = r.chunk.text if len(r.chunk.text) <= 400 else r.chunk.text[:400].rsplit(" ", 1)[0] + "…"
        lines.append(
            f"**[{n}] {html.escape(r.chunk.source)}, page {r.chunk.page_number}** "
            f"(similarity {r.score:.2f})\n\n> {html.escape(snippet)}"
        )
    return "\n\n".join(lines)


def index_documents(files, pipeline):
    paths = [f if isinstance(f, str) else f.name for f in (files or [])]
    if not paths:
        return pipeline, "Upload one or more PDFs first.", [], ""
    pipeline = pipeline or _new_session_pipeline()
    report = pipeline.index_files(paths)
    if not report.files:
        status = "No documents could be indexed.\n\n" + "\n".join(f"- {w}" for w in report.warnings)
        return pipeline, status, [], ""
    status = (
        f"Indexed **{len(report.files)}** file(s), **{report.pages}** pages, "
        f"**{report.chunks}** chunks in {report.seconds:.1f}s: {', '.join(report.files)}"
    )
    if report.warnings:
        status += "\n\n" + "\n".join(f"- {w}" for w in report.warnings)
    return pipeline, status, [], ""


def load_sample(pipeline):
    return index_documents([str(SAMPLE_PDF)], pipeline)


def respond(message, chat_history, pipeline):
    chat_history = chat_history or []
    if not message or not message.strip():
        return "", chat_history, gr.update()
    if pipeline is None:
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Upload and index a PDF first, or click **Try the sample**."},
        ]
        return "", chat_history, ""

    answer = pipeline.ask(message, history=chat_history)
    reply = answer.text + (f"\n\n*{answer.note}*" if answer.note else "")
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return "", chat_history, _format_sources(answer)


llm_label = SETTINGS.llm_model if _shared.llm else "extractive mode (set HF_TOKEN to enable the LLM)"

with gr.Blocks(title="DocuQuery: chat with your PDFs") as demo:
    gr.Markdown(
        "# DocuQuery\n"
        "Ask questions about your PDFs. Answers come only from your documents and cite the page they used.\n\n"
        f"<small>Embeddings: `{SETTINGS.embedding_model}` · Vector search: "
        f"`{_shared.store.backend}` · LLM: `{llm_label}`</small>"
    )
    session = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            files = gr.File(label="PDFs", file_count="multiple", file_types=[".pdf"], type="filepath")
            with gr.Row():
                index_btn = gr.Button("Index documents", variant="primary")
                sample_btn = gr.Button("Try the sample")
            status = gr.Markdown("No documents indexed yet.")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=460)
            with gr.Row():
                question = gr.Textbox(
                    placeholder="e.g. What problem does retrieval-augmented generation solve?",
                    show_label=False,
                    scale=5,
                )
                send = gr.Button("Ask", variant="primary", scale=1)
            sources = gr.Markdown()

    index_btn.click(index_documents, [files, session], [session, status, chatbot, sources])
    sample_btn.click(load_sample, [session], [session, status, chatbot, sources])
    question.submit(respond, [question, chatbot, session], [question, chatbot, sources])
    send.click(respond, [question, chatbot, session], [question, chatbot, sources])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
