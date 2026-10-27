# Code walkthrough and interview prep

Read this alongside the code. You should be able to explain every file without notes.

## Request flow in one paragraph

A user uploads PDFs; `app.index_documents` creates a per-session `RAGPipeline` that shares the
LLM client (and embedder, if dense mode is on). `pipeline.index_files` calls `loader.load_pdf` (one `Page` per page),
`chunker.chunk_pages` (overlapping, sentence-aligned `Chunk`s tagged with file and page), and adds
the chunks to a `BM25Index` (default) or embeds them in one batch into a FAISS `VectorStore`. On a
question, `pipeline.ask` searches the index, drops results below the threshold (`BM25_MIN_SCORE` or
`MIN_SCORE`), builds a prompt with numbered passages (`llm.build_messages`) and calls the LLM
(Groq's free OpenAI-compatible API by default). Any LLM failure falls back to
`ExtractiveAnswerer`. The UI shows the answer and a sources panel.

## File by file

- **config.py**: frozen dataclass; every field reads an env var so the same code runs locally and on Render.
- **loader.py**: pypdf text extraction; regexes repair hyphenated line breaks and wrapped lines. Image-only pages return no text and are skipped.
- **chunker.py**: splits into sentences, packs them up to `chunk_size`, carries trailing sentences forward as overlap, hard-splits very long "sentences" (tables) on spaces. Chunks never cross pages.
- **bm25.py**: the default retriever. Tokenises with stopword removal and light stemming, scores with BM25 (k1=1.2, b=0.75) and normalises scores to 0..1 against the best achievable score, so a fixed threshold rejects off-topic questions.
- **embeddings.py**: optional dense retrieval. `HFInferenceEmbedder` calls the Hugging Face Inference API in batches with retries; `SentenceTransformerEmbedder` runs locally; `make_embedder` returns None for BM25. The `Embedder` Protocol is what lets tests inject a fake.
- **vector_store.py**: (dense mode) FAISS `IndexFlatIP` (exact search). With normalised vectors, inner product = cosine similarity. NumPy fallback gives identical rankings; a test proves it.
- **llm.py**: the system prompt (answer only from passages, cite them, admit when not found), message building with recent history, HF `InferenceClient.chat_completion`, and the extractive fallback.
- **pipeline.py**: orchestration, error handling per file, relevance threshold, fallback logic.
- **app.py**: Gradio Blocks. Models load once at start-up; `gr.State` keeps each visitor's index separate.

## Questions you will likely be asked

**Why chunk at all? Why 800 characters with 150 overlap?**
Embedding models have input limits and one vector can't represent a whole document well. Small
chunks match precisely but lose context; large ones dilute the vector and waste prompt space. 800/150
is a common starting point for prose; I tuned it with the evaluation script (quote your numbers).

**Why all-MiniLM-L6-v2?**
Small (about 80 MB), fast on a free CPU, 384 dimensions, good quality for semantic search in English.
A trade-off against larger models such as bge or e5 that are more accurate but slower.

**Why cosine similarity and why normalise?**
Cosine compares direction, not length, so long and short texts are comparable. After L2
normalisation, the inner product equals cosine, which is what `IndexFlatIP` computes.

**Flat index? Won't that be slow?**
Flat is exact and fine up to hundreds of thousands of vectors. For millions I'd switch to an
approximate index such as IVF or HNSW, trading a little recall for much faster search.

**How do you reduce hallucination?**
The prompt restricts the model to numbered passages and asks for citations; low-similarity results
are filtered so it can say "not found"; temperature is low (0.2); sources are shown so users can verify.

**How do you know retrieval works?**
Hit rate @k and MRR on questions with known answer pages (`scripts/evaluate_retrieval.py`).
For generation I'd add faithfulness checks, for example an LLM judge or RAGAS.

**How did you test it without calling models?**
Dependency injection: the pipeline accepts any embedder and LLM. Tests use a hashing bag-of-words
embedder and a fake LLM, so the suite runs in seconds with no network, including the failure path.

**What would you improve?**
Hybrid search (BM25 + vectors) for exact terms like codes and names, a cross-encoder re-ranker,
OCR for scanned PDFs, streaming responses, and a persistent vector database.

**How is it deployed?**
GitHub Actions runs pytest on every push; Render redeploys the free web service from `render.yaml`.
The inference token is a Render environment secret, never in the code.

## Things to try yourself (so it's genuinely your work)

1. Run the evaluation with different `CHUNK_SIZE` and `TOP_K` values and record the results in the README.
2. Add BM25 hybrid search (the `rank_bm25` package) and see whether hit rate improves.
3. Add a "Clear documents" button to the UI.

## Why BM25 by default?

The demo has to run on free hosting: 512 MB RAM and a tenth of a CPU. A local embedding model
needs PyTorch (too big) and a hosted embedding API needs paid credits. BM25 needs neither, and
for questions that reuse the document's wording it is a strong baseline: 100% hit rate @4 on the
sample set. Its weakness is synonyms ("car" vs "vehicle"), which is exactly what dense embeddings
fix, so the dense path stays one environment variable away. In production you would combine both
(hybrid search with reciprocal rank fusion).
