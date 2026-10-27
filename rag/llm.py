"""Answer generation.

`OpenAICompatibleLLM` calls any OpenAI-compatible chat API; the default is
Groq's free tier (no card needed). `HFInferenceLLM` uses Hugging Face
Inference Providers (needs HF credits).
`ExtractiveAnswerer` needs no model at all: it returns the most relevant
sentences from the retrieved chunks. The pipeline falls back to it when no
token is configured or the API call fails, so the app always answers.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Protocol

from rag.chunker import split_sentences
from rag.vector_store import SearchResult

SYSTEM_PROMPT = (
    "You answer questions about the user's documents using ONLY the numbered "
    "context passages provided. Cite the passages you use inline as [1], [2], etc. "
    "If the passages do not contain the answer, say you could not find it in the "
    "documents. Do not use outside knowledge and do not invent citations. "
    "Be concise and clear."
)


def format_context(results: list[SearchResult]) -> str:
    blocks = []
    for n, r in enumerate(results, start=1):
        blocks.append(f"[{n}] (source: {r.chunk.source}, page {r.chunk.page_number})\n{r.chunk.text}")
    return "\n\n".join(blocks)


def build_messages(
    question: str, results: list[SearchResult], history: list[dict] | None = None, max_history: int = 4
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Recent turns let the model resolve follow-ups like "and the second one?".
    for turn in (history or [])[-max_history:]:
        if turn.get("role") in {"user", "assistant"} and isinstance(turn.get("content"), str):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append(
        {"role": "user", "content": f"Context passages:\n\n{format_context(results)}\n\nQuestion: {question}"}
    )
    return messages


class LLM(Protocol):
    name: str

    def generate(self, messages: list[dict]) -> str: ...


class OpenAICompatibleLLM:
    """Minimal client for `POST {base_url}/chat/completions` using only the
    standard library, so it adds no dependencies."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        timeout: float = 60,
        opener=None,
    ):
        self.name = model
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._open = opener or urllib.request.urlopen

    def _payload(self, messages: list[dict]) -> dict:
        payload = {
            "model": self.name,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if "gpt-oss" in self.name:
            # Reasoning models spend tokens thinking first; keep that short.
            payload["reasoning_effort"] = "low"
        return payload

    def generate(self, messages: list[dict]) -> str:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(self._payload(messages)).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "DocuQuery/1.0",
            },
            method="POST",
        )
        try:
            with self._open(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"LLM API error {exc.code}: {detail}") from exc
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        if not content or not content.strip():
            raise RuntimeError("model returned an empty answer")
        return content.strip()


class HFInferenceLLM:
    def __init__(self, model: str, token: str, max_new_tokens: int = 512, temperature: float = 0.2):
        from huggingface_hub import InferenceClient

        self.name = model
        self.client = InferenceClient(model=model, token=token, timeout=60)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, messages: list[dict]) -> str:
        response = self.client.chat_completion(
            messages=messages, max_tokens=self.max_new_tokens, temperature=self.temperature
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("model returned an empty answer")
        return content.strip()


_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its of on or "
    "that the their this to was were what when where which who why will with you your".split()
)


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}


class ExtractiveAnswerer:
    """Picks the sentences from the retrieved chunks that share the most
    terms with the question, keeping their citation numbers."""

    name = "extractive (no LLM)"

    def answer(self, question: str, results: list[SearchResult], max_sentences: int = 3) -> str:
        q_terms = _terms(question)
        scored = []
        for n, r in enumerate(results, start=1):
            for position, sentence in enumerate(split_sentences(r.chunk.text)):
                overlap = len(q_terms & _terms(sentence))
                # Ties go to higher-ranked chunks, then earlier sentences.
                scored.append((overlap + r.score, -n, -position, sentence, n, overlap))
        scored.sort(reverse=True)
        picked = [(s, n) for _, _, _, s, n, overlap in scored if overlap > 0][:max_sentences]
        if not picked:
            return "I could not find an answer to that in the uploaded documents."
        return " ".join(f"{s} [{n}]" for s, n in picked)
