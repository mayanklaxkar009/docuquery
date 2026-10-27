import io
import json
import urllib.error

import pytest

from rag.llm import OpenAICompatibleLLM


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener(body=None, error=None, seen=None):
    def open_(request, timeout):
        if seen is not None:
            seen.append(request)
        if error:
            raise error
        return FakeResponse(json.dumps(body).encode())

    return open_


def test_sends_openai_style_request_and_returns_content():
    seen = []
    body = {"choices": [{"message": {"content": "  RAG grounds answers [1]. "}}]}
    llm = OpenAICompatibleLLM("openai/gpt-oss-120b", "key", "https://example.com/v1/", opener=_opener(body, seen=seen))
    assert llm.generate([{"role": "user", "content": "hi"}]) == "RAG grounds answers [1]."
    request = seen[0]
    assert request.full_url == "https://example.com/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer key"
    payload = json.loads(request.data)
    assert payload["model"] == "openai/gpt-oss-120b" and payload["reasoning_effort"] == "low"


def test_non_reasoning_model_has_no_reasoning_effort():
    llm = OpenAICompatibleLLM("llama-3.1-8b-instant", "key")
    assert "reasoning_effort" not in llm._payload([])


def test_empty_answer_and_http_errors_raise():
    empty = OpenAICompatibleLLM("m", "k", opener=_opener({"choices": [{"message": {"content": ""}}]}))
    with pytest.raises(RuntimeError):
        empty.generate([])
    err = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, io.BytesIO(b"rate limited"))
    with pytest.raises(RuntimeError, match="429"):
        OpenAICompatibleLLM("m", "k", opener=_opener(error=err)).generate([])
