import numpy as np
import pytest

from rag.embeddings import HFInferenceEmbedder, KeywordEmbedder, make_embedder


class FakeClient:
    """Mimics InferenceClient.feature_extraction."""

    def __init__(self, dim=8, fail_times=0, per_token=False, single_only=False):
        self.dim, self.fail_times, self.per_token, self.single_only = dim, fail_times, per_token, single_only
        self.calls = []

    def feature_extraction(self, text):
        self.calls.append(text)
        if self.fail_times:
            self.fail_times -= 1
            raise ConnectionError("model loading")
        texts = [text] if isinstance(text, str) else text
        if self.single_only and not isinstance(text, str):
            raise ValueError("list input not supported")
        rows = [np.full(self.dim, len(t), dtype=np.float32) + np.arange(self.dim) for t in texts]
        out = np.stack(rows)
        if self.per_token:
            out = np.stack([out, out], axis=1)  # (n, tokens, dim)
        return out[0] if isinstance(text, str) else out


def test_api_embedder_batches_and_normalises():
    client = FakeClient()
    e = HFInferenceEmbedder("m", "t", batch_size=2, client=client)
    v = e.embed(["a", "bb", "ccc"])
    assert v.shape == (3, 8)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert len(client.calls) == 2  # 3 texts in batches of 2


def test_api_embedder_mean_pools_token_vectors():
    v = HFInferenceEmbedder("m", "t", client=FakeClient(per_token=True)).embed(["x", "yy"])
    assert v.shape == (2, 8)


def test_api_embedder_falls_back_to_one_text_per_call():
    client = FakeClient(single_only=True)
    v = HFInferenceEmbedder("m", "t", client=client).embed(["x", "yy"])
    assert v.shape == (2, 8)


def test_api_embedder_retries(monkeypatch):
    monkeypatch.setattr("rag.embeddings.time.sleep", lambda s: None)
    client = FakeClient(fail_times=2)
    assert HFInferenceEmbedder("m", "t", retries=3, client=client).embed(["x"]).shape == (1, 8)
    with pytest.raises(RuntimeError):
        HFInferenceEmbedder("m", "t", retries=2, client=FakeClient(fail_times=5)).embed(["x"])


def test_keyword_embedder_and_no_token_fallback():
    e = make_embedder("m", token=None)
    assert isinstance(e, KeywordEmbedder)
    v = e.embed(["vector search with faiss", "faiss vector index", "banana bread"])
    assert v[0] @ v[1] > v[0] @ v[2]
    assert e.embed([]).shape == (0, e.dim)
