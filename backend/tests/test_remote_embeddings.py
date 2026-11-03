"""远程嵌入：API 进程不加载 BGE，dense 走 HTTP，sparse 可由服务返回或本地哈希补齐。"""
import io
import json
import threading
import urllib.request

import pytest

from backend.infrastructure.config import Settings, settings
from backend.infrastructure.encoding import (
    EncodingCancelled,
    _hash_sparse,
    encode_documents,
    encode_query,
    uses_remote_embeddings,
)


@pytest.fixture
def remote_settings(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "remote")
    monkeypatch.setattr(settings, "EMBEDDING_API_BASE_URL", "http://embed.example/v1")
    monkeypatch.setattr(settings, "EMBEDDING_API_KEY", "secret")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setattr(settings, "EMBEDDING_API_TIMEOUT", 12)


def _urlopen(response, captured):
    def fake(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["auth"] = req.get_header("Authorization")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return io.BytesIO(json.dumps(response).encode("utf-8"))

    return fake


def test_hash_sparse_is_stable_and_nonempty():
    first = _hash_sparse("混合检索 hybrid search")
    second = _hash_sparse("混合检索 hybrid search")
    assert first == second
    assert first
    assert all(isinstance(k, int) and isinstance(v, float) for k, v in first.items())


def test_remote_dense_only_response_hashes_sparse_and_skips_local_model(remote_settings, monkeypatch):
    captured = {}
    response = {
        "data": [
            {"index": 1, "embedding": [0.2, 0.3]},
            {"index": 0, "embedding": [0.1, 0.4]},
        ]
    }

    def boom():
        raise AssertionError("local BGE should not load")

    monkeypatch.setattr("backend.infrastructure.encoding.get_bge_m3", boom)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen(response, captured))

    dense, sparse = encode_documents(["甲", "乙"])
    assert uses_remote_embeddings() is True
    assert dense == [[0.1, 0.4], [0.2, 0.3]]
    assert sparse[0] == _hash_sparse("甲")
    assert sparse[1] == _hash_sparse("乙")
    assert captured["url"] == "http://embed.example/v1/embeddings"
    assert captured["auth"] == "Bearer secret"
    assert captured["timeout"] == 12
    assert captured["body"]["model"] == "BAAI/bge-m3"
    assert captured["body"]["input"] == ["甲", "乙"]


def test_remote_sparse_from_service_is_kept(remote_settings, monkeypatch):
    captured = {}
    response = {
        "data": [
            {"index": 0, "embedding": [0.5], "sparse": {"7": 0.25, "8": 0.5}},
        ]
    }
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen(response, captured))
    dense, sparse = encode_query("问题")
    assert dense == [0.5]
    assert sparse == {7: 0.25, 8: 0.5}


def test_remote_cancel_does_not_call_the_service(remote_settings, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("cancelled encode must not call HTTP")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    cancel_event = threading.Event()
    cancel_event.set()
    with pytest.raises(EncodingCancelled):
        encode_documents(["hello"], cancel_event)


def test_remote_provider_requires_base_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/ragmate")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "remote")
    monkeypatch.setenv("EMBEDDING_API_BASE_URL", "")
    with pytest.raises(ValueError, match="EMBEDDING_API_BASE_URL"):
        Settings(_env_file=None)
