"""Reranker provider 单元测试：本地 CrossEncoder 与远程 /rerank API 两种打分路径。"""
import io
import json
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from backend.core import retriever
from backend.infrastructure.config import Settings, settings


class TestLocalProvider:
    def test_scores_are_sigmoid_of_logits(self, monkeypatch):
        monkeypatch.setattr(settings, "RERANKER_PROVIDER", "local")
        fake = MagicMock()
        fake.predict.return_value = [0.0, 2.0, -2.0]
        with patch.object(retriever, "get_reranker", return_value=fake):
            scores = retriever.score_pairs("q", ["a", "b", "c"])
        assert scores[0] == pytest.approx(0.5)
        assert scores[1] == pytest.approx(0.881, abs=0.001)
        assert scores[2] == pytest.approx(0.119, abs=0.001)

    def test_empty_texts_short_circuit(self):
        assert retriever.score_pairs("q", []) == []


def _fake_urlopen_factory(response_obj, captured):
    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["auth"] = req.get_header("Authorization")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        raw = json.dumps(response_obj).encode("utf-8")
        return io.BytesIO(raw)

    return fake_urlopen


@pytest.fixture
def api_settings(monkeypatch):
    monkeypatch.setattr(settings, "RERANKER_PROVIDER", "api")
    monkeypatch.setattr(settings, "RERANKER_API_BASE_URL", "https://rerank.example.com/v1")
    monkeypatch.setattr(settings, "RERANKER_API_KEY", "secret-key")
    monkeypatch.setattr(settings, "RERANKER_MODEL", "bge-reranker-v2-m3")


class TestApiProvider:
    def test_scores_scattered_by_index_and_request_shape(self, api_settings):
        captured = {}
        response = {
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.3},
                {"index": 1, "relevance_score": 0.6},
            ]
        }
        with patch.object(
            urllib.request, "urlopen", side_effect=_fake_urlopen_factory(response, captured)
        ):
            scores = retriever.score_pairs("问题", ["甲", "乙", "丙"])
        assert scores == [0.3, 0.6, 0.9]
        assert captured["url"] == "https://rerank.example.com/v1/rerank"
        assert captured["auth"] == "Bearer secret-key"
        assert captured["body"]["model"] == "bge-reranker-v2-m3"
        assert captured["body"]["query"] == "问题"
        assert captured["body"]["documents"] == ["甲", "乙", "丙"]
        assert captured["body"]["top_n"] == 3

    def test_missing_index_scores_zero(self, api_settings):
        captured = {}
        response = {"results": [{"index": 1, "relevance_score": 0.7}]}
        with patch.object(
            urllib.request, "urlopen", side_effect=_fake_urlopen_factory(response, captured)
        ):
            scores = retriever.score_pairs("q", ["a", "b", "c"])
        assert scores == [0.0, 0.7, 0.0]

    def test_no_auth_header_without_key(self, api_settings, monkeypatch):
        monkeypatch.setattr(settings, "RERANKER_API_KEY", "")
        captured = {}
        with patch.object(
            urllib.request,
            "urlopen",
            side_effect=_fake_urlopen_factory({"results": []}, captured),
        ):
            retriever.score_pairs("q", ["a"])
        assert captured["auth"] is None


def test_api_provider_requires_base_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/ragmate")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("RERANKER_PROVIDER", "api")
    monkeypatch.setenv("EMBEDDING_API_BASE_URL", "")
    monkeypatch.setenv("RERANKER_API_BASE_URL", "")
    with pytest.raises(ValueError, match="RERANKER_API_BASE_URL is required when RERANKER_PROVIDER=api"):
        Settings(_env_file=None)


class TestCallSites:
    def test_rerank_and_score_uses_score_pairs(self):
        candidates = [{"text": "甲"}, {"text": "乙"}, {"text": "丙"}]
        with patch.object(retriever, "score_pairs", return_value=[0.2, 0.9, 0.5]):
            scored, top = retriever._rerank_and_score("q", candidates)
        assert [c["text"] for c in scored] == ["乙", "丙", "甲"]
        assert top == 0.9

    def test_compression_failure_keeps_chunk_unchanged(self):
        block = "第一句话讲背景。第二句话是核心内容，非常相关。第三句话是无关的闲聊内容需要被压缩处理掉才行啊。"
        long_text = block * 7  # 超过 300 字压缩门槛，压缩必须真的跑打分
        candidates = [{"text": long_text, "source": "a.md"}]
        with patch.object(retriever, "score_pairs", side_effect=RuntimeError("api down")):
            result = retriever._compress_chunks("核心", candidates)
        assert result[0]["text"] == long_text


def _http_error_429(retry_after=None):
    from email.message import Message

    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError(
        "https://rerank.example.com/v1/rerank", 429, "Too Many Requests",
        headers, io.BytesIO(b""),
    )


class TestApiProviderRetry429:
    def test_retries_429_with_retry_after_then_backoff(self, api_settings, monkeypatch):
        sleeps = []
        monkeypatch.setattr("backend.core.retriever.time.sleep", sleeps.append)
        response = {"results": [{"index": 0, "relevance_score": 0.8}]}
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise _http_error_429(retry_after=7 if calls["n"] == 1 else None)
            return io.BytesIO(json.dumps(response).encode("utf-8"))

        with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            scores = retriever.score_pairs("q", ["a"])
        assert scores == [0.8]
        assert calls["n"] == 3
        assert sleeps == [7.0, 10.0]

    def test_gives_up_after_max_attempts(self, api_settings, monkeypatch):
        sleeps = []
        monkeypatch.setattr("backend.core.retriever.time.sleep", sleeps.append)
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            raise _http_error_429()

        with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            with pytest.raises(urllib.error.HTTPError):
                retriever.score_pairs("q", ["a"])
        assert calls["n"] == 5
        assert sleeps == [5.0, 10.0, 20.0, 40.0]

    def test_non_429_raises_without_retry(self, api_settings, monkeypatch):
        sleeps = []
        monkeypatch.setattr("backend.core.retriever.time.sleep", sleeps.append)
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            raise urllib.error.HTTPError(
                "https://rerank.example.com/v1/rerank", 500, "Internal Error",
                None, io.BytesIO(b""),
            )

        with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            with pytest.raises(urllib.error.HTTPError):
                retriever.score_pairs("q", ["a"])
        assert calls["n"] == 1
        assert sleeps == []
