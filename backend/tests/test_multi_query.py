"""多查询扩展（MULTI_QUERY_ENABLED）单元测试：变体生成、多路 RRF 融合、retrieve 接入。"""
from unittest.mock import MagicMock, patch

import pytest

from backend.core import retriever
from backend.infrastructure.config import settings


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


@pytest.fixture
def enable_multi(monkeypatch):
    monkeypatch.setattr(settings, "MULTI_QUERY_ENABLED", True)
    monkeypatch.setattr(settings, "MULTI_QUERY_COUNT", 3)


class TestExpandQueries:
    def test_disabled_returns_original_only(self, monkeypatch):
        monkeypatch.setattr(settings, "MULTI_QUERY_ENABLED", False)
        assert retriever.expand_queries("问题") == ["问题"]

    def test_parses_variants_and_dedupes(self, enable_multi):
        llm = MagicMock()
        llm.invoke.return_value = _FakeResponse(
            "1. 变体一\n变体一\n2、变体二\n\"问题\"\n\n变体三"
        )
        with patch("backend.infrastructure.model_factory.get_llm", return_value=llm):
            result = retriever.expand_queries("问题")
        # 原样重复、与原 query 相同、空行都被剔除；编号前缀被剥掉
        assert result == ["问题", "变体一", "变体二", "变体三"]

    def test_caps_at_configured_count(self, enable_multi, monkeypatch):
        monkeypatch.setattr(settings, "MULTI_QUERY_COUNT", 2)
        llm = MagicMock()
        llm.invoke.return_value = _FakeResponse("变体一\n变体二\n变体三\n变体四")
        with patch("backend.infrastructure.model_factory.get_llm", return_value=llm):
            assert retriever.expand_queries("问题") == ["问题", "变体一", "变体二"]

    def test_llm_failure_falls_back_to_original(self, enable_multi):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("llm down")
        with patch("backend.infrastructure.model_factory.get_llm", return_value=llm):
            assert retriever.expand_queries("问题") == ["问题"]


def _hit(source: str, chunk_index: int, text: str):
    return {
        "entity": {
            "text": text,
            "parent_text": text,
            "metadata": {"source": source, "page": 0, "chunk_index": chunk_index},
        }
    }


class TestSearchMulti:
    def test_rrf_fusion_prefers_chunks_seen_in_multiple_queries(self):
        # A 在两路都出现（第 2、1 名），B 只在第一路第 1 名，C 只在第二路第 2 名
        per_query = {
            "q1": [[_hit("b.md", 0, "B"), _hit("a.md", 0, "A")]],
            "q2": [[_hit("a.md", 0, "A"), _hit("c.md", 0, "C")]],
        }
        current = {}

        def fake_encode(q):
            current["q"] = q
            return [0.0], {}

        def fake_search(client, dense, sparse, over_fetch):
            return per_query[current["q"]]

        with patch.object(retriever, "encode_query", side_effect=fake_encode), patch.object(
            retriever, "_do_search", side_effect=fake_search
        ):
            merged = retriever._search_multi(object(), ["q1", "q2"], over_fetch=30)

        assert [c["source"] for c in merged] == ["a.md", "b.md", "c.md"]

    def test_truncates_to_over_fetch(self):
        results = [[_hit(f"doc{i}.md", 0, f"T{i}") for i in range(10)]]
        with patch.object(retriever, "encode_query", return_value=([0.0], {})), patch.object(
            retriever, "_do_search", return_value=results
        ):
            merged = retriever._search_multi(object(), ["q"], over_fetch=3)
        assert len(merged) == 3


class TestRetrieveIntegration:
    def test_retrieve_uses_multi_search_when_expanded(self, enable_multi):
        llm = MagicMock()
        llm.invoke.return_value = _FakeResponse("变体一")
        fake_reranker = MagicMock()
        fake_reranker.predict.side_effect = lambda pairs: [2.0] * len(pairs)

        def fake_search(client, dense, sparse, over_fetch):
            return [[_hit("a.md", 0, "短文本")]]

        with patch("backend.infrastructure.model_factory.get_llm", return_value=llm), patch.object(
            retriever, "init_milvus", return_value=object()
        ), patch.object(retriever, "encode_query", return_value=([0.0], {})), patch.object(
            retriever, "_do_search", side_effect=fake_search
        ), patch.object(
            retriever, "get_reranker", return_value=fake_reranker
        ):
            results = retriever.retrieve("问题", k=5)

        assert len(results) == 1
        assert results[0]["source"] == "a.md"
