"""父子分块（small-to-big）检索侧行为的回归测试。

背景：parent_text 曾被写入 Milvus 却从未在检索时请求，`or` 回退把缺陷静默吞掉，
导致 README 宣称的 "small-to-big" 实际退化为纯小片段检索。这组测试锁定三件事：

1. _do_search 必须显式请求 parent_text（否则父片段永远取不到）
2. _extract_candidates 必须把「打分文本」与「交付文本」分开保存
3. 压缩必须作用在父片段上，而不是用于打分的小片段
"""
from unittest.mock import MagicMock, patch

import pytest

from backend.core.retriever import (
    _do_search,
    _expand_to_parent,
    _extract_candidates,
    retrieve,
)


def _hit(text: str, parent_text: str | None = None, source: str = "a.pdf",
         page: int = 0, chunk_index: int = 0) -> dict:
    """构造一条 Milvus 搜索命中。parent_text=None 表示未请求该字段。"""
    entity = {
        "text": text,
        "metadata": {"source": source, "page": page, "chunk_index": chunk_index},
    }
    if parent_text is not None:
        entity["parent_text"] = parent_text
    return {"entity": entity}


# ── 检索时必须请求 parent_text ────────────────────────────────────────────────


class TestDoSearchRequestsParentText:
    """回归：output_fields 漏掉 parent_text 时，父片段会被静默丢弃。"""

    @patch("backend.core.retriever.RRFRanker")
    @patch("backend.core.retriever.AnnSearchRequest")
    def test_hybrid_search_requests_parent_text(self, mock_ann, mock_ranker, override_settings):
        override_settings(HYBRID_SEARCH_ENABLED=True)
        client = MagicMock()
        client.hybrid_search.return_value = [[]]

        _do_search(client, [0.1], MagicMock(), over_fetch=5)

        output_fields = client.hybrid_search.call_args.kwargs["output_fields"]
        assert "parent_text" in output_fields
        assert "text" in output_fields

    def test_dense_only_search_requests_parent_text(self, override_settings):
        override_settings(HYBRID_SEARCH_ENABLED=False)
        client = MagicMock()
        client.search.return_value = [[]]

        _do_search(client, [0.1], MagicMock(), over_fetch=5)

        output_fields = client.search.call_args.kwargs["output_fields"]
        assert "parent_text" in output_fields
        assert "text" in output_fields


# ── 候选必须同时保留小片段与父片段 ────────────────────────────────────────────


class TestExtractCandidates:
    def test_splits_scoring_text_and_delivery_context(self):
        """text 保存小片段（打分用），context 保存父片段（交付用）。"""
        results = [[_hit("小片段", parent_text="更大的父片段内容")]]

        candidates = _extract_candidates(results)

        assert len(candidates) == 1
        assert candidates[0]["text"] == "小片段"
        assert candidates[0]["context"] == "更大的父片段内容"

    def test_falls_back_when_parent_text_absent(self):
        """未请求到 parent_text 时，context 回退为小片段而非 None。"""
        results = [[_hit("小片段")]]

        candidates = _extract_candidates(results)

        assert candidates[0]["text"] == "小片段"
        assert candidates[0]["context"] == "小片段"

    def test_preserves_metadata_fields(self):
        results = [[_hit("t", parent_text="p", source="b.pdf", page=3, chunk_index=7)]]

        candidates = _extract_candidates(results)

        assert candidates[0]["source"] == "b.pdf"
        assert candidates[0]["page"] == 3
        assert candidates[0]["chunk_index"] == 7

    def test_handles_multiple_hits(self):
        results = [[_hit("t1", parent_text="p1"), _hit("t2", parent_text="p2")]]

        candidates = _extract_candidates(results)

        assert [c["text"] for c in candidates] == ["t1", "t2"]
        assert [c["context"] for c in candidates] == ["p1", "p2"]


# ── 展开只换交付文本，不动打分结果 ────────────────────────────────────────────


class TestExpandToParent:
    def test_replaces_text_with_parent(self):
        candidates = [{"text": "小片段", "context": "父片段", "score": 0.9}]

        result = _expand_to_parent(candidates)

        assert result[0]["text"] == "父片段"
        assert result[0]["expanded"] is True

    def test_keeps_score_and_source(self):
        """阈值判断与来源去重已经完成，展开不得改动这些字段。"""
        candidates = [{"text": "小", "context": "大", "score": 0.42, "source": "a.pdf"}]

        result = _expand_to_parent(candidates)

        assert result[0]["score"] == 0.42
        assert result[0]["source"] == "a.pdf"

    def test_no_marker_when_context_equals_text(self):
        """父片段与小片段相同时不标记 expanded，也不改文本。"""
        candidates = [{"text": "同一段", "context": "同一段", "score": 0.9}]

        result = _expand_to_parent(candidates)

        assert result[0]["text"] == "同一段"
        assert "expanded" not in result[0]

    def test_tolerates_missing_context_key(self):
        """兼容不含 context 的旧结构（例如其他调用方直接构造的候选）。"""
        candidates = [{"text": "小片段", "score": 0.9}]

        result = _expand_to_parent(candidates)

        assert result[0]["text"] == "小片段"
        assert "expanded" not in result[0]

    def test_empty_list(self):
        assert _expand_to_parent([]) == []


# ── 执行顺序：压缩必须作用在父片段上 ──────────────────────────────────────────


class TestCompressionRunsOnParentText:
    """回归：压缩的目标是"减少传给 LLM 的噪声"，必须作用于交付内容。"""

    def test_compression_receives_parent_text(self, override_settings):
        parent = "父片段内容。" * 50
        small = "小片段"
        candidate = {"text": small, "context": parent, "source": "a.pdf", "score": 0.9}

        with patch("backend.core.retriever.init_milvus") as mock_init, \
             patch("backend.core.retriever.encode_query", return_value=([0.1], None)), \
             patch("backend.core.retriever._do_search", return_value=[[]]), \
             patch("backend.core.retriever._extract_candidates", return_value=[dict(candidate)]), \
             patch("backend.core.retriever._rerank_and_score", return_value=([dict(candidate)], 0.9)), \
             patch("backend.core.retriever._filter_and_dedup", return_value=[dict(candidate)]), \
             patch("backend.core.retriever._compress_chunks", side_effect=lambda q, c: c) as mock_compress:
            mock_init.return_value = MagicMock()

            retrieve("测试查询", k=1)

        passed_candidates = mock_compress.call_args[0][1]
        assert passed_candidates[0]["text"] == parent, (
            "压缩收到的是小片段，说明 _expand_to_parent 没有在压缩之前执行"
        )

    def test_returns_parent_text_to_caller(self, override_settings):
        """最终返回给调用方（agent / 评测）的必须是父片段。"""
        parent = "父片段内容。" * 50
        candidate = {"text": "小片段", "context": parent, "source": "a.pdf", "score": 0.9}

        with patch("backend.core.retriever.init_milvus") as mock_init, \
             patch("backend.core.retriever.encode_query", return_value=([0.1], None)), \
             patch("backend.core.retriever._do_search", return_value=[[]]), \
             patch("backend.core.retriever._extract_candidates", return_value=[dict(candidate)]), \
             patch("backend.core.retriever._rerank_and_score", return_value=([dict(candidate)], 0.9)), \
             patch("backend.core.retriever._filter_and_dedup", return_value=[dict(candidate)]), \
             patch("backend.core.retriever._compress_chunks", side_effect=lambda q, c: c):
            mock_init.return_value = MagicMock()

            result = retrieve("测试查询", k=1)

        assert result[0]["text"] == parent
        assert result[0]["source"] == "a.pdf"
