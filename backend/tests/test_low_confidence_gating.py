"""低置信度 gating：忠实度校验与追问改写只在检索置信度低时跑。

置信度来自 calculate_confidence：high（top_score>=0.7 且 >=3 片段）、
medium（>=0.4 且 >=1 片段），其余为 low。证据扎实时跳过额外的 LLM 调用。
"""
import asyncio

from backend.application.chat import _pop_retrieval_signals, _retrieve_for_answer
from backend.infrastructure.config import settings

_HISTORY = [
    {"role": "user", "content": "价目表里有什么？"},
    {"role": "assistant", "content": "有价格。"},
    {"role": "user", "content": "多少钱？"},
]

_RESULT = [{"text": "价格是 10 元", "source": "价目表.pdf", "page": 0, "chunk_index": 1}]


def _rdata(top_score, returned):
    return {
        "metrics": [{"top_score": top_score, "returned": returned, "candidates": 4}],
        "texts": ["价格是 10 元"],
        "citations": [{"source": "价目表.pdf", "page": 0, "chunk_index": 1}],
        "debug": [],
        "results": list(_RESULT),
    }


def test_faithfulness_skipped_on_high_confidence(monkeypatch):
    monkeypatch.setattr(settings, "FAITHFULNESS_CHECK", True)

    def _boom(*args, **kwargs):
        raise AssertionError("faithfulness check should be skipped on high confidence")

    monkeypatch.setattr("backend.core.retriever.check_faithfulness", _boom)
    confidence, unsupported, citations, debug = asyncio.run(
        _pop_retrieval_signals("sess-1", "价格是 10 元。", False, rdata=_rdata(0.91, 5))
    )
    assert confidence["level"] == "high"
    assert unsupported is None


def test_faithfulness_runs_on_low_confidence(monkeypatch):
    monkeypatch.setattr(settings, "FAITHFULNESS_CHECK", True)
    calls = []

    def _fake_check(response, texts):
        calls.append((response, texts))
        return [{"claim": "价格是 10 元", "supported": False}]

    monkeypatch.setattr("backend.core.retriever.check_faithfulness", _fake_check)
    confidence, unsupported, citations, debug = asyncio.run(
        _pop_retrieval_signals("sess-1", "价格是 10 元。", False, rdata=_rdata(0.1, 1))
    )
    assert confidence["level"] == "low"
    assert len(calls) == 1
    assert unsupported == [{"claim": "价格是 10 元", "supported": False}]


def test_contextualize_skipped_on_high_confidence(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_CONTEXTUALIZE", True)
    calls = {"retrieve": [], "rewrite": 0}

    def _fake_retrieve(query, k=None):
        calls["retrieve"].append(query)
        return list(_RESULT)

    def _fake_metrics():
        return {"top_score": 0.91, "returned": 5, "candidates": 4}

    def _boom_rewrite(query, messages):
        calls["rewrite"] += 1
        raise AssertionError("should not rewrite on high confidence")

    monkeypatch.setattr("backend.core.retriever.retrieve", _fake_retrieve)
    monkeypatch.setattr("backend.core.retriever.get_retrieval_metrics", _fake_metrics)
    monkeypatch.setattr("backend.core.retriever.contextualize_query", _boom_rewrite)
    bundle = _retrieve_for_answer("多少钱？", _HISTORY, False)
    assert calls["retrieve"] == ["多少钱？"]
    assert calls["rewrite"] == 0
    assert len(bundle["results"]) == 1


def test_contextualize_retries_on_low_confidence(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_CONTEXTUALIZE", True)
    calls = {"retrieve": [], "rewrite": 0}

    def _fake_retrieve(query, k=None):
        calls["retrieve"].append(query)
        return list(_RESULT)

    def _fake_metrics():
        return {"top_score": 0.1, "returned": 1, "candidates": 4}

    def _fake_rewrite(query, messages):
        calls["rewrite"] += 1
        return "价目表里那个东西多少钱？"

    monkeypatch.setattr("backend.core.retriever.retrieve", _fake_retrieve)
    monkeypatch.setattr("backend.core.retriever.get_retrieval_metrics", _fake_metrics)
    monkeypatch.setattr("backend.core.retriever.contextualize_query", _fake_rewrite)
    bundle = _retrieve_for_answer("多少钱？", _HISTORY, False)
    assert calls["retrieve"] == ["多少钱？", "价目表里那个东西多少钱？"]
    assert calls["rewrite"] == 1
    assert len(bundle["results"]) == 1


def test_contextualize_flag_off_never_rewrites(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_CONTEXTUALIZE", False)
    calls = {"retrieve": [], "rewrite": 0}

    def _fake_retrieve(query, k=None):
        calls["retrieve"].append(query)
        return list(_RESULT)

    def _fake_metrics():
        return {"top_score": 0.1, "returned": 1, "candidates": 4}

    def _boom_rewrite(query, messages):
        calls["rewrite"] += 1
        raise AssertionError("flag off must never rewrite")

    monkeypatch.setattr("backend.core.retriever.retrieve", _fake_retrieve)
    monkeypatch.setattr("backend.core.retriever.get_retrieval_metrics", _fake_metrics)
    monkeypatch.setattr("backend.core.retriever.contextualize_query", _boom_rewrite)
    bundle = _retrieve_for_answer("多少钱？", _HISTORY, False)
    assert calls["retrieve"] == ["多少钱？"]
    assert calls["rewrite"] == 0
    assert len(bundle["results"]) == 1
