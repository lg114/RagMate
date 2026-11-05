"""默认对话是一次检索、一次生成、再接地。零片段拒答，不调用回答模型。"""
import asyncio

from backend.application.chat import chat, chat_stream
from backend.infrastructure.config import settings

_EMPTY = {
    "metrics": [{"top_score": 0.0, "returned": 0, "candidates": 0}],
    "texts": [],
    "citations": [],
    "debug": [],
    "results": [],
}

_HIT = {
    "metrics": [{"top_score": 0.91, "returned": 1, "candidates": 4}],
    "texts": ["价格是 10 元"],
    "citations": [{"source": "价目表.pdf", "page": 0, "chunk_index": 1}],
    "debug": [],
    "results": [{"text": "价格是 10 元", "source": "价目表.pdf", "page": 0, "chunk_index": 1}],
}


def _patch_session(monkeypatch):
    saved = {}

    async def load_session(session_id):
        return []

    async def save_session(session_id, history):
        saved["history"] = history

    async def persist(session_id, messages):
        saved["msgs"] = messages

    monkeypatch.setattr("backend.application.chat.load_session", load_session)
    monkeypatch.setattr("backend.application.chat.save_session", save_session)
    monkeypatch.setattr("backend.application.chat._persist_messages", persist)
    return saved


def _collect(message):
    async def _run():
        return [event async for event in chat_stream(message, session_id="sess-1")]

    return asyncio.run(_run())


def test_chat_refuses_without_calling_the_answer_model(monkeypatch):
    saved = _patch_session(monkeypatch)
    monkeypatch.setattr("backend.application.chat._retrieve_for_answer", lambda *args, **kwargs: _EMPTY)

    def invoke(_messages):
        raise AssertionError("zero retrieval must not generate an answer")

    monkeypatch.setattr("backend.application.chat._invoke_answer", invoke)

    result = asyncio.run(chat("什么是向量索引？", session_id="sess-1"))

    assert result["grounding"] == "refused"
    assert result["confidence"]["chunks"] == 0
    assert "没有找到足够相关的资料" in result["response"]
    assert saved["history"][-1] == {"role": "assistant", "content": result["response"]}
    assert saved["msgs"][-1] == ("assistant", result["response"])


def test_stream_zero_retrieval_does_not_emit_then_replace(monkeypatch):
    saved = _patch_session(monkeypatch)
    monkeypatch.setattr("backend.application.chat._retrieve_for_answer", lambda *args, **kwargs: _EMPTY)

    def tokens(_messages):
        raise AssertionError("zero retrieval must not stream an answer")
        yield ""  # pragma: no cover

    monkeypatch.setattr("backend.application.chat._iter_answer_tokens", tokens)

    events = _collect("什么是向量索引？")

    assert events
    assert all("token" not in event for event in events)
    done = events[-1]
    assert done["done"] is True
    assert done["grounding"] == "refused"
    assert "没有找到足够相关的资料" in done["response"]
    assert saved["history"][-1]["content"] == done["response"]


def test_chat_retrieves_once_then_generates_once(monkeypatch):
    monkeypatch.setattr(settings, "FAITHFULNESS_CHECK", False)
    _patch_session(monkeypatch)
    calls = {"retrieve": 0, "generate": 0}

    def retrieve(*args, **kwargs):
        calls["retrieve"] += 1
        return _HIT

    def invoke(_messages):
        calls["generate"] += 1
        return "价格是 10 元。"

    monkeypatch.setattr("backend.application.chat._retrieve_for_answer", retrieve)
    monkeypatch.setattr("backend.application.chat._invoke_answer", invoke)

    result = asyncio.run(chat("价格是多少？", session_id="sess-1"))

    assert calls == {"retrieve": 1, "generate": 1}
    assert result["grounding"] == "grounded"
    assert "价格是 10 元。" in result["response"]
    assert "价目表.pdf" in result["response"]
    assert result["citations"] == [{"source": "价目表.pdf", "page": 1}]


def test_stream_stores_answer_verbatim(monkeypatch):
    monkeypatch.setattr(settings, "FAITHFULNESS_CHECK", False)
    saved = _patch_session(monkeypatch)
    monkeypatch.setattr("backend.application.chat._retrieve_for_answer", lambda *args, **kwargs: _HIT)

    def tokens(_messages):
        yield "价格是 10 元。"

    monkeypatch.setattr("backend.application.chat._iter_answer_tokens", tokens)

    events = _collect("价格是多少？")
    streamed = "".join(event["token"] for event in events if "token" in event)
    stored = saved["history"][-1]["content"]

    assert streamed == "价格是 10 元。"
    assert stored == events[-1]["response"]
    assert "价目表.pdf" in stored
