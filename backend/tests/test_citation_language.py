"""引用跟问题语言走，气泡只保留模型真正引用到的页。"""
import asyncio
import re

from backend.application.chat import (
    _ground_response,
    chat,
    chat_stream,
    citation_label,
    normalize_citations,
)
from backend.application.chat_citations import (
    citations_for_answer,
    extract_model_citations,
    question_language,
)
from backend.application.chat_retrieval import _bundle_from_results
from backend.infrastructure.config import settings

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _patch_session(monkeypatch):
    async def load_session(session_id):
        return []

    async def save_session(session_id, history):
        return None

    async def persist(session_id, messages):
        return None

    monkeypatch.setattr("backend.application.chat.load_session", load_session)
    monkeypatch.setattr("backend.application.chat.save_session", save_session)
    monkeypatch.setattr("backend.application.chat._persist_messages", persist)
    monkeypatch.setattr(settings, "FAITHFULNESS_CHECK", False)


def _bundle(items):
    citations = []
    results = []
    for index, (name, page, score) in enumerate(items):
        citations.append({
            "source": name,
            "page": page,
            "chunk_index": index,
            "score": score,
        })
        results.append({
            "text": f"text about {name}",
            "source": name,
            "page": page,
            "chunk_index": index,
            "score": score,
        })
    top = max(score for _, _, score in items)
    return {
        "metrics": [{"top_score": top, "returned": len(items), "candidates": len(items)}],
        "texts": [item["text"] for item in results],
        "citations": citations,
        "debug": [],
        "results": results,
    }


def _ask(monkeypatch, question, items, answer):
    _patch_session(monkeypatch)
    captured = {}

    def retrieve(*args, **kwargs):
        return _bundle(items)

    def invoke(messages):
        captured["messages"] = messages
        return answer

    monkeypatch.setattr("backend.application.chat._retrieve_for_answer", retrieve)
    monkeypatch.setattr("backend.application.chat._invoke_answer", invoke)
    result = asyncio.run(chat(question, session_id="sess-cite"))
    return result, captured


def test_question_language_and_labels_round_trip():
    assert question_language("What is the price?") == "en"
    assert question_language("How does pgvector work?") == "en"
    assert question_language("价格是多少？") == "zh"
    assert question_language("") == "zh"

    assert citation_label({"source": "/data/manual.pdf", "page": 0}, "en") == "[manual.pdf, page 1]"
    assert citation_label({"source": "notes.md", "page": None}, "en") == "[notes.md]"
    assert citation_label({"source": "/data/手册.pdf", "page": 0}) == "【手册.pdf, 第1页】"

    english = "See " + citation_label({"source": "manual.pdf", "page": 0}, "en") + " now."
    assert extract_model_citations(english) == [("manual.pdf", 1)]
    chinese = "见" + citation_label({"source": "手册.pdf", "page": 1}) + "。"
    assert extract_model_citations(chinese) == [("手册.pdf", 2)]
    # 中文上下文标签把页码写在括号外面，复制过来也要对上那一页。
    assert extract_model_citations("见【手册.pdf】 第3页。") == [("手册.pdf", 3)]


def test_single_leftover_marker_stripped():
    zh = normalize_citations("结论见【手册.pdf, 第2页】。")
    body, sep, rest = zh.partition("数据来源：")
    assert sep
    assert "【" not in body
    assert "【手册.pdf, 第2页】" in rest

    en = normalize_citations("See [guide.pdf, page 2].")
    body, sep, rest = en.partition("Sources:")
    assert sep
    assert "[guide.pdf, page 2]" not in body
    assert "[guide.pdf, page 2]" in rest
    assert "【" not in en
    assert "数据来源" not in en

    bare = normalize_citations("Note [readme.md] here.")
    body, sep, rest = bare.partition("Sources:")
    assert sep
    assert "[readme.md]" not in body
    assert "[readme.md]" in rest


def test_english_question_uses_english_labels(monkeypatch):
    items = [
        ("manual.pdf", 0, 0.50),
        ("other.pdf", 1, 0.99),
    ]
    result, captured = _ask(
        monkeypatch,
        "What is the price?",
        items,
        "The price is 10 yuan.【manual.pdf, 第1页】",
    )
    prompt = "\n".join(message["content"] for message in captured["messages"])
    assert "[filename, page N]" in prompt
    assert "same language as the question" in prompt
    assert "Retrieved context:" in prompt
    assert "[manual.pdf, page 1]" in prompt
    assert "[other.pdf, page 2]" in prompt
    assert "【" not in prompt
    assert "第" not in prompt

    assert result["grounding"] == "grounded"
    assert not _CJK_RE.search(result["response"])
    assert "Sources: [manual.pdf, page 1]" in result["response"]
    assert "other.pdf" not in result["response"]
    assert result["citations"] == [{"source": "manual.pdf", "page": 1}]


def test_stream_english_question_uses_english_labels(monkeypatch):
    _patch_session(monkeypatch)
    items = [("manual.pdf", 0, 0.50), ("other.pdf", 1, 0.99)]
    monkeypatch.setattr(
        "backend.application.chat._retrieve_for_answer",
        lambda *args, **kwargs: _bundle(items),
    )

    def tokens(_messages):
        yield "The price is 10 yuan.【manual.pdf, 第1页】"

    monkeypatch.setattr("backend.application.chat._iter_answer_tokens", tokens)

    async def _run():
        return [event async for event in chat_stream("What is the price?", session_id="sess-cite")]

    done = asyncio.run(_run())[-1]
    assert done["done"] is True
    assert not _CJK_RE.search(done["response"])
    assert "Sources: [manual.pdf, page 1]" in done["response"]
    assert done["citations"] == [{"source": "manual.pdf", "page": 1}]


def test_chinese_question_keeps_chinese_labels(monkeypatch):
    items = [
        ("价目表.pdf", 0, 0.40),
        ("其他.pdf", 2, 0.99),
    ]
    result, captured = _ask(
        monkeypatch,
        "价格是多少？",
        items,
        "价格是 10 元。【价目表.pdf, 第1页】",
    )
    prompt = "\n".join(message["content"] for message in captured["messages"])
    assert "【文件名, 第X页】" in prompt
    assert "相同的语言" in prompt
    assert "检索上下文：" in prompt
    assert "【价目表.pdf】 第1页" in prompt

    body, sep, _rest = result["response"].partition("数据来源：")
    assert sep
    assert "【" not in body
    assert "数据来源：【价目表.pdf, 第1页】" in result["response"]
    assert "其他.pdf" not in result["response"]
    assert "Sources:" not in result["response"]
    assert result["citations"] == [{"source": "价目表.pdf", "page": 1}]


def _fifteen():
    items = [(f"f{i:02d}.pdf", i, 0.05) for i in range(15)]
    items[1] = ("f01.pdf", 1, 0.97)
    items[9] = ("f09.pdf", 9, 0.95)
    items[4] = ("f04.pdf", 4, 0.91)
    return items


def test_public_citations_are_only_pages_the_model_cited(monkeypatch):
    result, _captured = _ask(
        monkeypatch,
        "Which pages mention the limit?",
        _fifteen(),
        "Use [f14.pdf, page 15] before [f02.pdf, page 3].",
    )
    assert result["citations"] == [
        {"source": "f14.pdf", "page": 15},
        {"source": "f02.pdf", "page": 3},
    ]
    assert "f00.pdf" not in result["response"]
    assert "f01.pdf" not in result["response"]


def test_no_parseable_citation_falls_back_to_top_three(monkeypatch):
    result, _captured = _ask(
        monkeypatch,
        "What does the document say?",
        _fifteen(),
        "The document does not name a source file.",
    )
    assert result["citations"] == [
        {"source": "f01.pdf", "page": 2},
        {"source": "f09.pdf", "page": 10},
        {"source": "f04.pdf", "page": 5},
    ]
    assert "Sources: [f01.pdf, page 2]; [f09.pdf, page 10]; [f04.pdf, page 5]" in result["response"]
    assert not _CJK_RE.search(result["response"])


def test_unmatched_citation_does_not_fall_back():
    rdata = {
        "metrics": [{"top_score": 0.9, "returned": 2, "candidates": 2}],
        "texts": ["alpha", "beta"],
        "citations": [
            {"source": "real.pdf", "page": 0, "score": 0.9},
            {"source": "other.pdf", "page": 1, "score": 0.8},
        ],
        "debug": [],
    }
    text, extra = asyncio.run(_ground_response(
        "sess",
        "See [missing.pdf, page 1] for this.",
        False,
        rdata=rdata,
        question="Where is it?",
    ))
    assert extra["grounding"] == "grounded"
    assert "citations" not in extra
    assert "missing.pdf" not in text
    assert "real.pdf" not in text
    assert "Sources:" not in text


def test_refused_answer_publishes_no_citations():
    rdata = {
        "metrics": [{"top_score": 0.0, "returned": 0, "candidates": 3}],
        "texts": [],
        "citations": [{"source": "a.pdf", "page": 0, "score": 0.9}],
        "debug": [],
    }
    text, extra = asyncio.run(_ground_response(
        "sess",
        "Some answer the model should not keep.",
        False,
        rdata=rdata,
        question="What is the price?",
    ))
    assert extra["grounding"] == "refused"
    assert "citations" not in extra
    assert "a.pdf" not in text


def test_bundle_keeps_score_for_citation_fallback():
    bundle = _bundle_from_results(
        [{"text": "a", "source": "a.pdf", "page": 0, "chunk_index": 1, "score": 0.42}],
        {"top_score": 0.42, "returned": 1, "candidates": 1},
        None,
        False,
    )
    assert bundle["citations"][0]["score"] == 0.42
    ranked = citations_for_answer("No citation here.", bundle["citations"])
    assert [item["source"] for item in ranked] == ["a.pdf"]
