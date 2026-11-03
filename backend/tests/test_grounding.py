"""默认接地：引用、低置信拒答/保留、忠实度校验。"""
from backend.application.chat import (
    apply_grounding,
    attach_citations,
    citation_label,
    note_unsupported_claims,
    public_citations,
)
from backend.core.agent import remember_retrieval
from backend.core.retriever import check_faithfulness
from backend.infrastructure.config import settings


def test_citation_label_uses_one_based_page_and_basename():
    assert citation_label({"source": "/data/手册.pdf", "page": 0}) == "【手册.pdf, 第1页】"
    assert citation_label({"source": "notes.md", "page": None}) == "【notes.md】"


def test_public_citations_drop_text_and_dedupe():
    raw = [
        {"source": "a/手册.pdf", "page": 2, "chunk_index": 1},
        {"source": "手册.pdf", "page": 2, "chunk_index": 4},
        {"source": "手册.pdf", "page": 0},
    ]
    assert public_citations(raw) == [
        {"source": "手册.pdf", "page": 3},
        {"source": "手册.pdf", "page": 1},
    ]


def test_attach_citations_replaces_model_source_line():
    text = "结论见这里。\n\n数据来源：【不存在.pdf】"
    out = attach_citations(text, [{"source": "真实.pdf", "page": 4}])
    assert "不存在.pdf" not in out
    assert out.endswith("数据来源：【真实.pdf, 第5页】")
    assert out.startswith("结论见这里。")


def test_no_confidence_leaves_answer_unchanged():
    text = "你好，我是助手。"
    assert apply_grounding(text, None, None) == (text, None)


def test_low_confidence_with_no_chunks_refuses():
    text = "根据常识，价格是 99 元。"
    out, grounding = apply_grounding(
        text,
        {"level": "low", "score": 0.0, "chunks": 0},
        [],
    )
    assert grounding == "refused"
    assert "99" not in out
    assert "没有找到足够相关的资料" in out


def test_low_confidence_with_chunks_hedges_and_cites():
    out, grounding = apply_grounding(
        "价格可能是 10 元。",
        {"level": "low", "score": 0.31, "chunks": 1},
        [{"source": "价目表.pdf", "page": 1}],
    )
    assert grounding == "hedged"
    assert out.startswith("检索置信度较低")
    assert "价格可能是 10 元。" in out
    assert "数据来源：【价目表.pdf, 第2页】" in out


def test_high_confidence_attaches_citations_without_hedge():
    out, grounding = apply_grounding(
        "价格是 10 元。",
        {"level": "high", "score": 0.9, "chunks": 4},
        [{"source": "价目表.pdf", "page": 0}],
    )
    assert grounding == "grounded"
    assert not out.startswith("检索置信度较低")
    assert "数据来源：【价目表.pdf, 第1页】" in out


def test_unsupported_note_is_appended_once():
    noted = note_unsupported_claims("答案", [{"claim": "x", "supported": False}])
    assert "未能在检索资料中找到依据" in noted
    assert note_unsupported_claims(noted, [{"claim": "x", "supported": False}]) == noted
    assert note_unsupported_claims("答案", None) == "答案"


def test_remember_retrieval_records_empty_hit_as_low_signal():
    ctx = {"metrics": [], "texts": []}
    remember_retrieval(ctx, [], None)
    assert ctx["metrics"] == [{"top_score": 0.0, "returned": 0, "candidates": 0}]
    assert "citations" not in ctx


def test_remember_retrieval_keeps_sources_for_citations():
    ctx = {}
    remember_retrieval(
        ctx,
        [{"text": "正文", "source": "a.pdf", "page": 3, "chunk_index": 1, "score": 0.8}],
        {"top_score": 0.8, "returned": 1, "candidates": 4},
    )
    assert ctx["texts"] == ["正文"]
    assert ctx["citations"] == [{"source": "a.pdf", "page": 3, "chunk_index": 1}]
    assert "score" not in ctx["citations"][0]


def test_faithfulness_disabled_skips_llm(monkeypatch):
    monkeypatch.setattr(settings, "FAITHFULNESS_CHECK", False)

    def _boom():
        raise AssertionError("should not call LLM")

    monkeypatch.setattr("backend.infrastructure.model_factory.get_llm", _boom)
    assert check_faithfulness("这是一个足够长的声明句子。", ["证据"]) is None


def test_faithfulness_parses_unsupported_claim(monkeypatch):
    monkeypatch.setattr(settings, "FAITHFULNESS_CHECK", True)

    class _Resp:
        content = '说明\n[{"claim": "这是一个足够长的声明句子。", "supported": false}]'

    class _LLM:
        def invoke(self, prompt):
            assert "证据正文" in prompt
            return _Resp()

    monkeypatch.setattr("backend.infrastructure.model_factory.get_llm", lambda: _LLM())
    claims = check_faithfulness("这是一个足够长的声明句子。", ["证据正文"])
    assert claims == [{"claim": "这是一个足够长的声明句子。", "supported": False}]


def test_faithfulness_defaults_on():
    assert type(settings).model_fields["FAITHFULNESS_CHECK"].default is True
