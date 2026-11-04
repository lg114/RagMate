"""检索调试只在请求打开 debug 时出现，并标出过滤丢掉候选的原因。"""
from backend.core.retriever import (
    _filter_and_dedup,
    build_retrieval_debug,
    reset_retrieval_debug,
    set_retrieval_debug,
    take_retrieval_debug,
)


def _chunk(score, source="a.pdf", text="正文", page=0):
    return {"score": score, "source": source, "text": text, "page": page, "chunk_index": 0}


def test_debug_off_does_not_annotate_or_return_trace():
    low = _chunk(0.05, text="应被阈值丢掉")
    high = _chunk(0.9, text="留下")
    kept = _filter_and_dedup([high, low], 0.3, 5)
    assert kept == [high]
    assert "dropped_by" not in low
    assert "dropped_by" not in high
    assert take_retrieval_debug() is None


def test_debug_on_marks_threshold_and_builds_public_rows():
    low = _chunk(0.05, source="docs/低分.pdf", text="不相关的句子" * 5, page=1)
    high = _chunk(0.92, source="docs/高分.pdf", text="相关段落", page=0)
    token = set_retrieval_debug(True)
    try:
        kept = _filter_and_dedup([high, low], 0.3, 5)
        assert kept == [high]
        assert low["dropped_by"] == "threshold"
        assert high.get("dropped_by") is None
        high["_debug_preview"] = "相关段落"
        entry = build_retrieval_debug("问题", [high, low], kept, 0.3, 0.92)
    finally:
        reset_retrieval_debug(token)

    assert entry["returned"] == 1
    assert entry["candidates"] == 2
    assert entry["chunks"][0]["kept"] is True
    assert entry["chunks"][0]["source"] == "高分.pdf"
    assert entry["chunks"][0]["page"] == 1
    assert entry["chunks"][0]["dropped_by"] is None
    dropped = entry["chunks"][1]
    assert dropped["kept"] is False
    assert dropped["dropped_by"] == "threshold"
    assert dropped["source"] == "低分.pdf"
    assert dropped["page"] == 2
    assert len(dropped["preview"]) <= 240
    assert "score" in dropped


def test_take_retrieval_debug_discards_trace_when_disabled():
    from backend.core import retriever

    retriever._last_debug.entry = {"query": "secret", "chunks": [{"preview": "内部"}]}
    assert take_retrieval_debug() is None
    assert retriever._last_debug.entry is None


def test_score_gap_drop_reason():
    top = _chunk(0.9, source="a.pdf")
    cliff = _chunk(0.6, source="b.pdf")
    token = set_retrieval_debug(True)
    try:
        kept = _filter_and_dedup([top, cliff], 0.05, 10)
        assert kept == [top]
        assert cliff["dropped_by"] == "score_gap"
    finally:
        reset_retrieval_debug(token)
