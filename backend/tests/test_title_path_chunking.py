"""title_path 切块：路径拼装、前缀剥离、与 parent_child 块边界对齐。"""
import os
import tempfile

from langchain_core.documents import Document

from backend.application.ingest.pipeline import (
    _apply_title_path,
    _split_markdown_section,
    _title_path_from_metadata,
    strip_title_path_prefix,
)
from backend.infrastructure.config import settings


def test_title_path_from_metadata_joins_headers():
    assert _title_path_from_metadata({
        "Header 1": "A", "Header 2": "B", "Header 3": "C",
    }) == "A > B > C"
    assert _title_path_from_metadata({"Header 1": "Only"}) == "Only"
    assert _title_path_from_metadata({}) == ""


def test_apply_and_strip_title_path_roundtrip():
    body = "正文第一段\n\n第二段"
    wrapped = _apply_title_path(body, "H1 > H2")
    assert wrapped.startswith("标题路径: H1 > H2\n\n")
    assert strip_title_path_prefix(wrapped) == body
    assert strip_title_path_prefix(body) == body


def test_title_path_and_parent_child_share_body_boundaries(monkeypatch):
    section = Document(
        page_content="短正文，不会再被切成多块。" * 3,
        metadata={
            "source": "demo.md",
            "Header 1": "根",
            "Header 2": "节",
            "Header 3": "子节",
        },
    )
    monkeypatch.setattr(settings, "CHUNK_SIZE", 1000)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP", 200)
    monkeypatch.setattr(settings, "CHUNKING_STRATEGY", "parent_child")
    parent_chunks = _split_markdown_section(section, ".md")
    monkeypatch.setattr(settings, "CHUNKING_STRATEGY", "title_path")
    title_chunks = _split_markdown_section(section, ".md")
    assert len(parent_chunks) == len(title_chunks) == 1
    assert strip_title_path_prefix(title_chunks[0].page_content) == parent_chunks[0].page_content
    assert "标题路径: 根 > 节 > 子节" in title_chunks[0].page_content
    assert title_chunks[0].metadata["chunking_strategy"] == "title_path"


def test_corpus_bodies_align_under_both_strategies(monkeypatch):
    """样本语料上两种策略块数相同、去前缀后正文一致 → gold 三元组可复用。"""
    from backend.application.ingest import pipeline as ingest_pipeline

    corpus = os.path.join(
        os.path.dirname(__file__), "..", "..", "eval", "testsets", "sample", "corpus"
    )
    corpus = os.path.abspath(corpus)
    files = sorted(f for f in os.listdir(corpus) if f.endswith(".md"))
    monkeypatch.setattr(ingest_pipeline, "set_ingest_status_sync", lambda data: None)

    monkeypatch.setattr(settings, "CHUNKING_STRATEGY", "parent_child")
    pc, _, _ = ingest_pipeline._load_and_split(corpus, files)
    monkeypatch.setattr(settings, "CHUNKING_STRATEGY", "title_path")
    tp, _, _ = ingest_pipeline._load_and_split(corpus, files)

    assert len(pc) == len(tp) == 67
    for a, b in zip(pc, tp):
        assert a.metadata["chunk_index"] == b.metadata["chunk_index"]
        assert strip_title_path_prefix(b.page_content) == a.page_content
