"""title_path 切块：路径拼装、前缀剥离、_split_markdown_section 行为。"""
import pytest
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


def test_split_markdown_section_title_path(monkeypatch):
    """_split_markdown_section 在 title_path 策略下：子块带标题路径、parent_text 完整、metadata 正确。"""
    monkeypatch.setattr(settings, "CHUNKING_STRATEGY", "title_path")
    monkeypatch.setattr(settings, "CHUNK_SIZE", 1000)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP", 200)

    section = Document(
        page_content="短正文，不会再被切成多块。" * 3,
        metadata={
            "source": "demo.md",
            "Header 1": "根",
            "Header 2": "节",
            "Header 3": "子节",
        },
    )
    chunks = _split_markdown_section(section, ".md")

    assert len(chunks) == 1
    c = chunks[0]
    assert c.metadata["chunking_strategy"] == "title_path"
    assert c.metadata["title_path"] == "根 > 节 > 子节"
    assert "标题路径: 根 > 节 > 子节" in c.page_content
    assert strip_title_path_prefix(c.page_content) == section.page_content
    assert c.metadata["parent_text"].startswith("标题路径: 根 > 节 > 子节")


def test_split_markdown_section_rejects_unknown_strategy(monkeypatch):
    """未知策略应抛错，不静默回退。"""
    monkeypatch.setattr(settings, "CHUNKING_STRATEGY", "unknown_strategy")
    section = Document(page_content="内容", metadata={"source": "demo.md"})

    with pytest.raises(ValueError, match="Unknown CHUNKING_STRATEGY"):
        _split_markdown_section(section, ".md")
