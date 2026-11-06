"""title_path 切块：路径拼装、前缀剥离。"""
from langchain_core.documents import Document

from backend.application.ingest.pipeline import (
    _apply_title_path,
    _title_path_from_metadata,
    strip_title_path_prefix,
)


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
