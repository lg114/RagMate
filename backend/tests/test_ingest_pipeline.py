"""测试 ingest pipeline 的增量检测与去重逻辑。"""
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from backend.application.ingest.pipeline import (
    _detect_new_files,
    _discover_files,
    _get_text_splitter,
    _reconstruct_with_headers,
)


class TestDiscoverFiles:
    """测试 _discover_files 函数。"""

    def test_discover_all_files(self):
        """发现目录下所有支持的文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试文件
            (open(os.path.join(tmpdir, "test1.pdf"), "w")).close()
            (open(os.path.join(tmpdir, "test2.docx"), "w")).close()
            (open(os.path.join(tmpdir, "test3.txt"), "w")).close()
            (open(os.path.join(tmpdir, "test4.md"), "w")).close()
            (open(os.path.join(tmpdir, "test5.xlsx"), "w")).close()
            (open(os.path.join(tmpdir, "unsupported.xyz"), "w")).close()

            files = _discover_files(tmpdir, None)

            assert len(files) == 5
            assert "test1.pdf" in files
            assert "test2.docx" in files
            assert "test3.txt" in files
            assert "test4.md" in files
            assert "test5.xlsx" in files
            assert "unsupported.xyz" not in files

    def test_discover_with_filter(self):
        """按文件名过滤。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (open(os.path.join(tmpdir, "test1.pdf"), "w")).close()
            (open(os.path.join(tmpdir, "test2.pdf"), "w")).close()
            (open(os.path.join(tmpdir, "test3.pdf"), "w")).close()

            files = _discover_files(tmpdir, ["test1.pdf", "test3.pdf"])

            assert len(files) == 2
            assert "test1.pdf" in files
            assert "test3.pdf" in files
            assert "test2.pdf" not in files

    def test_discover_sorted(self):
        """文件应按字母顺序排序。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (open(os.path.join(tmpdir, "c.pdf"), "w")).close()
            (open(os.path.join(tmpdir, "a.pdf"), "w")).close()
            (open(os.path.join(tmpdir, "b.pdf"), "w")).close()

            files = _discover_files(tmpdir, None)

            assert files == ["a.pdf", "b.pdf", "c.pdf"]

    def test_discover_empty_directory(self):
        """空目录应返回空列表。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = _discover_files(tmpdir, None)
            assert files == []


class TestGetTextSplitter:
    """测试 _get_text_splitter 函数。"""

    def test_pdf_splitter(self):
        """PDF 应返回对应的 splitter。"""
        splitter = _get_text_splitter(".pdf")
        assert splitter is not None
        assert splitter._chunk_size == 600  # CHUNK_SIZE_PDF default
        assert splitter._chunk_overlap == 100  # CHUNK_OVERLAP_PDF default

    def test_docx_splitter(self):
        """DOCX 应返回对应的 splitter。"""
        splitter = _get_text_splitter(".docx")
        assert splitter is not None
        assert splitter._chunk_size == 800  # CHUNK_SIZE_DOCX default
        assert splitter._chunk_overlap == 150  # CHUNK_OVERLAP_DOCX default

    def test_txt_splitter(self):
        """TXT 应返回对应的 splitter。"""
        splitter = _get_text_splitter(".txt")
        assert splitter is not None
        assert splitter._chunk_size == 1000  # CHUNK_SIZE_TXT default
        assert splitter._chunk_overlap == 200  # CHUNK_OVERLAP_TXT default

    def test_md_splitter(self):
        """Markdown 应返回对应的 splitter。"""
        splitter = _get_text_splitter(".md")
        assert splitter is not None
        assert splitter._chunk_size == 1000  # CHUNK_SIZE default
        assert splitter._chunk_overlap == 200  # CHUNK_OVERLAP default

    def test_excel_returns_none(self):
        """Excel 文件应返回 None（不切分）。"""
        assert _get_text_splitter(".xlsx") is None
        assert _get_text_splitter(".xls") is None


class TestReconstructWithHeaders:
    """测试 _reconstruct_with_headers 函数。"""

    def test_no_headers(self):
        """无标题时直接返回原文。"""
        doc = MagicMock()
        doc.metadata = {}
        doc.page_content = "这是内容"

        result = _reconstruct_with_headers(doc)
        assert result == "这是内容"

    def test_single_header(self):
        """单个标题应正确还原。"""
        doc = MagicMock()
        doc.metadata = {"Header 1": "第一章"}
        doc.page_content = "这是内容"

        result = _reconstruct_with_headers(doc)
        assert result == "# 第一章\n\n这是内容"

    def test_multiple_headers(self):
        """多个标题应按层级还原。"""
        doc = MagicMock()
        doc.metadata = {
            "Header 1": "第一章",
            "Header 2": "第一节",
            "Header 3": "小节",
        }
        doc.page_content = "这是内容"

        result = _reconstruct_with_headers(doc)
        assert result == "# 第一章\n## 第一节\n### 小节\n\n这是内容"

    def test_partial_headers(self):
        """部分标题应正确还原。"""
        doc = MagicMock()
        doc.metadata = {"Header 1": "第一章", "Header 3": "小节"}
        doc.page_content = "这是内容"

        result = _reconstruct_with_headers(doc)
        assert result == "# 第一章\n### 小节\n\n这是内容"


class TestDetectNewFiles:
    """测试 _detect_new_files 函数。"""

    @patch("backend.application.ingest.pipeline.SyncSession")
    def test_all_new_files(self, mock_session_class):
        """所有文件都是新的（数据库中无记录）。"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            (open(os.path.join(tmpdir, "test1.pdf"), "w")).close()
            (open(os.path.join(tmpdir, "test2.pdf"), "w")).close()

            new_files, ingested_info = _detect_new_files(tmpdir, ["test1.pdf", "test2.pdf"])

            assert len(new_files) == 2
            assert "test1.pdf" in new_files
            assert "test2.pdf" in new_files
            assert ingested_info == {}

    @patch("backend.application.ingest.pipeline.SyncSession")
    def test_all_already_ingested(self, mock_session_class):
        """所有文件已索引且未修改。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建文件并获取 mtime
            file1 = os.path.join(tmpdir, "test1.pdf")
            file2 = os.path.join(tmpdir, "test2.pdf")
            (open(file1, "w")).close()
            (open(file2, "w")).close()
            mtime1 = os.path.getmtime(file1)
            mtime2 = os.path.getmtime(file2)

            # Mock 数据库返回已索引的文件信息
            mock_session = MagicMock()
            mock_session_class.return_value.__enter__.return_value = mock_session
            mock_result = MagicMock()
            mock_result.fetchall.return_value = [
                ("test1.pdf", mtime1),
                ("test2.pdf", mtime2),
            ]
            mock_session.execute.return_value = mock_result

            new_files, ingested_info = _detect_new_files(tmpdir, ["test1.pdf", "test2.pdf"])

            assert len(new_files) == 0
            assert len(ingested_info) == 2

    @patch("backend.application.ingest.pipeline.SyncSession")
    def test_modified_file_detected(self, mock_session_class):
        """修改过的文件应被检测为需要重新索引。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "test1.pdf")
            (open(file1, "w")).close()
            current_mtime = os.path.getmtime(file1)

            # Mock 数据库返回旧的 mtime（比当前早 100 秒）
            mock_session = MagicMock()
            mock_session_class.return_value.__enter__.return_value = mock_session
            mock_result = MagicMock()
            mock_result.fetchall.return_value = [
                ("test1.pdf", current_mtime - 100),
            ]
            mock_session.execute.return_value = mock_result

            new_files, ingested_info = _detect_new_files(tmpdir, ["test1.pdf"])

            assert len(new_files) == 1
            assert "test1.pdf" in new_files

    @patch("backend.application.ingest.pipeline.SyncSession")
    def test_mixed_new_and_existing(self, mock_session_class):
        """混合新文件和已索引文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "existing.pdf")
            (open(file1, "w")).close()
            mtime1 = os.path.getmtime(file1)
            (open(os.path.join(tmpdir, "new.pdf"), "w")).close()

            mock_session = MagicMock()
            mock_session_class.return_value.__enter__.return_value = mock_session
            mock_result = MagicMock()
            mock_result.fetchall.return_value = [
                ("existing.pdf", mtime1),
            ]
            mock_session.execute.return_value = mock_result

            new_files, ingested_info = _detect_new_files(tmpdir, ["existing.pdf", "new.pdf"])

            assert len(new_files) == 1
            assert "new.pdf" in new_files
            assert "existing.pdf" not in new_files
