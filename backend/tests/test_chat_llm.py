"""测试 chat_llm.py 的文本提取 helper。"""
from backend.application.chat_llm import extract_text_content


class TestExtractTextContent:
    """测试 extract_text_content 函数。"""

    def test_string_content(self):
        """字符串内容应直接返回。"""
        assert extract_text_content("Hello world") == "Hello world"
        assert extract_text_content("") == ""

    def test_list_content_single_block(self):
        """列表格式的单块内容应正确提取。"""
        content = [
            {"type": "text", "text": "Hello world"}
        ]
        assert extract_text_content(content) == "Hello world"

    def test_list_content_multiple_blocks(self):
        """列表格式的多块内容应正确拼接。"""
        content = [
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
        ]
        assert extract_text_content(content) == "First part\nSecond part"

    def test_list_content_mixed_types(self):
        """混合类型的内容应只提取 text 类型。"""
        content = [
            {"type": "text", "text": "Text content"},
            {"type": "image", "data": "base64..."},
            {"type": "text", "text": "More text"},
        ]
        assert extract_text_content(content) == "Text content\nMore text"

    def test_list_content_empty(self):
        """空列表应返回空字符串。"""
        assert extract_text_content([]) == ""

    def test_list_content_no_text_blocks(self):
        """没有 text 类型的块应返回空字符串。"""
        content = [
            {"type": "image", "data": "base64..."},
        ]
        assert extract_text_content(content) == ""

    def test_list_content_missing_text_field(self):
        """text 类型但缺少 text 字段应被跳过。"""
        content = [
            {"type": "text"},  # 缺少 text 字段
            {"type": "text", "text": "Valid text"},
        ]
        assert extract_text_content(content) == "Valid text"

    def test_non_standard_type(self):
        """非标准类型应返回空字符串。"""
        assert extract_text_content(123) == ""
        assert extract_text_content(None) == ""
        assert extract_text_content({}) == ""
