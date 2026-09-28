"""测试 chat.py 的查询路由逻辑。"""
import pytest

from backend.application.chat import _is_simple_query


class TestIsSimpleQuery:
    """测试 _is_simple_query 函数的路由判断。"""

    @pytest.mark.parametrize(
        "message",
        [
            "你好",
            "hello",
            "Hello",
            "HELLO",
            "hi",
            "hey",
            "嗨",
            "哈喽",
            "哈啰",
            "你好！",
            "hello?",
            "你好。",
        ],
    )
    def test_greetings(self, message):
        """问候语应被识别为简单查询。"""
        assert _is_simple_query(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "谢谢",
            "感谢",
            "thanks",
            "Thank you",
            "THANKS",
            "谢谢！",
            "thanks.",
        ],
    )
    def test_thanks(self, message):
        """感谢语应被识别为简单查询。"""
        assert _is_simple_query(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "好的",
            "好",
            "ok",
            "OK",
            "嗯",
            "知道了",
            "明白",
            "收到",
            "了解",
            "好的！",
            "ok.",
        ],
    )
    def test_acknowledgments(self, message):
        """简单确认应被识别为简单查询。"""
        assert _is_simple_query(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "再见",
            "拜拜",
            "bye",
            "Bye",
            "goodbye",
            "再见！",
            "bye.",
        ],
    )
    def test_farewells(self, message):
        """告别语应被识别为简单查询。"""
        assert _is_simple_query(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "你是谁",
            "你叫什么",
            "你能做什么",
            "介绍一下你自己",
            "你是哪个模型",
        ],
    )
    def test_identity_questions(self, message):
        """身份询问应被识别为简单查询。"""
        assert _is_simple_query(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "什么是机器学习",
            "帮我写一个Python脚本",
            "总结一下这个文档",
            "这个文件说了什么",
            "如何配置数据库",
            "请解释一下RAG",
            "有哪些最佳实践",
            "对比一下这两个方案",
            "为什么会出现这个错误",
            "怎么优化性能",
        ],
    )
    def test_complex_queries(self, message):
        """复杂查询不应被识别为简单查询。"""
        assert _is_simple_query(message) is False

    @pytest.mark.parametrize(
        "message",
        [
            "  你好  ",
            "\nhello\n",
            "\thi\t",
        ],
    )
    def test_whitespace_handling(self, message):
        """前后空白应被正确处理。"""
        assert _is_simple_query(message) is True

    def test_partial_match_not_matched(self):
        """部分匹配不应被识别为简单查询。"""
        assert _is_simple_query("你好世界") is False
        assert _is_simple_query("hello world") is False
        assert _is_simple_query("谢谢你帮我") is False
