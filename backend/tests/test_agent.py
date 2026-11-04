"""测试 agent.py 的 Deep Agent 相关逻辑。"""
import sys

import pytest
from unittest.mock import MagicMock, patch

from backend.core.agent import (
    extract_text_content,
    _prepare_session,
    pop_retrieval_data,
    get_confidence,
    _contexts,
    _contexts_lock,
)


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


class TestSessionContext:
    """测试 session 上下文管理。"""

    def setup_method(self):
        """每个测试前清理上下文。"""
        with _contexts_lock:
            _contexts.clear()

    def teardown_method(self):
        """每个测试后清理上下文。"""
        with _contexts_lock:
            _contexts.clear()

    def test_prepare_session(self):
        """_prepare_session 应初始化 session 上下文。"""
        messages = [{"role": "user", "content": "Hello"}]
        _prepare_session("test-session", messages)

        assert "test-session" in _contexts
        ctx = _contexts["test-session"]
        assert ctx["messages"] == messages
        assert ctx["metrics"] == []
        assert ctx["texts"] == []

    def test_pop_retrieval_data(self):
        """pop_retrieval_data 应返回并清除 session 数据。"""
        messages = [{"role": "user", "content": "Hello"}]
        _prepare_session("test-session", messages)

        # 添加一些检索数据
        _contexts["test-session"]["metrics"] = [{"top_score": 0.8, "returned": 3}]
        _contexts["test-session"]["texts"] = ["text1", "text2"]

        # 弹出数据
        data = pop_retrieval_data("test-session")

        assert data is not None
        assert data["metrics"] == [{"top_score": 0.8, "returned": 3}]
        assert data["texts"] == ["text1", "text2"]
        assert "test-session" not in _contexts

    def test_pop_retrieval_data_nonexistent(self):
        """弹出不存在的 session 应返回 None。"""
        data = pop_retrieval_data("nonexistent")
        assert data is None

    def test_multiple_sessions(self):
        """多个 session 应独立管理。"""
        _prepare_session("session1", [{"role": "user", "content": "Hello"}])
        _prepare_session("session2", [{"role": "user", "content": "Hi"}])

        assert "session1" in _contexts
        assert "session2" in _contexts
        assert _contexts["session1"]["messages"][0]["content"] == "Hello"
        assert _contexts["session2"]["messages"][0]["content"] == "Hi"


class TestGetConfidence:
    """测试 get_confidence 函数。"""

    def setup_method(self):
        """每个测试前清理上下文。"""
        with _contexts_lock:
            _contexts.clear()

    def teardown_method(self):
        """每个测试后清理上下文。"""
        with _contexts_lock:
            _contexts.clear()

    def test_confidence_no_context(self):
        """没有上下文应返回 None。"""
        assert get_confidence("nonexistent") is None

    def test_confidence_no_metrics(self):
        """没有检索指标应返回 None。"""
        _prepare_session("test-session", [])
        assert get_confidence("test-session") is None

    def test_confidence_with_metrics(self):
        """有检索指标应返回置信度。"""
        _prepare_session("test-session", [])
        _contexts["test-session"]["metrics"] = [
            {"top_score": 0.8, "returned": 5},
        ]

        result = get_confidence("test-session")
        assert result is not None
        assert result["level"] == "high"
        assert result["score"] == 0.8
        assert result["chunks"] == 5

    def test_confidence_multiple_metrics(self):
        """多个检索指标应使用最高分。"""
        _prepare_session("test-session", [])
        _contexts["test-session"]["metrics"] = [
            {"top_score": 0.5, "returned": 2},
            {"top_score": 0.9, "returned": 4},
            {"top_score": 0.6, "returned": 3},
        ]

        result = get_confidence("test-session")
        assert result is not None
        assert result["score"] == 0.9
        assert result["chunks"] == 9


class TestRetrievalToolLimits:
    """测试检索工具的调用限制。"""

    def test_max_tool_retrievals_constant(self):
        """MAX_TOOL_RETRIEVALS 应定义为 8。"""
        from backend.core.agent import MAX_TOOL_RETRIEVALS
        assert MAX_TOOL_RETRIEVALS == 8


class TestRunAgentStreaming:
    """工具规划文本不能拼进最终回复。"""

    def setup_method(self):
        with _contexts_lock:
            _contexts.clear()

    def teardown_method(self):
        with _contexts_lock:
            _contexts.clear()

    def test_direct_answer_is_kept(self, monkeypatch):
        from langchain_core.messages import AIMessageChunk

        from backend.core.agent import run_agent_streaming

        agent = MagicMock()
        agent.stream.return_value = [(AIMessageChunk(content="直接回答"), {})]
        monkeypatch.setattr("backend.core.agent.get_agent", lambda: agent)

        assert "".join(run_agent_streaming([{"role": "user", "content": "hi"}], "sid")) == "直接回答"

    def test_tool_planning_text_is_dropped(self, monkeypatch):
        from langchain_core.messages import AIMessageChunk, ToolMessage

        from backend.core.agent import run_agent_streaming

        planning = AIMessageChunk(content="我先检索一下。")
        call = AIMessageChunk(
            content="检索参数",
            tool_call_chunks=[{"name": "retrieval_tool", "args": "", "id": "c1", "index": 0, "type": "tool_call_chunk"}],
        )
        tool = ToolMessage(content="未找到相关文档", tool_call_id="c1")
        final = AIMessageChunk(content="没有找到足够相关的资料。")
        agent = MagicMock()
        agent.stream.return_value = [(planning, {}), (call, {}), (tool, {}), (final, {})]
        monkeypatch.setattr("backend.core.agent.get_agent", lambda: agent)

        text = "".join(run_agent_streaming([{"role": "user", "content": "问题"}], "sid"))
        assert text == "没有找到足够相关的资料。"
        assert "检索" not in text


class TestGetAgentWithoutDeepagents:
    """deepagents 未装时，get_agent 应报出指向 agents extra 的明确错误，而不是 ModuleNotFoundError。"""

    def test_missing_deepagents_raises_helpful_error(self, monkeypatch):
        import backend.core.agent as agent_mod

        monkeypatch.setattr(agent_mod, "_agent", None)
        monkeypatch.setitem(sys.modules, "deepagents", None)

        with pytest.raises(ImportError, match=r"\[agents\]"):
            agent_mod.get_agent()
