import threading
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

from deepagents import create_deep_agent

from model_factory import get_llm
from retriever import retrieve

# ── System Prompt ───────────────────────────────────────────────────────────
def _load_system_prompt() -> str:
    return (Path(__file__).parent / "prompts" / "researcher.md").read_text(encoding="utf-8")


# ── Tool ────────────────────────────────────────────────────────────────────
MAX_TOOL_RETRIEVALS = 8  # 单次 agent 调用中 retrieval_tool 的最大调用次数

_tool_call_state = threading.local()


def _reset_tool_counter():
    """重置当前线程的工具调用计数器（每次 agent 调用前执行）。"""
    _tool_call_state.retrieval_count = 0


@tool
def retrieval_tool(query: str) -> str:
    """检索相关文档片段来回答用户问题。输入是用户的问题，返回相关文档内容。"""
    from config import settings
    from errors import AppError

    count = getattr(_tool_call_state, "retrieval_count", 0) + 1
    _tool_call_state.retrieval_count = count

    if count > MAX_TOOL_RETRIEVALS:
        return "已达到最大检索次数，请基于已有信息回答用户。"

    try:
        results = retrieve(query, k=settings.FINAL_CONTEXT_K)
    except AppError:
        return "检索服务暂时不可用，请稍后重试"
    if not results:
        return "未找到相关文档"

    parts = []
    for r in results:
        source = r.get("source", "unknown")
        page = r.get("page")
        chunk_idx = r.get("chunk_index")
        loc = f"【{source}】"
        if page is not None:
            loc += f" 第{page + 1}页"
        if chunk_idx is not None:
            loc += f" 片段{chunk_idx}"
        parts.append(f"{loc}\n{r['text']}")
    return "\n\n---\n\n".join(parts)


# ── Agent 实例 ──────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_agent():
    """延迟创建 Deep Agent 实例（单例，首次调用时才初始化 LLM 连接）"""
    return create_deep_agent(
        model=get_llm(),
        tools=[retrieval_tool],
        system_prompt=_load_system_prompt(),
    )


# ── 内部工具函数 ────────────────────────────────────────────────────────────
def extract_text_content(content) -> str:
    """从 AIMessage.content 中提取文本，支持 str 和 list 两种格式。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text" and "text" in block]
        return "\n".join(parts)
    return ""


# ── 公开 API ────────────────────────────────────────────────────────────────
def run_agent(messages: list[dict], thread_id: str = "default") -> dict:
    """运行 agent，支持多轮对话。messages 格式: [{"role": "user", "content": "..."}, ...]"""
    _reset_tool_counter()
    return get_agent().invoke(
        {"messages": messages},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,
        },
    )


def run_agent_streaming(messages: list[dict], thread_id: str = "default"):
    """流式运行 agent，只返回最终回复。过滤掉 agent thinking 和 tool_call 参数。

    - 直接回答（无工具调用）：正常流式输出
    - 工具调用后回答：丢弃中间过程，只输出工具调用后的最终回复
    """
    from langchain_core.messages import AIMessageChunk, ToolMessage

    _reset_tool_counter()
    for msg_chunk, _ in get_agent().stream(
        {"messages": messages},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,
        },
        stream_mode="messages",
    ):
        if isinstance(msg_chunk, ToolMessage):
            continue
        if isinstance(msg_chunk, AIMessageChunk):
            if getattr(msg_chunk, "tool_calls", None):
                continue
            text = extract_text_content(getattr(msg_chunk, "content", ""))
            if text:
                yield text

