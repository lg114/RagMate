from pathlib import Path

from langchain_core.tools import tool

from deepagents import create_deep_agent

from model_factory import get_llm
from retriever import retrieve

# ── System Prompt ───────────────────────────────────────────────────────────
def _load_system_prompt() -> str:
    return (Path(__file__).parent / "prompts" / "researcher.md").read_text(encoding="utf-8")


# ── Tool ────────────────────────────────────────────────────────────────────
@tool
def retrieval_tool(query: str) -> str:
    """检索相关文档片段来回答用户问题。输入是用户的问题，返回相关文档内容。"""
    from config import settings
    from errors import ValidationError

    try:
        results = retrieve(query, k=settings.FINAL_CONTEXT_K)
    except ValidationError:
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
_agent = None


def get_agent():
    """延迟创建 Deep Agent 实例（首次调用时才初始化 LLM 连接）"""
    global _agent
    if _agent is None:
        _agent = create_deep_agent(
            model=get_llm(),
            tools=[retrieval_tool],
            system_prompt=_load_system_prompt(),
        )
    return _agent


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
    return get_agent().invoke(
        {"messages": messages},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,
        },
    )


def run_agent_streaming(messages: list[dict], thread_id: str = "default"):
    """流式运行 agent，只返回最终回复。过滤掉 agent thinking 和 tool_call 参数。"""
    from langchain_core.messages import AIMessageChunk, ToolMessage

    can_yield = False

    for msg_chunk, _ in get_agent().stream(
        {"messages": messages},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,
        },
        stream_mode="messages",
    ):
        if isinstance(msg_chunk, ToolMessage):
            can_yield = True
            continue
        if isinstance(msg_chunk, AIMessageChunk):
            if getattr(msg_chunk, "tool_calls", None):
                can_yield = False
                continue
            if can_yield:
                text = extract_text_content(getattr(msg_chunk, "content", ""))
                if text:
                    yield text
