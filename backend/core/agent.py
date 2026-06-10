import threading
from pathlib import Path

from langchain_core.tools import tool

from deepagents import create_deep_agent

from backend.infrastructure.model_factory import get_llm
from backend.core.retriever import retrieve

# ── 跨线程上下文存储 ─────────────────────────────────────────────────────────
# retrieval_tool 在 LangGraph 线程池中执行，与 run_agent 调用者不在同一线程，
# 因此不能用 threading.local()。使用模块级 dict（以 session_id 为 key）做跨线程通信。
_contexts: dict[str, dict] = {}
_contexts_lock = threading.Lock()
_current_session_id: str | None = None  # 每次 agent 调用前设置


def _prepare_session(session_id: str, messages: list[dict]):
    """agent 调用前，初始化该 session 的上下文。"""
    global _current_session_id
    _current_session_id = session_id
    with _contexts_lock:
        _contexts[session_id] = {"messages": messages, "metrics": [], "texts": []}


def pop_retrieval_data(session_id: str) -> dict | None:
    """agent 调用后，取出并清除该 session 的检索数据。"""
    with _contexts_lock:
        return _contexts.pop(session_id, None)


# ── System Prompt ───────────────────────────────────────────────────────────
def _load_system_prompt() -> str:
    return (Path(__file__).parent / "prompts" / "researcher.md").read_text(encoding="utf-8")


# ── Tool ────────────────────────────────────────────────────────────────────
MAX_TOOL_RETRIEVALS = 8  # 单次 agent 调用中 retrieval_tool 的最大调用次数

_retrieval_counts: dict[str, int] = {}  # session_id → count
_counts_lock = threading.Lock()


@tool
def retrieval_tool(query: str) -> str:
    """检索相关文档片段来回答用户问题。输入是用户的问题，返回相关文档内容。"""
    from backend.infrastructure.config import settings
    from backend.domain.errors import AppError

    sid = _current_session_id or "default"

    # 检索次数限制
    with _counts_lock:
        count = _retrieval_counts.get(sid, 0) + 1
        _retrieval_counts[sid] = count
    if count > MAX_TOOL_RETRIEVALS:
        return "已达到最大检索次数，请基于已有信息回答用户。"

    # 查询上下文化：用对话历史改写追问为自包含 query
    ctx = _contexts.get(sid)
    if ctx:
        history = ctx.get("messages", [])
        if len(history) > 1:
            from backend.core.retriever import contextualize_query
            query = contextualize_query(query, history)

    try:
        results = retrieve(query, k=settings.FINAL_CONTEXT_K)
    except AppError:
        return "检索服务暂时不可用，请稍后重试"
    if not results:
        return "未找到相关文档"

    # 收集检索质量指标和结果（用于 confidence 和 faithfulness check）
    from backend.core.retriever import get_retrieval_metrics
    metrics = get_retrieval_metrics()
    if ctx and metrics:
        ctx["metrics"].append(metrics)
    if ctx:
        ctx["texts"].extend(r.get("text", "") for r in results)

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
_agent_lock = threading.Lock()


def get_agent():
    """获取 Deep Agent 单例。首次调用时创建。"""
    global _agent
    if _agent is not None:
        return _agent
    with _agent_lock:
        if _agent is None:
            _agent = create_deep_agent(
                model=get_llm(),
                tools=[retrieval_tool],
                system_prompt=_load_system_prompt(),
            )
        return _agent


def get_confidence(session_id: str) -> dict | None:
    """根据检索指标计算置信度。"""
    ctx = _contexts.get(session_id)
    if not ctx:
        return None
    metrics_list = ctx.get("metrics", [])
    if not metrics_list:
        return None

    best_score = max(m["top_score"] for m in metrics_list)
    total_returned = sum(m["returned"] for m in metrics_list)

    if best_score >= 0.7 and total_returned >= 3:
        level = "high"
    elif best_score >= 0.4 and total_returned >= 1:
        level = "medium"
    else:
        level = "low"

    return {"level": level, "score": best_score, "chunks": total_returned}


def clear_agent_cache():
    """清除 Agent 和 LLM 缓存，下次调用时重新创建。"""
    global _agent
    with _agent_lock:
        _agent = None
    from backend.infrastructure.model_factory import clear_llm_cache
    clear_llm_cache()


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
    from backend.infrastructure.config import settings
    _prepare_session(thread_id, messages)
    try:
        return get_agent().invoke(
            {"messages": messages},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": settings.AGENT_RECURSION_LIMIT,
            },
        )
    finally:
        with _counts_lock:
            _retrieval_counts.pop(thread_id, None)


def run_agent_streaming(messages: list[dict], thread_id: str = "default"):
    """流式运行 agent，只返回最终回复。过滤掉 agent thinking 和 tool_call 参数。

    - 直接回答（无工具调用）：正常流式输出
    - 工具调用后回答：丢弃中间过程，只输出工具调用后的最终回复
    """
    from langchain_core.messages import AIMessageChunk, ToolMessage

    from backend.infrastructure.config import settings
    _prepare_session(thread_id, messages)
    try:
        for msg_chunk, _ in get_agent().stream(
            {"messages": messages},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": settings.AGENT_RECURSION_LIMIT,
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
    finally:
        with _counts_lock:
            _retrieval_counts.pop(thread_id, None)
