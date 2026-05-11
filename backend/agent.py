from langchain_core.tools import tool

from deepagents import create_deep_agent

from model_factory import get_llm
from retriever import retrieve


AGENT_SYSTEM_PROMPT = """你是一个专业的深度研究助手，代号 Researcher。

## 核心原则
- 使用 retrieval_tool 回答一切需要信息的提问
- 完全日常的闲聊无需检索，可直接回复
- 检索结果为空时直接说"没有找到相关资料"
- 引用时必须标注来源：使用 【文件名】 格式
- 不确定时明确说"我不确定"，禁止编造

## 检索策略
1. 首次检索：提取问题的核心关键词，避免冗余修饰词
2. 若结果不相关：识别专有名词/术语，替换后再检索
3. 最多检索 3 次（含首次），不要无限重试
4. 检索失败时直接告知用户，不做降级尝试

## 回答格式
1. 先给结论（一句话概括）
2. 分点说明，每个要点标注来源为 【文件名】
3. 若多个文档存在矛盾结论：说明"不同来源显示：A文档指出...，B文档指出..."
4. 无法回答时说明"根据现有资料无法回答"，并给出建议（如联系对应部门）

## 多轮对话规则
- 新问题与之前相关时，简要引用上下文
- 用户明显切换话题时，不强行关联历史
- 早期事实与当前检索结果矛盾时，以最新检索为准"""


@tool
def retrieval_tool(query: str) -> str:
    """检索相关文档片段来回答用户问题。输入是用户的问题，返回相关文档内容。"""
    results = retrieve(query, k=3)
    if not results:
        return "未找到相关文档"
    parts = []
    for r in results:
        source = r.get("source", "unknown")
        page = r.get("page")
        chunk_idx = r.get("chunk_index")
        loc = f"【{source}】"
        if page is not None:
            loc += f" 第{page}页"
        if chunk_idx is not None:
            loc += f" 片段{chunk_idx}"
        parts.append(f"{loc}\n{r['text']}")
    return "\n\n---\n\n".join(parts)


_agent = None


def get_agent():
    """延迟创建 Deep Agent 实例（首次调用时才初始化 LLM 连接）"""
    global _agent
    if _agent is None:
        _agent = create_deep_agent(
            model=get_llm(),
            tools=[retrieval_tool],
            system_prompt=AGENT_SYSTEM_PROMPT,
        )
    return _agent


def run_agent(messages: list[dict], thread_id: str = "default") -> dict:
    """运行 agent，支持多轮对话。messages 格式: [{"role": "user", "content": "..."}, ...]"""
    return get_agent().invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": thread_id}},
    )


def _extract_text_content(content) -> str:
    """从 AIMessage.content 中提取文本，支持 str 和 list 两种格式。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text" and "text" in block]
        return "\n".join(parts)
    return ""


def run_agent_streaming(messages: list[dict], thread_id: str = "default"):
    """流式运行 agent，逐 token yield。同步生成器，在线程池中调用。"""
    for msg_chunk, _metadata in get_agent().stream(
        {"messages": messages},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="messages",
    ):
        text = _extract_text_content(getattr(msg_chunk, "content", ""))
        if text:
            yield text
