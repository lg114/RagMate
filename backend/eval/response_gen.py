"""评估用响应生成。"""
import re
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

RAG_USER_PROMPT = """上下文：
{contexts}

问题：{question}

请基于上述上下文回答问题。"""


@lru_cache(maxsize=1)
def _load_eval_system_prompt() -> str:
    """从 prompts/researcher.md 加载评估用 system prompt，剥离工具调用相关段落。"""
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "researcher.md"
    if not prompt_path.exists():
        return "你是一个知识库问答助手。根据提供的上下文回答用户的问题。只基于提供的上下文回答，不要编造信息。"
    text = prompt_path.read_text(encoding="utf-8")
    sections_to_remove = [
        r"## 工具调用边界.*?(?=## |\Z)",
        r"## 检索与信息收集策略.*?(?=## |\Z)",
        r"## 核心工作流.*?(?=## |\Z)",
    ]
    for pattern in sections_to_remove:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def generate_response(question: str, contexts: list[str]) -> str:
    """Generate an answer using LLM based on retrieved contexts."""
    from model_factory import get_llm

    llm = get_llm()
    context_text = "\n\n---\n\n".join(contexts) if contexts else "（未找到相关上下文）"

    messages = [
        SystemMessage(content=_load_eval_system_prompt()),
        HumanMessage(content=RAG_USER_PROMPT.format(contexts=context_text, question=question)),
    ]

    response = llm.invoke(messages)
    return response.content
