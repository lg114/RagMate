"""直接调 LLM 的几处：简单查询路由、回答生成（同步/流式）。"""

import re
from pathlib import Path

from langchain_core.messages import AIMessage


def extract_text_content(content) -> str:
    """从 AIMessage.content 中提取文本，支持 str 和 list 两种格式。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text" and "text" in block]
        return "\n".join(parts)
    return ""


def _load_system_prompt() -> str:
    return (Path(__file__).resolve().parent.parent / "core" / "prompts" / "researcher.md").read_text(encoding="utf-8")


def extract_text(response: dict) -> str:
    """从 AIMessage 中提取最终回复文本。"""
    messages = response.get("messages", [])
    if not messages:
        return "没有收到回复"

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage):
        return str(last_msg)

    text = extract_text_content(last_msg.content)
    return text if text else str(last_msg.content)


# ── 查询路由：明显的非 RAG 查询，跳过 agent 直接回复 ─────────────────────────
_NON_RAG_RE = re.compile(
    r"^(?:"
    r"你好|hello|hi|hey|嗨|哈喽|哈啰"
    r"|谢谢|感谢|thanks|thank you"
    r"|你是谁|你叫什么|你能做什么|介绍一下你自己|你是哪个模型"
    r"|好的?|ok|嗯|知道了|明白|收到|了解"
    r"|再见|拜拜|bye|goodbye"
    r")\s*[!！?？.。]*$",
    re.IGNORECASE,
)


def _is_simple_query(message: str) -> bool:
    """判断是否为简单的非 RAG 查询（问候、感谢等）。"""
    return bool(_NON_RAG_RE.match(message.strip()))


def _direct_llm_reply(history: list[dict]) -> str:
    """直接调用 LLM 回复简单查询，不经过检索（同步）。"""
    from backend.infrastructure.model_factory import get_llm

    llm = get_llm()
    messages = [{"role": "system", "content": _load_system_prompt()}] + history
    response = llm.invoke(messages)
    return response.content


def _invoke_answer(messages: list[dict]) -> str:
    from backend.infrastructure.model_factory import get_llm

    response = get_llm().invoke(messages)
    text = extract_text_content(getattr(response, "content", ""))
    return text or "没有收到回复"


def _iter_answer_tokens(messages: list[dict]):
    """只产出回答正文。不包含 reasoning 字段，也不包含工具规划。"""
    from backend.infrastructure.model_factory import get_llm

    for chunk in get_llm().stream(messages):
        text = extract_text_content(getattr(chunk, "content", "") or "")
        if text:
            yield text
