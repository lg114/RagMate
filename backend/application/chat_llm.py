"""直接调 LLM 的几处：回答生成（同步/流式）。"""


def extract_text_content(content) -> str:
    """从 AIMessage.content 中提取文本，支持 str 和 list 两种格式。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text" and "text" in block]
        return "\n".join(parts)
    return ""


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
