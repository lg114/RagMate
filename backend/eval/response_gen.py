"""评估用响应生成：短超时 + 瞬时错误有限重试。"""
import hashlib
import logging
import re
import time
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("ragmate")

RAG_USER_PROMPT = """上下文：
{contexts}

问题：{question}

请基于上述上下文回答问题。"""

# OpenAI / Ollama / 网关常见瞬时失败码与关键字
_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
_TRANSIENT_MARKERS = (
    "error code: 408",
    "error code: 425",
    "error code: 429",
    "error code: 500",
    "error code: 502",
    "error code: 503",
    "error code: 504",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "temporarily unavailable",
    "service unavailable",
    "server error",
)


@lru_cache(maxsize=1)
def _load_eval_system_prompt() -> str:
    """从 prompts/researcher.md 加载评估用 system prompt，剥离工具调用相关段落。"""
    prompt_path = Path(__file__).resolve().parent.parent / "core" / "prompts" / "researcher.md"
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


def answer_prompt_fingerprint() -> dict:
    """生成侧 prompt 指纹，写入 raw meta，保证打分与生成配置可追溯。"""
    system = _load_eval_system_prompt()
    blob = system + "\n---\n" + RAG_USER_PROMPT
    return {
        "answer_system_prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest()[:16],
        "answer_user_prompt_sha256": hashlib.sha256(RAG_USER_PROMPT.encode("utf-8")).hexdigest()[:16],
        "answer_prompt_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
    }


def is_transient_llm_error(exc: BaseException) -> bool:
    """判断是否值得短等后重试（503/429/超时等）。"""
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    if status in _TRANSIENT_STATUS:
        return True
    text = str(exc).lower()
    return any(m in text for m in _TRANSIENT_MARKERS)


def _eval_llm():
    """评测专用 LLM：短超时、关闭 SDK 自动重试（重试由本模块管）。"""
    from backend.infrastructure.config import settings
    from backend.infrastructure.streaming_llm import create_llm

    return create_llm(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_API_BASE_URL,
        timeout=settings.EVAL_LLM_TIMEOUT,
        max_retries=0,
        streaming=False,
    )


def generate_response(question: str, contexts: list[str]) -> str:
    """Generate an answer using LLM based on retrieved contexts.

    Uses EVAL_LLM_TIMEOUT / EVAL_LLM_RETRIES so a hung 503 cannot stall
    the whole generate-only run for tens of minutes.
    """
    from backend.infrastructure.config import settings

    context_text = "\n\n---\n\n".join(contexts) if contexts else "（未找到相关上下文）"
    messages = [
        SystemMessage(content=_load_eval_system_prompt()),
        HumanMessage(content=RAG_USER_PROMPT.format(contexts=context_text, question=question)),
    ]

    attempts = 1 + max(0, int(settings.EVAL_LLM_RETRIES))
    backoff = float(settings.EVAL_LLM_RETRY_BACKOFF)
    last_exc: BaseException | None = None
    llm = _eval_llm()

    for attempt in range(1, attempts + 1):
        try:
            response = llm.invoke(messages)
            content = getattr(response, "content", None)
            if not content or not str(content).strip():
                raise RuntimeError("LLM returned empty content")
            return str(content)
        except Exception as e:
            last_exc = e
            if attempt >= attempts or not is_transient_llm_error(e):
                raise
            wait = backoff * attempt
            logger.warning(
                "Transient LLM error on generate attempt %d/%d: %s; retry in %.1fs",
                attempt, attempts, e, wait,
            )
            time.sleep(wait)

    assert last_exc is not None
    raise last_exc
