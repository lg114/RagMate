import logging
from functools import lru_cache

from backend.infrastructure.config import settings
from backend.infrastructure.streaming_llm import create_llm


@lru_cache(maxsize=1)
def get_llm():
    """创建 LLM 实例（单例）"""
    if not (settings.LLM_MODEL or "").strip() or not (settings.LLM_API_KEY or "").strip():
        from backend.domain.errors import ServiceUnavailableError
        raise ServiceUnavailableError("LLM model or API key is not configured. Please configure them in Settings.")
    return create_llm(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_API_BASE_URL,
    )
