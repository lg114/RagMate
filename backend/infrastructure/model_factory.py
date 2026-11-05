import logging
import warnings
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from backend.infrastructure.config import (
    LOCAL_EMBEDDING_PROVIDERS,
    REMOTE_EMBEDDING_PROVIDERS,
    settings,
)
from backend.infrastructure.streaming_llm import create_llm


@lru_cache(maxsize=1)
def get_llm():
    """创建 LLM 实例（单例）"""
    return create_llm(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_API_BASE_URL,
    )


def clear_llm_cache():
    """清除 LLM 单例缓存，下次调用时重新创建。"""
    get_llm.cache_clear()


@lru_cache(maxsize=1)
def get_embeddings():
    """根据 EMBEDDING_PROVIDER 创建 Embeddings 实例（单例，只加载一次）。

    remote / openai 使用 OpenAI 兼容 HTTP，本函数不会 import torch。
    huggingface / local / flagembedding 才在本进程加载 HuggingFace 权重。
    入库和查询的 dense 编码走 encoding.py，不经过这里。
    """
    if settings.EMBEDDING_PROVIDER in REMOTE_EMBEDDING_PROVIDERS:
        kwargs = {"model": settings.EMBEDDING_MODEL}
        if settings.EMBEDDING_API_KEY:
            kwargs["api_key"] = settings.EMBEDDING_API_KEY
        if settings.EMBEDDING_API_BASE_URL:
            kwargs["base_url"] = settings.EMBEDDING_API_BASE_URL
        return OpenAIEmbeddings(**kwargs)
    if settings.EMBEDDING_PROVIDER in LOCAL_EMBEDDING_PROVIDERS:
        # 延迟到真正走本地嵌入时再加载，避免默认远程路径把 torch 拉进 API 进程。
        try:
            import torch
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Local embedding models need the 'local' extra: "
                'pip install -e ".[local]"'
            ) from exc

        kwargs = {"model": settings.EMBEDDING_MODEL}
        device = settings.EMBEDDING_DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logging.getLogger(__name__).warning("CUDA requested but not available, falling back to CPU")
        model_kw = {"trust_remote_code": True}
        if device:
            model_kw["device"] = device
        kwargs["model_kwargs"] = model_kw
        if settings.EMBEDDING_NORMALIZE:
            kwargs["encode_kwargs"] = {"normalize_embeddings": True}
        if settings.HF_TOKEN:
            kwargs["model_kwargs"]["token"] = settings.HF_TOKEN

        # 抑制 HuggingFace 模型加载时的无害警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
            warnings.filterwarnings("ignore", message=".*unauthenticated.*")
            return HuggingFaceEmbeddings(**kwargs)
    raise NotImplementedError(f"Unsupported EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")
