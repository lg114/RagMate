import logging
import os
import warnings
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

import torch

from config import settings
from streaming_llm import create_llm


@lru_cache(maxsize=1)
def get_llm():
    """创建 LLM 实例（单例）"""
    return create_llm(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_API_BASE_URL,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """根据 EMBEDDING_PROVIDER 创建 Embeddings 实例（单例，只加载一次）"""
    if settings.EMBEDDING_PROVIDER == "openai":
        kwargs = {"model": settings.EMBEDDING_MODEL}
        if settings.EMBEDDING_API_KEY:
            kwargs["api_key"] = settings.EMBEDDING_API_KEY
        if settings.EMBEDDING_API_BASE_URL:
            kwargs["base_url"] = settings.EMBEDDING_API_BASE_URL
        return OpenAIEmbeddings(**kwargs)
    elif settings.EMBEDDING_PROVIDER == "huggingface":
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
