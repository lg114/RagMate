import logging
import os
import warnings
from functools import lru_cache

from langchain_litellm import ChatLiteLLM
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

import torch

from config import settings


class ThinkingChatLiteLLM(ChatLiteLLM):
    """支持 thinking 模式的 ChatLiteLLM 子类。

    部分模型（如 DeepSeek V4）的 thinking 模式要求：
    1. content 中不能有 thinking 类型块（API 会拒绝 unknown variant `thinking`）
    2. reasoning_content 必须通过顶层字段传回（否则 API 报错要求传回）

    langchain_litellm 的 _convert_message_to_dict 不会处理 thinking 块和
    reasoning_content，所以在此子类中覆盖 _create_message_dicts 来修复。
    """

    def _create_message_dicts(self, messages, stop=None):
        message_dicts, params = super()._create_message_dicts(messages, stop)

        for msg, msg_dict in zip(messages, message_dicts):
            # 1. 从 content 中移除 thinking 块
            content = msg_dict.get("content")
            if isinstance(content, list):
                text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
                msg_dict["content"] = "".join(b.get("text", "") for b in text_blocks) if text_blocks else ""

            # 2. 将 additional_kwargs.reasoning_content 提升为顶层字段
            reasoning = getattr(msg, "additional_kwargs", {}).get("reasoning_content")
            if reasoning:
                msg_dict["reasoning_content"] = reasoning

        return message_dicts, params


@lru_cache(maxsize=1)
def get_llm():
    """根据 LLM_PROVIDER 创建 LLM 实例，支持任意 LiteLLM 识别的 provider/model"""
    model_string = f"{settings.LLM_PROVIDER}/{settings.LLM_MODEL}"
    kwargs = dict(model=model_string, temperature=0, request_timeout=60)
    if settings.LLM_API_BASE_URL:
        kwargs["api_base"] = settings.LLM_API_BASE_URL
    if settings.LLM_API_KEY:
        kwargs["api_key"] = settings.LLM_API_KEY

    # thinking 模型使用子类来自动过滤 thinking 内容块、回传 reasoning_content
    if "deepseek" in settings.LLM_MODEL.lower():
        return ThinkingChatLiteLLM(**kwargs)
    return ChatLiteLLM(**kwargs)


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
        if device:
            kwargs["model_kwargs"] = {"device": device}
        if settings.EMBEDDING_NORMALIZE:
            kwargs["encode_kwargs"] = {"normalize_embeddings": True}
        if settings.HF_TOKEN:
            kwargs["model_kwargs"] = {**kwargs.get("model_kwargs", {}), "token": settings.HF_TOKEN}

        # 抑制 HuggingFace 模型加载时的无害警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
            warnings.filterwarnings("ignore", message=".*unauthenticated.*")
            return HuggingFaceEmbeddings(**kwargs)
    raise NotImplementedError(f"Unsupported EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")
