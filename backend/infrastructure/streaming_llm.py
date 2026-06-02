"""基于 langchain-openai ChatOpenAI 的 LLM 工厂，兼容各厂商自定义字段。"""

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI


class ChatOpenAICompatible(ChatOpenAI):
    """兼容各厂商自定义字段的 ChatOpenAI 子类。

    - 自动捕获 reasoning_content（DeepSeek 等 thinking mode）
    - 自动在后续请求中回传 reasoning_content
    - 通用设计：任何 additional_kwargs 中的自定义字段都会被自动处理
    """

    def _get_request_payload(self, input_: LanguageModelInput, *, stop=None, **kwargs) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        for i, msg in enumerate(messages):
            if isinstance(msg, (AIMessage, AIMessageChunk)) and msg.additional_kwargs:
                for key, value in msg.additional_kwargs.items():
                    if value and key not in ("refusal",) and i < len(payload.get("messages", [])):
                        payload["messages"][i][key] = value
        return payload

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        try:
            if hasattr(response, "choices") and response.choices:
                msg = response.choices[0].message
                if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                    result.generations[0].message.additional_kwargs["reasoning_content"] = msg.reasoning_content
        except (AttributeError, IndexError):
            pass
        return result

    def _convert_chunk_to_generation_chunk(
        self, chunk, default_chunk_class, base_generation_info
    ) -> ChatGenerationChunk | None:
        """重写 chunk 转换，从原始 delta 中捕获 reasoning_content。

        仅在父类转换结果之上注入 reasoning_content，其余逻辑完全继承。
        """
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info,
        )
        if generation_chunk is None:
            return None

        # 从原始 chunk 的 delta 中提取 reasoning_content 并注入
        choices = chunk.get("choices", [])
        if choices and choices[0].get("delta"):
            reasoning_content = choices[0]["delta"].get("reasoning_content", "") or ""
            if reasoning_content and isinstance(generation_chunk.message, AIMessageChunk):
                generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning_content

        return generation_chunk


def create_llm(model: str, api_key: str, base_url: str, temperature: float = None) -> ChatOpenAICompatible:
    """创建 LLM 实例，支持任意 OpenAI 兼容 API。"""
    from backend.infrastructure.config import settings
    if temperature is None:
        temperature = settings.LLM_TEMPERATURE
    return ChatOpenAICompatible(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        streaming=True,
    )
