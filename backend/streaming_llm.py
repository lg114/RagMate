"""基于 langchain-openai ChatOpenAI 的 LLM 工厂，兼容各厂商自定义字段。"""

import openai
import warnings
from typing import Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _handle_openai_bad_request, _handle_openai_api_error


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
        # 调试：打印发给 API 的 assistant 消息
        import sys, logging
        for i, m in enumerate(payload.get("messages", [])):
            if m.get("role") == "assistant":
                has_tc = "tool_calls" in m
                content_preview = str(m.get("content", ""))[:80]
                logging.getLogger("ragmate").info(f"API msg[{i}] assistant: has_tool_calls={has_tc}, content='{content_preview}'")
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

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """重写流式处理，从原始 chunks 中捕获 reasoning_content。"""
        self._ensure_sync_client_available()
        kwargs["stream"] = True
        stream_usage = self._should_stream_usage(None, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        base_generation_info = {}
        response = None

        try:
            if "response_format" in payload:
                payload.pop("stream")
                response_stream = self.root_client.beta.chat.completions.stream(**payload)
                context_manager = response_stream
            else:
                response = self.client.create(**payload)
                context_manager = response

            with context_manager as stream:
                is_first_chunk = True
                for chunk in stream:
                    if not isinstance(chunk, dict):
                        chunk = chunk.model_dump()

                    # 从原始 chunk 中提取 reasoning_content
                    reasoning_content = ""
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("delta"):
                        reasoning_content = choices[0]["delta"].get("reasoning_content", "") or ""

                    generation_chunk = self._convert_chunk_to_generation_chunk(
                        chunk,
                        default_chunk_class,
                        base_generation_info if is_first_chunk else {},
                    )
                    if generation_chunk is None:
                        continue
                    default_chunk_class = generation_chunk.message.__class__

                    # 将 reasoning_content 注入到 chunk
                    if reasoning_content and isinstance(generation_chunk.message, AIMessageChunk):
                        generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning_content

                    logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                            logprobs=logprobs,
                        )
                    is_first_chunk = False
                    yield generation_chunk

        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)

        if hasattr(response, "get_final_completion") and "response_format" in payload:
            final_completion = response.get_final_completion()
            generation_chunk = self._get_generation_chunk_from_completion(final_completion)
            if run_manager:
                run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
            yield generation_chunk


def create_llm(model: str, api_key: str, base_url: str, temperature: float = 0.01) -> ChatOpenAICompatible:
    """创建 LLM 实例，支持任意 OpenAI 兼容 API。"""
    return ChatOpenAICompatible(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        streaming=True,
    )
