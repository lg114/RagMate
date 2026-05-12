"""支持真正 token 级流式的 LiteLLM Chat Model。"""

import json
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import litellm
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr


def _convert_messages(messages: list[BaseMessage]) -> list[dict]:
    """将 LangChain 消息转为 LiteLLM 格式。"""
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            d = {"role": "user", "content": msg.content}
        elif isinstance(msg, SystemMessage):
            d = {"role": "system", "content": msg.content}
        elif isinstance(msg, ToolMessage):
            d = {"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id}
        elif isinstance(msg, AIMessage):
            d = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {"id": tc["id"], "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}, "type": "function"}
                    for tc in msg.tool_calls
                ]
        else:
            d = {"role": "user", "content": str(msg.content)}
        result.append(d)
    return result


def _parse_response_message(msg) -> AIMessage:
    """将 LiteLLM 响应消息转为 LangChain AIMessage，支持 tool_calls。"""
    content = msg.content or ""
    tool_calls = []
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append({"id": tc.id, "name": tc.function.name, "args": args})
    return AIMessage(content=content, tool_calls=tool_calls)


class StreamingLiteLLM(BaseChatModel):
    """支持真正 token 级流式的 LiteLLM Chat Model。"""

    model: str
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.01
    request_timeout: float = 60
    _bound_tools: list[dict] | None = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "streaming-litellm"

    def bind_tools(self, tools, **kwargs):
        """绑定工具，返回新的模型实例。"""
        new = self.model_copy()
        tool_defs = []
        for t in tools:
            schema = t.args_schema.model_json_schema() if hasattr(t, "args_schema") and t.args_schema else {"type": "object", "properties": {}}
            tool_defs.append({
                "type": "function",
                "function": {"name": t.name, "description": t.description or "", "parameters": schema},
            })
        new._bound_tools = tool_defs
        return new

    def _build_call_kwargs(self, messages, stop=None, stream=False):
        kwargs = dict(model=self.model, messages=_convert_messages(messages), temperature=self.temperature, stream=stream)
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if stop:
            kwargs["stop"] = stop
        if self._bound_tools:
            kwargs["tools"] = self._bound_tools
        return kwargs

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = litellm.completion(**self._build_call_kwargs(messages, stop))
        ai_msg = _parse_response_message(response.choices[0].message)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        response = litellm.completion(**self._build_call_kwargs(messages, stop, stream=True))
        for chunk in response:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            tool_calls = None
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                tool_calls = []
                for tc in delta.tool_calls:
                    args_raw = tc.function.arguments if tc.function else ""
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) and args_raw else args_raw if isinstance(args_raw, dict) else {}
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append({"id": tc.id or "", "name": tc.function.name if tc.function else "", "args": args})
            chunk_msg = AIMessageChunk(content=content, tool_calls=tool_calls or [])
            yield ChatGenerationChunk(message=chunk_msg)
            if run_manager and content:
                run_manager.on_llm_new_token(content)

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}
