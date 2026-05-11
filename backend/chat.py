import asyncio
import uuid

from langchain_core.messages import AIMessage

from agent import run_agent, run_agent_streaming
from database import async_session
from models import ChatHistory
from redis_client import load_session, save_session


def extract_text(response: dict) -> str:
    """从 deepagents 的 AIMessage 中提取最终回复文本。

    AIMessage.content 有两种形式：
    - str: 直接返回
    - list[dict]: 多块内容，只取 type=="text" 的块拼接
    """
    messages = response.get("messages", [])
    if not messages:
        return "没有收到回复"

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage):
        return str(last_msg)

    content = last_msg.content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = [block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text" and "text" in block]
        return "\n".join(parts) if parts else ""

    return str(content)


async def _persist_messages(session_id: str, messages: list[tuple[str, str]]):
    """批量持久化消息到 PostgreSQL（role, content pairs）。"""
    async with async_session() as session:
        session.add_all(
            ChatHistory(session_id=session_id, role=role, content=content)
            for role, content in messages
        )
        await session.commit()


AGENT_TIMEOUT = 120  # Agent 调用超时（秒）


# 错误提示前缀集合，用于判断 response_text 是否为错误而非正常回复
_ERROR_PREFIXES = (
    "请求超时",
    "请求被取消",
    "API 认证失败",
    "连接超时",
    "处理请求时出错",
)


def _is_error_response(text: str) -> bool:
    return text.startswith(_ERROR_PREFIXES)


def _classify_error(e: Exception) -> str:
    """根据异常类型和消息内容返回用户友好的错误提示"""
    msg = str(e).lower()

    if isinstance(e, asyncio.TimeoutError):
        return "请求超时，请稍后重试"
    if isinstance(e, asyncio.CancelledError):
        return "请求被取消"
    if any(k in msg for k in ("rate_limit", "429", "rate limit")):
        return "请求频率超限，请稍后重试"
    if any(k in msg for k in ("auth", "401", "api key", "unauthorized")):
        return "API 认证失败，请检查 API Key 配置"
    if any(k in msg for k in ("timeout", "timed out", "connection")):
        return "连接超时，请检查网络或 LLM 服务状态"
    if any(k in msg for k in ("insufficient_quota", "quota")):
        return "API 配额不足，请检查账户余额"

    return f"处理请求时出错: {msg}"


async def chat(message: str, session_id: str | None = None) -> dict:
    """处理用户聊天消息，支持多轮对话。返回 {"response": str, "session_id": str}"""
    if not session_id:
        session_id = uuid.uuid4().hex

    # 1. 加载会话历史
    history = await load_session(session_id)

    # 2. 追加用户消息
    history.append({"role": "user", "content": message})

    # 3. 调用 Agent（同步，放到线程中执行，带超时）
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(run_agent, history, session_id),
            timeout=AGENT_TIMEOUT,
        )
        response_text = extract_text(result)
    except asyncio.TimeoutError:
        response_text = "请求超时，请稍后重试"
    except asyncio.CancelledError:
        raise  # 上层取消，不吞掉
    except Exception as e:
        response_text = _classify_error(e)

    # 4. 追加 assistant 消息 + 持久化（只有正常回复才写入，错误提示直接丢弃）
    is_error = _is_error_response(response_text)
    if not is_error:
        history.append({"role": "assistant", "content": response_text})
        await save_session(session_id, history)

    msgs = [("user", message)]
    if not is_error:
        msgs.append(("assistant", response_text))
    await _persist_messages(session_id, msgs)

    return {"response": response_text, "session_id": session_id}


_SENTINEL = object()


async def chat_stream(message: str, session_id: str | None = None):
    """流式聊天，逐 token yield。返回格式: {"token": str} 或 {"done": True, "session_id": str}"""
    if not session_id:
        session_id = uuid.uuid4().hex

    history = await load_session(session_id)
    history.append({"role": "user", "content": message})

    queue: asyncio.Queue = asyncio.Queue()
    full_response: list[str] = []

    def _run():
        try:
            for token in run_agent_streaming(history, session_id):
                full_response.append(token)
                queue.put_nowait(token)
        except Exception as e:
            queue.put_nowait(_classify_error(e))
        finally:
            queue.put_nowait(_SENTINEL)

    task = asyncio.get_running_loop().run_in_executor(None, _run)

    try:
        while True:
            token = await queue.get()
            if token is _SENTINEL:
                break
            yield {"token": token}
    except asyncio.CancelledError:
        task.cancel()
        raise

    response_text = "".join(full_response)
    is_error = _is_error_response(response_text)

    if not is_error:
        history.append({"role": "assistant", "content": response_text})
        await save_session(session_id, history)

    msgs = [("user", message)]
    if not is_error:
        msgs.append(("assistant", response_text))
    await _persist_messages(session_id, msgs)

    yield {"done": True, "session_id": session_id}
