import asyncio
import logging
import re
import time
import uuid

from langchain_core.messages import AIMessage
from sqlalchemy import delete, select

from backend.core.agent import run_agent, run_agent_streaming, extract_text_content
from backend.infrastructure.database import async_session
from backend.domain.models import ChatHistory
from backend.infrastructure.redis_client import load_session, save_session

logger = logging.getLogger("ragmate")


def extract_text(response: dict) -> str:
    """从 deepagents 的 AIMessage 中提取最终回复文本。"""
    messages = response.get("messages", [])
    if not messages:
        return "没有收到回复"

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage):
        return str(last_msg)

    text = extract_text_content(last_msg.content)
    return text if text else str(last_msg.content)


async def _persist_messages(session_id: str, messages: list[tuple[str, str]]):
    """批量持久化消息到 PostgreSQL（role, content pairs）。"""
    async with async_session() as session:
        session.add_all(
            ChatHistory(session_id=session_id, role=role, content=content)
            for role, content in messages
        )
        await session.commit()


async def _delete_last_persisted_turn(session_id: str):
    """删除 PostgreSQL 中最后一轮 user/assistant，用于重试或重新生成。"""
    async with async_session() as session:
        result = await session.execute(
            select(ChatHistory.id, ChatHistory.role)
            .where(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.created_at.desc(), ChatHistory.id.desc())
            .limit(2)
        )
        rows = result.fetchall()

        ids_to_delete = []
        if rows and rows[0].role == "assistant":
            ids_to_delete.append(rows[0].id)
            if len(rows) > 1 and rows[1].role == "user":
                ids_to_delete.append(rows[1].id)
        elif rows and rows[0].role == "user":
            ids_to_delete.append(rows[0].id)

        if ids_to_delete:
            await session.execute(delete(ChatHistory).where(ChatHistory.id.in_(ids_to_delete)))
            await session.commit()


AGENT_TIMEOUT = 120  # Agent 调用超时（秒）

# ── 查询路由：明显的非 RAG 查询，跳过 agent 直接回复 ─────────────────────────
_NON_RAG_RE = re.compile(
    r"^(?:"
    r"你好|hello|hi|hey|嗨|哈喽|哈啰"
    r"|谢谢|感谢|thanks|thank you"
    r"|你是谁|你叫什么|你能做什么|介绍一下你自己|你是哪个模型"
    r"|好的?|ok|嗯|知道了|明白|收到|了解"
    r"|再见|拜拜|bye|goodbye"
    r")\s*[!！?？.。]*$",
    re.IGNORECASE,
)


def _is_simple_query(message: str) -> bool:
    """判断是否为简单的非 RAG 查询（问候、感谢等）。"""
    return bool(_NON_RAG_RE.match(message.strip()))


def _direct_llm_reply(history: list[dict]) -> str:
    """直接调用 LLM 回复简单查询，不经过 agent（同步）。"""
    from backend.infrastructure.model_factory import get_llm
    from backend.core.agent import _load_system_prompt

    llm = get_llm()
    messages = [{"role": "system", "content": _load_system_prompt()}] + history
    response = llm.invoke(messages)
    return response.content


# 错误哨兵前缀，用于判断 response_text 是否为错误而非正常回复
# 使用 UUID 标记确保不会与 LLM 自然输出冲突
_ERROR_SENTINEL = "\x00ERR\x00"


_CITATION_RE = re.compile(r"【([^】]+\.(?:pdf|docx?|xlsx?|xls|txt|md))】", re.IGNORECASE)
_SOURCE_LINE_RE = re.compile(r"^\s*数据来源[:：].*$", re.MULTILINE)


def normalize_citations(text: str) -> str:
    """Move repeated file citations into one source line for readability."""
    matches = _CITATION_RE.findall(text)
    if len(matches) < 2:
        return text
    sources = list(dict.fromkeys(matches))

    cleaned = _SOURCE_LINE_RE.sub("", text)
    for source in sources:
        cleaned = cleaned.replace(f"【{source}】", "")

    cleaned = re.sub(r"[ \t]+([，。；：、,.!?！？])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    source_line = "数据来源：" + "、".join(f"【{source}】" for source in sources)
    return f"{cleaned}\n\n{source_line}" if cleaned else source_line


def _is_error_response(text: str) -> bool:
    return text.startswith(_ERROR_SENTINEL)


def _strip_error_sentinel(text: str) -> str:
    """移除错误哨兵前缀，返回用户友好的错误消息。"""
    if text.startswith(_ERROR_SENTINEL):
        return text[len(_ERROR_SENTINEL):]
    return text


def _classify_error(e: Exception) -> str:
    """根据异常类型和消息内容返回用户友好的错误提示（带哨兵前缀）"""
    msg = str(e).lower()

    if isinstance(e, asyncio.TimeoutError):
        return f"{_ERROR_SENTINEL}请求超时，请稍后重试"
    if isinstance(e, asyncio.CancelledError):
        return f"{_ERROR_SENTINEL}请求被取消"
    if any(k in msg for k in ("rate_limit", "429", "rate limit")):
        return f"{_ERROR_SENTINEL}请求频率超限，请稍后重试"
    if any(k in msg for k in ("auth", "401", "api key", "unauthorized")):
        return f"{_ERROR_SENTINEL}API 认证失败，请检查 API Key 配置"
    if any(k in msg for k in ("timeout", "timed out", "connection")):
        return f"{_ERROR_SENTINEL}连接超时，请检查网络或 LLM 服务状态"
    if any(k in msg for k in ("insufficient_quota", "quota")):
        return f"{_ERROR_SENTINEL}API 配额不足，请检查账户余额"

    logger.error("Unhandled error in chat", exc_info=True)
    return f"{_ERROR_SENTINEL}处理请求时出错，请稍后重试"


def _strip_last_turn(history: list[dict]) -> list[dict]:
    """移除历史中最后一轮 user/assistant（用于重试/重新生成）。"""
    if not history:
        return history

    trimmed = history[:]
    if trimmed[-1].get("role") == "assistant":
        trimmed.pop()
    if trimmed and trimmed[-1].get("role") == "user":
        trimmed.pop()
    return trimmed


async def chat(message: str, session_id: str | None = None, replace_last: bool = False) -> dict:
    """处理用户聊天消息，支持多轮对话。返回 {"response": str, "session_id": str}"""
    if not session_id:
        session_id = str(uuid.uuid4())

    # 1. 加载会话历史
    history = await load_session(session_id)

    # 2. 重试/重新生成时，移除旧的一轮对话
    if replace_last:
        history = _strip_last_turn(history)
        await _delete_last_persisted_turn(session_id)

    # 3. 追加用户消息
    history.append({"role": "user", "content": message})

    # 3.5 查询路由：简单查询跳过 agent
    from backend.infrastructure.config import settings
    if settings.QUERY_ROUTING_ENABLED and _is_simple_query(message):
        t0 = time.monotonic()
        try:
            response_text = await asyncio.to_thread(_direct_llm_reply, history)
            response_text = normalize_citations(response_text)
        except Exception as e:
            response_text = _classify_error(e)
        elapsed = time.monotonic() - t0

        is_error = _is_error_response(response_text)
        if is_error:
            logger.warning(f"chat route error: session={session_id[:8]} elapsed={elapsed:.1f}s err={_strip_error_sentinel(response_text)}")
        else:
            history.append({"role": "assistant", "content": response_text})
            logger.info(f"chat route ok: session={session_id[:8]} resp_len={len(response_text)} elapsed={elapsed:.1f}s")
        await save_session(session_id, history)

        msgs = [("user", message)]
        if not is_error:
            msgs.append(("assistant", response_text))
        await _persist_messages(session_id, msgs)
        return {"response": _strip_error_sentinel(response_text), "session_id": session_id}

    # 4. 调用 Agent（同步，放到线程中执行，带超时）
    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(run_agent, history, session_id),
            timeout=AGENT_TIMEOUT,
        )
        response_text = normalize_citations(extract_text(result))
    except asyncio.TimeoutError:
        response_text = f"{_ERROR_SENTINEL}请求超时，请稍后重试"
    except asyncio.CancelledError:
        raise  # 上层取消，不吞掉
    except Exception as e:
        response_text = _classify_error(e)
    elapsed = time.monotonic() - t0

    # 4. 追加 assistant 消息 + 持久化（错误提示不写入历史，但用户消息需要保留）
    is_error = _is_error_response(response_text)
    if is_error:
        logger.warning(f"chat error: session={session_id[:8]} msg_len={len(message)} elapsed={elapsed:.1f}s err={_strip_error_sentinel(response_text)}")
    else:
        history.append({"role": "assistant", "content": response_text})
        logger.info(f"chat ok: session={session_id[:8]} msg_len={len(message)} resp_len={len(response_text)} elapsed={elapsed:.1f}s")
    await save_session(session_id, history)

    msgs = [("user", message)]
    if not is_error:
        msgs.append(("assistant", response_text))
    await _persist_messages(session_id, msgs)

    return {"response": _strip_error_sentinel(response_text), "session_id": session_id}


_SENTINEL = object()


async def chat_stream(message: str, session_id: str | None = None, replace_last: bool = False):
    """流式聊天，逐 token yield。返回格式: {"token": str} 或 {"done": True, "session_id": str}"""
    if not session_id:
        session_id = str(uuid.uuid4())

    t0 = time.monotonic()
    history = await load_session(session_id)

    if replace_last:
        history = _strip_last_turn(history)
        await _delete_last_persisted_turn(session_id)

    history.append({"role": "user", "content": message})

    # 查询路由：简单查询跳过 agent
    from backend.infrastructure.config import settings
    if settings.QUERY_ROUTING_ENABLED and _is_simple_query(message):
        t0 = time.monotonic()
        try:
            response_text = await asyncio.to_thread(_direct_llm_reply, history)
            response_text = normalize_citations(response_text)
        except Exception as e:
            response_text = _classify_error(e)
        elapsed = time.monotonic() - t0

        is_error = _is_error_response(response_text)
        if is_error:
            logger.warning(f"chat_stream route error: session={session_id[:8]} elapsed={elapsed:.1f}s err={_strip_error_sentinel(response_text)}")
            await save_session(session_id, history)
            yield {"error": _strip_error_sentinel(response_text)}
            return

        history.append({"role": "assistant", "content": response_text})
        logger.info(f"chat_stream route ok: session={session_id[:8]} resp_len={len(response_text)} elapsed={elapsed:.1f}s")
        await save_session(session_id, history)
        await _persist_messages(session_id, [("user", message), ("assistant", response_text)])
        yield {"token": response_text}
        yield {"done": True, "session_id": session_id}
        return

    queue: asyncio.Queue = asyncio.Queue()
    full_response: list[str] = []
    error_msg: str | None = None
    loop = asyncio.get_running_loop()

    def _run():
        nonlocal error_msg
        try:
            for token in run_agent_streaming(history, session_id):
                full_response.append(token)
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as e:
            error_msg = _classify_error(e)
        finally:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)
            except RuntimeError:
                pass  # event loop already closed

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
    try:
        await task
    except Exception:
        pass  # 线程异常已通过 error_msg 捕获

    # 错误单独处理，不混入 token 队列
    if error_msg:
        elapsed = time.monotonic() - t0
        logger.warning(f"chat_stream error: session={session_id[:8]} msg_len={len(message)} elapsed={elapsed:.1f}s err={_strip_error_sentinel(error_msg)}")
        # 保存用户消息到 Redis，避免重试时 LLM 丢失上下文
        await save_session(session_id, history)
        yield {"error": _strip_error_sentinel(error_msg)}
        return

    response_text = normalize_citations("".join(full_response))
    is_error = _is_error_response(response_text)
    elapsed = time.monotonic() - t0

    if is_error:
        logger.warning(f"chat_stream error: session={session_id[:8]} msg_len={len(message)} elapsed={elapsed:.1f}s err={_strip_error_sentinel(response_text)}")
    else:
        history.append({"role": "assistant", "content": response_text})
        logger.info(f"chat_stream ok: session={session_id[:8]} msg_len={len(message)} resp_len={len(response_text)} elapsed={elapsed:.1f}s")
    await save_session(session_id, history)

    msgs = [("user", message)]
    if not is_error:
        msgs.append(("assistant", response_text))
    await _persist_messages(session_id, msgs)

    yield {"done": True, "session_id": session_id}
