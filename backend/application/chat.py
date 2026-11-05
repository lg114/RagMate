"""问答编排：chat / chat_stream 两个入口。

实现细节已拆到同目录的 chat_* 模块；这里只留编排（会话加载、
路由、检索-生成-接地、流式），并重新导出被移走的名字——
backend.api.chat 从这里导入 chat/chat_stream，测试从这里
patch/导入内部函数，两边都不用改。
"""

import asyncio
import logging
import time
import uuid

from backend.infrastructure.config import settings
from backend.infrastructure.session_store import load_session, save_session

from backend.application.chat_citations import (
    apply_grounding,
    attach_citations,
    citation_label,
    normalize_citations,
    note_unsupported_claims,
    public_citations,
)
from backend.application.chat_errors import (
    _ERROR_SENTINEL,
    _classify_error,
    _error_from_exception,
    _is_error_response,
    _strip_error_sentinel,
)
from backend.application.chat_grounding import _ground_response, _pop_retrieval_signals
from backend.application.chat_history import (
    _delete_last_persisted_turn,
    _persist_messages,
    _strip_last_turn,
    _trim_history,
)
from backend.application.chat_llm import (
    _direct_llm_reply,
    _invoke_answer,
    _is_simple_query,
    _iter_answer_tokens,
    extract_text,
)
from backend.application.chat_retrieval import (
    _generation_messages,
    _refusal_from_bundle,
    _retrieval_rdata,
    _retrieve_for_answer,
)

__all__ = [
    "chat",
    "chat_stream",
    "apply_grounding",
    "attach_citations",
    "citation_label",
    "normalize_citations",
    "note_unsupported_claims",
    "public_citations",
    "extract_text",
    "_ERROR_SENTINEL",
    "_classify_error",
    "_error_from_exception",
    "_is_error_response",
    "_strip_error_sentinel",
    "_ground_response",
    "_pop_retrieval_signals",
    "_delete_last_persisted_turn",
    "_persist_messages",
    "_strip_last_turn",
    "_trim_history",
    "_direct_llm_reply",
    "_invoke_answer",
    "_is_simple_query",
    "_iter_answer_tokens",
    "_generation_messages",
    "_refusal_from_bundle",
    "_retrieval_rdata",
    "_retrieve_for_answer",
]

logger = logging.getLogger("ragmate")

async def _store_turn(session_id: str, history: list[dict], message: str, visible: str, is_error: bool) -> None:
    """写入会话。错误只保留用户消息，成功时存落地后的回答，不存检索规划。"""
    if not is_error:
        history.append({"role": "assistant", "content": visible})
    await save_session(session_id, history)
    msgs = [("user", message)]
    if not is_error:
        msgs.append(("assistant", visible))
    await _persist_messages(session_id, msgs)


async def _finish_error(session_id: str, history: list[dict], message: str, exc: Exception, t0: float, where: str) -> dict:
    visible = _strip_error_sentinel(_error_from_exception(exc))
    elapsed = time.monotonic() - t0
    logger.warning(f"{where} error: session={session_id[:8]} elapsed={elapsed:.1f}s err={visible}")
    await _store_turn(session_id, history, message, visible, True)
    return {"response": visible, "session_id": session_id}


async def _chat_single(message: str, session_id: str, history: list[dict], debug: bool) -> dict:
    """默认链路：一次检索，零片段则拒答，否则一次生成，再接地。"""
    t0 = time.monotonic()
    try:
        bundle = await asyncio.wait_for(
            asyncio.to_thread(_retrieve_for_answer, message, history, debug),
            timeout=settings.AGENT_TIMEOUT,
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return await _finish_error(session_id, history, message, e, t0, "chat")

    if not bundle["results"]:
        visible, extra = _refusal_from_bundle(bundle, debug)
        elapsed = time.monotonic() - t0
        logger.info(
            f"chat refused: session={session_id[:8]} msg_len={len(message)} elapsed={elapsed:.1f}s"
        )
        await _store_turn(session_id, history, message, visible, False)
        return {"response": visible, "session_id": session_id, **extra}

    try:
        response_text = await asyncio.wait_for(
            asyncio.to_thread(_invoke_answer, _generation_messages(history, bundle["results"])),
            timeout=settings.AGENT_TIMEOUT,
        )
        response_text = normalize_citations(response_text)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return await _finish_error(session_id, history, message, e, t0, "chat")

    visible, extra = await _ground_response(
        session_id, response_text, False, debug=debug, rdata=_retrieval_rdata(bundle),
    )
    elapsed = time.monotonic() - t0
    logger.info(
        f"chat ok: session={session_id[:8]} msg_len={len(message)} resp_len={len(visible)} elapsed={elapsed:.1f}s"
    )
    await _store_turn(session_id, history, message, visible, False)
    return {"response": visible, "session_id": session_id, **extra}


async def _chat_stream_single(message: str, session_id: str, history: list[dict], debug: bool, t0: float):
    """默认流式链路。零片段时只发最终拒答，不先吐规划文本再替换。"""
    try:
        bundle = await asyncio.to_thread(_retrieve_for_answer, message, history, debug)
    except asyncio.CancelledError:
        await save_session(session_id, history)
        raise
    except Exception as e:
        visible = _strip_error_sentinel(_error_from_exception(e))
        logger.warning(
            f"chat_stream error: session={session_id[:8]} elapsed={time.monotonic() - t0:.1f}s err={visible}"
        )
        await save_session(session_id, history)
        yield {"error": visible}
        return

    if not bundle["results"]:
        visible, extra = _refusal_from_bundle(bundle, debug)
        logger.info(
            f"chat_stream refused: session={session_id[:8]} elapsed={time.monotonic() - t0:.1f}s"
        )
        await _store_turn(session_id, history, message, visible, False)
        yield {"done": True, "session_id": session_id, "response": visible, **extra}
        return

    queue: asyncio.Queue = asyncio.Queue()
    full_response: list[str] = []
    error_msg: str | None = None
    loop = asyncio.get_running_loop()
    messages = _generation_messages(history, bundle["results"])

    def _run():
        nonlocal error_msg
        try:
            for token in _iter_answer_tokens(messages):
                full_response.append(token)
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as e:
            error_msg = _error_from_exception(e)
        finally:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)
            except RuntimeError:
                pass

    task = loop.run_in_executor(None, _run)
    try:
        while True:
            token = await queue.get()
            if token is _SENTINEL:
                break
            yield {"token": token}
    except asyncio.CancelledError:
        task.cancel()
        await save_session(session_id, history)
        raise
    try:
        await task
    except Exception:
        pass

    if error_msg:
        visible = _strip_error_sentinel(error_msg)
        logger.warning(
            f"chat_stream error: session={session_id[:8]} elapsed={time.monotonic() - t0:.1f}s err={visible}"
        )
        await save_session(session_id, history)
        yield {"error": visible}
        return

    response_text = normalize_citations("".join(full_response))
    visible, extra = await _ground_response(
        session_id, response_text, False, debug=debug, rdata=_retrieval_rdata(bundle),
    )
    logger.info(
        f"chat_stream ok: session={session_id[:8]} resp_len={len(visible)} elapsed={time.monotonic() - t0:.1f}s"
    )
    await _store_turn(session_id, history, message, visible, False)
    yield {"done": True, "session_id": session_id, "response": visible, **extra}


async def chat(message: str, session_id: str | None = None, replace_last: bool = False, debug: bool = False) -> dict:
    """处理用户聊天消息，支持多轮对话。返回 {"response": str, "session_id": str}"""
    if not session_id:
        session_id = str(uuid.uuid4())

    # 1. 加载会话历史
    history = await load_session(session_id)

    # 2. 重试/重新生成时，移除旧的一轮对话
    if replace_last:
        history = _strip_last_turn(history)
        await _delete_last_persisted_turn(session_id)

    # 2.5 截断历史，避免 prompt 膨胀
    history = _trim_history(history)

    # 3. 追加用户消息
    history.append({"role": "user", "content": message})

    # 3.5 查询路由：简单查询直接回复
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

    # 4. 默认链路：一次检索、一次生成、再接地。
    return await _chat_single(message, session_id, history, debug)


_SENTINEL = object()


async def chat_stream(message: str, session_id: str | None = None, replace_last: bool = False, debug: bool = False):
    """流式聊天，逐 token yield。返回格式: {"token": str} 或 {"done": True, "session_id": str}"""
    if not session_id:
        session_id = str(uuid.uuid4())

    t0 = time.monotonic()
    history = await load_session(session_id)

    if replace_last:
        history = _strip_last_turn(history)
        await _delete_last_persisted_turn(session_id)

    history = _trim_history(history)
    history.append({"role": "user", "content": message})

    # 查询路由：简单查询直接回复
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

    # 默认流式链路。
    async for event in _chat_stream_single(message, session_id, history, debug, t0):
        yield event

