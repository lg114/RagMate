"""聊天端点。"""
import json
import logging

from fastapi import APIRouter, Request
from sqlalchemy import delete, func, select
from starlette.responses import StreamingResponse

from backend.application.chat import chat, chat_stream
from backend.infrastructure.database import async_session
from backend.domain.models import ChatHistory
from backend.infrastructure.rate_limiter import check_rate_limit
from backend.infrastructure.redis_client import get_redis
from backend.domain.schemas import ChatRequest, ChatResponse, validate_session_id

logger = logging.getLogger("ragmate")
router = APIRouter()


def _get_client_ip(request: Request) -> str:
    """获取客户端 IP，支持反向代理。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, request: Request):
    check_rate_limit(_get_client_ip(request))
    result = await chat(body.message, body.session_id, replace_last=body.replace_last)
    return ChatResponse(response=result["response"], session_id=result["session_id"])


@router.post("/chat/stream")
async def chat_stream_endpoint(body: ChatRequest, request: Request):
    check_rate_limit(_get_client_ip(request))

    async def event_generator():
        async for chunk in chat_stream(body.message, body.session_id, replace_last=body.replace_last):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chat/sessions")
async def list_sessions():
    async with async_session() as session:
        first_msg = (
            select(
                ChatHistory.session_id,
                ChatHistory.content,
                ChatHistory.created_at,
                func.row_number().over(
                    partition_by=ChatHistory.session_id,
                    order_by=ChatHistory.created_at,
                ).label("rn"),
            )
            .where(ChatHistory.role == "user")
            .subquery()
        )
        result = await session.execute(
            select(
                first_msg.c.session_id,
                first_msg.c.content,
                first_msg.c.created_at,
            ).where(first_msg.c.rn == 1)
            .order_by(first_msg.c.created_at.desc())
            .limit(50)
        )
        sessions = [
            {
                "session_id": row.session_id,
                "first_message": row.content[:80] + ("..." if len(row.content) > 80 else ""),
                "created_at": row.created_at.isoformat(),
            }
            for row in result.fetchall()
        ]
    return {"sessions": sessions}


@router.get("/chat/sessions/{session_id}")
async def get_history(session_id: str):
    validate_session_id(session_id)
    async with async_session() as session:
        result = await session.execute(
            select(ChatHistory)
            .where(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.created_at)
        )
        messages = [
            {"role": row.role, "content": row.content, "created_at": row.created_at.isoformat()}
            for row in result.scalars()
        ]
    return {"session_id": session_id, "messages": messages}


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    validate_session_id(session_id)
    try:
        r = await get_redis()
        await r.delete(f"ragmate:session:{session_id}")
    except Exception:
        logger.debug("Failed to delete Redis session cache", exc_info=True)

    async with async_session() as session:
        await session.execute(delete(ChatHistory).where(ChatHistory.session_id == session_id))
        await session.commit()

    return {"success": True}
