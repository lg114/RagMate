"""会话工作集放在 PostgreSQL 的 chat_history 里，不再用 Redis。

任意 worker 都读同一张表。一次请求里先 save_session（按整段历史对齐后只补末尾），
再 _persist_messages；后者如果发现末尾已经是这一轮，就不再插第二次。
"""
from sqlalchemy import select

from backend.domain.models import ChatHistory
from backend.infrastructure.config import settings


def pending_tail(existing: list[dict], desired: list[dict]) -> list[dict]:
    """返回 desired 末尾里还没出现在 existing 末尾的消息。

    desired 可能只保留最近几轮，所以可以比 existing 短。
    连续两轮正文相同也按整段前缀对齐，不会把新的一轮误判成已经写过。
    """
    def same(left: dict, right: dict) -> bool:
        return left.get("role") == right.get("role") and left.get("content") == right.get("content")

    for new_count in range(0, len(desired) + 1):
        head = desired if new_count == 0 else desired[:-new_count]
        if not head:
            return list(desired[-new_count:]) if new_count else []
        if len(existing) < len(head):
            continue
        tail = existing[-len(head):]
        if all(same(a, b) for a, b in zip(tail, head)):
            if new_count == 0:
                return []
            return list(desired[-new_count:])
    return list(desired)


def _ends_with(existing: list[dict], fresh: list[dict]) -> bool:
    if not fresh or len(existing) < len(fresh):
        return False
    tail = existing[-len(fresh):]
    return all(
        a.get("role") == b.get("role") and a.get("content") == b.get("content")
        for a, b in zip(tail, fresh)
    )


async def _fetch(session, session_id: str) -> list[dict]:
    result = await session.execute(
        select(ChatHistory.role, ChatHistory.content)
        .where(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.created_at.desc(), ChatHistory.id.desc())
        .limit(settings.MAX_SESSION_MESSAGES)
    )
    rows = list(reversed(result.fetchall()))
    return [{"role": row.role, "content": row.content} for row in rows]


def _normalize(messages: list[dict]) -> list[dict]:
    capped = messages[-settings.MAX_SESSION_MESSAGES:]
    return [{"role": item.get("role"), "content": item.get("content")} for item in capped]


async def load_session(session_id: str) -> list[dict]:
    """读取会话最近的消息。来源是 PostgreSQL，重启后仍然在。"""
    from backend.infrastructure.database import async_session

    async with async_session() as session:
        return await _fetch(session, session_id)


async def save_session(session_id: str, messages: list[dict]) -> None:
    """把历史末尾还没落库的消息写入 PostgreSQL。"""
    desired = _normalize(messages)
    if not desired:
        return
    from backend.infrastructure.database import async_session

    async with async_session() as session:
        existing = await _fetch(session, session_id)
        tail = pending_tail(existing, desired)
        if not tail:
            return
        session.add_all(
            ChatHistory(session_id=session_id, role=item["role"], content=item["content"] or "")
            for item in tail
            if item.get("role") and item.get("content") is not None
        )
        await session.commit()


async def append_messages(session_id: str, messages: list[tuple[str, str]]) -> None:
    """追加一轮消息。若 save_session 已经写过同样的末尾，则什么都不做。"""
    fresh = [{"role": role, "content": content} for role, content in messages]
    if not fresh:
        return
    from backend.infrastructure.database import async_session

    async with async_session() as session:
        existing = await _fetch(session, session_id)
        if _ends_with(existing, fresh):
            return
        session.add_all(
            ChatHistory(session_id=session_id, role=item["role"], content=item["content"])
            for item in fresh
        )
        await session.commit()
