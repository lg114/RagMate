"""会话历史：截断、删最后一轮、PostgreSQL 持久化。"""

from sqlalchemy import delete, select

from backend.domain.models import ChatHistory
from backend.infrastructure.config import settings
from backend.infrastructure.database import async_session
from backend.infrastructure.session_store import append_messages

def _trim_history(history: list[dict]) -> list[dict]:
    """截断历史到最近 N 轮，避免 prompt 膨胀。"""
    max_turns = settings.MAX_HISTORY_TURNS
    if len(history) <= max_turns * 2:
        return history
    return history[-(max_turns * 2):]


async def _persist_messages(session_id: str, messages: list[tuple[str, str]]):
    """把这一轮写入 PostgreSQL。save_session 已经写过同样的末尾时不会再插一次。"""
    await append_messages(session_id, messages)


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
