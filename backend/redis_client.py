import json
from datetime import datetime, timezone

import redis
import redis.asyncio as aioredis

from config import settings

_redis: aioredis.Redis | None = None
_sync_redis: redis.Redis | None = None

MAX_SESSION_MESSAGES = 200  # 单 session 最大消息数，超出则截断旧消息


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


def get_sync_redis() -> redis.Redis:
    global _sync_redis
    if _sync_redis is None:
        _sync_redis = redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=5)
    return _sync_redis


def _session_key(session_id: str) -> str:
    return f"ragmate:session:{session_id}"


async def load_session(session_id: str) -> list[dict]:
    r = await get_redis()
    data = await r.get(_session_key(session_id))
    if not data:
        return []
    try:
        messages = json.loads(data)
        # 截断超长 session，防止内存和传输开销
        return messages[-MAX_SESSION_MESSAGES:] if len(messages) > MAX_SESSION_MESSAGES else messages
    except json.JSONDecodeError:
        return []


async def save_session(session_id: str, messages: list[dict], ttl: int = 86400):
    r = await get_redis()
    await r.setex(_session_key(session_id), ttl, json.dumps(messages, ensure_ascii=False))


# ── Ingest distributed lock ──

INGEST_LOCK_KEY = "ragmate:ingest:lock"
INGEST_STATUS_KEY = "ragmate:ingest:status"
INGEST_LOCK_TTL = 600  # 10 minutes max, auto-release on crash


async def acquire_ingest_lock() -> bool:
    """获取入库分布式锁。返回 True 表示获取成功。"""
    r = await get_redis()
    return await r.set(INGEST_LOCK_KEY, "1", nx=True, ex=INGEST_LOCK_TTL)


async def release_ingest_lock():
    r = await get_redis()
    await r.delete(INGEST_LOCK_KEY)


async def get_ingest_status() -> dict:
    r = await get_redis()
    data = await r.get(INGEST_STATUS_KEY)
    if not data:
        return {"status": "idle", "last_ingest": None}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {"status": "idle", "last_ingest": None}


async def set_ingest_status(data: dict):
    data["last_ingest"] = datetime.now(timezone.utc).isoformat()
    r = await get_redis()
    await r.setex(INGEST_STATUS_KEY, INGEST_LOCK_TTL, json.dumps(data, default=str))


def set_ingest_status_sync(data: dict):
    """同步版本，供 ingest 后台任务使用"""
    data["last_ingest"] = datetime.now(timezone.utc).isoformat()
    r = get_sync_redis()
    r.setex(INGEST_STATUS_KEY, INGEST_LOCK_TTL, json.dumps(data, default=str))
