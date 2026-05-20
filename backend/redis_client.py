import asyncio
import json
import threading
import uuid
from datetime import datetime, timezone

import redis
import redis.asyncio as aioredis

from config import settings

_redis: aioredis.Redis | None = None
_sync_redis: redis.Redis | None = None
_redis_lock = asyncio.Lock()
_sync_redis_lock = threading.Lock()

MAX_SESSION_MESSAGES = 200  # 单 session 最大消息数，超出则截断旧消息


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        async with _redis_lock:
            if _redis is None:
                _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


def get_sync_redis() -> redis.Redis:
    global _sync_redis
    if _sync_redis is not None:
        return _sync_redis
    with _sync_redis_lock:
        if _sync_redis is None:
            _sync_redis = redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=5)
    return _sync_redis


def close_sync_redis():
    """关闭同步 Redis 连接（应用关闭时调用）"""
    global _sync_redis
    if _sync_redis is not None:
        _sync_redis.close()
        _sync_redis = None


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
INGEST_LOCK_TTL = 600  # 10 分钟自动过期，崩溃时自动释放

# Lua 脚本：仅当 value 匹配 token 时才删除 key（防止误删他人锁）
_RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


async def acquire_ingest_lock() -> str | None:
    """获取入库分布式锁。返回 token 表示获取成功，返回 None 表示失败。"""
    r = await get_redis()
    token = uuid.uuid4().hex
    ok = await r.set(INGEST_LOCK_KEY, token, nx=True, ex=INGEST_LOCK_TTL)
    return token if ok else None


async def release_ingest_lock(token: str):
    """释放入库分布式锁。只有持有正确 token 的进程才能释放。"""
    r = await get_redis()
    await r.eval(_RELEASE_LOCK_SCRIPT, 1, INGEST_LOCK_KEY, token)


async def force_release_ingest_lock():
    """强制释放入库锁（用于启动时清理遗留锁）。"""
    r = await get_redis()
    await r.delete(INGEST_LOCK_KEY)


async def renew_ingest_lock(token: str):
    """续期入库锁（延长 TTL）。"""
    r = await get_redis()
    current = await r.get(INGEST_LOCK_KEY)
    if current == token:
        await r.expire(INGEST_LOCK_KEY, INGEST_LOCK_TTL)


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
    payload = {**data, "last_ingest": datetime.now(timezone.utc).isoformat()}
    r = await get_redis()
    await r.setex(INGEST_STATUS_KEY, INGEST_LOCK_TTL, json.dumps(payload, default=str))


def set_ingest_status_sync(data: dict):
    """同步版本，供 ingest 后台任务使用"""
    payload = {**data, "last_ingest": datetime.now(timezone.utc).isoformat()}
    r = get_sync_redis()
    r.setex(INGEST_STATUS_KEY, INGEST_LOCK_TTL, json.dumps(payload, default=str))
