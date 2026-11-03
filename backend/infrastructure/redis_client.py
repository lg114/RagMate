import asyncio
import json
import threading
import uuid
from datetime import datetime, timezone

import redis
import redis.asyncio as aioredis

from backend.infrastructure.config import settings

_redis: aioredis.Redis | None = None
_sync_redis: redis.Redis | None = None
_redis_lock: asyncio.Lock | None = None
_sync_redis_lock = threading.Lock()


async def get_redis() -> aioredis.Redis:
    global _redis, _redis_lock
    if _redis is None:
        if _redis_lock is None:
            _redis_lock = asyncio.Lock()
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
        max_msgs = settings.MAX_SESSION_MESSAGES
        return messages[-max_msgs:] if len(messages) > max_msgs else messages
    except json.JSONDecodeError:
        return []


async def save_session(session_id: str, messages: list[dict], ttl: int | None = None):
    r = await get_redis()
    await r.setex(_session_key(session_id), ttl or settings.SESSION_TTL, json.dumps(messages, ensure_ascii=False))


# ── Ingest distributed lock ──

INGEST_LOCK_KEY = "ragmate:ingest:lock"
INGEST_STATUS_KEY = "ragmate:ingest:status"

# Lua 脚本：仅当 value 匹配 token 时才删除 key（防止误删他人锁）
_RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

# Lua 脚本：仅当 value 匹配 token 时才续期（原子操作，消除 TOCTOU 竞态）
_RENEW_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("expire", KEYS[1], ARGV[2])
else
    return 0
end
"""


async def acquire_ingest_lock() -> str | None:
    """获取入库分布式锁。返回 token 表示获取成功，返回 None 表示失败。"""
    r = await get_redis()
    token = uuid.uuid4().hex
    ok = await r.set(INGEST_LOCK_KEY, token, nx=True, ex=settings.INGEST_LOCK_TTL)
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
    """续期入库锁（原子操作，仅 token 匹配时才续期）。"""
    r = await get_redis()
    await r.eval(_RENEW_LOCK_SCRIPT, 1, INGEST_LOCK_KEY, token, settings.INGEST_LOCK_TTL)


async def get_ingest_status() -> dict:
    """读取入库状态原文，不判断持锁进程是否还活着。"""
    r = await get_redis()
    data = await r.get(INGEST_STATUS_KEY)
    if not isinstance(data, str) or not data:
        return {"status": "idle", "last_ingest": None}
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return {"status": "idle", "last_ingest": None}
    if not isinstance(parsed, dict):
        return {"status": "idle", "last_ingest": None}
    return parsed


async def set_ingest_status(data: dict):
    payload = {**data, "last_ingest": datetime.now(timezone.utc).isoformat()}
    r = await get_redis()
    await r.setex(INGEST_STATUS_KEY, settings.INGEST_STATUS_TTL, json.dumps(payload, default=str))


def set_ingest_status_sync(data: dict):
    """同步版本，供 ingest 后台任务使用"""
    payload = {**data, "last_ingest": datetime.now(timezone.utc).isoformat()}
    r = get_sync_redis()
    r.setex(INGEST_STATUS_KEY, settings.INGEST_STATUS_TTL, json.dumps(payload, default=str))


# ── Ingest task record（跨进程 / 跨 worker 的任务句柄） ──
# 锁只表示「有人在跑」；任务哈希保存 token、心跳和取消标志，
# 这样状态查询和取消请求不必打到持有 asyncio.Task 的那个进程。

INGEST_TASK_KEY = "ragmate:ingest:task"

_HEARTBEAT_TASK_SCRIPT = """
if redis.call("hget", KEYS[1], "token") == ARGV[1] then
    redis.call("hset", KEYS[1], "heartbeat_at", ARGV[2])
    redis.call("expire", KEYS[1], ARGV[3])
    return 1
end
return 0
"""

_CANCEL_TASK_SCRIPT = """
if redis.call("exists", KEYS[1]) == 1 then
    redis.call("hset", KEYS[1], "cancel", "1")
    return 1
end
return 0
"""

_CLEAR_TASK_SCRIPT = """
if redis.call("hget", KEYS[1], "token") == ARGV[1] then
    return redis.call("del", KEYS[1])
end
return 0
"""


async def get_ingest_lock_token() -> str | None:
    """读取当前入库锁的 token。没有锁时返回 None。"""
    r = await get_redis()
    token = await r.get(INGEST_LOCK_KEY)
    return token if isinstance(token, str) else None


async def get_ingest_lock_ttl() -> int | None:
    """返回入库锁剩余秒数。锁不存在时返回 None。"""
    r = await get_redis()
    ttl = await r.ttl(INGEST_LOCK_KEY)
    if isinstance(ttl, int) and ttl >= 0:
        return ttl
    return None


async def save_ingest_task(token: str, filenames: list[str] | None, worker_id: str) -> None:
    """写入入库任务记录并设置与锁相同的 TTL。"""
    now = datetime.now(timezone.utc).isoformat()
    r = await get_redis()
    await r.delete(INGEST_TASK_KEY)
    await r.hset(INGEST_TASK_KEY, mapping={
        "token": token,
        "worker_id": worker_id,
        "filenames": json.dumps(filenames, ensure_ascii=False),
        "started_at": now,
        "heartbeat_at": now,
        "cancel": "0",
    })
    await r.expire(INGEST_TASK_KEY, settings.INGEST_LOCK_TTL)


async def get_ingest_task() -> dict | None:
    """读取入库任务记录。不存在时返回 None。"""
    r = await get_redis()
    data = await r.hgetall(INGEST_TASK_KEY)
    if not isinstance(data, dict) or not data.get("token"):
        return None
    return data


async def heartbeat_ingest_task(token: str) -> bool:
    """刷新任务心跳并续期。token 不匹配时返回 False。"""
    r = await get_redis()
    now = datetime.now(timezone.utc).isoformat()
    renewed = await r.eval(
        _HEARTBEAT_TASK_SCRIPT, 1, INGEST_TASK_KEY, token, now, settings.INGEST_LOCK_TTL,
    )
    return bool(renewed)


async def mark_ingest_cancel() -> bool:
    """请求取消当前入库任务。任何 worker 都可调用。没有任务时返回 False。"""
    r = await get_redis()
    flagged = await r.eval(_CANCEL_TASK_SCRIPT, 1, INGEST_TASK_KEY)
    return bool(flagged)


async def clear_ingest_task(token: str) -> None:
    """仅当 token 匹配时删除任务记录。"""
    r = await get_redis()
    await r.eval(_CLEAR_TASK_SCRIPT, 1, INGEST_TASK_KEY, token)


def ingest_cancel_requested_sync() -> bool:
    """同步读取取消标志，供入库线程在阶段边界轮询。Redis 不可用时返回 False。"""
    try:
        flag = get_sync_redis().hget(INGEST_TASK_KEY, "cancel")
    except Exception:
        return False
    return flag == "1"
