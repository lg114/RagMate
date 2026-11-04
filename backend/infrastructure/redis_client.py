"""可选的 Redis 连接。

入库状态和会话历史都在 PostgreSQL。只有设置了 REDIS_URL 时才连接 Redis，
用来在多个 worker 之间共享限流计数。留空则限流退回本进程内存，核心路径不需要 Redis。
"""
import asyncio

import redis.asyncio as aioredis

from backend.infrastructure.config import settings

_redis: aioredis.Redis | None = None
_redis_lock: asyncio.Lock | None = None


def redis_enabled() -> bool:
    """REDIS_URL 非空时才启用 Redis。"""
    return bool(settings.REDIS_URL and settings.REDIS_URL.strip())


async def get_redis() -> aioredis.Redis:
    """返回异步 Redis 客户端。未配置 REDIS_URL 时直接报错，避免静默连到默认地址。"""
    global _redis, _redis_lock
    if not redis_enabled():
        raise RuntimeError(
            "REDIS_URL is empty, so Redis is disabled. "
            "Ingest and chat do not need it. Set REDIS_URL only for shared rate limits."
        )
    if _redis is None:
        if _redis_lock is None:
            _redis_lock = asyncio.Lock()
        async with _redis_lock:
            if _redis is None:
                _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


async def close_redis() -> None:
    """关闭已建立的异步连接。没连过就什么都不做。"""
    global _redis
    client = _redis
    _redis = None
    if client is not None:
        await client.aclose()
