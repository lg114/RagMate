"""频率限制：每 IP 每窗口最多 N 次。

REDIS_URL 有值时用 Redis 固定窗口，多个 worker 共用计数。
留空时用本进程内存，单机演示不需要 Redis；多 worker 各自计数，不跨进程。
配了 Redis 但连不上时返回明确的 503，不把请求悄悄放行。
"""
import asyncio
import time

from backend.domain.errors import ServiceUnavailableError, ValidationError
from backend.infrastructure.config import settings
from backend.infrastructure.redis_client import redis_enabled

# Lua 脚本：原子 INCR + 首次请求设置 EXPIRE（真正固定窗口，非滑动窗口）
# KEYS[1]: 限流 key, ARGV[1]: 窗口秒数, ARGV[2]: 最大请求数
# 返回 1 表示允许，0 表示拒绝
_RATE_LIMIT_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], tonumber(ARGV[1]))
end
if count > tonumber(ARGV[2]) then
    return 0
end
return 1
"""

_buckets: dict[str, tuple[int, float]] = {}
_bucket_lock = asyncio.Lock()
_REDIS_DOWN = "已配置 REDIS_URL，但 Redis 不可用。多 worker 共享限流需要它；入库和会话不需要。请启动 Redis，或把 REDIS_URL 留空改用进程内限流。"


def _too_many() -> ValidationError:
    return ValidationError("请求过于频繁，请稍后重试")


async def _check_memory(ip: str) -> None:
    now = time.monotonic()
    window = settings.RATE_LIMIT_WINDOW
    async with _bucket_lock:
        if len(_buckets) > 4096:
            expired = [key for key, (_, start) in _buckets.items() if now - start >= window]
            for key in expired:
                _buckets.pop(key, None)
        count, start = _buckets.get(ip, (0, now))
        if now - start >= window:
            count, start = 0, now
        count += 1
        _buckets[ip] = (count, start)
        if count > settings.RATE_LIMIT_MAX:
            raise _too_many()


async def check_rate_limit(ip: str):
    if not redis_enabled():
        await _check_memory(ip)
        return

    from backend.infrastructure.redis_client import get_redis

    key = f"ragmate:rate:{ip}"
    try:
        client = await get_redis()
        allowed = await client.eval(_RATE_LIMIT_SCRIPT, 1, key, settings.RATE_LIMIT_WINDOW, settings.RATE_LIMIT_MAX)
    except Exception as exc:
        raise ServiceUnavailableError(_REDIS_DOWN) from exc
    if not allowed:
        raise _too_many()
