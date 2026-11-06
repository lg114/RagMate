"""频率限制：每 IP 每窗口最多 N 次。

本进程内存固定窗口，单机演示够用。
"""
import asyncio
import time

from backend.infrastructure.config import settings
from backend.domain.errors import ValidationError

_buckets: dict[str, tuple[int, float]] = {}
_bucket_lock = asyncio.Lock()


def _too_many() -> ValidationError:
    return ValidationError("请求过于频繁，请稍后重试")


async def check_rate_limit(ip: str) -> None:
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
