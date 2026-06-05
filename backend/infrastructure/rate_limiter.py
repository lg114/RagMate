"""Redis 频率限制：每 IP 每分钟最多 30 次。基于固定窗口计数器，多 Worker 共享。"""
from backend.domain.errors import ValidationError

_RATE_LIMIT_MAX = 30
_RATE_LIMIT_WINDOW = 60  # 秒


async def check_rate_limit(ip: str):
    from backend.infrastructure.redis_client import get_redis

    key = f"ragmate:rate:{ip}"
    r = await get_redis()
    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, _RATE_LIMIT_WINDOW)
    count, _ = await pipe.execute()
    if count > _RATE_LIMIT_MAX:
        raise ValidationError("请求过于频繁，请稍后重试")
