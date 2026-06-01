"""Redis 频率限制：每 IP 每分钟最多 30 次。基于固定窗口计数器，多 Worker 共享。"""
from errors import ValidationError

_RATE_LIMIT_MAX = 30
_RATE_LIMIT_WINDOW = 60  # 秒


def check_rate_limit(ip: str):
    from redis_client import get_sync_redis

    key = f"ragmate:rate:{ip}"
    r = get_sync_redis()
    count = r.incr(key)
    if count == 1:
        r.expire(key, _RATE_LIMIT_WINDOW)
    if count > _RATE_LIMIT_MAX:
        raise ValidationError("请求过于频繁，请稍后重试")
