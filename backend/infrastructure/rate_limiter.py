"""Redis 频率限制：每 IP 每分钟最多 30 次。基于固定窗口计数器，多 Worker 共享。"""
from backend.domain.errors import ValidationError

_RATE_LIMIT_MAX = 30
_RATE_LIMIT_WINDOW = 60  # 秒

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


async def check_rate_limit(ip: str):
    from backend.infrastructure.redis_client import get_redis

    key = f"ragmate:rate:{ip}"
    r = await get_redis()
    allowed = await r.eval(_RATE_LIMIT_SCRIPT, 1, key, _RATE_LIMIT_WINDOW, _RATE_LIMIT_MAX)
    if not allowed:
        raise ValidationError("请求过于频繁，请稍后重试")
