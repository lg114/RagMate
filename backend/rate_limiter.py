"""内存频率限制：每 IP 每分钟最多 30 次。"""
from errors import ValidationError

_rate_limit: dict[str, list[float]] = {}
_RATE_LIMIT_MAX = 30
_RATE_LIMIT_WINDOW = 60.0


def check_rate_limit(ip: str):
    import time
    now = time.time()
    if len(_rate_limit) > 1000:
        for k in list(_rate_limit):
            _rate_limit[k] = [t for t in _rate_limit[k] if now - t < _RATE_LIMIT_WINDOW]
            if not _rate_limit[k]:
                del _rate_limit[k]
    timestamps = _rate_limit.get(ip, [])
    timestamps = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
    _rate_limit[ip] = timestamps
    if len(timestamps) >= _RATE_LIMIT_MAX:
        raise ValidationError("请求过于频繁，请稍后重试")
    timestamps.append(now)
    _rate_limit[ip] = timestamps
