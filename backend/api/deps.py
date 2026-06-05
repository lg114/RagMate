"""API 层共享依赖。"""
from fastapi import Request


def get_client_ip(request: Request) -> str:
    """获取客户端 IP，支持反向代理（X-Forwarded-For）。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
