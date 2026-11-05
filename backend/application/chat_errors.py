"""错误哨兵与错误分类：把异常翻译成用户友好的错误提示。"""

import asyncio
import logging

from backend.domain.errors import AppError

logger = logging.getLogger("ragmate")

# 错误哨兵前缀，用于判断 response_text 是否为错误而非正常回复
# 使用 UUID 标记确保不会与 LLM 自然输出冲突
_ERROR_SENTINEL = "\x00ERR\x00"


def _is_error_response(text: str) -> bool:
    return text.startswith(_ERROR_SENTINEL)


def _strip_error_sentinel(text: str) -> str:
    """移除错误哨兵前缀，返回用户友好的错误消息。"""
    if text.startswith(_ERROR_SENTINEL):
        return text[len(_ERROR_SENTINEL):]
    return text


def _classify_error(e: Exception) -> str:
    """根据异常类型和消息内容返回用户友好的错误提示（带哨兵前缀）"""
    
    # 优先检查异常类型（更可靠）
    if isinstance(e, asyncio.TimeoutError):
        return f"{_ERROR_SENTINEL}请求超时，请稍后重试"
    if isinstance(e, asyncio.CancelledError):
        return f"{_ERROR_SENTINEL}请求被取消"
    
    # 检查 LLM SDK 的异常类名
    exc_type_name = type(e).__name__
    if "RateLimit" in exc_type_name or "TooManyRequests" in exc_type_name:
        return f"{_ERROR_SENTINEL}请求频率超限，请稍后重试"
    if "Authentication" in exc_type_name or "Unauthorized" in exc_type_name:
        return f"{_ERROR_SENTINEL}API 认证失败，请检查 API Key 配置"
    if "InsufficientQuota" in exc_type_name or "QuotaExceeded" in exc_type_name:
        return f"{_ERROR_SENTINEL}API 配额不足，请检查账户余额"
    
    # 降级到消息内容匹配
    msg = str(e).lower()
    if any(k in msg for k in ("rate_limit", "429", "rate limit")):
        return f"{_ERROR_SENTINEL}请求频率超限，请稍后重试"
    if any(k in msg for k in ("auth", "401", "api key", "unauthorized")):
        return f"{_ERROR_SENTINEL}API 认证失败，请检查 API Key 配置"
    if any(k in msg for k in ("timeout", "timed out", "connection")):
        return f"{_ERROR_SENTINEL}连接超时，请检查网络或 LLM 服务状态"
    if any(k in msg for k in ("insufficient_quota", "quota")):
        return f"{_ERROR_SENTINEL}API 配额不足，请检查账户余额"

    logger.error("Unhandled error in chat", exc_info=True)
    return f"{_ERROR_SENTINEL}处理请求时出错，请稍后重试"


def _error_from_exception(exc: Exception) -> str:
    """业务错误用原文；其余走通用分类。返回值带错误哨兵。"""
    if isinstance(exc, AppError):
        return f"{_ERROR_SENTINEL}{exc.message}"
    return _classify_error(exc)
