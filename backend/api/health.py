"""健康检查端点。"""
import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import select

from backend.infrastructure.database import async_session
from backend.infrastructure.redis_client import get_redis
from backend.infrastructure.milvus import check_milvus_available

logger = logging.getLogger("ragmate")
router = APIRouter()

_HEALTH_CHECK_TIMEOUT = 5  # 每个依赖检查的超时秒数


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready():
    status = {"milvus": False, "postgresql": False, "redis": False}
    try:
        milvus_ok = await asyncio.wait_for(
            asyncio.to_thread(check_milvus_available), timeout=_HEALTH_CHECK_TIMEOUT
        )
        status["milvus"] = milvus_ok
    except Exception as e:
        logger.warning(f"Milvus health check failed: {e}")
    try:
        async with async_session() as session:
            await asyncio.wait_for(session.execute(select(1)), timeout=_HEALTH_CHECK_TIMEOUT)
        status["postgresql"] = True
    except Exception as e:
        logger.warning(f"PostgreSQL health check failed: {e}")
    try:
        r = await get_redis()
        await asyncio.wait_for(r.ping(), timeout=_HEALTH_CHECK_TIMEOUT)
        status["redis"] = True
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")

    all_ok = all(status.values())
    body = {"status": "ready" if all_ok else "degraded", "checks": status}
    return body if all_ok else JSONResponse(status_code=503, content=body)
