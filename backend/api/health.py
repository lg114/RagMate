"""健康检查端点。"""
import asyncio
import logging

from fastapi import APIRouter
from sqlalchemy import select

from backend.infrastructure.database import async_session
from backend.infrastructure.redis_client import get_redis
from backend.infrastructure.milvus import check_milvus_available

logger = logging.getLogger("ragmate")
router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready():
    status = {"milvus": False, "postgresql": False, "redis": False}
    try:
        milvus_ok = await asyncio.to_thread(check_milvus_available)
        status["milvus"] = milvus_ok
    except Exception as e:
        logger.warning(f"Milvus health check failed: {e}")
    try:
        async with async_session() as session:
            await session.execute(select(1))
        status["postgresql"] = True
    except Exception as e:
        logger.warning(f"PostgreSQL health check failed: {e}")
    try:
        r = await get_redis()
        await r.ping()
        status["redis"] = True
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
    return {"status": "ready" if all(status.values()) else "degraded", "checks": status}
