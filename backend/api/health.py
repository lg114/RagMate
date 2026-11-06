"""健康检查端点。"""
import asyncio
import logging
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import select

from backend.infrastructure.config import settings
from backend.infrastructure.database import async_session
from backend.infrastructure.vector_store import check_vector_store_available

logger = logging.getLogger("ragmate")
router = APIRouter()

_HEALTH_CHECK_TIMEOUT = 5  # 每个依赖检查的超时秒数
_LLM_CHECK_TIMEOUT = 3  # 模型探活更短：它只列一下模型，不该拖慢 /ready
# 这些 host 视为"模型跑在本机"，其余按远程 API 处理。
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal", "host.lima.internal"}


def _model_source() -> str:
    """模型部署在本地还是远程 API。显式配置 LLM_IS_LOCAL 优先于按 host 推断。"""
    if settings.LLM_IS_LOCAL is not None:
        return "local" if settings.LLM_IS_LOCAL else "remote"
    base = (settings.LLM_API_BASE_URL or "").strip()
    if not base:
        return "remote"  # 未指定 base_url：走 SDK 默认的官方端点
    host = (urlparse(base).hostname or "").lower()
    return "local" if host in _LOCAL_HOSTS else "remote"


async def _check_llm(model: str, source: str) -> tuple[bool, str]:
    """探一下模型端点，返回 (是否就绪, 原因码)。

    只看端点活着不够——本地 Ollama 服务起来了但模型没 pull 时照样列不出目标模型，
    那种情况下提问才会 404。所以要求"能列出模型"且"目标模型在列表里"。
    """
    if not model:
        return False, "not_configured"
    base = (settings.LLM_API_BASE_URL or "").strip().rstrip("/")
    # 与 OpenAI SDK 的约定一致：base_url 已含 /v1，模型列表就挂在它下面。
    url = f"{base}/models" if base else "https://api.openai.com/v1/models"
    headers = {}
    if settings.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"
    try:
        # 本地部署不走系统代理：代理会把 127.0.0.1 的请求转发出去而失败。
        async with httpx.AsyncClient(timeout=_LLM_CHECK_TIMEOUT, trust_env=source != "local") as client:
            resp = await client.get(url, headers=headers)
    except Exception as e:
        logger.warning(f"LLM endpoint {url} unreachable: {e}")
        return False, "unreachable"
    if resp.status_code != 200:
        logger.warning(f"LLM endpoint {url} returned HTTP {resp.status_code}")
        return False, "http_error"
    try:
        ids = {m.get("id") for m in resp.json().get("data", []) if isinstance(m, dict)}
    except Exception as e:
        logger.warning(f"LLM endpoint {url} returned an unexpected body: {e}")
        return False, "http_error"
    if model not in ids:
        logger.warning(f"LLM endpoint {url} is up but '{model}' is not among {sorted(i for i in ids if i)}")
        return False, "model_missing"
    return True, ""


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready():
    model = (settings.LLM_MODEL or "").strip()
    source = _model_source() if model else "unknown"

    async def check_postgres() -> bool:
        try:
            async with async_session() as session:
                await asyncio.wait_for(session.execute(select(1)), timeout=_HEALTH_CHECK_TIMEOUT)
            return True
        except Exception as e:
            logger.warning(f"PostgreSQL health check failed: {e}")
            return False

    async def check_pgvector() -> bool:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(check_vector_store_available), timeout=_HEALTH_CHECK_TIMEOUT
            )
        except Exception as e:
            logger.warning(f"pgvector health check failed: {e}")
            return False

    # 三项并发：串行时全挂要等 13 秒，健康检查不该这么慢。
    pg_ok, vectors_ok, (llm_ok, llm_detail) = await asyncio.gather(
        check_postgres(), check_pgvector(), _check_llm(model, source)
    )
    status = {"postgresql": pg_ok, "pgvector": vectors_ok, "llm": llm_ok}

    # 只有基础设施决定 status / HTTP 码：模型挂了应用仍能列文档、看历史，
    # 不该让外部探针把整个服务判死（否则容器会被反复重启）。
    infra_ok = pg_ok and vectors_ok
    body = {
        "status": "ready" if infra_ok else "degraded",
        "checks": status,
        "llm_detail": llm_detail,
        "model": model,
        "model_source": source,
    }
    return body if infra_ok else JSONResponse(status_code=503, content=body)
