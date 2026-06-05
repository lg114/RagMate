"""入库端点。"""
from fastapi import APIRouter, Request

from backend.api.deps import get_client_ip
from backend.application.ingest_manager import start_ingest
from backend.infrastructure.redis_client import get_ingest_status
from backend.infrastructure.rate_limiter import check_rate_limit
from backend.domain.schemas import IngestRequest

router = APIRouter()


@router.post("/ingest")
async def trigger_ingest(request: Request, body: IngestRequest | None = None):
    await check_rate_limit(get_client_ip(request))
    filenames = body.filenames if body and body.filenames else None
    return await start_ingest(filenames)


@router.get("/ingest/status")
async def ingest_status():
    return await get_ingest_status()
