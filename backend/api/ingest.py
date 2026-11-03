"""入库端点。"""
from fastapi import APIRouter, Request

from backend.api.deps import get_client_ip
from backend.application.ingest_manager import reconcile_stale_ingest, request_ingest_cancellation, start_ingest
from backend.infrastructure.rate_limiter import check_rate_limit
from backend.domain.schemas import IngestRequest

router = APIRouter()


@router.post("/ingest")
async def trigger_ingest(request: Request, body: IngestRequest | None = None):
    await check_rate_limit(get_client_ip(request))
    filenames = body.filenames if body and body.filenames else None
    return await start_ingest(filenames)


@router.post("/ingest/cancel")
async def cancel_ingest_endpoint(request: Request):
    await check_rate_limit(get_client_ip(request))
    return await request_ingest_cancellation()


@router.get("/ingest/status")
async def ingest_status():
    """返回入库状态。running 但心跳已死时会收成 interrupted。"""
    return await reconcile_stale_ingest()
