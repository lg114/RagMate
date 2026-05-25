"""入库端点。"""
from fastapi import APIRouter

from ingest_manager import start_ingest
from redis_client import get_ingest_status

router = APIRouter()


@router.post("/ingest")
async def trigger_ingest(body: dict | None = None):
    filenames = body.get("filenames") if body else None
    return await start_ingest(filenames)


@router.get("/ingest/status")
async def ingest_status():
    return await get_ingest_status()
