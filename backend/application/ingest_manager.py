"""入库任务生命周期管理：锁获取、后台执行、续期、清理。"""

import asyncio
import logging

from backend.application.ingest.pipeline import ingest_documents
from backend.infrastructure.redis_client import (
    acquire_ingest_lock,
    release_ingest_lock,
    renew_ingest_lock,
    set_ingest_status,
)

logger = logging.getLogger("ragmate")

_ingest_task: asyncio.Task | None = None
_ingest_lock_token: str | None = None
_start_lock = asyncio.Lock()


async def _run_ingest(filenames: list[str] | None = None):
    """在后台线程中执行入库，通过 Redis 管理状态与锁。"""
    global _ingest_lock_token

    async def _renew_loop():
        while True:
            await asyncio.sleep(300)
            if _ingest_lock_token:
                try:
                    await renew_ingest_lock(_ingest_lock_token)
                except Exception:
                    logger.debug("Failed to renew ingest lock", exc_info=True)

    renew_task = asyncio.create_task(_renew_loop())
    try:
        await set_ingest_status({"status": "running"})
        result = await asyncio.to_thread(ingest_documents, None, filenames)
        await set_ingest_status(result)
    except asyncio.CancelledError:
        await set_ingest_status({"status": "idle"})
        raise
    except Exception as e:
        logger.exception("Ingestion failed")
        await set_ingest_status({"status": "failed", "error": str(e)})
    finally:
        renew_task.cancel()
        await asyncio.gather(renew_task, return_exceptions=True)
        global _ingest_task
        _ingest_task = None
        if _ingest_lock_token:
            try:
                await release_ingest_lock(_ingest_lock_token)
            except Exception:
                logger.debug("Failed to release ingest lock", exc_info=True)
            _ingest_lock_token = None


async def start_ingest(filenames: list[str] | None = None) -> dict:
    """启动入库任务（如果未在运行）。返回状态。"""
    global _ingest_task, _ingest_lock_token
    async with _start_lock:
        if _ingest_task and not _ingest_task.done():
            return {"status": "already_running"}
        token = await acquire_ingest_lock()
        if not token:
            return {"status": "already_running"}
        _ingest_lock_token = token
        _ingest_task = asyncio.create_task(_run_ingest(filenames))
    return {"status": "started"}


async def cancel_ingest():
    """取消正在运行的入库任务（用于应用关闭时清理）。"""
    global _ingest_task
    if _ingest_task and not _ingest_task.done():
        _ingest_task.cancel()
        try:
            await _ingest_task
        except (asyncio.CancelledError, Exception):
            pass
