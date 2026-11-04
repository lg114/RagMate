"""入库任务生命周期：Postgres 里的锁、心跳、取消标志，以及本进程里的执行任务。

跨 worker 的事实来源是 ingest_control 这一行，而不是进程内的 asyncio.Task。
持锁进程负责跑入库循环并写进度；任意进程都能读状态、请求取消。
持锁进程退出后，心跳过期，下一次状态查询会把 running 收成 interrupted 并放开锁。
终态和失败原因留在这一行上，不按 TTL 删除。
"""

import asyncio
import logging
import os
import socket
import threading
import uuid
from datetime import datetime, timezone

from backend.application.ingest.pipeline import ingest_documents
from backend.infrastructure.config import settings
from backend.infrastructure.ingest_store import (
    acquire_ingest_lock,
    clear_ingest_task,
    force_release_ingest_lock,
    get_ingest_lock_token,
    get_ingest_lock_ttl,
    get_ingest_status,
    get_ingest_task,
    heartbeat_ingest_task,
    mark_ingest_cancel,
    release_ingest_lock,
    renew_ingest_lock,
    save_ingest_task,
    set_ingest_status,
)

logger = logging.getLogger("ragmate")

_ingest_task: asyncio.Task | None = None
_ingest_lock_token: str | None = None
_ingest_cancel_event: threading.Event | None = None
_start_lock = asyncio.Lock()
_WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

_INTERRUPTED_ERROR = "入库任务已中断（进程重启或 worker 退出），请重新入库。"


def ingest_task_is_live(
    task: dict | None,
    lock_token: str | None,
    *,
    now: datetime,
    stale_seconds: int,
) -> bool:
    """持锁记录的心跳仍在有效期内时，认为入库任务还活着。"""
    if not task or not lock_token or task.get("token") != lock_token:
        return False
    raw = task.get("heartbeat_at")
    if not isinstance(raw, str) or not raw:
        return False
    try:
        heartbeat = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    age = (now - heartbeat).total_seconds()
    return age <= stale_seconds


async def reconcile_stale_ingest() -> dict:
    """返回入库状态。running 但心跳已死时改写成 interrupted，并释放对应的锁。"""
    status = await get_ingest_status()
    if status.get("status") != "running":
        return status

    task = await get_ingest_task()
    lock_token = await get_ingest_lock_token()
    now = datetime.now(timezone.utc)
    if ingest_task_is_live(task, lock_token, now=now, stale_seconds=settings.INGEST_HEARTBEAT_STALE):
        return status

    interrupted = {k: v for k, v in status.items() if k != "status"}
    interrupted["status"] = "interrupted"
    interrupted["error"] = interrupted.get("error") or _INTERRUPTED_ERROR
    await set_ingest_status(interrupted)
    if lock_token and task and task.get("token") == lock_token:
        try:
            await release_ingest_lock(lock_token)
        except Exception:
            logger.debug("Failed to release stale ingest lock", exc_info=True)
        try:
            await clear_ingest_task(lock_token)
        except Exception:
            logger.debug("Failed to clear stale ingest task", exc_info=True)
    elif lock_token and task is None:
        # 还没有任务记录时无法对 token。刚 acquire、尚未写入任务的窗口里租约仍接近满值；
        # 超过一个心跳周期还没人续期，就当成崩溃遗留的锁直接删掉。
        ttl = await get_ingest_lock_ttl()
        stale_ttl = settings.INGEST_LOCK_TTL - settings.INGEST_HEARTBEAT_STALE
        if ttl is not None and ttl <= stale_ttl:
            try:
                await force_release_ingest_lock()
            except Exception:
                logger.debug("Failed to release orphan ingest lock", exc_info=True)
    return interrupted


async def _watch_lock(token: str, cancel_event: threading.Event):
    """续期锁和心跳，并在数据库取消标志出现或锁丢失时置位本地取消事件。"""
    while not cancel_event.is_set():
        await asyncio.sleep(settings.INGEST_HEARTBEAT_INTERVAL)
        if cancel_event.is_set():
            return
        try:
            renewed = await renew_ingest_lock(token)
            alive = await heartbeat_ingest_task(token)
        except Exception:
            logger.debug("Failed to refresh ingest lock", exc_info=True)
            continue
        if not renewed or not alive:
            logger.warning("Ingest lock lost; stopping local ingest task")
            cancel_event.set()
            return
        try:
            current = await get_ingest_task()
        except Exception:
            logger.debug("Failed to read ingest task", exc_info=True)
            continue
        if current and current.get("cancel") == "1":
            cancel_event.set()
            return


async def _run_ingest(filenames: list[str] | None, token: str):
    """在后台线程中执行入库。锁与取消标志以 Postgres 的 ingest_control 为准。"""
    global _ingest_lock_token, _ingest_cancel_event, _ingest_task

    cancel_event = threading.Event()
    _ingest_cancel_event = cancel_event
    _ingest_lock_token = token
    watch_task = asyncio.create_task(_watch_lock(token, cancel_event))
    try:
        await set_ingest_status({"status": "running"})
        result = await asyncio.to_thread(
            ingest_documents, settings.DOCUMENTS_DIR, filenames, cancel_event,
        )
        if cancel_event.is_set() or (isinstance(result, dict) and result.get("message") == "Cancelled"):
            await set_ingest_status({"status": "cancelled", "message": "Cancelled"})
        else:
            await set_ingest_status(result)
    except asyncio.CancelledError:
        cancel_event.set()
        try:
            await set_ingest_status({"status": "cancelled", "message": "Cancelled"})
        except Exception:
            logger.debug("Failed to set cancelled ingest status", exc_info=True)
        raise
    except Exception as e:
        logger.exception("Ingestion failed")
        try:
            await set_ingest_status({"status": "failed", "error": str(e)})
        except Exception:
            logger.debug("Failed to set ingest failure status", exc_info=True)
    finally:
        watch_task.cancel()
        await asyncio.gather(watch_task, return_exceptions=True)
        _ingest_task = None
        _ingest_cancel_event = None
        try:
            await clear_ingest_task(token)
        except Exception:
            logger.debug("Failed to clear ingest task", exc_info=True)
        try:
            await release_ingest_lock(token)
        except Exception:
            logger.debug("Failed to release ingest lock", exc_info=True)
        if _ingest_lock_token == token:
            _ingest_lock_token = None


async def start_ingest(filenames: list[str] | None = None) -> dict:
    """启动入库任务（如果未在运行）。返回状态。"""
    global _ingest_task, _ingest_lock_token
    async with _start_lock:
        if _ingest_task and not _ingest_task.done():
            return {"status": "already_running"}
        await reconcile_stale_ingest()
        token = await acquire_ingest_lock()
        if not token:
            return {"status": "already_running"}
        try:
            await save_ingest_task(token, filenames, _WORKER_ID)
        except Exception:
            try:
                await release_ingest_lock(token)
            except Exception:
                logger.debug("Failed to release lock after task save error", exc_info=True)
            raise
        _ingest_lock_token = token
        _ingest_task = asyncio.create_task(_run_ingest(filenames, token))
    return {"status": "started"}


async def request_ingest_cancellation() -> dict:
    """请求取消正在运行的入库。任意 worker 都可以调用。

    持锁进程在心跳循环里看到数据库标志后停止；若取消请求正好落在本进程，也会立刻置位本地事件。
    """
    requested = await mark_ingest_cancel()
    if _ingest_cancel_event is not None:
        _ingest_cancel_event.set()
    if requested or (_ingest_task and not _ingest_task.done()):
        return {"status": "cancelling"}
    return {"status": "idle"}


async def cancel_ingest():
    """取消本进程正在运行的入库任务（应用关闭时调用）。"""
    try:
        await request_ingest_cancellation()
    except Exception:
        logger.debug("Failed to publish ingest cancel flag", exc_info=True)
        if _ingest_cancel_event is not None:
            _ingest_cancel_event.set()
    if _ingest_task and not _ingest_task.done():
        _ingest_task.cancel()
        try:
            await _ingest_task
        except (asyncio.CancelledError, Exception):
            pass
