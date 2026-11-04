"""入库控制面：Postgres 里的一行，代替原来的 Redis 锁 / 状态 / 取消标志。

同一时间只有一个持锁 worker 写进度和终态。任意 worker 都可以读这一行、
把 cancel_requested 置上。心跳停了以后，下一次状态查询把 running 收成
interrupted。失败原因写在 error 列上，没有 TTL，进程重启也不会被清掉。

锁的互斥靠单行 UPDATE 的 WHERE：Postgres 会把并发更新串行化，只有一个
能把空锁（或已经过期的租约）换成自己的 token。
"""
import json
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, update
from sqlalchemy.exc import IntegrityError

from backend.domain.models import IngestControl
from backend.infrastructure.config import settings

_ROW_ID = 1


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _iso(value: datetime | None) -> str | None:
    aware = _aware(value)
    return aware.isoformat() if aware is not None else None


def _ensure(session) -> None:
    if session.get(IngestControl, _ROW_ID) is not None:
        return
    try:
        with session.begin_nested():
            session.add(
                IngestControl(
                    id=_ROW_ID,
                    status="idle",
                    detail={"status": "idle", "last_ingest": None},
                    cancel_requested=False,
                )
            )
            session.flush()
    except IntegrityError:
        pass


def _steal_before(now: datetime) -> datetime:
    """租约剩余时间短于「满 TTL 减一个心跳超时」时，视为没人续期，可以抢走。"""
    window = settings.INGEST_LOCK_TTL - settings.INGEST_HEARTBEAT_STALE
    return now + timedelta(seconds=window)


def _acquire(session) -> str | None:
    _ensure(session)
    token = uuid.uuid4().hex
    now = _now()
    expires = now + timedelta(seconds=settings.INGEST_LOCK_TTL)
    result = session.execute(
        update(IngestControl)
        .where(IngestControl.id == _ROW_ID)
        .where(
            or_(
                IngestControl.lock_token.is_(None),
                IngestControl.lock_expires_at.is_(None),
                IngestControl.lock_expires_at <= _steal_before(now),
            )
        )
        .values(
            lock_token=token,
            lock_expires_at=expires,
            task_token=None,
            worker_id=None,
            filenames_json=None,
            started_at=None,
            heartbeat_at=None,
            cancel_requested=False,
        )
    )
    if result.rowcount != 1:
        return None
    return token


def _release(session, token: str) -> None:
    session.execute(
        update(IngestControl)
        .where(IngestControl.id == _ROW_ID, IngestControl.lock_token == token)
        .values(lock_token=None, lock_expires_at=None)
    )


def _force_release(session) -> None:
    _ensure(session)
    session.execute(
        update(IngestControl)
        .where(IngestControl.id == _ROW_ID)
        .values(lock_token=None, lock_expires_at=None)
    )


def _renew(session, token: str) -> bool:
    now = _now()
    result = session.execute(
        update(IngestControl)
        .where(IngestControl.id == _ROW_ID, IngestControl.lock_token == token)
        .values(lock_expires_at=now + timedelta(seconds=settings.INGEST_LOCK_TTL))
    )
    return result.rowcount == 1


def _write_status(session, data: dict) -> None:
    """整份状态替换写入。不修改调用方传入的 dict，也不碰锁和取消标志。"""
    _ensure(session)
    now = _now()
    # default=str 与旧的 Redis 写入一致，也不改调用方手里的 dict。
    payload = json.loads(json.dumps({**data, "last_ingest": now.isoformat()}, default=str))
    status = str(payload.get("status") or "idle")
    error = payload.get("error")
    error_text = str(error) if error else None
    session.execute(
        update(IngestControl)
        .where(IngestControl.id == _ROW_ID)
        .values(status=status, detail=payload, error=error_text, last_ingest=now)
    )


def _read_status(session) -> dict:
    _ensure(session)
    row = session.get(IngestControl, _ROW_ID)
    if row is None:
        return {"status": "idle", "last_ingest": None}
    detail = dict(row.detail) if isinstance(row.detail, dict) else {}
    detail["status"] = row.status or detail.get("status") or "idle"
    if row.error:
        detail["error"] = row.error
    elif "error" in detail and not detail["error"]:
        detail.pop("error", None)
    detail["last_ingest"] = _iso(row.last_ingest)
    return detail


def _lock_token(session) -> str | None:
    row = session.get(IngestControl, _ROW_ID)
    if row is None or not row.lock_token:
        return None
    return row.lock_token


def _lock_ttl(session) -> int | None:
    row = session.get(IngestControl, _ROW_ID)
    if row is None or not row.lock_token or row.lock_expires_at is None:
        return None
    expires = _aware(row.lock_expires_at)
    remaining = int((expires - _now()).total_seconds())
    return max(remaining, 0)


def _save_task(session, token: str, filenames, worker_id: str) -> None:
    _ensure(session)
    now = _now()
    result = session.execute(
        update(IngestControl)
        .where(IngestControl.id == _ROW_ID, IngestControl.lock_token == token)
        .values(
            task_token=token,
            worker_id=worker_id,
            filenames_json=json.dumps(filenames, ensure_ascii=False),
            started_at=now,
            heartbeat_at=now,
            cancel_requested=False,
            lock_expires_at=now + timedelta(seconds=settings.INGEST_LOCK_TTL),
        )
    )
    if result.rowcount != 1:
        raise RuntimeError("ingest lock token does not match; task was not saved")


def _task_view(session) -> dict | None:
    row = session.get(IngestControl, _ROW_ID)
    if row is None or not row.task_token:
        return None
    return {
        "token": row.task_token,
        "worker_id": row.worker_id or "",
        "filenames": row.filenames_json if row.filenames_json is not None else "null",
        "started_at": _iso(row.started_at) or "",
        "heartbeat_at": _iso(row.heartbeat_at) or "",
        "cancel": "1" if row.cancel_requested else "0",
    }


def _heartbeat(session, token: str) -> bool:
    now = _now()
    result = session.execute(
        update(IngestControl)
        .where(IngestControl.id == _ROW_ID, IngestControl.task_token == token)
        .values(
            heartbeat_at=now,
            lock_expires_at=now + timedelta(seconds=settings.INGEST_LOCK_TTL),
        )
    )
    return result.rowcount == 1


def _mark_cancel(session) -> bool:
    result = session.execute(
        update(IngestControl)
        .where(IngestControl.id == _ROW_ID, IngestControl.task_token.is_not(None))
        .values(cancel_requested=True)
    )
    return result.rowcount == 1


def _read_cancel(session) -> bool:
    row = session.get(IngestControl, _ROW_ID)
    return bool(row and row.task_token and row.cancel_requested)


def _clear_task(session, token: str) -> None:
    session.execute(
        update(IngestControl)
        .where(IngestControl.id == _ROW_ID, IngestControl.task_token == token)
        .values(
            task_token=None,
            worker_id=None,
            filenames_json=None,
            started_at=None,
            heartbeat_at=None,
            cancel_requested=False,
        )
    )


def _run_sync(fn, *args):
    from backend.infrastructure.database import SyncSession

    with SyncSession() as session:
        result = fn(session, *args)
        session.commit()
        return result


async def _run_async(fn, *args):
    from backend.infrastructure.database import async_session

    async with async_session() as session:
        result = await session.run_sync(lambda sync_session: fn(sync_session, *args))
        await session.commit()
        return result


async def acquire_ingest_lock() -> str | None:
    """取得入库锁。成功返回 token；已有未过期的锁时返回 None。"""
    return await _run_async(_acquire)


async def release_ingest_lock(token: str) -> None:
    """仅当 token 匹配时释放锁。状态和失败原因保留。"""
    await _run_async(_release, token)


async def force_release_ingest_lock() -> None:
    """无条件清掉锁租约。用于启动后清理没人续期的遗留锁。"""
    await _run_async(_force_release)


async def renew_ingest_lock(token: str) -> bool:
    """续期锁。token 不匹配时返回 False，调用方应停止本地任务。"""
    return bool(await _run_async(_renew, token))


async def get_ingest_status() -> dict:
    """读取入库状态。终态一直留着，不会因为 TTL 被删掉。"""
    return await _run_async(_read_status)


async def set_ingest_status(data: dict) -> None:
    await _run_async(_write_status, data)


def set_ingest_status_sync(data: dict) -> None:
    """同步版本，供入库线程在阶段边界上报进度和终态。"""
    _run_sync(_write_status, data)


async def get_ingest_lock_token() -> str | None:
    return await _run_async(_lock_token)


async def get_ingest_lock_ttl() -> int | None:
    """锁剩余秒数。没有锁时返回 None。过期但还没清掉时返回 0。"""
    return await _run_async(_lock_ttl)


async def save_ingest_task(token: str, filenames: list[str] | None, worker_id: str) -> None:
    """在已持有的锁上写下任务、心跳和取消标志。token 不对则抛错。"""
    await _run_async(_save_task, token, filenames, worker_id)


async def get_ingest_task() -> dict | None:
    return await _run_async(_task_view)


async def heartbeat_ingest_task(token: str) -> bool:
    """刷新心跳并续期。token 不匹配时返回 False。"""
    return bool(await _run_async(_heartbeat, token))


async def mark_ingest_cancel() -> bool:
    """请求取消。任何 worker 都可调用。没有进行中的任务时返回 False。"""
    return bool(await _run_async(_mark_cancel))


async def clear_ingest_task(token: str) -> None:
    """仅当 token 匹配时清掉任务记录。状态行本身保留。"""
    await _run_async(_clear_task, token)


def ingest_cancel_requested_sync() -> bool:
    """同步读取取消标志，供入库线程在阶段边界轮询。数据库不可用时返回 False。"""
    try:
        return bool(_run_sync(_read_cancel))
    except Exception:
        return False


def acquire_ingest_lock_sync() -> str | None:
    return _run_sync(_acquire)


def save_ingest_task_sync(token: str, filenames, worker_id: str) -> None:
    _run_sync(_save_task, token, filenames, worker_id)


def get_ingest_status_sync() -> dict:
    return _run_sync(_read_status)


def mark_ingest_cancel_sync() -> bool:
    return bool(_run_sync(_mark_cancel))


def get_ingest_task_sync() -> dict | None:
    return _run_sync(_task_view)


def release_ingest_lock_sync(token: str) -> None:
    _run_sync(_release, token)
