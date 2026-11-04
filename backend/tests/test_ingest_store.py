"""入库控制行在 SQLite 上的行为：互斥、取消、终态不过期。"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.domain.models import IngestControl
from backend.infrastructure.config import settings
from backend.infrastructure.database import Base
from backend.infrastructure.ingest_store import (
    _lock_ttl,
    _run_sync,
    acquire_ingest_lock_sync,
    get_ingest_status_sync,
    get_ingest_task_sync,
    ingest_cancel_requested_sync,
    mark_ingest_cancel_sync,
    release_ingest_lock_sync,
    save_ingest_task_sync,
    set_ingest_status_sync,
)


@pytest.fixture
def ingest_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'ingest.db'}")
    Base.metadata.create_all(engine, tables=[IngestControl.__table__])
    monkeypatch.setattr("backend.infrastructure.database.get_sync_engine", lambda: engine)
    return engine


def test_terminal_failure_survives_a_new_session(ingest_db):
    original = {"status": "failed", "error": "编码器崩溃", "stage": "encoding", "progress": 2, "total": 4}
    set_ingest_status_sync(original)
    assert "last_ingest" not in original

    again = get_ingest_status_sync()
    assert again["status"] == "failed"
    assert again["error"] == "编码器崩溃"
    assert again["stage"] == "encoding"
    assert again["progress"] == 2
    assert again["last_ingest"]

    # 再开一次连接，模拟进程重启。没有 TTL 把终态清掉。
    restarted = get_ingest_status_sync()
    assert restarted["status"] == "failed"
    assert restarted["error"] == "编码器崩溃"


def test_lock_is_exclusive_until_the_lease_ages(ingest_db):
    token = acquire_ingest_lock_sync()
    assert token
    assert acquire_ingest_lock_sync() is None
    ttl = _run_sync(_lock_ttl)
    assert ttl is not None and ttl > settings.INGEST_LOCK_TTL - 5

    aged = datetime.now(timezone.utc) + timedelta(
        seconds=settings.INGEST_LOCK_TTL - settings.INGEST_HEARTBEAT_STALE - 1
    )
    with Session(ingest_db) as session:
        row = session.get(IngestControl, 1)
        row.lock_expires_at = aged
        session.commit()

    stolen = acquire_ingest_lock_sync()
    assert stolen and stolen != token


def test_release_requires_the_matching_token(ingest_db):
    token = acquire_ingest_lock_sync()
    release_ingest_lock_sync("someone-else")
    assert acquire_ingest_lock_sync() is None
    release_ingest_lock_sync(token)
    assert acquire_ingest_lock_sync()


def test_cancel_flag_is_visible_to_a_sync_reader(ingest_db):
    token = acquire_ingest_lock_sync()
    save_ingest_task_sync(token, ["a.txt"], "worker-a")
    assert ingest_cancel_requested_sync() is False
    assert mark_ingest_cancel_sync() is True
    assert ingest_cancel_requested_sync() is True
    task = get_ingest_task_sync()
    assert task["token"] == token
    assert task["cancel"] == "1"
    assert task["worker_id"] == "worker-a"


def test_cancel_is_idle_without_a_task(ingest_db):
    assert mark_ingest_cancel_sync() is False
    assert get_ingest_task_sync() is None
    status = get_ingest_status_sync()
    assert status["status"] == "idle"
