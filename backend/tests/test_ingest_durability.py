"""入库状态、心跳和取消标志放在 Redis，而不是进程内的 task 句柄。"""
import asyncio
import threading
from datetime import datetime, timedelta, timezone

import pytest

from backend.application import ingest_manager
from backend.application.ingest import pipeline
from backend.infrastructure.encoding import EncodingCancelled, encode_documents


def test_task_is_live_when_heartbeat_is_fresh():
    now = datetime(2026, 11, 3, tzinfo=timezone.utc)
    task = {"token": "abc", "heartbeat_at": now.isoformat()}
    assert ingest_manager.ingest_task_is_live(task, "abc", now=now, stale_seconds=30) is True


def test_task_is_dead_when_heartbeat_is_stale():
    now = datetime(2026, 11, 3, tzinfo=timezone.utc)
    old = (now - timedelta(seconds=31)).isoformat()
    task = {"token": "abc", "heartbeat_at": old}
    assert ingest_manager.ingest_task_is_live(task, "abc", now=now, stale_seconds=30) is False


def test_task_is_dead_without_lock_or_on_token_mismatch():
    now = datetime(2026, 11, 3, tzinfo=timezone.utc)
    task = {"token": "abc", "heartbeat_at": now.isoformat()}
    assert ingest_manager.ingest_task_is_live(task, None, now=now, stale_seconds=30) is False
    assert ingest_manager.ingest_task_is_live(task, "other", now=now, stale_seconds=30) is False
    assert ingest_manager.ingest_task_is_live(None, "abc", now=now, stale_seconds=30) is False


def test_reconcile_interrupts_dead_task_and_releases_its_lock(monkeypatch):
    stored = {}

    async def fake_status():
        return {"status": "running", "stage": "encoding", "progress": 1, "total": 4}

    async def fake_task():
        return {"token": "t", "heartbeat_at": "2000-01-01T00:00:00+00:00"}

    async def fake_lock():
        return "t"

    async def fake_set(data):
        stored["status"] = data

    async def fake_release(token):
        stored["released"] = token

    async def fake_clear(token):
        stored["cleared"] = token

    monkeypatch.setattr(ingest_manager, "get_ingest_status", fake_status)
    monkeypatch.setattr(ingest_manager, "get_ingest_task", fake_task)
    monkeypatch.setattr(ingest_manager, "get_ingest_lock_token", fake_lock)
    monkeypatch.setattr(ingest_manager, "set_ingest_status", fake_set)
    monkeypatch.setattr(ingest_manager, "release_ingest_lock", fake_release)
    monkeypatch.setattr(ingest_manager, "clear_ingest_task", fake_clear)

    result = asyncio.run(ingest_manager.reconcile_stale_ingest())
    assert result["status"] == "interrupted"
    assert result["progress"] == 1
    assert result["total"] == 4
    assert "重新入库" in result["error"]
    assert stored["released"] == "t"
    assert stored["cleared"] == "t"
    assert stored["status"]["status"] == "interrupted"


def test_reconcile_releases_orphan_lock_once_its_ttl_has_aged(monkeypatch):
    released = {"yes": False}

    async def fake_status():
        return {"status": "running", "progress": 2, "total": 5}

    async def fake_task():
        return None

    async def fake_lock():
        return "orphan"

    async def fake_ttl():
        return 500  # 600 - 30 = 570，低于阈值视为没人续期

    async def fake_set(_data):
        return None

    async def fake_force():
        released["yes"] = True

    monkeypatch.setattr(ingest_manager, "get_ingest_status", fake_status)
    monkeypatch.setattr(ingest_manager, "get_ingest_task", fake_task)
    monkeypatch.setattr(ingest_manager, "get_ingest_lock_token", fake_lock)
    monkeypatch.setattr(ingest_manager, "get_ingest_lock_ttl", fake_ttl)
    monkeypatch.setattr(ingest_manager, "set_ingest_status", fake_set)
    monkeypatch.setattr(ingest_manager, "force_release_ingest_lock", fake_force)

    result = asyncio.run(ingest_manager.reconcile_stale_ingest())
    assert result["status"] == "interrupted"
    assert released["yes"] is True


def test_reconcile_keeps_a_fresh_lock_that_has_no_task_yet(monkeypatch):
    async def fake_status():
        return {"status": "running"}

    async def fake_task():
        return None

    async def fake_lock():
        return "brand-new"

    async def fake_ttl():
        return 599

    async def fake_set(_data):
        return None

    async def fail_force():
        raise AssertionError("fresh lock must stay")

    monkeypatch.setattr(ingest_manager, "get_ingest_status", fake_status)
    monkeypatch.setattr(ingest_manager, "get_ingest_task", fake_task)
    monkeypatch.setattr(ingest_manager, "get_ingest_lock_token", fake_lock)
    monkeypatch.setattr(ingest_manager, "get_ingest_lock_ttl", fake_ttl)
    monkeypatch.setattr(ingest_manager, "set_ingest_status", fake_set)
    monkeypatch.setattr(ingest_manager, "force_release_ingest_lock", fail_force)

    result = asyncio.run(ingest_manager.reconcile_stale_ingest())
    assert result["status"] == "interrupted"


def test_reconcile_leaves_a_live_task_untouched(monkeypatch):
    now = datetime.now(timezone.utc)

    async def fake_status():
        return {"status": "running", "progress": 2, "total": 4}

    async def fake_task():
        return {"token": "t", "heartbeat_at": now.isoformat()}

    async def fake_lock():
        return "t"

    async def fail_set(data):
        raise AssertionError("live task must not be rewritten")

    monkeypatch.setattr(ingest_manager, "get_ingest_status", fake_status)
    monkeypatch.setattr(ingest_manager, "get_ingest_task", fake_task)
    monkeypatch.setattr(ingest_manager, "get_ingest_lock_token", fake_lock)
    monkeypatch.setattr(ingest_manager, "set_ingest_status", fail_set)

    result = asyncio.run(ingest_manager.reconcile_stale_ingest())
    assert result["status"] == "running"
    assert result["progress"] == 2


def test_cancel_request_reaches_redis_without_a_local_task(monkeypatch):
    called = {"n": 0}

    async def fake_mark():
        called["n"] += 1
        return True

    monkeypatch.setattr(ingest_manager, "mark_ingest_cancel", fake_mark)
    ingest_manager._ingest_task = None
    ingest_manager._ingest_cancel_event = None

    result = asyncio.run(ingest_manager.request_ingest_cancellation())
    assert result == {"status": "cancelling"}
    assert called["n"] == 1


def test_cancel_request_is_idle_when_nothing_is_running(monkeypatch):
    async def fake_mark():
        return False

    monkeypatch.setattr(ingest_manager, "mark_ingest_cancel", fake_mark)
    ingest_manager._ingest_task = None
    ingest_manager._ingest_cancel_event = None
    result = asyncio.run(ingest_manager.request_ingest_cancellation())
    assert result == {"status": "idle"}


def test_local_task_short_circuits_start(monkeypatch):
    async def scenario():
        async def hang():
            await asyncio.sleep(30)

        ingest_manager._ingest_task = asyncio.create_task(hang())
        try:
            async def fail_acquire():
                raise AssertionError("must not acquire a second lock")

            monkeypatch.setattr(ingest_manager, "acquire_ingest_lock", fail_acquire)
            result = await ingest_manager.start_ingest(["a.pdf"])
            assert result == {"status": "already_running"}
        finally:
            ingest_manager._ingest_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await ingest_manager._ingest_task
            ingest_manager._ingest_task = None

    asyncio.run(scenario())


def test_load_and_split_stops_before_touching_files(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("hello")

    def boom(_data):
        raise AssertionError("should not report progress after cancel")

    monkeypatch.setattr(pipeline, "set_ingest_status_sync", boom)
    cancel_event = threading.Event()
    cancel_event.set()
    chunks, counter, failed = pipeline._load_and_split(str(tmp_path), ["a.txt"], cancel_event)
    assert chunks == []
    assert failed == []
    assert counter["a.txt"] == 0


def test_encode_documents_honours_cancel_without_loading_the_model(monkeypatch):
    def boom():
        raise AssertionError("model should not load")

    monkeypatch.setattr("backend.infrastructure.encoding.get_bge_m3", boom)
    cancel_event = threading.Event()
    cancel_event.set()
    with pytest.raises(EncodingCancelled):
        encode_documents(["hello"], cancel_event)
