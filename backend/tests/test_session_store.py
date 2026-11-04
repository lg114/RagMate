"""会话末尾对齐，以及未配置 Redis 时的进程内限流。"""
import asyncio

import pytest

from backend.domain.errors import ServiceUnavailableError, ValidationError
from backend.infrastructure.config import settings
from backend.infrastructure.rate_limiter import _buckets, check_rate_limit
from backend.infrastructure.session_store import pending_tail


def test_pending_tail_appends_only_the_new_turn():
    existing = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    desired = existing + [{"role": "user", "content": "next"}]
    assert pending_tail(existing, desired) == [{"role": "user", "content": "next"}]


def test_pending_tail_keeps_a_repeated_turn():
    existing = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    desired = existing + [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    assert pending_tail(existing, desired) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]


def test_pending_tail_is_empty_when_the_trimmed_window_matches():
    existing = [{"role": "user", "content": f"m{i}"} for i in range(30)]
    desired = existing[-4:]
    assert pending_tail(existing, desired) == []


def test_pending_tail_writes_a_brand_new_session():
    desired = [{"role": "user", "content": "hi"}]
    assert pending_tail([], desired) == desired


def test_memory_rate_limit_blocks_after_the_window_fills(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", "")
    monkeypatch.setattr(settings, "RATE_LIMIT_MAX", 2)
    monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW", 60)
    _buckets.clear()

    async def run():
        await check_rate_limit("203.0.113.5")
        await check_rate_limit("203.0.113.5")
        with pytest.raises(ValidationError):
            await check_rate_limit("203.0.113.5")

    asyncio.run(run())


def test_configured_redis_outage_is_explicit(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", "redis://127.0.0.1:6399/0")

    async def down():
        raise ConnectionError("redis down")

    monkeypatch.setattr("backend.infrastructure.redis_client.get_redis", down)

    async def run():
        with pytest.raises(ServiceUnavailableError) as caught:
            await check_rate_limit("203.0.113.9")
        assert "REDIS_URL" in caught.value.message

    asyncio.run(run())
