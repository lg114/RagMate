"""会话末尾对齐，以及进程内限流。"""
import asyncio

import pytest

from backend.domain.errors import ValidationError
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
    monkeypatch.setattr(settings, "RATE_LIMIT_MAX", 2)
    monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW", 60)
    _buckets.clear()

    async def run():
        await check_rate_limit("203.0.113.5")
        await check_rate_limit("203.0.113.5")
        with pytest.raises(ValidationError):
            await check_rate_limit("203.0.113.5")

    asyncio.run(run())
