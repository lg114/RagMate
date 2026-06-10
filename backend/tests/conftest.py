"""共享 fixture。"""
import pytest

from backend.infrastructure.config import settings


@pytest.fixture
def override_settings(monkeypatch):
    """临时修改 settings 属性，测完自动还原。"""

    def _override(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setattr(settings, key, value)

    return _override
