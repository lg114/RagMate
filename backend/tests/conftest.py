"""共享 fixture。"""
import os
from pathlib import Path

import pytest

# Settings() 在 import 时校验。默认嵌入和重排走远程 HTTP，URL 为空会拒绝启动。
# 干净 checkout（CI、没有 backend/.env）补上占位地址，测试自己 mock HTTP，不会真的去连。
# 已有 .env 或环境变量时不覆盖，避免盖掉本机配置。
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if not _ENV_FILE.exists():
    os.environ.setdefault(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/ragmate",
    )
    os.environ.setdefault("LLM_API_KEY", "test-key")
    os.environ.setdefault("EMBEDDING_API_BASE_URL", "http://embed.test/v1")
    os.environ.setdefault("RERANKER_API_BASE_URL", "http://rerank.test/v1")

from backend.infrastructure.config import settings


@pytest.fixture
def override_settings(monkeypatch):
    """临时修改 settings 属性，测完自动还原。"""

    def _override(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setattr(settings, key, value)

    return _override
