"""可选 API Token 鉴权（API_AUTH_TOKEN）单元测试。

覆盖：未设置时不拦、设置后无 token/错 token 返回 401、Bearer 与 X-API-Key
两种带法都放行、健康检查不拦。
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.infrastructure.config import settings


@pytest.fixture
def client():
    """与 test_api_smoke 相同的 mock 基础设施，避免依赖真实 DB。"""
    with patch("backend.app.init_db", new_callable=AsyncMock), patch(
        "backend.infrastructure.database.async_session"
    ) as mock_session, patch(
        "backend.infrastructure.vector_store.check_vector_store_available", return_value=True
    ), patch(
        "backend.application.chat.load_session", new=AsyncMock(return_value=[])
    ), patch(
        "backend.application.chat.save_session", new=AsyncMock()
    ), patch(
        "backend.application.chat._persist_messages", new=AsyncMock()
    ):
        session_instance = MagicMock()
        session_instance.__enter__ = MagicMock(return_value=session_instance)
        session_instance.__exit__ = MagicMock(return_value=None)
        session_instance.execute = AsyncMock()
        session_instance.commit = AsyncMock()
        mock_session.return_value = session_instance

        from backend.app import create_app

        # raise_server_exceptions=False：本文件只断言鉴权中间件是否放行，
        # 放行后下游 handler 的错误应表现为 500 响应而非在测试里抛出。
        yield TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture
def with_token(monkeypatch):
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-secret")


class TestAuthDisabled:
    def test_no_token_required_when_unset(self, client, monkeypatch):
        monkeypatch.setattr(settings, "API_AUTH_TOKEN", "")
        response = client.post("/chat", json={"message": "hi"})
        assert response.status_code != 401


class TestAuthEnabled:
    def test_missing_token_rejected(self, client, with_token):
        response = client.post("/chat", json={"message": "hi"})
        assert response.status_code == 401
        assert response.json()["code"] == "UNAUTHORIZED"

    def test_wrong_token_rejected(self, client, with_token):
        response = client.post(
            "/chat",
            json={"message": "hi"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401

    def test_bearer_token_accepted(self, client, with_token):
        response = client.post(
            "/chat",
            json={"message": "hi"},
            headers={"Authorization": "Bearer test-secret"},
        )
        assert response.status_code != 401

    def test_x_api_key_accepted(self, client, with_token):
        response = client.post(
            "/chat",
            json={"message": "hi"},
            headers={"X-API-Key": "test-secret"},
        )
        assert response.status_code != 401

    def test_documents_and_ingest_also_protected(self, client, with_token):
        assert client.get("/documents").status_code == 401
        assert client.get("/ingest/status").status_code == 401

    def test_health_not_protected(self, client, with_token):
        response = client.get("/health")
        assert response.status_code == 200
