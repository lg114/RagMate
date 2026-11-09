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

    def test_remote_request_without_token_blocked(self, client, monkeypatch):
        monkeypatch.setattr(settings, "API_AUTH_TOKEN", "")
        response = client.post("/chat", json={"message": "hi"}, headers={"Host": "remote.external.com"})
        assert response.status_code == 403
        assert response.json()["code"] == "REMOTE_ACCESS_DISABLED" 

    def test_docker_bridge_gateway_with_localhost_host_is_accepted(self, client, monkeypatch):
        """Docker 内部网桥转发的请求（如 172.19.0.1）携带 Host: localhost 视为本机请求免密放行。"""
        monkeypatch.setattr(settings, "API_AUTH_TOKEN", "")
        client_with_docker_ip = TestClient(client.app, client=("172.19.0.1", 45678))
        response = client_with_docker_ip.get("/auth/status", headers={"Host": "localhost:8000"})
        assert response.status_code == 200
        data = response.json()
        assert data["is_local"] is True
        assert data["auth_required"] is False
        assert data["authenticated"] is True


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

    def test_login_and_cookie_session_flow(self, client, with_token):
        # 1. Login with bad password -> 401
        res_bad = client.post("/auth/login", json={"password": "wrong"})
        assert res_bad.status_code == 401

        # 2. Login with correct password -> 200 and set cookie
        res_ok = client.post("/auth/login", json={"password": "test-secret"})
        assert res_ok.status_code == 200
        assert res_ok.json()["ok"] is True
        session_token = res_ok.json()["token"]
        assert len(session_token) > 20

        # 3. Access protected endpoint using the returned session cookie
        res_chat = client.post("/chat", json={"message": "hi"})
        assert res_chat.status_code != 401

        # 4. Access protected endpoint using Bearer session token
        res_bearer = client.post(
            "/chat",
            json={"message": "hi"},
            headers={"Authorization": f"Bearer {session_token}"},
        )
        assert res_bearer.status_code != 401

        # 5. Logout invalidates session
        res_logout = client.post("/auth/logout")
        assert res_logout.status_code == 200

        # Now calling with old session token gets rejected
        client.cookies.clear()
        res_revoked = client.post(
            "/chat",
            json={"message": "hi"},
            headers={"Authorization": f"Bearer {session_token}"},
        )
        assert res_revoked.status_code == 401

    def test_auth_status_endpoint(self, client, with_token):
        res = client.get("/auth/status")
        assert res.status_code == 200
        data = res.json()
        assert data["auth_required"] is True
        assert data["authenticated"] is False

        # After logging in
        client.post("/auth/login", json={"password": "test-secret"})
        res_after = client.get("/auth/status")
        assert res_after.json()["authenticated"] is True
