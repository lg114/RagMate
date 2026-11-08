"""Unit tests for Settings API (PR 1: Security, Validation, and Immutability)."""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.app import app
from backend.infrastructure.config import settings, _ENV_FILE


@pytest.fixture
def client():
    return TestClient(app)


def test_management_disabled_by_default_when_no_token(client, monkeypatch):
    """When API_AUTH_TOKEN is not configured on the server, management API must return 403."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "")
    
    res = client.get("/settings")
    assert res.status_code == 403
    data = res.json()
    assert data.get("code") == "MANAGEMENT_DISABLED"
    assert "not configured" in data.get("detail", "").lower()

    res_put = client.put("/settings", json={"settings": {"LLM_MODEL": "gpt-4o"}})
    assert res_put.status_code == 403
    assert res_put.json().get("code") == "MANAGEMENT_DISABLED"


def test_management_auth_with_token(client, monkeypatch):
    """When API_AUTH_TOKEN is configured, access requires valid bearer token or X-API-Key."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-admin-secret-token")

    # 1. No token -> 401
    res_no_auth = client.get("/settings")
    assert res_no_auth.status_code == 401
    assert res_no_auth.json().get("code") == "UNAUTHORIZED"

    # 2. Wrong token -> 401
    res_bad_auth = client.get("/settings", headers={"Authorization": "Bearer wrong-token"})
    assert res_bad_auth.status_code == 401
    assert res_bad_auth.json().get("code") == "UNAUTHORIZED"

    # 3. Valid Bearer Token -> 200
    res_ok_bearer = client.get(
        "/settings", headers={"Authorization": "Bearer test-admin-secret-token"}
    )
    assert res_ok_bearer.status_code == 200

    # 4. Valid X-API-Key header -> 200
    res_ok_key = client.get(
        "/settings", headers={"X-API-Key": "test-admin-secret-token"}
    )
    assert res_ok_key.status_code == 200


def test_database_url_not_leaked(client, monkeypatch):
    """GET /settings must NEVER leak DATABASE_URL or database credentials."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}

    res = client.get("/settings", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "DATABASE_URL" not in data["settings"]


def test_secrets_masked_completely(client, monkeypatch):
    """Secrets must be empty string and expose boolean configured status only."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}

    res = client.get("/settings", headers=headers)
    assert res.status_code == 200
    data = res.json()
    s = data["settings"]

    assert s["LLM_API_KEY"] == ""
    assert isinstance(s["LLM_API_KEY_configured"], bool)
    assert s["EMBEDDING_API_KEY"] == ""
    assert isinstance(s["EMBEDDING_API_KEY_configured"], bool)
    assert s["RERANKER_API_KEY"] == ""
    assert isinstance(s["RERANKER_API_KEY_configured"], bool)


def test_non_editable_fields_rejected(client, monkeypatch):
    """Attempting to edit deployment-managed parameters (DB, chunking, CORS) must return 400."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}

    # Attempt to change DATABASE_URL
    res_db = client.put(
        "/settings",
        json={"settings": {"DATABASE_URL": "postgresql://hacker:pw@localhost/db"}},
        headers=headers,
    )
    assert res_db.status_code == 400
    assert "not editable" in res_db.json()["detail"].lower()

    # Attempt to change CHUNK_SIZE
    res_chunk = client.put(
        "/settings",
        json={"settings": {"CHUNK_SIZE": 500}},
        headers=headers,
    )
    assert res_chunk.status_code == 400
    assert "not editable" in res_chunk.json()["detail"].lower()

    # Attempt to change CORS_ORIGINS
    res_cors = client.put(
        "/settings",
        json={"settings": {"CORS_ORIGINS": "*"}},
        headers=headers,
    )
    assert res_cors.status_code == 400
    assert "not editable" in res_cors.json()["detail"].lower()


def test_temperature_range_validation(client, monkeypatch):
    """LLM_TEMPERATURE must be between 0.0 and 2.0."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}

    res_too_high = client.put(
        "/settings",
        json={"settings": {"LLM_TEMPERATURE": 2.5}},
        headers=headers,
    )
    assert res_too_high.status_code == 400
    assert "temperature" in res_too_high.json()["detail"].lower()

    res_negative = client.put(
        "/settings",
        json={"settings": {"LLM_TEMPERATURE": -0.5}},
        headers=headers,
    )
    assert res_negative.status_code == 400


def test_url_validation(client, monkeypatch):
    """Invalid URLs must be rejected before touching .env."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}

    res = client.put(
        "/settings",
        json={"settings": {"LLM_API_BASE_URL": "ftp://not-http"}},
        headers=headers,
    )
    assert res.status_code == 400
    assert "valid http or https url" in res.json()["detail"].lower()


def test_remote_embedding_requires_url(client, monkeypatch):
    """Setting EMBEDDING_PROVIDER to remote without a URL must be rejected."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}

    res = client.put(
        "/settings",
        json={"settings": {"EMBEDDING_PROVIDER": "remote", "EMBEDDING_API_BASE_URL": ""}},
        headers=headers,
    )
    assert res.status_code == 400
    assert "embedding_api_base_url is required" in res.json()["detail"].lower()


def test_optimistic_locking_conflict(client, monkeypatch):
    """Submitting an outdated version hash must return 409 Conflict."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}

    res = client.put(
        "/settings",
        json={"settings": {"LLM_TEMPERATURE": 0.5}, "version": "outdated_stale_version_hash"},
        headers=headers,
    )
    assert res.status_code == 409
    assert "modified in another session" in res.json()["detail"].lower()


def test_reset_endpoint_removed(client, monkeypatch):
    """POST /settings/reset must no longer exist (404 or 405)."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}

    res = client.post("/settings/reset", headers=headers)
    assert res.status_code in (404, 405)
