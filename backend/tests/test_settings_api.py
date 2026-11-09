"""Unit tests for Settings API (PR 1: Security, Validation, and Immutability)."""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.app import app
from backend.infrastructure.config import settings, _ENV_FILE


@pytest.fixture
def client():
    return TestClient(app)


def test_management_disabled_for_remote_when_no_token(client, monkeypatch):
    """When API_AUTH_TOKEN is not configured, remote callers receive 403."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "")
    
    res = client.get("/settings", headers={"Host": "remote.external.com"})
    assert res.status_code == 403
    data = res.json()
    assert data.get("code") == "MANAGEMENT_DISABLED"
    assert "not configured" in data.get("detail", "").lower()

    res_put = client.put("/settings", json={"settings": {"LLM_MODEL": "gpt-4o"}}, headers={"Host": "remote.external.com"})
    assert res_put.status_code == 403
    assert res_put.json().get("code") == "MANAGEMENT_DISABLED"


def test_management_allowed_for_localhost_when_no_token(client, monkeypatch):
    """When API_AUTH_TOKEN is not configured, localhost requests are allowed directly."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "")
    res = client.get("/settings")
    assert res.status_code == 200
    assert "settings" in res.json()


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


def test_settings_returns_sources_and_index_status(client, monkeypatch):
    """GET /settings must report field_sources, index_status, and pending_restart."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}

    res = client.get("/settings", headers=headers)
    assert res.status_code == 200
    data = res.json()

    assert "field_sources" in data
    assert "index_status" in data
    assert "pending_restart" in data

    sources = data["field_sources"]
    assert sources.get("DATABASE_URL") == "environment"  # Non-editable is deployment managed

    index_status = data["index_status"]
    assert "has_indexed_docs" in index_status
    assert "chunk_count" in index_status
    assert index_status.get("dimension") == 1024


def test_environment_managed_field_rejected(client, monkeypatch):
    """Fields managed by the deployment environment cannot be updated via web API."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}

    # Simulate DEPLOY_MANAGED_KEYS protecting LLM_MODEL
    monkeypatch.setenv("DEPLOY_MANAGED_KEYS", "LLM_MODEL")

    res = client.put(
        "/settings",
        json={"settings": {"LLM_MODEL": "gpt-5-future"}},
        headers=headers,
    )
    assert res.status_code == 400
    assert "managed by the deployment environment" in res.json()["detail"].lower()


def test_index_locking_prevents_changing_embedding_model(client, monkeypatch):
    """When vector store contains indexed chunks, changing embedding model/provider is blocked."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}

    # Mock indexed documents exist
    monkeypatch.setattr("backend.api.settings.get_indexed_chunks_count", lambda *args: 42)
    monkeypatch.setattr(
        "backend.api.settings.get_vector_index_metadata",
        lambda *args: {
            "embedding_provider": "remote",
            "embedding_model": "BAAI/bge-m3",
            "embedding_base_url": "http://127.0.0.1:8081/v1",
            "dimension": 1024,
        },
    )

    # 1. Attempting to change EMBEDDING_MODEL must fail with 400
    res = client.put(
        "/settings",
        json={"settings": {"EMBEDDING_MODEL": "text-embedding-3-small"}},
        headers=headers,
    )
    assert res.status_code == 400
    assert "cannot modify 'embedding_model'" in res.json()["detail"].lower()
    assert "re-indexing" in res.json()["detail"].lower()

    # 2. Attempting to change EMBEDDING_PROVIDER must fail with 400
    res_prov = client.put(
        "/settings",
        json={"settings": {"EMBEDDING_PROVIDER": "huggingface"}},
        headers=headers,
    )
    assert res_prov.status_code == 400
    assert "cannot modify 'embedding_provider'" in res_prov.json()["detail"].lower()

    # 3. But updating EMBEDDING_API_KEY (secret) is allowed
    res_key = client.put(
        "/settings",
        json={"settings": {"EMBEDDING_API_KEY": "new-secret-key"}},
        headers=headers,
    )
    assert res_key.status_code == 200
    assert "EMBEDDING_API_KEY" in res_key.json()["updated"]


def test_pending_restart_tracking(client, monkeypatch):
    """Modifying a setting on disk indicates it is pending restart."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    monkeypatch.setattr(settings, "LLM_TEMPERATURE", 0.01)
    headers = {"Authorization": "Bearer test-token"}

    res = client.put(
        "/settings",
        json={"settings": {"LLM_TEMPERATURE": 0.85}},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert "LLM_TEMPERATURE" in data["pending_restart"]


def test_vector_index_compatibility_logic(monkeypatch):
    """Test validate_index_compatibility function under various scenarios."""
    from backend.infrastructure.vector_store import validate_index_compatibility

    # Case 1: No indexed chunks
    monkeypatch.setattr("backend.infrastructure.vector_store.PgVectorStore.count_chunks", lambda self: 0)
    ok, reason = validate_index_compatibility("test_table")
    assert ok is True

    # Case 2: Indexed chunks exist but metadata missing -> unconfirmed
    monkeypatch.setattr("backend.infrastructure.vector_store.PgVectorStore.count_chunks", lambda self: 100)
    monkeypatch.setattr("backend.infrastructure.vector_store.PgVectorStore.get_index_metadata", lambda self: None)
    ok, reason = validate_index_compatibility("test_table")
    assert ok is False
    assert "metadata is missing" in reason

    # Case 3: Metadata dimension mismatch
    monkeypatch.setattr(
        "backend.infrastructure.vector_store.PgVectorStore.get_index_metadata",
        lambda self: {
            "embedding_provider": "remote",
            "embedding_model": "BAAI/bge-m3",
            "dimension": 512,
        },
    )
    ok, reason = validate_index_compatibility("test_table")
    assert ok is False
    assert "dimension mismatch" in reason.lower()

    # Case 4: Model mismatch
    monkeypatch.setattr(
        "backend.infrastructure.vector_store.PgVectorStore.get_index_metadata",
        lambda self: {
            "embedding_provider": "remote",
            "embedding_model": "old-model",
            "embedding_base_url": "",
            "dimension": 1024,
        },
    )
    ok, reason = validate_index_compatibility("test_table", model="new-model", provider="remote")
    assert ok is False
    assert "model mismatch" in reason.lower()

    # Case 5: Fully compatible
    monkeypatch.setattr(
        "backend.infrastructure.vector_store.PgVectorStore.get_index_metadata",
        lambda self: {
            "embedding_provider": "remote",
            "embedding_model": "BAAI/bge-m3",
            "embedding_base_url": "http://127.0.0.1:8081/v1",
            "dimension": 1024,
        },
    )
    ok, reason = validate_index_compatibility(
        "test_table",
        model="BAAI/bge-m3",
        provider="remote",
        base_url="http://127.0.0.1:8081/v1",
    )
    assert ok is True


def test_test_endpoints_require_auth(client, monkeypatch):
    """Connection test endpoints require authentication."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")

    # 1. Without auth header -> 401
    res = client.post("/settings/test/llm")
    assert res.status_code == 401

    # 2. When server token is empty and caller is remote -> 403 MANAGEMENT_DISABLED
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "")
    res_disabled = client.post("/settings/test/llm", headers={"Host": "remote.external.com"})
    assert res_disabled.status_code == 403
    assert res_disabled.json()["code"] == "MANAGEMENT_DISABLED" 


def test_test_embedding_checks_1024_dimension(client, monkeypatch):
    """Embedding test must reject endpoints returning anything other than 1024 dimensions."""
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}

    class _MockResponse:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data

        def json(self):
            return self._data

    # Case A: returns 1536 dimensions (e.g. text-embedding-3-large)
    async def mock_post_1536(self, *args, **kwargs):
        return _MockResponse(200, {"data": [{"embedding": [0.0] * 1536}]})

    monkeypatch.setattr("httpx.AsyncClient.post", mock_post_1536)
    res = client.post("/settings/test/embedding", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is False
    assert data["dimension"] == 1536
    assert "incompatible dimension" in data["detail"].lower()

    # Case B: returns exactly 1024 dimensions
    async def mock_post_1024(self, *args, **kwargs):
        return _MockResponse(200, {"data": [{"embedding": [0.0] * 1024}]})

    monkeypatch.setattr("httpx.AsyncClient.post", mock_post_1024)
    res_ok = client.post("/settings/test/embedding", headers=headers)
    assert res_ok.status_code == 200
    data_ok = res_ok.json()
    assert data_ok["ok"] is True
    assert data_ok["dimension"] == 1024

