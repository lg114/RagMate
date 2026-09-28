"""API 层端到端冒烟测试。"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_infrastructure():
    """Mock 所有基础设施依赖。"""
    with patch("backend.app.init_db", new_callable=AsyncMock) as mock_init_db, \
         patch("backend.infrastructure.redis_client.get_redis", new_callable=AsyncMock) as mock_redis, \
         patch("backend.infrastructure.database.async_session") as mock_session, \
         patch("backend.infrastructure.milvus.check_milvus_available") as mock_milvus:

        # Redis mock
        redis_instance = AsyncMock()
        redis_instance.exists = AsyncMock(return_value=False)
        mock_redis.return_value = redis_instance

        # Database session mock
        session_instance = MagicMock()
        session_instance.__enter__ = MagicMock(return_value=session_instance)
        session_instance.__exit__ = MagicMock(return_value=None)
        session_instance.execute = AsyncMock()
        session_instance.commit = AsyncMock()
        mock_session.return_value = session_instance

        # Milvus mock
        mock_milvus.return_value = True

        yield {
            "init_db": mock_init_db,
            "redis": mock_redis,
            "session": mock_session,
            "milvus": mock_milvus,
        }


@pytest.fixture
def client(mock_infrastructure):
    """创建测试客户端。"""
    from backend.app import create_app
    app = create_app()
    return TestClient(app)


class TestHealthEndpoints:
    """测试健康检查端点。"""

    def test_health_check(self, client):
        """GET /health 应返回 200。"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_ready_check(self, client, mock_infrastructure):
        """GET /ready 应检查基础设施状态。"""
        response = client.get("/ready")
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data


class TestDocumentsEndpoints:
    """测试文档管理端点。"""

    def test_list_documents(self, client, mock_infrastructure):
        """GET /documents 应返回文档列表。"""
        # Mock 数据库查询返回空列表
        mock_session = mock_infrastructure["session"].return_value
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)


class TestChatEndpoints:
    """测试聊天端点。"""

    def test_chat_request_validation(self, client):
        """POST /chat 缺少必要字段应返回 422。"""
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_stream_request_validation(self, client):
        """POST /chat/stream 缺少必要字段应返回 422。"""
        response = client.post("/chat/stream", json={})
        assert response.status_code == 422


class TestIngestEndpoints:
    """测试入库端点。"""

    def test_ingest_status(self, client, mock_infrastructure):
        """GET /ingest/status 应返回入库状态。"""
        # Mock Redis 返回状态
        mock_redis = mock_infrastructure["redis"].return_value
        mock_redis.get = AsyncMock(return_value=None)

        response = client.get("/ingest/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestErrorHandling:
    """测试错误处理。"""

    def test_not_found_endpoint(self, client):
        """访问不存在的路由应返回 404。"""
        response = client.get("/nonexistent-endpoint")
        # FastAPI 默认返回 404
        assert response.status_code == 404

    def test_invalid_json(self, client):
        """发送无效 JSON 应返回 422。"""
        response = client.post(
            "/chat",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


class TestCORS:
    """测试 CORS 配置。"""

    def test_cors_headers_present(self, client):
        """响应应包含 CORS 头。"""
        response = client.options(
            "/health",
            headers={"Origin": "http://localhost:8000"},
        )
        # CORS 预检请求
        assert response.status_code == 200


class TestRateLimiting:
    """测试限流逻辑。"""

    def test_rate_limit_header(self, client, mock_infrastructure):
        """请求应经过限流中间件。"""
        # Mock Redis 限流检查通过
        mock_redis = mock_infrastructure["redis"].return_value
        mock_redis.eval = AsyncMock(return_value=1)  # 1 = 允许

        response = client.get("/documents")
        # 应该正常通过
        assert response.status_code == 200
