"""RagMate 应用工厂：创建 FastAPI 实例，注册中间件、路由、异常处理。"""
import asyncio
import contextvars
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.infrastructure.config import settings
from backend.infrastructure.database import engine, init_db
from backend.domain.errors import AppError

# 请求级上下文：每个请求生成唯一 request_id，贯穿整个调用链
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


# 配置 ragmate logger
logger = logging.getLogger("ragmate")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] [%(request_id)s] %(message)s"))
_handler.addFilter(_RequestIdFilter())
logger.addHandler(_handler)
logger.propagate = False

# LangSmith Tracing
if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.application.ingest_manager import cancel_ingest, reconcile_stale_ingest

    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

    # 表就绪后再对账：心跳已死的 running 收成 interrupted，并放开对应的锁。
    try:
        state = await reconcile_stale_ingest()
        if state.get("status") == "interrupted":
            logger.warning("Reconciled a stale ingest task on startup: %s", state.get("error"))
    except Exception as e:
        logger.warning(f"Failed to reconcile ingest state on startup: {e}")

    # 后台预热本地模型。远程嵌入 / 远程 rerank 不在 API 进程里加载权重。
    def _warmup_models():
        if settings.RERANKER_PROVIDER == "local":
            try:
                from backend.core.retriever import get_reranker
                get_reranker()
                logger.info("Reranker model warmed up")
            except Exception as e:
                logger.warning(f"Reranker warmup failed: {e}")
        else:
            logger.info("Skipping local reranker warmup (RERANKER_PROVIDER=%s)", settings.RERANKER_PROVIDER)
        from backend.infrastructure.encoding import get_bge_m3, uses_remote_embeddings
        if uses_remote_embeddings():
            logger.info("Skipping local BGE-M3 warmup (EMBEDDING_PROVIDER=%s)", settings.EMBEDDING_PROVIDER)
        else:
            try:
                get_bge_m3()
                logger.info("BGE-M3 model warmed up")
            except Exception as e:
                logger.warning(f"BGE-M3 warmup failed: {e}")

    _warmup_task = asyncio.get_running_loop().run_in_executor(None, _warmup_models)

    try:
        yield
    finally:
        await cancel_ingest()
        try:
            await engine.dispose()
        except Exception:
            logger.debug("Failed to dispose async engine on shutdown", exc_info=True)
        try:
            from backend.infrastructure.database import get_sync_engine
            sync_engine = get_sync_engine()
            if sync_engine is not None:
                await asyncio.to_thread(sync_engine.dispose)
        except Exception:
            logger.debug("Failed to dispose sync engine on shutdown", exc_info=True)


def create_app() -> FastAPI:
    app = FastAPI(title="RagMate API", lifespan=lifespan)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
        max_age=3600,
    )

    # 可选 API Token 鉴权中间件：API_AUTH_TOKEN 为空时不启用（默认）。
    # 启用后，/chat、/documents、/ingest 下的接口必须带
    # Authorization: Bearer <token> 或 X-API-Key: <token>，否则 401。
    # 健康检查与前端静态资源不拦（探针与页面本身没有 token）。
    # API Token 鉴权中间件：
    # 1. 管理接口（/settings）：
    #    - 若服务端未配置 API_AUTH_TOKEN，默认关闭管理功能（403 MANAGEMENT_DISABLED），防止公网匿名篡改。
    #    - 若已配置，必须携带匹配的 Bearer Token 或 X-API-Key（401 UNAUTHORIZED）。
    # 2. 业务接口（/chat, /documents, /ingest）：
    #    - API_AUTH_TOKEN 为空时不鉴权（单机/内网默认）；非空时需验证。
    # 3. 探针与静态资源（/health, /ready, 静态文件）：不拦截。
    from backend.api.auth import is_local_request, is_authenticated

    _ALL_PROTECTED_PREFIXES = ("/chat", "/documents", "/ingest", "/settings")

    @app.middleware("http")
    async def _require_api_token(request, call_next):
        token = settings.API_AUTH_TOKEN
        path = request.url.path

        # 探针、认证端点、静态前端不拦截
        if (
            path in ("/health", "/ready")
            or path.startswith("/auth/")
            or not any(path == p or path.startswith(p + "/") for p in _ALL_PROTECTED_PREFIXES)
        ):
            return await call_next(request)

        is_local = is_local_request(request)

        # 场景 1：服务端未配置 API_AUTH_TOKEN（密码）
        if not token:
            if not is_local:
                # 局域网或外部请求在未设密码时严格禁止访问受保护接口
                return JSONResponse(
                    status_code=403,
                    content={
                        "code": "MANAGEMENT_DISABLED" if path.startswith("/settings") else "REMOTE_ACCESS_DISABLED",
                        "detail": "Remote access is disabled because API_AUTH_TOKEN is not configured on the server.",
                    },
                )
            # 仅本机使用（127.0.0.1 / localhost）：免密放行！
            return await call_next(request)

        # 场景 2：服务端已配置 API_AUTH_TOKEN -> 要求统一密码鉴权
        if not is_authenticated(request, token):
            return JSONResponse(
                status_code=401,
                content={"code": "UNAUTHORIZED", "detail": "Missing or invalid API token or session."},
            )

        return await call_next(request)

    # 上传大小限制中间件：包装 ASGI receive，流式检查实际 body 大小
    # 不依赖 Content-Length 头（可伪造），而是检查实际接收的字节数

    class _PayloadTooLarge(Exception):
        pass

    @app.middleware("http")
    async def _limit_upload_size(request, call_next):
        if request.url.path == "/documents/upload":
            received = 0
            original_receive = request._receive

            async def limiting_receive():
                nonlocal received
                msg = await original_receive()
                if msg.get("type") == "http.request":
                    received += len(msg.get("body", b""))
                    if received > settings.MAX_UPLOAD_SIZE:
                        raise _PayloadTooLarge()
                return msg

            request._receive = limiting_receive
            try:
                return await call_next(request)
            except _PayloadTooLarge:
                # 丢弃剩余 body，避免连接处于脏状态
                while True:
                    msg = await original_receive()
                    if msg.get("type") != "http.request" or not msg.get("more_body"):
                        break
                max_mb = settings.MAX_UPLOAD_SIZE // (1024 * 1024)
                return JSONResponse(
                    status_code=413,
                    content={"code": "PAYLOAD_TOO_LARGE", "detail": f"File exceeds {max_mb}MB limit"},
                )

        return await call_next(request)

    # Request ID 中间件
    @app.middleware("http")
    async def _set_request_id(request, call_next):
        token = request_id_var.set(request.headers.get("x-request-id", uuid.uuid4().hex[:12]))
        try:
            return await call_next(request)
        finally:
            request_id_var.reset(token)

    # 异常处理
    @app.exception_handler(AppError)
    async def app_error_handler(request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content={"code": exc.code, "detail": exc.message})

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error("Unhandled exception", exc_info=(type(exc), exc, exc.__traceback__))
        return JSONResponse(status_code=500, content={"code": "INTERNAL_ERROR", "detail": "An unexpected error occurred"})

    # 注册路由
    from backend.api.health import router as health_router
    from backend.api.chat import router as chat_router
    from backend.api.documents import router as documents_router
    from backend.api.ingest import router as ingest_router
    from backend.api.settings import router as settings_router
    from backend.api.auth import router as auth_router

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(settings_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    app.include_router(ingest_router)

    # 静态文件（必须最后注册）
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    if os.path.exists(frontend_dir):
        from starlette.responses import FileResponse

        @app.exception_handler(404)
        async def not_found_fallback(request, exc):
            path = request.url.path
            if path.startswith(("/chat", "/documents", "/ingest", "/health", "/ready")):
                return JSONResponse(status_code=404, content={"code": "NOT_FOUND", "detail": "Endpoint not found"})
            index_path = os.path.join(frontend_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return JSONResponse(status_code=404, content={"code": "NOT_FOUND", "detail": "Not found"})

        app.mount("/", StaticFiles(directory=frontend_dir), name="frontend")

    return app


app = create_app()
