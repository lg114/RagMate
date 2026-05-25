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

from config import settings
from database import engine, init_db
from errors import AppError

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
    from redis_client import get_redis, close_sync_redis
    from ingest_manager import cancel_ingest

    # 检查是否有遗留的 ingest 锁
    try:
        r = await get_redis()
        lock_exists = await r.exists("ragmate:ingest:lock")
        if lock_exists:
            logger.warning("Found existing ingest lock on startup — another instance may be running, or a previous run crashed.")
    except Exception as e:
        logger.warning(f"Failed to check ingest lock on startup: {e}")

    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

    # 后台预热模型
    def _warmup_models():
        try:
            from retriever import get_reranker
            get_reranker()
            logger.info("Reranker model warmed up")
        except Exception as e:
            logger.warning(f"Reranker warmup failed: {e}")
        try:
            from ingest import get_bge_m3
            get_bge_m3()
            logger.info("BGE-M3 model warmed up")
        except Exception as e:
            logger.warning(f"BGE-M3 warmup failed: {e}")

    asyncio.get_running_loop().run_in_executor(None, _warmup_models)

    try:
        yield
    finally:
        await cancel_ingest()
        try:
            r = await get_redis()
            await r.aclose()
        except Exception:
            logger.debug("Failed to close async Redis on shutdown", exc_info=True)
        try:
            close_sync_redis()
        except Exception:
            logger.debug("Failed to close sync Redis on shutdown", exc_info=True)
        try:
            await engine.dispose()
        except Exception:
            logger.debug("Failed to dispose async engine on shutdown", exc_info=True)
        try:
            from database import get_sync_engine
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
    )

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
    from routes.health import router as health_router
    from routes.chat import router as chat_router
    from routes.documents import router as documents_router
    from routes.ingest import router as ingest_router

    app.include_router(health_router)
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
