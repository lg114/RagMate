import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from sqlalchemy import delete, func, select
from starlette.responses import StreamingResponse

from chat import chat, chat_stream
from config import settings
from database import async_session, engine, init_db, sync_engine
import document_service
from errors import ConflictError, NotFoundError, ServiceUnavailableError, ValidationError
from ingest import ingest_documents
from models import ChatHistory
from redis_client import get_redis, acquire_ingest_lock, release_ingest_lock, force_release_ingest_lock, renew_ingest_lock, get_ingest_status, set_ingest_status
from retriever import _check_milvus_available, get_milvus_client

# LangSmith Tracing
if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

_ingest_task: asyncio.Task | None = None
_ingest_lock_token: str | None = None

# 简单的内存频率限制：每 session 每分钟最多 10 次
_rate_limit: dict[str, list[float]] = {}
_RATE_LIMIT_MAX = 10
_RATE_LIMIT_WINDOW = 60.0


def _check_rate_limit(session_id: str):
    import time
    now = time.time()
    # 定期清理过期条目，防止内存泄漏
    if len(_rate_limit) > 1000:
        for sid in list(_rate_limit):
            _rate_limit[sid] = [t for t in _rate_limit[sid] if now - t < _RATE_LIMIT_WINDOW]
            if not _rate_limit[sid]:
                del _rate_limit[sid]
    timestamps = _rate_limit.get(session_id, [])
    timestamps = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
    if len(timestamps) >= _RATE_LIMIT_MAX:
        from errors import ValidationError
        raise ValidationError("请求过于频繁，请稍后重试")
    timestamps.append(now)
    _rate_limit[session_id] = timestamps


async def _run_ingest():
    """在后台线程中执行入库，通过 Redis 管理状态与锁"""
    global _ingest_lock_token

    # 后台续期任务：每 5 分钟延长锁 TTL
    async def _renew_loop():
        while True:
            await asyncio.sleep(300)
            if _ingest_lock_token:
                try:
                    await renew_ingest_lock(_ingest_lock_token)
                except Exception:
                    logging.getLogger("ragmate").debug("Failed to renew ingest lock", exc_info=True)

    renew_task = asyncio.create_task(_renew_loop())
    try:
        await set_ingest_status({"status": "running"})
        result = await asyncio.to_thread(ingest_documents, None, True)
        await set_ingest_status(result)
    except asyncio.CancelledError:
        await set_ingest_status({"status": "idle"})
        raise
    except Exception as e:
        await set_ingest_status({"status": "failed", "error": str(e)})
    finally:
        renew_task.cancel()
        global _ingest_task
        _ingest_task = None
        if _ingest_lock_token:
            try:
                await release_ingest_lock(_ingest_lock_token)
            except Exception:
                logging.getLogger("ragmate").debug("Failed to release ingest lock", exc_info=True)
            _ingest_lock_token = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 检查是否有遗留的 ingest 锁（不自动释放，避免多实例误删）
    try:
        r = await get_redis()
        lock_exists = await r.exists("ragmate:ingest:lock")
        if lock_exists:
            logging.getLogger("ragmate").warning("Found existing ingest lock on startup — another instance may be running, or a previous run crashed. Manual cleanup may be needed.")
    except Exception as e:
        logging.getLogger("ragmate").warning(f"Failed to check ingest lock on startup: {e}")
    try:
        await init_db()
        logging.getLogger("ragmate").info("Database initialized")
    except Exception as e:
        logging.getLogger("ragmate").error(f"Database initialization failed: {e}")
        raise  # fail fast — app cannot function without DB

    # 后台预热 Reranker 模型
    def _warmup_reranker():
        try:
            from retriever import get_reranker
            get_reranker()
            logging.getLogger("ragmate").info("Reranker model warmed up")
        except Exception as e:
            logging.getLogger("ragmate").warning(f"Reranker warmup failed: {e}")
    asyncio.get_running_loop().run_in_executor(None, _warmup_reranker)

    try:
        yield
    finally:
        # 取消正在运行的 ingest 任务
        global _ingest_task
        if _ingest_task and not _ingest_task.done():
            _ingest_task.cancel()
            try:
                await _ingest_task
            except asyncio.CancelledError:
                pass
        try:
            r = await get_redis()
            await r.aclose()
        except Exception:
            logging.getLogger("ragmate").debug("Failed to close Redis on shutdown", exc_info=True)
        try:
            await engine.dispose()
        except Exception:
            logging.getLogger("ragmate").debug("Failed to dispose async engine on shutdown", exc_info=True)
        try:
            await asyncio.to_thread(sync_engine.dispose)
        except Exception:
            logging.getLogger("ragmate").debug("Failed to dispose sync engine on shutdown", exc_info=True)


app = FastAPI(title="RagMate API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    return JSONResponse(status_code=exc.status_code, content={"code": exc.code, "detail": exc.message})


@app.exception_handler(NotFoundError)
async def not_found_handler(request, exc: NotFoundError):
    return JSONResponse(status_code=exc.status_code, content={"code": exc.code, "detail": exc.message})


@app.exception_handler(ConflictError)
async def conflict_handler(request, exc: ConflictError):
    return JSONResponse(status_code=exc.status_code, content={"code": exc.code, "detail": exc.message})


@app.exception_handler(ServiceUnavailableError)
async def service_unavailable_handler(request, exc: ServiceUnavailableError):
    return JSONResponse(status_code=exc.status_code, content={"code": exc.code, "detail": exc.message})


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.getLogger("ragmate").error("Unhandled exception", exc_info=(type(exc), exc, exc.__traceback__))
    return JSONResponse(status_code=500, content={"code": "INTERNAL_ERROR", "detail": "An unexpected error occurred"})


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v):
        if v is not None and not re.match(r'^[a-f0-9-]{36}$', v):
            raise ValueError("Invalid session_id format")
        return v


class ChatResponse(BaseModel):
    response: str
    session_id: str


# ── Endpoints ──

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    status = {"milvus": False, "postgresql": False, "redis": False}
    try:
        milvus_ok = await asyncio.to_thread(_check_milvus_available)
        status["milvus"] = milvus_ok
    except Exception as e:
        logging.getLogger("ragmate").warning(f"Milvus health check failed: {e}")
    try:
        async with async_session() as session:
            await session.execute(select(1))
        status["postgresql"] = True
    except Exception as e:
        logging.getLogger("ragmate").warning(f"PostgreSQL health check failed: {e}")
    try:
        r = await get_redis()
        await r.ping()
        status["redis"] = True
    except Exception as e:
        logging.getLogger("ragmate").warning(f"Redis health check failed: {e}")
    return {"status": "ready" if all(status.values()) else "degraded", "checks": status}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    _check_rate_limit(request.session_id or "anonymous")
    result = await chat(request.message, request.session_id)
    return ChatResponse(response=result["response"], session_id=result["session_id"])


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    _check_rate_limit(request.session_id or "anonymous")
    async def event_generator():
        async for chunk in chat_stream(request.message, request.session_id):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/chat/sessions")
async def list_sessions():
    async with async_session() as session:
        first_msg = (
            select(
                ChatHistory.session_id,
                ChatHistory.content,
                ChatHistory.created_at,
                func.row_number().over(
                    partition_by=ChatHistory.session_id,
                    order_by=ChatHistory.created_at,
                ).label("rn"),
            )
            .where(ChatHistory.role == "user")
            .subquery()
        )
        result = await session.execute(
            select(
                first_msg.c.session_id,
                first_msg.c.content,
                first_msg.c.created_at,
            ).where(first_msg.c.rn == 1)
            .order_by(first_msg.c.created_at.desc())
            .limit(50)
        )
        sessions = [
            {
                "session_id": row.session_id,
                "first_message": row.content[:80] + ("..." if len(row.content) > 80 else ""),
                "created_at": row.created_at.isoformat(),
            }
            for row in result.fetchall()
        ]
    return {"sessions": sessions}


@app.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(ChatHistory)
            .where(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.created_at)
        )
        messages = [
            {"role": row.role, "content": row.content, "created_at": row.created_at.isoformat()}
            for row in result.scalars()
        ]
    return {"session_id": session_id, "messages": messages}


@app.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定 session 的 Redis 会话缓存 + PostgreSQL 历史记录"""
    # 删除 Redis 会话缓存（Redis 挂了不影响主流程）
    try:
        r = await get_redis()
        await r.delete(f"ragmate:session:{session_id}")
    except Exception:
        pass

    # 删除 PostgreSQL 历史记录
    async with async_session() as session:
        await session.execute(delete(ChatHistory).where(ChatHistory.session_id == session_id))
        await session.commit()

    return {"success": True}


@app.get("/documents")
async def list_documents():
    async with async_session() as session:
        docs = await document_service.list_documents(settings.DOCUMENTS_DIR, session)
    return {"documents": docs}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    async with async_session() as session:
        return await document_service.save_document(
            filename=file.filename or "",
            content=content,
            docs_dir=settings.DOCUMENTS_DIR,
            session=session,
        )

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    async with async_session() as session:
        await document_service.delete_document(
            filename=filename,
            docs_dir=settings.DOCUMENTS_DIR,
            session=session,
            milvus_client=get_milvus_client(),
        )
    return {"success": True}


_ingest_start_lock = asyncio.Lock()


@app.post("/ingest")
async def start_ingest():
    global _ingest_task, _ingest_lock_token
    async with _ingest_start_lock:
        if _ingest_task and not _ingest_task.done():
            return {"status": "already_running"}
        token = await acquire_ingest_lock()
        if not token:
            return {"status": "already_running"}
        _ingest_lock_token = token
        _ingest_task = asyncio.create_task(_run_ingest())
    return {"status": "started"}


@app.get("/ingest/status")
async def ingest_status():
    return await get_ingest_status()


# ── Static files (frontend) — must be last ──
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
