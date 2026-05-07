# Changelog

All notable changes to this project are documented here.

## [Unreleased]

---

## Prototype 2 — 2025-05-07

### Retrieval Layer
- `retriever.py` — Confirmed current path is pure semantic search (Embedding → Milvus), agent uses k=3, no BM25 / keyword matching
- `config.py` / `model_factory.py` — Confirmed embedding model is `all-MiniLM-L6-v2` (384-dim, lightweight, moderate accuracy)

### Robustness Fixes
- `document_service.py` — `list_documents`: `os.path.exists(filepath)` reduced from 3 calls to 1 per document iteration, result cached in `path_exists` variable
- `document_service.py` — `delete_document`: Added Redis session cache cleanup (deletes `ragmate:session:{filename}` key, failure doesn't affect main flow)
- `chat.py` — Error responses no longer appended to Redis history list (previously not saved to PG but still written to Redis), unified as "error hints discarded directly", keeping history list clean
- `cli.py` — `handle_retrieve()` added 30s timeout protection, prevents permanent block when Milvus is unresponsive
- `cli.py` — `handle_chat()` added 180s timeout protection, user can retry after timeout

### Logging & Observability
- `ingest.py` — `__main__`: `print(result)` changed to `logging.getLogger(__name__).info(result)`, consistent with other logging in the file
- `main.py` — `/ready` endpoint: each `except` block changed from silent `pass` to `warning` log, recording specific errors for diagnosing service degradation

### Timeout & Connection Protection
- `redis_client.py` — `get_sync_redis()`: `from_url` added `socket_connect_timeout=5`, prevents call thread from blocking indefinitely when Redis is unresponsive
- `main.py` — `init_db` failure changed from `warning` + continue startup to `error` + `raise`, fail-fast avoids running with degraded state

### Other
- `main.py` — `lifespan` startup: clears leftover ingest lock, failure changed to `warning` log instead of silent ignore
- `document_service.py` — Removed in-function `from redis_client import get_redis`, now uses module-top-level imported `get_redis`, avoiding repeated module path resolution on each execution
- `main.py` — `/chat/history` endpoint: changed from `GET /chat/history?session_id=xxx` (query param) to `GET /chat/history/{session_id}` (path param), frontend `app.js` updated accordingly

### Technical Debt
- Confirmed global exception handler `@app.exception_handler(Exception)` signature is correct (FastAPI passes `request, exc`), was previously misreported as a bug
- `start_ingest` has race window between check and creation (check `done()` → `create_task`), needs `asyncio.Lock` refactoring, not yet modified
- `main.py` frontend directory existence is a module-level one-time evaluation, if backend starts before frontend directory is created it won't dynamically detect, accepted as limitation

---

## Prototype 1 — 2025-05-06

### Features
- New Web UI (pure HTML/CSS/JS), FastAPI hosted, zero build tools
- New PostgreSQL integration: document metadata tracking, chat history persistence
- New Redis integration: query cache (same question avoids re-retrieval), multi-turn session state
- Chat endpoint upgraded to multi-turn dialogue, supports session_id context
- New API endpoints: GET/POST/DELETE /documents, POST /ingest, GET /ingest/status
- Embedding dimension auto-detection, no longer hardcoded to 384, adapts to any model
- Embedding model changed to singleton load (@lru_cache), no longer reloaded on every query
- Agent changed to lazy initialization (get_agent), avoids creating LLM connection on import

### Configuration
- `config.py` — Added EMBEDDING_API_KEY / EMBEDDING_API_BASE_URL / DATABASE_URL / REDIS_URL
- Added `.gitignore`, excludes `.env`, `__pycache__`, etc.
- `.env.example` — All configuration items completed

### Architecture
- New DeepSeek V4 thinking mode compatibility fix
- New Milvus connection detection and timeout mechanism
- New Windows GBK encoding compatibility handling
- Migrated to `pyproject.toml` (PEP 621)
- Switched to Milvus vector database
- Added `docker-compose` complete infrastructure
- Supports `sentence-transformers` local Embedding

### Error Handling
- Refactored error handling: New `errors.py` typed error hierarchy (AppError / NotFoundError / ValidationError / ConflictError / ServiceUnavailableError), global exception handler no longer leaks internal error information
- New `document_service.py` service layer: document CRUD business logic split from controller, endpoints only handle HTTP concerns

### Ingest
- Ingest distributed lock: Removed `threading.Lock` + `ingest_status.json` file state, switched to Redis SETNX lock + Redis state storage, multi-worker deployment is safe
- Path traversal fix: `validate_filename()` three-layer check (raw input rejects path separators → Path().name extraction → comparison verification), replaced original weak check of only `..`

### CLI
- Fixed CLI chat: `handle_chat` changed to async, `chat()` call added await, fixed bug returning coroutine object instead of result

### Infrastructure
- Milvus client connection reuse: `get_milvus_client()` singleton, unified across retriever/ingest/main three places
- Fixed `extract_text` fragile isinstance chain, simplified to AIMessage.content dual format handling
- `DeepSeekChatLiteLLM` renamed to `ThinkingChatLiteLLM`, more general thinking mode compatibility class
- Frontend `localStorage` → `sessionStorage`, added CSP meta tag
- `.gitignore` added `volumes/` (Docker data directory)
- Cleaned up dead code (commented-out warning filters, etc.)