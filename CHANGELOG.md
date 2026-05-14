# Changelog

All notable changes to this project are documented here.

## [Unreleased]

## Prototype 9 — 2026-05-14
- `streaming_llm.py` — Replaced LiteLLM with LangChain `ChatOpenAI` + custom `ChatOpenAICompatible` subclass for multi-API compatibility
- `streaming_llm.py` — Auto-detect `tool_calls` support: first attempt with native format, auto-retry with text conversion on failure
- `streaming_llm.py` — Auto-capture and replay `reasoning_content` for DeepSeek thinking mode
- `model_factory.py` — Simplified LLM factory to use `create_llm()`
- `config.py` — Removed `LLM_PROVIDER`, added `RERANK_CANDIDATES`, `FINAL_CONTEXT_K`, `RERANK_SCORE_THRESHOLD`
- `retriever.py` — `HYBRID_SEARCH_ENABLED` now actually controls hybrid vs dense-only search
- `retriever.py` — Added rerank score threshold filtering (default 0.15)
- `retriever.py` — Added retrieval logging (query, top_score, filtered_count, returned)
- `retriever.py` — `_collection_loaded` flag now auto-resets on Milvus connection errors
- `source_utils.py` — Fixed `canonical_source` bug: `data_1.xlsx` and `data_2.xlsx` no longer treated as same file
- `agent.py` — Strengthened refusal prompt: "宁可拒答，不要低质量上下文"
- `app.js` — Stream rendering throttled via `requestAnimationFrame` (no longer re-parses on every token)
- `app.js` — Error display now includes retry button
- `app.js` — Removed dead `API.chat()` method
- `app.js` — Replaced `alert()` with inline error messages
- `docker-compose.yml` — `depends_on` now uses `condition: service_healthy`
- `pyproject.toml` — Added `[build-system]` section
- `pyproject.toml` — Removed `langchain-litellm` dependency
- `main.py` — CORS origins now strip whitespace after split
- `main.py` — Static file mount no longer swallows API 404s
- `main.py` — Rate limit dict now auto-cleans when exceeding 1000 entries
- `ingest.py` — Milvus IDs use UUID instead of timestamp (no collision risk)
- `ingest.py` — Milvus filter escaping now handles backslashes
- `document_service.py` — Removed dead Redis cleanup code, fixed Milvus filter escaping
- `chat.py` — User messages now persist to Redis even on error (context preserved)
- `config.py` — `MILVUS_PORT` type fixed: `str` → `int`

### Added
- `frontend/app.js` — Retry button on error messages
- `backend/documents/` — Test documents: `.md`, `.txt`, `.xlsx`, `.docx`
- `backend/eval/dataset.json` — 12 new eval questions for multi-format documents

### Fixed
- `app.js` — Regenerate button no longer creates duplicate user messages
- `retriever.py` — Milvus collection reload on restart (was stuck with stale `_collection_loaded` flag)
- `source_utils.py` — Source dedup no longer treats `data_1.xlsx` and `data_2.xlsx` as the same file
- `config.py` — `MILVUS_PORT` was `str`, caused `TypeError` in `socket.create_connection`

## Prototype 8 — 2026-05-11

### Performance
- `retriever.py` — `load_collection()` now cached, only called once per process
- `main.py` — Reranker model warmed up in background on startup
- `ingest.py` — `encode_documents()` now processes in batches of 64 to avoid memory overflow

### Reliability
- `chat.py` — Streaming errors now yielded as `{"error": msg}` events, no longer mixed into token queue
- `main.py` — Simple in-memory rate limiter: 10 requests per session per minute

### Security
- `.env.example` — Added CORS production warning comment
- `document_service.py` — Added magic bytes validation for PDF/DOCX/XLSX uploads
- `main.py` — Chat rate limiting on both `/chat` and `/chat/stream`

### Code Quality
- `ingest.py` — All `print()` replaced with `logging.getLogger("ragmate").info()`
- `app.js` — SSE `JSON.parse` now wrapped in try/catch

---

## Prototype 7 — 2026-05-11

### Security Hardening
- `main.py` — `session_id` now validated with regex `^[a-f0-9]{32}$` (Pydantic field_validator)
- `document_service.py` — Filename whitelist regex `^[\w\-. 一-鿿]+\.\w+$` prevents Milvus filter injection
- `main.py` / `ingest.py` — All silent `except Exception: pass` now log at `debug` level with `exc_info`
- Frontend `style.css` / `app.js` — Streaming indicator changed from purple block cursor to "思考中..." typing dots

---

## Prototype 6 — 2026-05-11

### RAG Quality Optimization
- `agent.py` — Strengthened system prompt: only answer current question, no unsolicited background; dates/prices/versions must come from retrieval; engineering suggestions marked as "可考虑"
- `agent.py` — Reworked citation rules to merge repeated same-source citations and improve answer readability
- `chat.py` / `frontend/app.js` — Added deterministic citation cleanup that moves repeated file citations into one `数据来源` line
- `frontend/app.js` / `frontend/style.css` — Refined chat layout: narrower reading column, lighter assistant answers, source chips, cleaner user bubbles, and RAG-specific streaming status
- `config.py` — New `RETRIEVAL_TOP_K=2` (real Q&A) separate from `RERANKER_TOP_K=3` (evaluation)
- `agent.py` — `retrieval_tool` now uses configurable `RETRIEVAL_TOP_K`
- `retriever.py` — Source dedup: same canonical source max 2 chunks; near-duplicate sources (e.g. `xxx.pdf`/`xxx_2.pdf`) normalized
- `eval/runner.py` — Detailed output per result: source, page, chunk_index, score, preview
- `eval/runner.py` — New metrics: `contains` (keyword match) and `must_not` (leak detection)
- `eval/metrics.py` — New `contains_check()` and `must_not_check()` functions
- `eval/metrics.py` / `retriever.py` — Shared canonical source normalization so `xxx.pdf` and `xxx_2.pdf` are evaluated consistently
- `eval/dataset.json` — Enhanced with `expected_contains` and `must_not_sources` fields

---

## Prototype 5 — 2026-05-11

### Critical Fixes
- `agent.py` — Fixed streaming leaking tool results to user: `run_agent_streaming()` now filters out `ToolMessage`, only yields `AIMessageChunk` text
- `agent.py` — Page numbers now display as 1-based (was 0-based from PyPDFLoader metadata)
- `cli.py` — CLI page display also fixed to 1-based
- `database.py` — `init_db()` now runs `ALTER TABLE ADD COLUMN IF NOT EXISTS file_mtime` for existing databases
- `ingest.py` — Fixed `NameError`: `ingested_filenames` renamed to `ingested_info`
- `ingest.py` — Old records with `file_mtime=NULL` now trigger re-ingestion to backfill mtime
- `main.py` — Startup no longer force-releases ingest lock (safe for multi-instance); logs warning instead
- `retriever.py` — Entire retrieval pipeline wrapped in try/except; `RetrievalError` raised on any failure
- `cli.py` — `handle_ingest()` now calls `init_db()` before ingestion (ensures schema migrations run)

### Stability Fixes
- `chat.py` — Fixed `asyncio.Queue` thread-safety: now uses `loop.call_soon_threadsafe()` for cross-thread writes
- `cli.py` — Fixed `retrieve()` dict handling in CLI menu (was slicing dict as string)
- `redis_client.py` — Added `renew_ingest_lock()` for lock TTL renewal during long ingestion
- `main.py` — Ingest lock renewal: background task extends TTL every 5 minutes during ingestion
- `main.py` — Startup lock cleanup now checks if lock exists before releasing (safe for multi-instance)
- `ingest.py` — Incremental ingest now detects file changes via `mtime` (not just filename)
- `models.py` — Added `file_mtime` field to Document model
- `retriever.py` — Raises `RetrievalError` on service failure instead of returning empty list
- `errors.py` — New `RetrievalError` for distinguishing "no results" from "service error"
- `agent.py` — `retrieval_tool` catches `RetrievalError` and returns user-friendly message
- Frontend `index.html` — Removed fake SRI integrity hashes from CDN scripts (browser was blocking)
- Frontend `index.html` — Upload text changed from "上传 PDF" to "上传文档" (supports multiple formats)

---

## Prototype 4 — 2026-05-11

### Code Review Fixes
- `streaming_llm.py` — Fixed streaming tool call JSON parse crash (partial chunks now handled gracefully)
- `streaming_llm.py` — `_bound_tools` now uses Pydantic `PrivateAttr` instead of private field
- `redis_client.py` — Fixed singleton race condition with `asyncio.Lock` double-checked locking
- `redis_client.py` — Distributed lock now uses UUID token + Lua script for safe release (only lock holder can release)
- `redis_client.py` — `set_ingest_status` no longer mutates the input dict (copies first)
- `redis_client.py` — Added `force_release_ingest_lock()` for startup cleanup
- `database.py` — Sync engine URL now uses SQLAlchemy `make_url()` instead of string replace
- `document_service.py` — Silent `except Exception: pass` now logs warnings with `exc_info`
- `document_service.py` — Added security comment for Milvus filter string construction
- `ingest.py` — Milvus collection now uses `auto_id=True` instead of manual ID offset (eliminates ID conflict risk)
- `main.py` — `start_ingest` race condition fixed with `asyncio.Lock`
- `main.py` — CORS origins now configurable via `CORS_ORIGINS` setting (default: `*`)
- `retriever.py` — `retrieve()` now catches exceptions and returns empty list with error log
- Frontend `index.html` — Locked marked.js to v15.0.7, added SRI integrity attributes to CDN scripts
- Frontend `style.css` — Added responsive layout (`@media max-width: 768px`)
- Frontend `style.css` — Added `prefers-reduced-motion` media query to disable animations

---

## Prototype 3 — 2026-05-11

### Streaming Output
- Chat now supports SSE streaming — tokens are displayed progressively instead of waiting for the complete response
- `agent.py` — New `run_agent_streaming()` function using LangGraph's `.stream(stream_mode="messages")` for token-level streaming
- `chat.py` — New `chat_stream()` async generator bridging sync agent streaming to async SSE via queue
- `main.py` — New `POST /chat/stream` SSE endpoint, original `/chat` endpoint preserved for backward compatibility
- `streaming_llm.py` — New `StreamingLiteLLM` class with true token-level streaming support (replaces `ThinkingChatLiteLLM` which buffers all tokens)
- `streaming_llm.py` — `bind_tools()` support for agent tool-calling flow
- `model_factory.py` — Switched from `ThinkingChatLiteLLM` to `StreamingLiteLLM`
- Frontend `app.js` — New `API.chatStream()` for SSE consumption, `ChatPanel.send()` rewritten for progressive rendering with blinking cursor
- Frontend `style.css` — New `.stream-cursor` animation for streaming indicator

### Multi-Format Document Support
- Now supports PDF, DOCX, XLSX, TXT, Markdown files (previously PDF only)
- `ingest.py` — New `load_document()` loader factory using pypdf, docx2txt, unstructured, openpyxl
- `ingest.py` — `SUPPORTED_EXTENSIONS` constant defines accepted file types
- `document_service.py` — `validate_filename()` updated to accept all supported formats
- Frontend `index.html` — File input `accept` attribute updated for multi-format upload
- Frontend `app.js` — Upload validation updated to check against supported extensions
- New dependencies: `docx2txt`, `unstructured`, `openpyxl`, `msoffcrypto-tool`, `xlrd`

### Embedding Upgrade
- Default embedding model changed from `all-MiniLM-L6-v2` (384-dim, English) to `BAAI/bge-m3` (1024-dim, multilingual)
- Significantly improved Chinese semantic retrieval quality
- `model_factory.py` — Added `trust_remote_code=True` for bge-m3 custom code support
- `config.py` — Default `EMBEDDING_MODEL` updated to `BAAI/bge-m3`
- `.env.example` / `README` updated with new default
- Note: existing Milvus collections need re-ingestion after model change (dimension mismatch 384 → 1024)

### Hybrid Search + Reranking
- Retrieval upgraded from pure dense vector search to hybrid search (dense + sparse) with RRF fusion + cross-encoder reranking
- `retriever.py` — Rewritten: uses `client.hybrid_search()` with dense + sparse `AnnSearchRequest` + `RRFRanker`, then `BAAI/bge-reranker-v2-m3` cross-encoder reranking
- `retriever.py` — `retrieve()` now returns `List[dict]` with `text`, `source`, `page`, `score` fields
- `ingest.py` — Uses `FlagEmbedding.BGEM3FlagModel` to encode both dense and sparse vectors per chunk
- `ingest.py` — Milvus collection schema updated with `SPARSE_FLOAT_VECTOR` field and `SPARSE_INVERTED_INDEX`
- `ingest.py` — New `encode_documents()` and `encode_query()` functions for bge-m3 dual-vector encoding
- `agent.py` — `retrieval_tool` updated to format source + page metadata in results
- `config.py` — New settings: `HYBRID_SEARCH_ENABLED`, `RERANKER_MODEL`, `RERANKER_TOP_K`
- New dependency: `FlagEmbedding`
- Note: Milvus collection schema changed — old data must be re-ingested

### Smarter Chunking
- Markdown files now use `MarkdownHeaderTextSplitter` to preserve document hierarchy (H1/H2/H3 as metadata)
- PDF page numbers preserved in chunk metadata
- All chunks now include `chunk_index` metadata for sequential tracking
- `retriever.py` — Returns `chunk_index` in results
- `agent.py` — `retrieval_tool` formats source + page + chunk_index in citations

### Evaluation System
- New `eval/` module for retrieval quality evaluation
- `eval/dataset.json` — Test Q&A dataset (question + expected sources)
- `eval/metrics.py` — Retrieval recall and precision metrics
- `eval/runner.py` — Evaluation runner with latency tracking and formatted report
- `cli.py` — New option "4. 评估" to run evaluation from CLI

### Bug Fixes
- Fixed delete confirmation dialog — replaced native `confirm()` with custom modal that works reliably
- Fixed session delete not clearing chat panel when deleting the active session

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
