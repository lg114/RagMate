<div align="center">

# RagMate

**Self-hosted RAG system for searchable documents and cited answers.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![pgvector](https://img.shields.io/badge/pgvector-Postgres-336791)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/license-MIT-4C566A)](LICENSE)

[English](README.md) · [中文](README_zh.md)

</div>

---

RagMate is a self-hosted RAG application that turns PDF, DOCX, XLSX, TXT, and Markdown files into a searchable knowledge base. It combines hybrid vector search, cross-encoder reranking, agentic reasoning, streaming responses, and optional faithfulness checks — while keeping your data and infrastructure under your own control.

## Contents

- [Highlights](#highlights)
- [Quick start](#quick-start)
- [First-use workflow](#first-use-workflow)
- [Project layout](#project-layout)
- [Architecture](#architecture)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Development](#development)
- [Data, backup, and security](#data-backup-and-security)
- [Vector storage](#vector-storage)
- [Known constraints](#known-constraints)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Highlights

- **Hybrid retrieval** — dense + sparse search fused with RRF (k=60) in pgvector. Embeddings come from an OpenAI-compatible API by default; in-process BGE-M3 is opt-in.
- **Reranked, filtered context** — remote reranking by default, plus dynamic score thresholds, per-source deduplication, score-gap truncation, and sentence-level contextual compression. A local cross-encoder is opt-in.
- **Title-path chunking** — markdown chunks carry an `H1 > H2 > H3` path for retrieval; the larger parent passage is still what the LLM sees. That strategy is frozen.
- **Grounded chat** — one retrieval, one answer, then citations. Nothing kept means a hard refusal and no answer-model call. Deep Agents is off unless `DEEP_AGENTS_ENABLED=true`.
- **Answer quality signals** — retrieval confidence (`high` / `medium` / `low`) on every RAG answer, citations taken from the chunks that were actually retrieved, a refusal or hedge when confidence is low, and faithfulness verification (on by default) that flags unsupported claims.
- **Incremental ingestion** — mtime-based change detection, content-hash deduplication, and crash-safe "insert new vectors before deleting old ones".
- **Retrieval regression guard** — `ragmate-eval retrieval-gate` replays the 98-question sample with deterministic BM25. Deterministic retrieval metrics are the CI gate; there is no LLM-judge scoring.
- **Self-hosted by design** — one Postgres with pgvector holds documents, chat, and vectors. Redis is optional and only used for shared rate limits.

---

## Quick start

### Prerequisites

- Python 3.12+
- Docker Desktop, or Docker Engine with Compose v2
- An OpenAI-compatible LLM endpoint and API key
- An OpenAI-compatible embeddings endpoint and a `/rerank` endpoint. Docker Compose runs Postgres with pgvector; it does not serve these models. Redis is not started unless you opt in.
- Local FlagEmbedding is optional. That path needs about 8 GB of RAM for CPU inference, and a GPU helps on larger collections.

### 1. Start the infrastructure

From the repository root:

```bash
docker compose up -d
docker compose ps
```

The stack exposes PostgreSQL (`5432`) with the pgvector extension. Redis is not part of this default path. The application itself runs on your host during development. On first start the API runs `CREATE EXTENSION vector` and creates the chunk table.

Ingest status, the lock, cancel, and failure reasons live in PostgreSQL. Chat history does too. Leave `REDIS_URL` empty. Rate limits then use an in-process counter, which is enough for one uvicorn process. To share that counter across workers, set `REDIS_URL=redis://localhost:6379/0` and start the optional service:

```bash
docker compose --profile redis up -d
```

To stop services without deleting data:

```bash
docker compose stop
```

### 2. Install the backend

```bash
cd backend
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e .
```

`ragmate-eval retrieval-gate` is part of that install. Tests and the retrieval gate need only the default install.

For a byte-identical environment, use the lockfile instead (from `backend/`): `uv sync --frozen --extra test`. Exact versions are pinned in `backend/uv.lock`; regenerate it with `uv lock` after changing dependencies in `pyproject.toml`.

DeepAgents support is an optional extra: `pip install -e ".[agents]"` (run from `backend/`). The default install excludes deepagents.

### 3. Configure the application

Stay in the `backend` directory:

```bash
cp .env.example .env
```

On Windows PowerShell, use `Copy-Item .env.example .env` instead.

> [!IMPORTANT]
> The process refuses to start unless these are set: `DATABASE_URL`, `LLM_API_KEY`, `EMBEDDING_API_BASE_URL`, and `RERANKER_API_BASE_URL`.
> Embedding and rerank API keys are optional when the service does not check them.
> To load BGE-M3 and the cross-encoder inside the API process instead, set `EMBEDDING_PROVIDER=huggingface` and `RERANKER_PROVIDER=local` (see the comments in `.env.example`). `local` and `flagembedding` are aliases of `huggingface`.

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ragmate
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o

# Optional: any OpenAI-compatible chat provider
LLM_API_BASE_URL=https://api.openai.com/v1

# Required with the default remote providers. Point both at your sidecar or hosted API.
EMBEDDING_PROVIDER=remote
EMBEDDING_API_BASE_URL=http://127.0.0.1:8081/v1
RERANKER_PROVIDER=api
RERANKER_API_BASE_URL=http://127.0.0.1:8081/v1
```

The default database URL matches the Docker Compose stack. Vectors live in that same database (`VECTOR_TABLE`, default `ragmate_docs`). `REDIS_URL` is empty unless you opted into the Redis profile. See [`backend/.env.example`](backend/.env.example) for the annotated full list, including the local FlagEmbedding opt-in.

### 4. Run RagMate

Return to the repository root, with the virtual environment still active:

```bash
uvicorn backend.app:app --reload --port 8000
```

A normal start does not download BGE-M3 or the reranker, and it does not import torch. The first ingest and the first query call your embedding and rerank URLs. If you opted into local models, that first start downloads the weights (a few GB) and warms them in the background.

Open [http://localhost:8000](http://localhost:8000) for the web UI. Interactive API docs are at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## First-use workflow

1. Open **文档** and upload one or more supported files.
2. Ingest the pending files and wait until the in-page status banner finishes.
3. Open **对话** and ask questions about the indexed documents.
4. Click a citation to open the source: file name, page, library status, and the retrieved passage when retrieval debug is on.
5. **设置** holds appearance, the shared retrieval-debug switch, an optional API token, and readiness for PostgreSQL and pgvector. Redis appears in that list only when `REDIS_URL` is set.

The UI is a thin demo shell: the zero-build page at `frontend/index.html`, served at `/`. No frontend build step. Retrieval debug is a dev-mode trace (`"debug": true`); leave it off for a normal demo. Retrieval knobs are not in the page — they are the frozen server defaults below.

Supported formats: `.pdf`, `.docx`, `.xlsx`, `.xls`, `.txt`, `.md`. Uploads are limited to 50 MB per file by default.

---

## Project layout

```
RagMate/
├── docker-compose.yml      # Postgres + pgvector; Redis is profile "redis"
├── backend/
│   ├── app.py              # App factory: middleware, exception handlers, static mount
│   ├── api/                # HTTP routes: chat, documents, ingest, health
│   ├── application/        # Use-case orchestration
│   │   └── ingest/         # pipeline (load → chunk → dedupe → encode), loaders, db_sync
│   ├── core/               # RAG core: retriever, agent, prompts/researcher.md
│   ├── domain/             # ORM models, Pydantic schemas, error types
│   ├── infrastructure/     # config, database, pgvector, optional redis, encoding, model_factory
│   ├── eval/               # retrieval-gate and deterministic retrieval evaluation
│   └── tests/
└── frontend/               # Zero-build UI (对话 / 文档 / 设置), served by FastAPI at /
```

---

## Architecture

The backend is strictly layered — each layer only calls downward:

| Layer | Directory | Responsibility |
|---|---|---|
| Interface | `api/` | HTTP concerns only: routing, request validation, rate limiting |
| Application | `application/` | Use-case orchestration: chat flow, document CRUD, ingestion lifecycle |
| Core | `core/` | RAG domain logic: retrieval pipeline, agent, system prompts |
| Domain | `domain/` | ORM models, Pydantic schemas, `AppError` hierarchy |
| Infrastructure | `infrastructure/` | External systems: PostgreSQL + pgvector, optional Redis, models, config |

```mermaid
flowchart LR
    D[Documents] --> P[Load and chunk]
    P --> E[Embed API]
    E --> V[(pgvector)]
    P --> M[(PostgreSQL metadata)]
    Q[User query] --> R[Hybrid search + RRF]
    R --> RR[Rerank, compress, filter]
    RR --> A[Agent and LLM]
    A --> O[Answer with citations]
```

**Ingestion path** — file discovery with mtime-based change detection → format-specific loader (PyPDF, Docx2txt, TextLoader, Excel to Markdown) → heading-aware chunking with optional title-path prefixes → content-hash deduplication against the pgvector table → dense + sparse encoding through the embeddings API (in-process FlagEmbedding only when you opt in) → insert new vectors → delete the previous vectors for that source → sync the `documents` table.

**Query path** — optional query contextualization (rewrites follow-ups into standalone queries) → dense HNSW plus exact sparse inner product, fused with RRF (k=60) → reranking (`POST /rerank` by default; a local cross-encoder converts logits with sigmoid) → sentence-level contextual compression → dynamic threshold, per-source deduplication, and score-gap truncation → one answer, then citation normalization → faithfulness check (on by default). RAG answers always include the retrieved sources and a confidence level. If retrieval kept no chunks, the answer is replaced with a refusal; a low score with some chunks is prefixed with an explicit hedge. There is no multi-query expansion: one query, one retrieval pass.

Simple greetings and other non-RAG messages are routed straight to the LLM, skipping the agent entirely.

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/chat` | Non-streaming answer |
| POST | `/chat/stream` | SSE streaming answer |
| GET | `/chat/sessions` | List the 50 most recent sessions |
| GET | `/chat/sessions/{session_id}` | Full message history for a session |
| DELETE | `/chat/sessions/{session_id}` | Delete a session from PostgreSQL |
| GET | `/documents` | List documents and their ingestion status |
| POST | `/documents/upload` | Upload a single file (50 MB max) |
| DELETE | `/documents/{filename}` | Delete a document, its file, and its vectors |
| POST | `/ingest` | Start ingestion (optionally for specific filenames) |
| POST | `/ingest/cancel` | Ask the worker that holds the ingest lock to stop |
| GET | `/ingest/status` | Current ingestion status and progress (survives restarts) |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe for PostgreSQL and pgvector. Redis is checked only when `REDIS_URL` is set |

### Chat request body

```jsonc
{
  "message": "What is RAG?",   // required, max 10 000 characters
  "session_id": "...",         // optional UUID; a new one is created when omitted
  "replace_last": false,       // retry / regenerate: drop the last turn first
  "debug": false                // include retrieved chunks, scores, and why filter dropped them
}
```

Example:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is RAG?"}'
```

### Streaming protocol

`/chat/stream` returns Server-Sent Events. Every `data:` line is a JSON object:

```jsonc
data: {"token": "Retrieval-augmented generation..."}   // incremental text

data: {"done": true, "session_id": "...",
       "response": "final grounded answer…",
       "confidence": {"level": "high", "score": 0.87, "chunks": 6},
       "citations": [{"source": "guide.pdf", "page": 3}],
       "grounding": "grounded",
       "unsupported_claims": [{"claim": "...", "supported": false}],
       "retrieval_debug": [{"query": "...", "chunks": [{"rank": 1, "score": 0.91, "kept": true, "dropped_by": null, "preview": "..."}]}]}

data: {"error": "..."}                                 // terminal error
```

Idle connections receive a `:heartbeat` comment every 15 seconds. `confidence`, `citations`, `grounding`, and `unsupported_claims` appear only on the `done` event. `response` on that event is the text after grounding (refusal, hedge, and the real source line) — clients should replace the streamed tokens with it. When retrieval keeps no chunks, the stream does not emit tokens first; the refusal arrives only on `done`. `unsupported_claims` is present when faithfulness checking flagged a claim. `grounding` is `refused`, `hedged`, `grounded`, or omitted when no retrieval ran.

The non-streaming `POST /chat` returns the same fields. `confidence` and `citations` are `null` when no retrieval took place — for example, a greeting routed straight to the LLM.

`retrieval_debug` is a dev-mode trace. It is omitted unless the request sets `"debug": true`. Each entry is one retrieval call: reranked candidates with score, a short preview, and `dropped_by` of `threshold`, `per_source`, `score_gap`, or `top_k` when the chunk was not kept. The composer and the **设置** page share a “检索调试” checkbox that sends this flag and remembers it in the browser. Leave it off for a normal demo; previews are excerpts of your documents.

---

## Configuration

Settings are read from `backend/.env` (see [`backend/.env.example`](backend/.env.example)). They are validated at startup, so invalid combinations fail fast rather than at query time.

### Essentials

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | **required** | Metadata, chat history, and ingest control |
| `LLM_API_KEY` | **required** | LLM provider credential |
| `EMBEDDING_API_BASE_URL` | **required** by default | OpenAI-compatible `/embeddings`. Not required when `EMBEDDING_PROVIDER` is `huggingface`, `local`, or `flagembedding` |
| `RERANKER_API_BASE_URL` | **required** by default | `/rerank` endpoint. Not required when `RERANKER_PROVIDER=local` |
| `LLM_MODEL` | `gpt-4o` | Model exposed by the provider |
| `LLM_API_BASE_URL` | empty | Custom OpenAI-compatible endpoint |
| `LLM_TEMPERATURE` | `0.01` | Kept low for factual answers |
| `REDIS_URL` | empty | Optional. Set only to share rate limits across workers. Ingest and chat do not use it |
| `VECTOR_TABLE` | `ragmate_docs` | pgvector table in the same database as `DATABASE_URL` |
| `DOCUMENTS_DIR` | `backend/documents` | Upload target directory |
| `CORS_ORIGINS` | `http://localhost:8000,...` | Comma-separated; restrict in production |

### Retrieval (frozen)

One retrieval config is validated. It is hybrid dense + sparse search, fused with RRF, then reranked. The fused-stage numbers are pinned in [`eval/reports/baseline/retrieval-baseline-98q-titlepath-rerun.json`](eval/reports/baseline/retrieval-baseline-98q-titlepath-rerun.json). Leave the defaults below unless you re-run `ragmate-eval retrieval` on that 98-question set and replace the floors in [`eval/README.md`](eval/README.md) and `backend/tests/test_retrieval_gate.py`.

| Piece | Frozen value |
|---|---|
| Hybrid dense + sparse | always on |
| Fusion | RRF with `RRF_K=60` |
| Chunking | `CHUNKING_STRATEGY=title_path`, `CHUNK_SIZE=1000`, overlap `200`, parent `2500` / `300` |
| Rerank then filter | `RERANK_CANDIDATES=30`, `RERANK_SCORE_THRESHOLD=0.3`, `FINAL_CONTEXT_K=15` |
| Query expansion | none (single query) |

`CHUNKING_STRATEGY=parent_child` is an environment variable only, frozen with that baseline. The filter and compression constants in the details are frozen literals in code. Changing one without a new deterministic baseline does not create a supported config. Do not add another fusion arm or a chunking strategy.

The 98-question report was collected with in-process BGE-M3 and a local cross-encoder. The algorithm above is what is frozen. Calling those models over HTTP (`EMBEDDING_PROVIDER=remote`, `RERANKER_PROVIDER=api`) is the deployment default and does not change fusion or chunking.

`DEEP_AGENTS_ENABLED` stays `false` (one retrieve, one answer, then grounding). `FAITHFULNESS_CHECK` stays `true` (checked only when retrieval confidence is low).

<details>
<summary><strong>Frozen env overrides</strong> — listed so a deployment can see them, not so they get tuned in place</summary>

**Chunking** — `CHUNK_SIZE` (`1000`), `CHUNK_OVERLAP` (`200`), `CHUNK_SIZE_PDF` (`600`), `CHUNK_OVERLAP_PDF` (`100`), `CHUNK_SIZE_DOCX` (`800`), `CHUNK_OVERLAP_DOCX` (`150`), `CHUNK_SIZE_TXT` (`1000`), `CHUNK_OVERLAP_TXT` (`200`), `CHUNK_SIZE_TABLE` (`1500`), `CHUNK_SIZE_PARENT` (`2500`), `CHUNK_OVERLAP_PARENT` (`300`), `CHUNKING_STRATEGY` (`title_path`). `parent_child` keeps the same boundaries and leaves the title path out of retrieval text. It is not a second supported strategy.

Overlap must be smaller than its chunk size, otherwise startup fails with a validation error.

**Filter and compression** — frozen literals: dynamic threshold `0.5`×top score, per-source cap `4` (min `2`), dominance `0.9`×`1.5`, score-gap `0.15`, high-score ratio `0.6`; compression threshold `0.4`, min `300` chars. `QUERY_CONTEXTUALIZE` (`true`, follow-up rewrite only when first-pass retrieval confidence is low, not multi-query), `QUERY_ROUTING_ENABLED` (`true`), `DROP_RATIO_SEARCH` (`0.2`).

**Embedding** — `EMBEDDING_PROVIDER` (`remote`), `EMBEDDING_MODEL` (`BAAI/bge-m3`), `EMBEDDING_NORMALIZE` (`true`), `HF_TOKEN`, `EMBEDDING_API_KEY` (optional), `EMBEDDING_API_BASE_URL` (required for `remote` / `openai`), `EMBEDDING_API_TIMEOUT` (`60`).

`remote` (also `openai`) calls `POST {EMBEDDING_API_BASE_URL}/embeddings`. Startup does not import torch or warm a local BGE model. `RERANKER_PROVIDER` defaults to `api` and calls `POST {RERANKER_API_BASE_URL}/rerank` (`RERANKER_API_KEY` optional, `RERANKER_API_TIMEOUT` `30`). If the embeddings service includes `sparse` (or `lexical_weights`) on each result, those weights are stored; otherwise RagMate fills a stable hashed sparse vector. That hash is not BGE-M3 lexical weight, so re-ingest after switching providers.

**Local opt-in** — first install the optional models with `pip install -e ".[local]"` (run from `backend/`), then set `EMBEDDING_PROVIDER=huggingface` (`local` and `flagembedding` are the same path) to load BGE-M3 with FlagEmbedding inside the API process. `EMBEDDING_DEVICE` (`cpu` or `cuda`) applies only then. Set `RERANKER_PROVIDER=local` to load the cross-encoder in-process. Turning either of those on imports torch on first use and warms the weights at startup. The commented block in [`backend/.env.example`](backend/.env.example) shows both lines.

CI does not need a GPU or a live embeddings service. Tests mock the HTTP client, and the retrieval gate replays BM25. The workflow still exports placeholder URLs because settings are validated at import.

**Agent** — `AGENT_RECURSION_LIMIT` (`30`).

**Tracing** — `LANGSMITH_TRACING` (`false`), `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` (`default`), `LANGSMITH_ENDPOINT`.

</details>

For local models, point `LLM_API_BASE_URL` at the OpenAI-compatible endpoint exposed by Ollama, LM Studio, or a similar server.

---

## Evaluation

The regression guard is `ragmate-eval retrieval-gate`. It ships with the base install. From `backend/`:

```bash
ragmate-eval retrieval-gate
```

It replays the committed 98-question sample with deterministic BM25 (no pgvector query, no BGE, no GPU) and fails when Recall@k or nDCG drops below the fused-stage floors pinned from `eval/reports/baseline/retrieval-baseline-98q-titlepath-rerun.json`. The floors, the 0.015 allowance, and the steps for replacing that baseline on purpose are in [`eval/README.md`](eval/README.md). GitHub Actions runs this command after pytest. The floors were not lowered for this store change.

RAGAS judge scoring was removed: it never produced a usable baseline and had been shelved from CI and the CLI main path. The deterministic retrieval gate in `eval/` is the regression guard.

---

## Development

```bash
cd backend
pip install -e ".[test]"
pytest -v
```

Tests live in `backend/tests/` and cover the retrieval filtering and deduplication logic. `conftest.py` provides an `override_settings` fixture for temporarily patching configuration. With no `backend/.env`, that file also fills placeholder `DATABASE_URL`, `LLM_API_KEY`, and the remote embed/rerank URLs so `pytest` does not need a GPU or a running model server. A checked-in `.env` is used as-is. `ragmate-eval retrieval-gate` does not go through `conftest`; export the same placeholder URLs (as CI does) or keep a valid `.env`.

> [!NOTE]
> `ruff` and `mypy` are not configured in `pyproject.toml` yet. The codebase follows PEP 8, 4-space indentation, and Chinese docstrings on public functions.

Every blocking call — model inference, vector I/O, ingestion — is pushed to a thread with `asyncio.to_thread`, so avoid adding blocking work directly to coroutines.

---

## Data, backup, and security

Docker volumes are written to `volumes/` and hold PostgreSQL data, including the pgvector table (plus Redis if that profile is running). For a consistent backup, stop the stack, copy the directory, then restart it. `volumes/etcd`, `volumes/minio`, and `volumes/milvus` are leftovers from the old stack and are not used.

Before exposing the service beyond localhost:

- Set `CORS_ORIGINS` to your real origin.
- Replace the default infrastructure credentials in `docker-compose.yml`.
- Put an authentication layer in front of the API.
- Review upload and rate-limit settings — currently 30 requests per IP per minute on chat, upload, delete, and ingest. Without `REDIS_URL` that counter is per process.

RagMate validates requests — filename allowlist, magic-byte checks, path-traversal rejection, parameterized vector filters, and a 50 MB upload cap — but it is not a complete internet-facing security boundary on its own.

---

## Vector storage

Chunks live in the Postgres table named by `VECTOR_TABLE` (default `ragmate_docs`). An existing `MILVUS_COLLECTION` value is still read as that table name when `VECTOR_TABLE` is empty.

To rebuild vectors from the files on disk, re-ingest. Incremental ingest skips a file whose `documents` row is already `ingested`, even when the vector table is empty. Clear that marker, then ingest again:

```sql
UPDATE documents SET status = 'uploaded';
```

Use the UI or `POST /ingest`. New vectors are written to pgvector. Deleting a document still deletes its vectors before the database row is committed.

An existing `volumes/postgres` directory from `postgres:15` can stay. Recreate the container on the pgvector image so the extension files are present:

```bash
docker compose up -d --force-recreate postgres
```

The API user must be allowed to run `CREATE EXTENSION vector`. The Compose `postgres` user can. `volumes/etcd`, `volumes/minio`, and `volumes/milvus` are leftovers from the old stack and can be deleted.

---

## Known constraints

- **Ingest runs on one worker at a time.** Status, the lock, progress, the cancel flag, and the failure reason live in one Postgres row (`ingest_control`), so any uvicorn worker can read progress and request cancellation. Only the worker that holds the lock writes progress and the terminal state. It refreshes a heartbeat (`INGEST_HEARTBEAT_INTERVAL`, default 5s). If that process dies, the next status read marks the job `interrupted` after `INGEST_HEARTBEAT_STALE` seconds (default 30s) and releases the lock. The row is not expired, so a failed or interrupted job is still there after a restart.
- **No authentication or multi-tenancy.** Every session and document is visible to anyone who can reach the API.
- **Vector schema changes need a manual rebuild.** If the pgvector table is missing columns or the `vector` / `sparsevec` types, startup raises a validation error instead of altering those columns. Drop the table and re-ingest.
- **Sparse search is exact.** Dense search uses an HNSW index (inner product). Sparse search is an exact inner product over `sparsevec`, because this pgvector version has no sparse ANN index. Query-side `DROP_RATIO_SEARCH` (default 0.2) still drops the smallest sparse weights before that scan. RRF stays at k=60.
- **No migration tool.** Schema changes are applied with `create_all` plus a small hand-written column-migration list in `infrastructure/database.py`.
- **In-process embedding and rerank are opt-in, and CPU inference is slow.** The default calls the remote embeddings and `/rerank` APIs and does not load torch. Set `EMBEDDING_PROVIDER=huggingface` and `RERANKER_PROVIDER=local` only when this process should load BGE-M3 and the cross-encoder.

---

## Troubleshooting

| Symptom | What to check |
|---|---|
| `/ready` reports `degraded` | `docker compose ps`, then `docker compose logs <service>` |
| Startup fails with "Required env vars not set" | `DATABASE_URL`, `LLM_API_KEY`, and — unless you opted into local models — `EMBEDDING_API_BASE_URL` and `RERANKER_API_BASE_URL` in `backend/.env` |
| Local model download or CPU inference is slow | Stay on the default remote providers, or set `EMBEDDING_DEVICE=cuda` for the local opt-in |
| Chat returns no useful context | Confirm ingestion finished and `DATABASE_URL` points at the Postgres that holds the pgvector table |
| Port conflicts | Change the host-side ports in `docker-compose.yml` and update the matching `.env` values |
| Ingestion stuck in `running` | Status is reconciled from the Postgres heartbeat. A dead worker becomes `interrupted` after `INGEST_HEARTBEAT_STALE` seconds (default 30); `POST /ingest/cancel` asks the live worker to stop |

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

[FastAPI](https://fastapi.tiangolo.com/) · [LangChain](https://python.langchain.com/) · [Deep Agents](https://github.com/langchain-ai/deepagents) · [pgvector](https://github.com/pgvector/pgvector) · [BGE](https://github.com/FlagOpen/FlagEmbedding)
