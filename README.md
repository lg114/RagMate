<div align="center">

# RagMate

**Self-hosted RAG system for searchable documents and cited answers.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Milvus](https://img.shields.io/badge/Milvus-2.5-2A6FB0)](https://milvus.io/)
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
- [Known constraints](#known-constraints)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Highlights

- **Hybrid retrieval** — dense + sparse search fused with RRF in Milvus. Embeddings come from an OpenAI-compatible API by default; in-process BGE-M3 is opt-in.
- **Reranked, filtered context** — remote reranking by default, plus dynamic score thresholds, per-source deduplication, score-gap truncation, and sentence-level contextual compression. A local cross-encoder is opt-in.
- **Title-path + parent-child chunking** — markdown chunks carry an `H1 > H2 > H3` path for retrieval; the larger parent passage is still what the LLM sees.
- **Grounded chat** — one retrieval, one answer, then citations. Nothing kept means a hard refusal and no answer-model call. Deep Agents is off unless `DEEP_AGENTS_ENABLED=true`.
- **Answer quality signals** — retrieval confidence (`high` / `medium` / `low`) on every RAG answer, citations taken from the chunks that were actually retrieved, a refusal or hedge when confidence is low, and faithfulness verification (on by default) that flags unsupported claims.
- **Incremental ingestion** — mtime-based change detection, content-hash deduplication, and crash-safe "insert new vectors before deleting old ones".
- **Built-in evaluation** — optional RAGAS CLI for test-set generation, scoring, and CI quality gates.
- **Self-hosted by design** — PostgreSQL, Milvus, and MinIO run through Docker Compose. Redis is optional and only used for shared rate limits.

---

## Quick start

### Prerequisites

- Python 3.12+
- Docker Desktop, or Docker Engine with Compose v2
- An OpenAI-compatible LLM endpoint and API key
- An OpenAI-compatible embeddings endpoint and a `/rerank` endpoint. Docker Compose runs Milvus, PostgreSQL, and MinIO; it does not serve these models. Redis is not started unless you opt in.
- Local FlagEmbedding is optional. That path needs about 8 GB of RAM for CPU inference, and a GPU helps on larger collections.

### 1. Start the infrastructure

From the repository root:

```bash
docker compose up -d
docker compose ps
```

The stack exposes PostgreSQL (`5432`), Milvus (`19530`), MinIO (`9000` / `9001`), and Attu (`8080`). Redis is not part of this default path. The application itself runs on your host during development.

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

For the evaluation CLI, install the optional extras:

```bash
pip install -e ".[eval]"
```

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

The default database and Milvus values match the Docker Compose stack. `REDIS_URL` is empty unless you opted into the Redis profile. See [`backend/.env.example`](backend/.env.example) for the annotated full list, including the local FlagEmbedding opt-in.

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
5. **设置** holds appearance, the shared retrieval-debug switch, an optional API token, and readiness for PostgreSQL and Milvus. Redis appears in that list only when `REDIS_URL` is set.

The UI is still the zero-build page at `frontend/index.html`, served at `/`. No frontend build step.

Supported formats: `.pdf`, `.docx`, `.xlsx`, `.xls`, `.txt`, `.md`. Uploads are limited to 50 MB per file by default.

---

## Project layout

```
RagMate/
├── docker-compose.yml      # PostgreSQL, Milvus, MinIO, etcd, Attu; Redis is profile "redis"
├── backend/
│   ├── app.py              # App factory: middleware, exception handlers, static mount
│   ├── api/                # HTTP routes: chat, documents, ingest, health
│   ├── application/        # Use-case orchestration
│   │   └── ingest/         # pipeline (load → chunk → dedupe → encode), loaders, db_sync
│   ├── core/               # RAG core: retriever, agent, prompts/researcher.md
│   ├── domain/             # ORM models, Pydantic schemas, error types
│   ├── infrastructure/     # config, database, optional redis, milvus, encoding, model_factory
│   ├── eval/               # RAGAS evaluation CLI (optional dependency)
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
| Infrastructure | `infrastructure/` | External systems: Milvus, PostgreSQL, optional Redis, models, config |

```mermaid
flowchart LR
    D[Documents] --> P[Load and chunk]
    P --> E[Embed API]
    E --> V[(Milvus vectors)]
    P --> M[(PostgreSQL metadata)]
    Q[User query] --> R[Hybrid search + RRF]
    R --> RR[Rerank, compress, filter]
    RR --> A[Agent and LLM]
    A --> O[Answer with citations]
```

**Ingestion path** — file discovery with mtime-based change detection → format-specific loader (PyPDF, Docx2txt, TextLoader, Excel to Markdown) → heading-aware chunking with optional title-path prefixes → content-hash deduplication against Milvus → dense + sparse encoding through the embeddings API (in-process FlagEmbedding only when you opt in) → insert new vectors → delete the previous vectors for that source → sync the `documents` table.

**Query path** — optional query contextualization (rewrites follow-ups into standalone queries) → dense + sparse ANN search with RRF fusion → reranking (`POST /rerank` by default; a local cross-encoder converts logits with sigmoid) → sentence-level contextual compression → dynamic threshold, per-source deduplication, and score-gap truncation → one answer, then citation normalization → faithfulness check (on by default). RAG answers always include the retrieved sources and a confidence level. If retrieval kept no chunks, the answer is replaced with a refusal; a low score with some chunks is prefixed with an explicit hedge.

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
| GET | `/ready` | Readiness probe for PostgreSQL and Milvus. Redis is checked only when `REDIS_URL` is set |

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

`retrieval_debug` is omitted unless the request sets `"debug": true`. Each entry is one retrieval call: reranked candidates with score, a short preview, and `dropped_by` of `threshold`, `per_source`, `score_gap`, or `top_k` when the chunk was not kept. The composer and the **设置** page share a “检索调试” checkbox that sends this flag and remembers it in the browser. Leave it off in normal use; previews are excerpts of your documents.

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
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | Vector database |
| `MILVUS_COLLECTION` | `ragmate_docs` | Collection name |
| `DOCUMENTS_DIR` | `backend/documents` | Upload target directory |
| `CORS_ORIGINS` | `http://localhost:8000,...` | Comma-separated; restrict in production |

### Retrieval and reranking

| Variable | Default | Notes |
|---|---|---|
| `HYBRID_SEARCH_ENABLED` | `true` | Dense + sparse; `false` falls back to dense only |
| `RERANK_CANDIDATES` | `30` | Candidates passed to the cross-encoder |
| `RERANK_SCORE_THRESHOLD` | `0.3` | Sigmoid probability floor for a chunk |
| `FINAL_CONTEXT_K` | `15` | Hard cap on chunks sent to the LLM |
| `DYNAMIC_THRESHOLD_RATIO` | `0.5` | Floor becomes `top_score × ratio` when higher |
| `MAX_PER_SOURCE` / `MIN_PER_SOURCE` | `4` / `2` | Per-source chunk limits |
| `SCORE_GAP_THRESHOLD` | `0.15` | Stop at the first large score cliff |
| `CONTEXTUAL_COMPRESSION` | `true` | Keep only relevant sentences inside a chunk |
| `QUERY_CONTEXTUALIZE` | `true` | Rewrite follow-ups into standalone queries |
| `QUERY_ROUTING_ENABLED` | `true` | Greetings skip retrieval and go straight to the LLM |
| `DEEP_AGENTS_ENABLED` | `false` | Opt in to the Deep Agents tool loop. Default is one retrieve, one answer, then grounding |
| `FAITHFULNESS_CHECK` | `true` | One extra LLM call per RAG answer; failures are ignored and the answer is still returned |

<details>
<summary><strong>Full variable list</strong> — chunking, embedding, agent, and tracing</summary>

**Chunking** — `CHUNK_SIZE` (`1000`), `CHUNK_OVERLAP` (`200`), `CHUNK_SIZE_PDF` (`600`), `CHUNK_OVERLAP_PDF` (`100`), `CHUNK_SIZE_DOCX` (`800`), `CHUNK_OVERLAP_DOCX` (`150`), `CHUNK_SIZE_TXT` (`1000`), `CHUNK_OVERLAP_TXT` (`200`), `CHUNK_SIZE_TABLE` (`1500`), `CHUNK_SIZE_PARENT` (`2500`), `CHUNK_OVERLAP_PARENT` (`300`), `CHUNKING_STRATEGY` (`title_path`; set `parent_child` to keep header paths out of retrieval text).

Overlap must be smaller than its chunk size, otherwise startup fails with a validation error.

**Reranker tuning** — `HIGH_SCORE_RATIO` (`0.6`), `SOURCE_DOMINANCE_THRESHOLD` (`0.9`), `SOURCE_DOMINANCE_BOOST` (`1.5`), `DROP_RATIO_SEARCH` (`0.2`), `COMPRESSION_SCORE_THRESHOLD` (`0.4`), `COMPRESSION_MIN_CHARS` (`300`).

**Embedding** — `EMBEDDING_PROVIDER` (`remote`), `EMBEDDING_MODEL` (`BAAI/bge-m3`), `EMBEDDING_NORMALIZE` (`true`), `HF_TOKEN`, `EMBEDDING_API_KEY` (optional), `EMBEDDING_API_BASE_URL` (required for `remote` / `openai`), `EMBEDDING_API_TIMEOUT` (`60`).

`remote` (also `openai`) calls `POST {EMBEDDING_API_BASE_URL}/embeddings`. Startup does not import torch or warm a local BGE model. `RERANKER_PROVIDER` defaults to `api` and calls `POST {RERANKER_API_BASE_URL}/rerank` (`RERANKER_API_KEY` optional, `RERANKER_API_TIMEOUT` `30`). If the embeddings service includes `sparse` (or `lexical_weights`) on each result, those weights are stored; otherwise RagMate fills a stable hashed sparse vector. That hash is not BGE-M3 lexical weight, so re-ingest after switching providers.

**Local opt-in** — set `EMBEDDING_PROVIDER=huggingface` (`local` and `flagembedding` are the same path) to load BGE-M3 with FlagEmbedding inside the API process. `EMBEDDING_DEVICE` (`cpu` or `cuda`) applies only then. Set `RERANKER_PROVIDER=local` to load the cross-encoder in-process. Turning either of those on imports torch on first use and warms the weights at startup. The commented block in [`backend/.env.example`](backend/.env.example) shows both lines.

CI does not need a GPU or a live embeddings service. Tests mock the HTTP client, and the retrieval gate replays BM25. The workflow still exports placeholder URLs because settings are validated at import.

**Agent** — `AGENT_RECURSION_LIMIT` (`30`).

**Tracing** — `LANGSMITH_TRACING` (`false`), `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` (`default`), `LANGSMITH_ENDPOINT`.

</details>

For local models, point `LLM_API_BASE_URL` at the OpenAI-compatible endpoint exposed by Ollama, LM Studio, or a similar server.

---

## Evaluation

Install the extras with `pip install -e ".[eval]"`, then run the interactive menu:

```bash
cd backend
ragmate-eval
```

Or drive it from a script:

```bash
ragmate-eval generate --size 50 --output ../eval/testsets/testset.json

ragmate-eval evaluate --testset ../eval/testsets/testset.json \
  --report ../eval/reports/report.json --threshold 0.75
```

Metrics cover faithfulness, answer relevancy, context precision, context recall, and factual correctness. `--threshold` makes the command exit non-zero below the score, which works well as a CI gate. Test sets and reports are written under `eval/` at the repository root and are git-ignored.

Retrieval quality on pull requests is a separate command, `ragmate-eval retrieval-gate`. It replays the committed 98-question sample with deterministic BM25 (no Milvus, no BGE, no GPU) and fails when Recall@k or nDCG drops below the fused-stage floors pinned from `eval/reports/baseline/retrieval-baseline-98q-titlepath-rerun.json`. The floors, the 0.015 allowance, and the steps for replacing that baseline on purpose are in `eval/README.md`. GitHub Actions runs this command after pytest.

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

Every blocking call — model inference, Milvus I/O, ingestion — is pushed to a thread with `asyncio.to_thread`, so avoid adding blocking work directly to coroutines.

---

## Data, backup, and security

Docker volumes are written to `volumes/` and hold PostgreSQL, Milvus, and MinIO data (plus Redis if that profile is running). For a consistent backup, stop the stack, copy the directory, then restart it.

Before exposing the service beyond localhost:

- Set `CORS_ORIGINS` to your real origin.
- Replace the default infrastructure credentials in `docker-compose.yml`.
- Put an authentication layer in front of the API.
- Review upload and rate-limit settings — currently 30 requests per IP per minute on chat, upload, delete, and ingest. Without `REDIS_URL` that counter is per process.

RagMate validates requests — filename allowlist, magic-byte checks, path-traversal rejection, Milvus filter-injection guards, and a 50 MB upload cap — but it is not a complete internet-facing security boundary on its own.

---

## Known constraints

- **Ingest runs on one worker at a time.** Status, the lock, progress, the cancel flag, and the failure reason live in one Postgres row (`ingest_control`), so any uvicorn worker can read progress and request cancellation. Only the worker that holds the lock writes progress and the terminal state. It refreshes a heartbeat (`INGEST_HEARTBEAT_INTERVAL`, default 5s). If that process dies, the next status read marks the job `interrupted` after `INGEST_HEARTBEAT_STALE` seconds (default 30s) and releases the lock. The row is not expired, so a failed or interrupted job is still there after a restart.
- **No authentication or multi-tenancy.** Every session and document is visible to anyone who can reach the API.
- **Schema changes need a manual rebuild.** If the Milvus collection schema does not match, startup raises a validation error instead of migrating; the collection must be dropped and recreated.
- **No migration tool.** Schema changes are applied with `create_all` plus a small hand-written column-migration list in `infrastructure/database.py`.
- **In-process embedding and rerank are opt-in, and CPU inference is slow.** The default calls the remote embeddings and `/rerank` APIs and does not load torch. Set `EMBEDDING_PROVIDER=huggingface` and `RERANKER_PROVIDER=local` only when this process should load BGE-M3 and the cross-encoder.

---

## Troubleshooting

| Symptom | What to check |
|---|---|
| `/ready` reports `degraded` | `docker compose ps`, then `docker compose logs <service>` |
| Startup fails with "Required env vars not set" | `DATABASE_URL`, `LLM_API_KEY`, and — unless you opted into local models — `EMBEDDING_API_BASE_URL` and `RERANKER_API_BASE_URL` in `backend/.env` |
| Local model download or CPU inference is slow | Stay on the default remote providers, or set `EMBEDDING_DEVICE=cuda` for the local opt-in |
| Chat returns no useful context | Confirm ingestion finished and `MILVUS_HOST` and `DATABASE_URL` point at the running stack |
| Port conflicts | Change the host-side ports in `docker-compose.yml` and update the matching `.env` values |
| Ingestion stuck in `running` | Status is reconciled from the Postgres heartbeat. A dead worker becomes `interrupted` after `INGEST_HEARTBEAT_STALE` seconds (default 30); `POST /ingest/cancel` asks the live worker to stop |

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

[FastAPI](https://fastapi.tiangolo.com/) · [LangChain](https://python.langchain.com/) · [Deep Agents](https://github.com/langchain-ai/deepagents) · [Milvus](https://milvus.io/) · [BGE](https://github.com/FlagOpen/FlagEmbedding) · [RAGAS](https://docs.ragas.io/)
