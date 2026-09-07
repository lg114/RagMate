<div align="center">

# RagMate

**Self-hosted RAG knowledge management for searchable documents and cited answers.**

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

- **Hybrid retrieval** — BGE-M3 dense + sparse search fused with RRF in Milvus.
- **Reranked, filtered context** — cross-encoder reranking, dynamic score thresholds, per-source deduplication, score-gap truncation, and sentence-level contextual compression.
- **Parent-child chunking** — retrieve on small chunks, feed the LLM the larger parent passage.
- **Agentic chat** — Deep Agents workflow with multi-turn sessions, SSE streaming, and retrieval exposed as a tool.
- **Answer quality signals** — retrieval confidence (`high` / `medium` / `low`) and optional faithfulness verification that flags unsupported claims.
- **Incremental ingestion** — mtime-based change detection, content-hash deduplication, and crash-safe "insert new vectors before deleting old ones".
- **Built-in evaluation** — optional RAGAS CLI for test-set generation, scoring, and CI quality gates.
- **Self-hosted by design** — PostgreSQL, Redis, Milvus, and MinIO run through Docker Compose.

---

## Quick start

### Prerequisites

- Python 3.12+
- Docker Desktop, or Docker Engine with Compose v2
- An OpenAI-compatible LLM endpoint and API key
- At least 8 GB RAM for CPU embedding; a GPU is recommended for larger collections

### 1. Start the infrastructure

From the repository root:

```bash
docker compose up -d
docker compose ps
```

The stack exposes PostgreSQL (`5432`), Redis (`6379`), Milvus (`19530`), MinIO (`9000` / `9001`), and Attu (`8080`). The application itself runs on your host during development.

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
> Two variables are required — the app refuses to start without them: `DATABASE_URL` and `LLM_API_KEY`.

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ragmate
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o

# Optional: any OpenAI-compatible provider
LLM_API_BASE_URL=https://api.openai.com/v1
```

The default database, Redis, and Milvus values match the Docker Compose stack. See [`backend/.env.example`](backend/.env.example) for the annotated full list.

### 4. Run RagMate

Return to the repository root, with the virtual environment still active:

```bash
uvicorn backend.app:app --reload --port 8000
```

The first start downloads BGE-M3 and the reranker model (a few GB) and warms both up in the background, so allow extra time before the first request.

Open [http://localhost:8000](http://localhost:8000) for the web UI. Interactive API docs are at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## First-use workflow

1. Open the **Documents** tab and upload one or more supported files.
2. Trigger ingestion and wait for the status to become `success`.
3. Open **Chat** and ask questions about the indexed documents.
4. Use the returned citations to inspect the source passages.

Supported formats: `.pdf`, `.docx`, `.xlsx`, `.xls`, `.txt`, `.md`. Uploads are limited to 50 MB per file by default.

---

## Project layout

```
RagMate/
├── docker-compose.yml      # PostgreSQL, Redis, Milvus, MinIO, etcd, Attu
├── backend/
│   ├── app.py              # App factory: middleware, exception handlers, static mount
│   ├── api/                # HTTP routes: chat, documents, ingest, health
│   ├── application/        # Use-case orchestration
│   │   └── ingest/         # pipeline (load → chunk → dedupe → encode), loaders, db_sync
│   ├── core/               # RAG core: retriever, agent, prompts/researcher.md
│   ├── domain/             # ORM models, Pydantic schemas, error types
│   ├── infrastructure/     # config, database, redis, milvus, encoding, model_factory
│   ├── eval/               # RAGAS evaluation CLI (optional dependency)
│   └── tests/
└── frontend/               # Zero-build vanilla HTML/CSS/JS, served by FastAPI
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
| Infrastructure | `infrastructure/` | External systems: Milvus, PostgreSQL, Redis, models, config |

```mermaid
flowchart LR
    D[Documents] --> P[Load and chunk]
    P --> E[BGE-M3 encode]
    E --> V[(Milvus vectors)]
    P --> M[(PostgreSQL metadata)]
    Q[User query] --> R[Hybrid search + RRF]
    R --> RR[Rerank, compress, filter]
    RR --> A[Agent and LLM]
    A --> O[Answer with citations]
```

**Ingestion path** — file discovery with mtime-based change detection → format-specific loader (PyPDF, Docx2txt, TextLoader, Excel to Markdown) → heading-aware and parent-child chunking → content-hash deduplication against Milvus → BGE-M3 dense + sparse encoding → insert new vectors → delete the previous vectors for that source → sync the `documents` table.

**Query path** — optional query contextualization (rewrites follow-ups into standalone queries) → dense + sparse ANN search with RRF fusion → cross-encoder reranking (logits converted with sigmoid) → sentence-level contextual compression → dynamic threshold, per-source deduplication, and score-gap truncation → agent and LLM → citation normalization → optional faithfulness check.

Simple greetings and other non-RAG messages are routed straight to the LLM, skipping the agent entirely.

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/chat` | Non-streaming answer |
| POST | `/chat/stream` | SSE streaming answer |
| GET | `/chat/sessions` | List the 50 most recent sessions |
| GET | `/chat/sessions/{session_id}` | Full message history for a session |
| DELETE | `/chat/sessions/{session_id}` | Delete a session (Redis cache + PostgreSQL) |
| GET | `/documents` | List documents and their ingestion status |
| POST | `/documents/upload` | Upload a single file (50 MB max) |
| DELETE | `/documents/{filename}` | Delete a document, its file, and its vectors |
| POST | `/ingest` | Start ingestion (optionally for specific filenames) |
| GET | `/ingest/status` | Current ingestion status and progress |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe across PostgreSQL, Redis, and Milvus |

### Chat request body

```jsonc
{
  "message": "What is RAG?",   // required, max 10 000 characters
  "session_id": "...",         // optional UUID; a new one is created when omitted
  "replace_last": false        // retry / regenerate: drop the last turn first
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
       "confidence": {"level": "high", "score": 0.87, "chunks": 6},
       "unsupported_claims": [{"claim": "...", "supported": false}]}

data: {"error": "..."}                                 // terminal error
```

Idle connections receive a `:heartbeat` comment every 15 seconds. `confidence` and `unsupported_claims` appear only on the `done` event; `unsupported_claims` is present only when `FAITHFULNESS_CHECK` is enabled and some claim failed verification.

---

## Configuration

Settings are read from `backend/.env` (see [`backend/.env.example`](backend/.env.example)). They are validated at startup, so invalid combinations fail fast rather than at query time.

### Essentials

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | **required** | Metadata and chat history |
| `LLM_API_KEY` | **required** | LLM provider credential |
| `LLM_MODEL` | `gpt-4o` | Model exposed by the provider |
| `LLM_API_BASE_URL` | empty | Custom OpenAI-compatible endpoint |
| `LLM_TEMPERATURE` | `0.01` | Kept low for factual answers |
| `REDIS_URL` | `redis://localhost:6379/0` | Sessions, ingestion lock, rate limits |
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
| `QUERY_ROUTING_ENABLED` | `true` | Skip the agent for greetings and chitchat |
| `FAITHFULNESS_CHECK` | `false` | Adds one extra LLM call per answer |

<details>
<summary><strong>Full variable list</strong> — chunking, embedding, agent, and tracing</summary>

**Chunking** — `CHUNK_SIZE` (`1000`), `CHUNK_OVERLAP` (`200`), `CHUNK_SIZE_PDF` (`600`), `CHUNK_OVERLAP_PDF` (`100`), `CHUNK_SIZE_DOCX` (`800`), `CHUNK_OVERLAP_DOCX` (`150`), `CHUNK_SIZE_TXT` (`1000`), `CHUNK_OVERLAP_TXT` (`200`), `CHUNK_SIZE_TABLE` (`1500`), `CHUNK_SIZE_PARENT` (`2500`), `CHUNK_OVERLAP_PARENT` (`300`).

Overlap must be smaller than its chunk size, otherwise startup fails with a validation error.

**Reranker tuning** — `HIGH_SCORE_RATIO` (`0.6`), `SOURCE_DOMINANCE_THRESHOLD` (`0.9`), `SOURCE_DOMINANCE_BOOST` (`1.5`), `DROP_RATIO_SEARCH` (`0.2`), `COMPRESSION_SCORE_THRESHOLD` (`0.4`), `COMPRESSION_MIN_CHARS` (`300`).

**Embedding** — `EMBEDDING_PROVIDER` (`huggingface`), `EMBEDDING_MODEL` (`BAAI/bge-m3`), `EMBEDDING_NORMALIZE` (`true`), `HF_TOKEN`, `EMBEDDING_API_KEY`, `EMBEDDING_API_BASE_URL`.

`EMBEDDING_DEVICE` (`cpu`) applies to the HuggingFace embeddings path used by the evaluation CLI. The indexing and retrieval path loads BGE-M3 through FlagEmbedding and does not currently honour this setting.

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

---

## Development

```bash
cd backend
pip install -e ".[test]"
pytest -v
```

Tests live in `backend/tests/` and cover the retrieval filtering and deduplication logic. `conftest.py` provides an `override_settings` fixture for temporarily patching configuration.

> [!NOTE]
> `ruff` and `mypy` are not configured in `pyproject.toml` yet. The codebase follows PEP 8, 4-space indentation, and Chinese docstrings on public functions.

Every blocking call — model inference, Milvus I/O, ingestion — is pushed to a thread with `asyncio.to_thread`, so avoid adding blocking work directly to coroutines.

---

## Data, backup, and security

Docker volumes are written to `volumes/` and hold PostgreSQL, Redis, Milvus, and MinIO data. For a consistent backup, stop the stack, copy the directory, then restart it.

Before exposing the service beyond localhost:

- Set `CORS_ORIGINS` to your real origin.
- Replace the default infrastructure credentials in `docker-compose.yml`.
- Put an authentication layer in front of the API.
- Review upload and rate-limit settings — currently 30 requests per IP per minute on chat, upload, delete, and ingest.

RagMate validates requests — filename allowlist, magic-byte checks, path-traversal rejection, Milvus filter-injection guards, and a 50 MB upload cap — but it is not a complete internet-facing security boundary on its own.

---

## Known constraints

- **Single process assumed.** The ingestion task handle and lock token live in process memory, so running multiple workers (for example `uvicorn --workers 4`) breaks ingestion status and cancellation. Run a single worker.
- **No authentication or multi-tenancy.** Every session and document is visible to anyone who can reach the API.
- **Schema changes need a manual rebuild.** If the Milvus collection schema does not match, startup raises a validation error instead of migrating; the collection must be dropped and recreated.
- **No migration tool.** Schema changes are applied with `create_all` plus a small hand-written column-migration list in `infrastructure/database.py`.
- **CPU inference is slow.** Embedding and reranking run locally; expect seconds per query on CPU-only machines.

---

## Troubleshooting

| Symptom | What to check |
|---|---|
| `/ready` reports `degraded` | `docker compose ps`, then `docker compose logs <service>` |
| Startup fails with "Required env vars not set" | `DATABASE_URL` and `LLM_API_KEY` in `backend/.env` |
| Model download or CPU inference is slow | Use CUDA, or a hosted embedding service |
| Chat returns no useful context | Confirm ingestion finished and `MILVUS_HOST`, `DATABASE_URL`, `REDIS_URL` point at the running stack |
| Port conflicts | Change the host-side ports in `docker-compose.yml` and update the matching `.env` values |
| Ingestion stuck in `running` | A previous crash may still hold the lock; it expires after 10 minutes, or restart the app |

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

[FastAPI](https://fastapi.tiangolo.com/) · [LangChain](https://python.langchain.com/) · [Deep Agents](https://github.com/langchain-ai/deepagents) · [Milvus](https://milvus.io/) · [BGE](https://github.com/FlagOpen/FlagEmbedding) · [RAGAS](https://docs.ragas.io/)
