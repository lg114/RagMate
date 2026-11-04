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

- **Hybrid retrieval** — BGE-M3 dense + sparse search fused with RRF in Milvus.
- **Reranked, filtered context** — cross-encoder reranking, dynamic score thresholds, per-source deduplication, score-gap truncation, and sentence-level contextual compression.
- **Title-path + parent-child chunking** — markdown chunks carry an `H1 > H2 > H3` path for retrieval; the larger parent passage is still what the LLM sees.
- **Grounded chat** — one retrieval, one answer, then citations. Nothing kept means a hard refusal and no answer-model call. Deep Agents is off unless `DEEP_AGENTS_ENABLED=true`.
- **Answer quality signals** — retrieval confidence (`high` / `medium` / `low`) on every RAG answer, citations taken from the chunks that were actually retrieved, a refusal or hedge when confidence is low, and faithfulness verification (on by default) that flags unsupported claims.
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

1. Open **文档** and upload one or more supported files.
2. Ingest the pending files and wait until the in-page status banner finishes.
3. Open **对话** and ask questions about the indexed documents.
4. Click a citation to open the source: file name, page, library status, and the retrieved passage when retrieval debug is on.
5. **设置** holds appearance, the shared retrieval-debug switch, an optional API token, and readiness for PostgreSQL, Redis, and Milvus.

The UI is still the zero-build page at `frontend/index.html`, served at `/`. No frontend build step.

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

**Ingestion path** — file discovery with mtime-based change detection → format-specific loader (PyPDF, Docx2txt, TextLoader, Excel to Markdown) → heading-aware chunking with optional title-path prefixes → content-hash deduplication against Milvus → BGE-M3 dense + sparse encoding → insert new vectors → delete the previous vectors for that source → sync the `documents` table.

**Query path** — optional query contextualization (rewrites follow-ups into standalone queries) → dense + sparse ANN search with RRF fusion → cross-encoder reranking (logits converted with sigmoid) → sentence-level contextual compression → dynamic threshold, per-source deduplication, and score-gap truncation → agent and LLM → citation normalization → faithfulness check (on by default). RAG answers always include the retrieved sources and a confidence level. If retrieval kept no chunks, the answer is replaced with a refusal; a low score with some chunks is prefixed with an explicit hedge.

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
| POST | `/ingest/cancel` | Ask the worker that holds the ingest lock to stop |
| GET | `/ingest/status` | Current ingestion status and progress (survives restarts) |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe across PostgreSQL, Redis, and Milvus |

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
| `QUERY_ROUTING_ENABLED` | `true` | Greetings skip retrieval and go straight to the LLM |
| `DEEP_AGENTS_ENABLED` | `false` | Opt in to the Deep Agents tool loop. Default is one retrieve, one answer, then grounding |
| `FAITHFULNESS_CHECK` | `true` | One extra LLM call per RAG answer; failures are ignored and the answer is still returned |

<details>
<summary><strong>Full variable list</strong> — chunking, embedding, agent, and tracing</summary>

**Chunking** — `CHUNK_SIZE` (`1000`), `CHUNK_OVERLAP` (`200`), `CHUNK_SIZE_PDF` (`600`), `CHUNK_OVERLAP_PDF` (`100`), `CHUNK_SIZE_DOCX` (`800`), `CHUNK_OVERLAP_DOCX` (`150`), `CHUNK_SIZE_TXT` (`1000`), `CHUNK_OVERLAP_TXT` (`200`), `CHUNK_SIZE_TABLE` (`1500`), `CHUNK_SIZE_PARENT` (`2500`), `CHUNK_OVERLAP_PARENT` (`300`), `CHUNKING_STRATEGY` (`title_path`; set `parent_child` to keep header paths out of retrieval text).

Overlap must be smaller than its chunk size, otherwise startup fails with a validation error.

**Reranker tuning** — `HIGH_SCORE_RATIO` (`0.6`), `SOURCE_DOMINANCE_THRESHOLD` (`0.9`), `SOURCE_DOMINANCE_BOOST` (`1.5`), `DROP_RATIO_SEARCH` (`0.2`), `COMPRESSION_SCORE_THRESHOLD` (`0.4`), `COMPRESSION_MIN_CHARS` (`300`).

**Embedding** — `EMBEDDING_PROVIDER` (`huggingface`), `EMBEDDING_MODEL` (`BAAI/bge-m3`), `EMBEDDING_NORMALIZE` (`true`), `HF_TOKEN`, `EMBEDDING_API_KEY`, `EMBEDDING_API_BASE_URL`, `EMBEDDING_API_TIMEOUT` (`60`).

`huggingface` (also `local` / `flagembedding`) loads BGE-M3 inside the API process with FlagEmbedding. `EMBEDDING_DEVICE` (`cpu` or `cuda`) is passed to that loader. `remote` or `openai` calls `POST {EMBEDDING_API_BASE_URL}/embeddings` instead, and startup does not warm a local BGE model. Pair it with `RERANKER_PROVIDER=api` so rerank stays off-process too. If the service includes `sparse` (or `lexical_weights`) on each result, those weights are stored; otherwise RagMate fills a stable hashed sparse vector. That hash is not BGE-M3 lexical weight, so re-ingest after switching providers. A sidecar can be any OpenAI-compatible embeddings server plus a `/rerank` endpoint (see the commented block in `backend/.env.example`). CI does not need a GPU or those services: the default remains local FlagEmbedding, and tests mock the HTTP client.

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

- **Ingest runs on one worker at a time.** Status, the lock, progress, and the cancel flag live in Redis, so any uvicorn worker can read progress and request cancellation. The worker that acquired the lock does the work and refreshes a heartbeat (`INGEST_HEARTBEAT_INTERVAL`, default 5s). If that process dies, the next status read marks the job `interrupted` after `INGEST_HEARTBEAT_STALE` seconds (default 30s) and releases the lock.
- **No authentication or multi-tenancy.** Every session and document is visible to anyone who can reach the API.
- **Schema changes need a manual rebuild.** If the Milvus collection schema does not match, startup raises a validation error instead of migrating; the collection must be dropped and recreated.
- **No migration tool.** Schema changes are applied with `create_all` plus a small hand-written column-migration list in `infrastructure/database.py`.
- **CPU inference is slow when models stay in-process.** The default loads BGE-M3 and the reranker locally. Point `EMBEDDING_PROVIDER=remote` and `RERANKER_PROVIDER=api` at a sidecar if the API process should not load those weights.

---

## Troubleshooting

| Symptom | What to check |
|---|---|
| `/ready` reports `degraded` | `docker compose ps`, then `docker compose logs <service>` |
| Startup fails with "Required env vars not set" | `DATABASE_URL` and `LLM_API_KEY` in `backend/.env` |
| Model download or CPU inference is slow | Use CUDA, or a hosted embedding service |
| Chat returns no useful context | Confirm ingestion finished and `MILVUS_HOST`, `DATABASE_URL`, `REDIS_URL` point at the running stack |
| Port conflicts | Change the host-side ports in `docker-compose.yml` and update the matching `.env` values |
| Ingestion stuck in `running` | Status is reconciled from the Redis heartbeat. A dead worker becomes `interrupted` after `INGEST_HEARTBEAT_STALE` seconds (default 30); `POST /ingest/cancel` asks the live worker to stop |

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

[FastAPI](https://fastapi.tiangolo.com/) · [LangChain](https://python.langchain.com/) · [Deep Agents](https://github.com/langchain-ai/deepagents) · [Milvus](https://milvus.io/) · [BGE](https://github.com/FlagOpen/FlagEmbedding) · [RAGAS](https://docs.ragas.io/)
