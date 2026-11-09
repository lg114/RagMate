<div align="center">

# RagMate

**Lightweight, self-hosted RAG application with hybrid retrieval, cross-encoder reranking, and verifiable citations.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.8.7-blue)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-4C566A)](LICENSE)

[English](README.md) · [中文](README_zh.md)

</div>

---

RagMate turns your private documents into an interactive, cited knowledge base. It combines dense semantic search and lexical BM25 fused with Reciprocal Rank Fusion (RRF), cross-encoder reranking, and streaming answers with document-level citations — while keeping 100% of your data and infrastructure under your own control.

---

## Highlights

- **Production-Grade Hybrid Retrieval** — Dense HNSW vectors and lexical BM25 fused via Reciprocal Rank Fusion (RRF, $k=60$).
- **Cross-Encoder Reranking & Compression** — Remote `/rerank` API by default (or optional in-process CrossEncoder), dynamic score thresholding, per-source deduplication, score-gap truncation, and sentence-level contextual compression.
- **Structure-Aware Title-Path Chunking** — Markdown chunks carry parent header paths (`H1 > H2 > H3`) for accurate retrieval, while delivering the larger parent passage to the LLM.
- **Grounded Citations & Zero-Evidence Refusal** — Direct source jumps (file name + page number), retrieval confidence scoring (`high` / `medium` / `low`), and automatic refusal when insufficient evidence is found.
- **Zero-Fuss Self-Hosting** — Single command `docker compose up -d` starts PostgreSQL 15 with pgvector and the web application. Schema and extensions auto-initialize on boot.
- **Localhost-First Security** — Loopback requests (`127.0.0.1`) require zero passwords; remote deployments are protected by a unified instance password with HttpOnly session cookies.
- **Zero-Build Bilingual UI** — Clean single-page interface (English / 中文 toggle) with streaming SSE, mobile drawer support, document ingestion tracking, and runtime model settings.
- **Deterministic CI Regression Guard** — `ragmate-eval retrieval-gate` replays a 130-question test set against a deterministic BM25 baseline with strict Recall@k and nDCG floors (no flaky LLM judges).

---

## Quick Start

The recommended standard deployment requires only **Docker** and **Compose v2** — no local Python environment or build step needed.

```bash
# 1. (Optional) Copy configuration template
cp .env.example .env    # Windows: copy .env.example .env

# 2. Launch the full stack
docker compose up -d
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

> **Tip**: You can enter without configuring `.env` first — RagMate boots gracefully in zero-friction mode, allowing you to configure model endpoints and API keys directly in the **Settings** view with instant connection testing.

- **Zero Database Setup**: PostgreSQL 15, pgvector extension, and database tables initialize automatically on first boot.
- **Persistent Data**: Vector embeddings, documents, and chat sessions persist across restarts in Docker named volumes (`postgres_data`, `ragmate_docs`).
- **Safe Defaults**: Web port binds to `127.0.0.1:8000` only; PostgreSQL is isolated within the internal Docker network.

---

## Web Interface Overview

RagMate provides a zero-build, responsive web application served directly from FastAPI:

1. **Chat View**
   - Real-time Server-Sent Events (SSE) streaming responses.
   - Interactive citation tags that highlight the exact source file and page.
   - Expandable **Retrieval Details** inspector (`"debug": true`) displaying candidate chunks, reranker scores, and filter decisions.
   - Pinned and searchable session history in the sidebar.

2. **Documents View**
   - Drag-and-drop file upload supporting 12+ document formats.
   - Live ingestion progress tracking with cancellation support.
   - Automatic mtime and content-hash deduplication (skips unmodified files).

3. **Settings View**
   - **Model Connections**: Configure Chat LLM, Embeddings, and Rerankers. Includes instant connection testing and index compatibility safeguards.
   - **Service Status & Data Destinations**: Real-time health monitors for Postgres, pgvector, and model endpoints, with transparent disclosure of where data is sent.
   - **Advanced Retrieval Specs**: Read-only display of active runtime retrieval parameters (chunk sizes, thresholds, RRF parameters).

---

## Supported Document Formats

RagMate includes dedicated extractors for all common business and technical document types:

| Category | File Extensions | Extractor Engine |
|---|---|---|
| Portable Documents | `.pdf` | PyPDF (page-aware text extraction) |
| Office Documents | `.docx`, `.pptx`, `.xlsx`, `.csv` | docx2txt, python-pptx, openpyxl, standard CSV |
| Web & Structured | `.html`, `.htm`, `.rtf`, `.json`, `.xml` | BeautifulSoup4, striprtf, standard JSON/XML |
| eBooks & Plaintext | `.epub`, `.txt`, `.md` | EbookLib, native utf-8 stream parser |

*Upload limit defaults to 50 MB per file (configurable via `MAX_UPLOAD_SIZE`).*

---

## Architecture

The system follows a strict layered architecture:

```mermaid
flowchart LR
    subgraph Ingestion ["Ingestion Pipeline"]
        direction TB
        F["Raw Files"] --> L["Format Loader"]
        L --> C["Title-Path Chunking"]
        C --> D["Hash Deduplication"]
        D --> E["Embedding API"]
        E --> V[("pgvector")]
    end

    subgraph Retrieval ["Hybrid Retrieval Pipeline"]
        direction TB
        Q["User Query"] --> H["Dense HNSW + BM25"]
        H --> RRF["RRF Fusion (k=60)"]
        RRF --> RERANK["Cross-Encoder Rerank"]
        RERANK --> FILTER["Gap & Threshold Filter"]
    end

    subgraph Generation ["Grounded Generation"]
        direction TB
        FILTER --> LLM["Chat LLM"]
        LLM --> ANS["Cited Answer (SSE)"]
    end

    Ingestion -.-> Retrieval
    Retrieval --> Generation
```

### Ingestion Path
File discovery (mtime detection)  →  format loader  →  title-path hierarchical chunking  →  content-hash deduplication against pgvector  →  dense embedding via API  →  atomic vector swap (insert new chunks before deleting old chunks)  →  document table sync.

### Retrieval Path
User query  →  parallel Dense HNSW (inner product) and lexical BM25  →  Reciprocal Rank Fusion (RRF, $k=60$)  →  cross-encoder reranking  →  score-gap truncation and dynamic thresholding (0.5 × top score)  →  single-pass grounded generation with verified citations.

---

## Configuration Reference

RagMate loads settings from `.env` in the project root (or `backend/.env`).

### Core Model Settings

| Variable | Description | Default |
|---|---|---|
| `LLM_MODEL` | Chat model name (e.g. `gpt-4o`, `deepseek-chat`, `qwen2.5:7b`) | `gpt-4o` |
| `LLM_API_KEY` | API key for chat model (can also be set in Web UI Settings) | *optional at boot* |
| `LLM_API_BASE_URL` | Base URL for OpenAI-compatible LLM endpoint | `https://api.openai.com/v1` |
| `LLM_TEMPERATURE` | Generation temperature (kept low for factual answers) | `0.01` |
| `EMBEDDING_MODEL` | Embedding model identifier | `BAAI/bge-m3` |
| `EMBEDDING_API_BASE_URL` | Base URL for embeddings endpoint (can also be set in Web UI Settings) | *optional at boot* |
| `EMBEDDING_API_KEY` | API key for embeddings (optional for local Ollama/sidecars) | *empty* |
| `RERANKER_MODEL` | Cross-encoder reranker model identifier | `BAAI/bge-reranker-v2-m3` |
| `RERANKER_API_BASE_URL` | Base URL for reranker endpoint (can also be set in Web UI Settings) | *optional at boot* |
| `RERANKER_API_KEY` | API key for reranker (optional for local sidecars) | *empty* |

### Security & Instance Access

| Variable | Description | Default |
|---|---|---|
| `API_AUTH_TOKEN` | Instance master password. If empty, localhost has direct passwordless access; external requests require this token | *empty* |
| `CORS_ORIGINS` | Allowed CORS origins for browser access | `http://localhost:8000,http://127.0.0.1:8000` |
| `MAX_UPLOAD_SIZE` | Maximum upload size per document (bytes) | `52428800` (50 MB) |

### Storage & Database

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string (auto-injected by Docker Compose) | `postgresql+asyncpg://postgres:postgres@localhost:5432/ragmate` |
| `VECTOR_TABLE` | Target pgvector table name | `ragmate_docs` |
| `DOCUMENTS_DIR` | Directory where uploaded source documents are stored | `/app/documents` |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat/stream` | Stream grounded answers with citations via Server-Sent Events (SSE) |
| `POST` | `/chat` | Non-streaming synchronous chat completion |
| `GET` | `/chat/sessions` | List recent conversation sessions |
| `GET` | `/chat/sessions/{id}` | Retrieve full turn-by-turn message history |
| `DELETE` | `/chat/sessions/{id}` | Permanently delete a conversation session |
| `GET` | `/documents` | List indexed documents, file sizes, and ingest statuses |
| `POST` | `/documents/upload` | Upload a document file (multipart/form-data) |
| `DELETE` | `/documents/{name}` | Delete a document, its source file, and associated vectors |
| `POST` | `/ingest` | Trigger incremental background ingestion |
| `POST` | `/ingest/cancel` | Cancel an in-progress ingestion run |
| `GET` | `/ingest/status` | Ingestion lifecycle state and progress metrics |
| `GET` | `/settings` | Retrieve active runtime settings and index compatibility status |
| `PUT` | `/settings` | Update model connection settings (atomic save with restart notice) |
| `POST` | `/settings/test/{type}` | Test connection for `llm`, `embedding`, or `reranker` |
| `GET` | `/health` | Liveness health check |
| `GET` | `/ready` | Readiness check (validates PostgreSQL and pgvector) |

---

## Evaluation & Retrieval Gate

RagMate includes an automated retrieval regression gate:

```bash
# Replay deterministic 130-question evaluation benchmark
ragmate-eval retrieval-gate
```

- Replays the 130-question curated sample using deterministic BM25.
- Evaluates **Recall@k** and **nDCG** against baseline floors pinned in `eval/reports/baseline/retrieval-baseline-130q-jina-rerun.json`.
- Serves as the primary CI regression test without relying on flaky, expensive LLM judges.

---

## Development & Local Testing

If you are developing backend features, running tests, or prefer running on the host machine:

### 1. Start Development Database
Expose PostgreSQL (`5432`) to the host using the development overlay:
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres
```

### 2. Python Environment Setup
```bash
cd backend
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\Activate.ps1   # Windows PowerShell

pip install -e ".[test]"
```

### 3. Run Development Server
From the project root:
```bash
# Convenience launcher (auto-checks database and port)
python run.py

# Or directly via Uvicorn
uvicorn backend.app:app --reload --port 8000
```

### 4. Run Test Suite
```bash
pytest backend/tests -v
```

---

## Data Persistence & Security

- **Persistent Volumes**: All embeddings and metadata are stored in Docker volume `postgres_data`. Source files reside in `ragmate_docs`. Upgrading images or restarting containers will never delete your data.
- **Backups**: To create a backup, run `docker compose stop`, copy the `postgres_data` volume or dump with `pg_dump`, then restart.
- **Network Isolation**: The database container exposes no ports to the host network by default, preventing unintended access.
- **Instance Security**: Set `API_AUTH_TOKEN` in `.env` whenever deploying RagMate on a shared network or server.

---

## Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| Cannot connect to Ollama from Docker | `localhost` inside container refers to container itself | Use `http://host.docker.internal:11434/v1` in `LLM_API_BASE_URL` or `EMBEDDING_API_BASE_URL`. |
| Port 8000 already in use | Another service is using port 8000 | In `docker-compose.yml`, change `"127.0.0.1:8000:8000"` to `"127.0.0.1:8080:8000"`. |
| Vector dimension mismatch | Switched embedding model with existing index | The Settings UI locks the embedding model when documents are indexed. Delete existing documents or rebuild index to change models. |
| Host `pytest` cannot connect to DB | Postgres port not exposed to host | Run `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres`. |

---

## License

This project is licensed under the [MIT License](LICENSE).
