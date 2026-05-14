# RagMate

[中文版](README_zh.md)

An enterprise-grade knowledge management system based on Retrieval-Augmented Generation (RAG). Users upload documents, and the system retrieves the most relevant content from the knowledge base via vector search and LLM inference to generate accurate answers.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/fastapi-0.115-green.svg)](https://fastapi.tiangolo.com/)

---

## Key Features

- **Hybrid Search** — Dense + Sparse vector hybrid search with RRF fusion + cross-encoder reranking, significantly outperforming pure vector retrieval
- **Deep Agent** — LangGraph-based multi-turn reasoning agent with tool calling, sub-agent delegation, and task planning
- **Streaming Output** — SSE real-time token-by-token streaming
- **Multi-Format Documents** — PDF, DOCX, XLSX, TXT, Markdown support
- **Smart Chunking** — Markdown split by heading hierarchy, PDF page numbers preserved, all chunks carry sequential metadata
- **Multilingual Embedding** — BAAI/bge-m3 (1024-dim), native dense + sparse dual vectors
- **Flexible LLM Integration** — Access any OpenAI-compatible API (OpenAI, Anthropic, DeepSeek, MiMo, etc.) via LangChain ChatOpenAI
- **Full Observability** — LangSmith tracing for agent execution
- **Self-Hosted** — All data on-premise, no external dependencies
- **Retrieval Evaluation** — Built-in evaluation system for quantifying recall and precision

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   FastAPI Server     │
│   POST /chat/stream  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Deep Agent         │
│   (LangGraph +       │
│    ChatOpenAI)       │
└──────────┬──────────┘
           │ tool_call: retrieval_tool
           ▼
┌─────────────────────┐
│   Hybrid Retrieval   │
│   Dense + Sparse     │
│   + RRF + Reranking  │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌─────────┐
│ Milvus │  │ Redis   │
│ Vector │  │ Session │
│ Store  │  │ Cache   │
└────┬───┘  └────┬────┘
     │           │
     ▼           ▼
┌────────┐  ┌─────────────┐
│ Doc    │  │ PostgreSQL  │
│ Ingest │  │ Chat History│
└────────┘  └─────────────┘
```

### Retrieval Flow

1. User query encoded to dense + sparse dual vectors via BGE-M3
2. Milvus parallel search: dense (AUTOINDEX, IP) + sparse (SPARSE_INVERTED_INDEX, IP)
3. RRF (Reciprocal Rank Fusion) merges both result sets
4. Cross-encoder (BAAI/bge-reranker-v2-m3) reranks candidates
5. Returns top-k results with text, source, page, chunk_index, and relevance score

### Ingestion Flow

1. Scans document directory, incremental processing of new files
2. Selects loader by format (PyPDF / Docx2txt / UnstructuredExcel / TextLoader)
3. Markdown uses `MarkdownHeaderTextSplitter` (preserves H1/H2/H3 hierarchy), others use `RecursiveCharacterTextSplitter`
4. BGE-M3 encodes to dense + sparse dual vectors
5. Stores in Milvus with metadata (source, page, chunk_index)
6. Syncs PostgreSQL document status

---

## Quick Start

### Prerequisites

- Python 3.12+
- Docker Desktop (for Milvus, PostgreSQL, Redis)

### 1. Start Infrastructure

```bash
docker-compose up -d
```

| Service | Port | Purpose |
|---------|------|---------|
| Milvus | 19530 | Vector database (dense + sparse) |
| Attu | 8080 | Milvus web admin UI |
| PostgreSQL | 5432 | Document metadata, chat history |
| Redis | 6379 | Session cache, distributed lock |
| MinIO | 9001 | Milvus object storage backend |

### 2. Install Dependencies

```bash
cd backend
pip install -e .
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` with your LLM API settings:

```env
LLM_MODEL=gpt-4o
LLM_API_KEY=your_api_key
LLM_API_BASE_URL=https://api.openai.com/v1
```

Supports any OpenAI-compatible API (DeepSeek, MiMo, etc.):

```env
LLM_MODEL=deepseek-chat
LLM_API_KEY=your_key
LLM_API_BASE_URL=https://api.deepseek.com/v1
```

### 4. Start Server

```bash
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 in browser.

---

## Usage

### Web UI

- **Chat** — Knowledge-base powered streaming Q&A with multi-turn conversation
- **Documents** — Upload documents, manage documents, trigger ingestion

### CLI

```bash
python backend/cli.py
```

| Option | Function |
|--------|----------|
| 1 | Ingest documents |
| 2 | Retrieve documents |
| 3 | Chat |
| 4 | Evaluate (retrieval quality test) |
| 5 | Exit |

### Evaluation System

```bash
python backend/cli.py
# Select 4
```

Loads `eval/dataset.json` test dataset, runs retrieval for each question, computes recall and precision, outputs formatted report. Expand the dataset by adding more Q&A entries to `eval/dataset.json`.

---

## API Reference

### Chat

```
POST /chat
Body: { "message": "...", "session_id": "optional" }
Response: { "response": "...", "session_id": "..." }
```

```
POST /chat/stream
Body: { "message": "...", "session_id": "optional" }
Response: text/event-stream
  data: {"token": "..."}
  data: {"done": true, "session_id": "..."}
```

```
GET /chat/sessions
Response: { "sessions": [{ "session_id": "...", "first_message": "...", "created_at": "..." }] }
```

```
GET /chat/history/{session_id}
Response: { "session_id": "...", "messages": [{ "role": "...", "content": "...", "created_at": "..." }] }
```

```
DELETE /chat/sessions/{session_id}
Response: { "success": true }
```

### Documents

```
GET /documents
Response: { "documents": [{ "filename": "...", "size_bytes": ..., "status": "...", "chunk_count": ... }] }
```

```
POST /documents/upload
Body: multipart/form-data, field name "file" (PDF/DOCX/XLSX/TXT/MD, max 50MB)
Response: { "filename": "...", "status": "uploaded" }
```

```
DELETE /documents/{filename}
Response: { "success": true }
```

### Ingestion

```
POST /ingest
Response: { "status": "started" | "already_running" }
```

```
GET /ingest/status
Response: { "status": "idle|running|success|failed", "document_count": ..., "chunk_count": ... }
```

### System

```
GET /health
Response: { "status": "ok" }

GET /ready
Response: { "status": "ready|degraded", "checks": { "milvus": ..., "postgresql": ..., "redis": ... } }
```

---

## Configuration

All settings are configured via `.env` file or environment variables, validated by `pydantic-settings`.

| Category | Variable | Default | Description |
|----------|----------|---------|-------------|
| **LLM** | `LLM_MODEL` | `gpt-4o` | Model name |
| | `LLM_API_KEY` | | API key |
| | `LLM_API_BASE_URL` | | Custom API endpoint |
| **Embedding** | `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` or `openai` |
| | `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model |
| | `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| | `EMBEDDING_NORMALIZE` | `true` | Normalize embeddings |
| | `HF_TOKEN` | | HuggingFace token |
| **Database** | `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| | `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| **Milvus** | `MILVUS_HOST` | `localhost` | Host |
| | `MILVUS_PORT` | `19530` | Port |
| | `MILVUS_COLLECTION` | `ragmate_docs` | Collection name |
| **Ingestion** | `CHUNK_SIZE` | `500` | Text chunk size |
| | `CHUNK_OVERLAP` | `50` | Chunk overlap |
| **Retrieval** | `HYBRID_SEARCH_ENABLED` | `true` | Enable hybrid search |
| | `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model |
| | `RERANK_CANDIDATES` | `20` | Rerank candidate pool size |
| | `FINAL_CONTEXT_K` | `4` | Chunks passed to the LLM |
| | `RERANK_SCORE_THRESHOLD` | `0.15` | Results below this score are discarded |
| **LangSmith** | `LANGSMITH_TRACING` | `false` | Enable tracing |
| | `LANGSMITH_API_KEY` | | LangSmith API key |

---

## Project Structure

```
RagMate/
├── docker-compose.yml
├── LICENSE
├── README.md
├── README_zh.md
├── CHANGELOG.md
├── .python-version
├── .gitignore
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── backend/
    ├── pyproject.toml
    ├── .env.example
    ├── config.py              # Configuration (pydantic-settings)
    ├── main.py                # FastAPI app + all endpoints
    ├── agent.py               # Deep Agent (system prompt + retrieval_tool)
    ├── chat.py                # Chat orchestration (sync + streaming)
    ├── retriever.py           # Hybrid search + Reranking
    ├── ingest.py              # Document processing + vector ingestion
    ├── streaming_llm.py       # ChatOpenAI factory
    ├── model_factory.py       # LLM / Embedding factory
    ├── document_service.py    # Document CRUD
    ├── database.py            # SQLAlchemy async/sync engines
    ├── models.py              # ORM models (Document, ChatHistory)
    ├── redis_client.py        # Redis session / lock / status
    ├── errors.py              # Typed error hierarchy
    ├── cli.py                 # CLI
    ├── eval/                  # Evaluation system
    │   ├── dataset.json       # Test Q&A dataset
    │   ├── metrics.py         # Recall / precision metrics
    │   └── runner.py          # Evaluation runner
    └── documents/             # Document storage directory
```

---

## Tech Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| Web Framework | FastAPI + Uvicorn | ASGI, serves frontend static files |
| Frontend | HTML/CSS/JS | Zero-dependency native frontend |
| LLM | LangChain ChatOpenAI | Access any OpenAI-compatible API |
| Embedding | BAAI/bge-m3 | 1024-dim, multilingual, dense + sparse dual vectors |
| Vector DB | Milvus 2.5 | Hybrid search (dense + sparse + RRF) |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder reranking |
| Agent | LangGraph (Deep Agents) | Multi-turn reasoning + tool calling |
| Tracing | LangSmith | Agent execution monitoring |
| Cache | Redis | Session state + distributed lock |
| Storage | PostgreSQL | Document metadata + chat history |

---

## License

MIT License — see [LICENSE](LICENSE).
