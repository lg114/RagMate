# RagMate

[中文版](README_zh.md)

---

A production-grade **Retrieval-Augmented Generation (RAG)** application that understands complex questions, retrieves relevant documents from a knowledge base, and generates accurate answers through LLM inference. Built for enterprise knowledge management.

> An intelligent knowledge partner that retrieves the most relevant content from a vast document library and has it read, reasoned about, and answered by a large language model.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/fastapi-0.110-green.svg)](https://fastapi.tiangolo.com/)

---

## 🌟 What it has

- **Intelligent Q&A** — RAG-based question answering with multi-turn conversation support
- **Semantic Vector Search** — Milvus-powered high-accuracy document retrieval at scale
- **Deep Agent** — Multi-turn reasoning assistant with sub-agent support and task planning
- **Multi-format Support** — PDF parsing, intelligent chunking, and vectorization
- **Production Architecture** — Milvus + PostgreSQL + Redis full-stack infrastructure
- **Flexible LLM/Embedding** — Unify OpenAI, Anthropic, DeepSeek via LiteLLM
- **Local Deployment** — All data self-hosted, no external dependencies
- **LangSmith Tracing** — Full-chain Agent execution monitoring and debugging

---

## 🏗️ How it works

```
PDF Upload                          User Query
    │                                    │
    ▼                                    ▼
┌─────────────────┐            ┌─────────────────┐
│  PDF Parser     │            │  FastAPI Server │
│  (PyPDFLoader)  │            └────────┬────────┘
└────────┬────────┘                     │
         ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│  Text Splitter  │            │  Deep Agent     │
│  (Recursive)    │            │  (retrieval_tool│
└────────┬────────┘            │   + LLM)        │
         ▼                     │                 │
┌─────────────────┐            └──────────┬──────┘
│  Embedding      │                       │
│  (all-MiniLM)   │                       ▼
└────────┬────────┘            ┌─────────────────┐
         │                     │  Redis Session  │
         ▼                     │  (multi-turn)   │
┌─────────────────┐            └────────┬────────┘
│  Milvus         │                     │
│  (vector store) │                     ▼
└────────┬────────┘            ┌─────────────────┐
         │                     │  PostgreSQL     │
         ▼                     │  (chat history) │
    ┌────────────┐             └─────────────────┘
    │ Retrieval  │
    │ (top-k)    │
    └─────┬──────┘
          │
          ▼
    ┌────────────┐
    │    LLM     │
    │  (answer)  │
    └────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** — Use `pyenv install 3.12`
- **Docker Desktop** — For Milvus, PostgreSQL, Redis

### 1. Install Dependencies

```bash
cd backend
pip install -e .
```

### 2. Configure Environment

```bash
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys
```

Key configuration items:

```env
# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=your_api_key
LLM_API_BASE_URL=https://api.openai.com/v1

# Embedding Configuration
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# Optional: LangSmith Tracing
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
```

### 3. Start Infrastructure

```bash
docker-compose up -d
```

| Service | Address | Purpose |
|---------|---------|---------|
| **Milvus** | localhost:19530 | Vector database for document embeddings |
| **Attu** | http://localhost:8080 | Milvus web UI for vector management |
| **PostgreSQL** | localhost:5432 | Document metadata, chat history |
| **Redis** | localhost:6379 | Session cache, distributed lock |
| **MinIO Console** | http://localhost:9001 | Object storage backend for Milvus |

### 4. Start Application

```bash
cd backend
uvicorn main:app --reload --port 8000
# Open http://localhost:8000 in browser
```

---

## 📖 Usage

### Web UI

Access `http://localhost:8000`. Use the sidebar tab to switch between:

- **Chat** — RAG-powered Q&A with multi-turn conversation
- **Documents** — Upload PDFs, manage documents, trigger ingestion

### CLI

```bash
python backend/cli.py
```

Menu:
- **1** — Ingest documents
- **2** — Retrieve documents
- **3** — Chat
- **4** — Exit

---

## 🔌 API Reference

### Chat

```
POST /chat
Body: { "message": "...", "session_id": "optional" }
Response: { "response": "...", "session_id": "..." }
```

```
GET /chat/sessions
Response: { "sessions": [{ "session_id": "...", "first_message": "...", "created_at": "..." }] }
```

```
GET /chat/history/{session_id}
Response: { "session_id": "...", "messages": [{ "role": "user|assistant", "content": "...", "created_at": "..." }] }
```

```
DELETE /chat/sessions/{session_id}
Response: { "success": true }
```

### Documents

```
GET /documents
Response: { "documents": [{ "filename": "...", "size_bytes": ..., "status": "...", "chunk_count": ..., "uploaded_at": "...", "exists_on_disk": true/false }] }
```

```
POST /documents/upload
Body: multipart/form-data, field name "file" (PDF only, max 50MB)
Response: { "filename": "...", "size_bytes": ..., "status": "uploaded", "uploaded_at": "..." }
```

```
DELETE /documents/{filename}
Response: { "success": true }
```

### Ingest

```
POST /ingest
Response: { "status": "started" | "already_running" }
```

```
GET /ingest/status
Response: { "status": "idle|running|success|failed", "document_count": ..., "chunk_count": ..., "last_ingest": "..." }
```

### System

```
GET /health
Response: { "status": "ok" }

GET /ready
Response: { "status": "ready|degraded", "checks": { "milvus": true/false, "postgresql": true/false, "redis": true/false } }
```

---

## 📁 Project Structure

```
RagMate/
├── .python-version
├── .gitignore
├── LICENSE                      # MIT License
├── README.md                    # This file (English)
├── README_zh.md                # Chinese version
├── CHANGELOG.md                 # Version history
├── docker-compose.yml
├── backend/
│   ├── pyproject.toml
│   ├── .env.example
│   ├── config.py                # Pydantic settings, all config validated at startup
│   ├── database.py              # SQLAlchemy async engine
│   ├── models.py                # Document / ChatHistory ORM models
│   ├── errors.py                # Typed error hierarchy
│   ├── redis_client.py          # Redis client + ingest lock
│   ├── model_factory.py         # LLM/Embedding factory
│   ├── retriever.py             # Milvus vector retrieval
│   ├── ingest.py                # PDF ingestion pipeline
│   ├── agent.py                 # Deep Agent with retrieval_tool
│   ├── chat.py                  # Chat orchestration
│   ├── document_service.py      # Document CRUD service layer
│   ├── main.py                 # FastAPI app + endpoints
│   ├── cli.py                  # CLI
│   └── documents/              # PDF storage
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

---

## 🛠️ Tech Stack

| Component | Technology | Note |
|-----------|------------|------|
| Web Framework | FastAPI + Uvicorn | ASGI, serves frontend static files |
| Frontend | HTML/CSS/JS | Zero-dependency native frontend |
| LLM | LiteLLM | Unified call for OpenAI/Anthropic/DeepSeek |
| Embedding | sentence-transformers | Local HuggingFace vectorization |
| Vector DB | Milvus | Production-grade vector retrieval |
| Agent | Deep Agents | Multi-turn reasoning + sub-agent |
| Tracing | LangSmith | Full-chain Agent execution debugging |
| Cache | Redis | Query cache + session state |
| Storage | PostgreSQL | Document metadata + chat history |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.