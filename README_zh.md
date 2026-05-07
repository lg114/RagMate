# RagMate

[English Version](README.md)

---

一个 **检索增强生成(RAG)** 应用，能自主理解复杂问题、从知识库中检索相关文档，并通过大语言模型推理生成准确答案。为企业知识管理而构建。

> 一个智能的知识伙伴，从海量文档中检索最相关的内容，交由大语言模型阅读、推理并给出准确答案。

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/fastapi-0.110-green.svg)](https://fastapi.tiangolo.com/)

---

## 🌟 功能特点

- **智能问答** — 基于检索增强生成的问答，支持多轮对话
- **语义向量检索** — Milvus 驱动的高精度大规模文档检索
- **深度 Agent** — 多轮推理助手，支持子 Agent 派生和任务规划
- **多格式支持** — PDF 解析、智能切分与向量化
- **生产级架构** — Milvus + PostgreSQL + Redis 全栈基础设施
- **灵活的 LLM/Embedding** — 通过 LiteLLM 统一接入 OpenAI、Anthropic、DeepSeek
- **本地部署** — 所有数据自托管，无外部依赖
- **LangSmith 追踪** — 全链路 Agent 执行监控与调试

---

## 🏗️ 架构

```
PDF 上传                              用户提问
    │                                    │
    ▼                                    ▼
┌─────────────────┐            ┌─────────────────┐
│  PDF 解析器      │           │  FastAPI 服务   │ 
│  (PyPDFLoader)  │            └────────┬────────┘
└────────┬────────┘                     │
         ▼                             ▼
┌─────────────────┐            ┌─────────────────┐
│  文本切分器     │            │  深度 Agent     │
│  (Recursive)    │            │ (检索工具 + LLM)│
└────────┬────────┘            │                 │
         ▼                     │                 │
┌─────────────────┐            └──────────┬──────┘
│  Embedding      │                       │
│  (all-MiniLM)   │                       ▼
└────────┬────────┘            ┌─────────────────┐
         │                     │  Redis 会话     │
         ▼                     │  (多轮对话)     │
┌─────────────────┐            └────────┬────────┘
│  Milvus         │                     │
│  (向量存储)     │                     ▼
└────────┬────────┘            ┌─────────────────┐
         │                     │  PostgreSQL     │
         ▼                     │  (聊天历史)     │
    ┌────────────┐             └─────────────────┘
    │  向量检索   │
    │  (top-k)   │
    └─────┬──────┘
          │
          ▼
    ┌────────────┐
    │    LLM     │
    │  (生成答案) │
    └────────────┘
```

---

## 🚀 快速开始

### 环境准备

- **Python 3.12+** — 推荐使用 `pyenv install 3.12`
- **Docker Desktop** — 用于运行 Milvus、PostgreSQL、Redis

### 1. 安装依赖

```bash
cd backend
pip install -e .
```

### 2. 配置文件

```bash
cp backend/.env.example backend/.env
# 编辑 backend/.env，配置你的 API Key
```

主要配置项：

```env
# LLM 配置
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=your_api_key
LLM_API_BASE_URL=https://api.openai.com/v1

# Embedding 配置
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# 可选：LangSmith 追踪
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
```

### 3. 启动基础设施

```bash
docker-compose up -d
```

| 服务 | 地址 | 用途 |
|------|------|------|
| **Milvus** | localhost:19530 | 向量数据库，存储文档 embedding |
| **Attu** | http://localhost:8080 | Milvus Web UI，可视化管理向量数据 |
| **PostgreSQL** | localhost:5432 | 文档元数据、对话历史 |
| **Redis** | localhost:6379 | 查询缓存、会话状态 |
| **MinIO Console** | http://localhost:9001 | Milvus 后端对象存储管理 |

### 4. 启动服务

```bash
cd backend
uvicorn main:app --reload --port 8000
# 浏览器打开 http://localhost:8000
```

---

## 📖 使用说明

### Web UI

访问 `http://localhost:8000`，使用侧边栏 Tab 切换功能：

- **对话** — 基于知识库的智能问答，支持多轮对话
- **文档** — 上传 PDF、管理文档、触发向量入库

### CLI

```bash
python backend/cli.py
```

菜单选项：
- **1** — 摄入文档
- **2** — 检索文档
- **3** — 聊天问答
- **4** — 退出

---

## 🔌 API 参考

### 对话

```
POST /chat
Body: { "message": "...", "session_id": "可选" }
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

### 文档

```
GET /documents
Response: { "documents": [{ "filename": "...", "size_bytes": ..., "status": "...", "chunk_count": ..., "uploaded_at": "...", "exists_on_disk": true/false }] }
```

```
POST /documents/upload
Body: multipart/form-data，字段名 "file"（仅 PDF，最大 50MB）
Response: { "filename": "...", "size_bytes": ..., "status": "uploaded", "uploaded_at": "..." }
```

```
DELETE /documents/{filename}
Response: { "success": true }
```

### 入库

```
POST /ingest
Response: { "status": "started" | "already_running" }
```

```
GET /ingest/status
Response: { "status": "idle|running|success|failed", "document_count": ..., "chunk_count": ..., "last_ingest": "..." }
```

### 系统

```
GET /health
Response: { "status": "ok" }

GET /ready
Response: { "status": "ready|degraded", "checks": { "milvus": true/false, "postgresql": true/false, "redis": true/false } }
```

---

## 📁 项目目录

```
RagMate/
├── .python-version
├── .gitignore
├── LICENSE                      # MIT 许可证
├── README.md                   # 英文版
├── README_zh.md                # 本文件（中文版）
├── CHANGELOG.md                # 版本历史
├── docker-compose.yml
├── backend/
│   ├── pyproject.toml
│   ├── .env.example
│   ├── config.py                # Pydantic 配置，启动时校验，失败即终止
│   ├── database.py              # SQLAlchemy 异步引擎
│   ├── models.py                # Document / ChatHistory ORM 模型
│   ├── errors.py                # 类型化错误层级
│   ├── redis_client.py          # Redis 客户端 + 入库分布式锁
│   ├── model_factory.py          # LLM/Embedding 工厂
│   ├── retriever.py             # Milvus 向量检索
│   ├── ingest.py                 # PDF 入库流程
│   ├── agent.py                 # 深度 Agent（集成 retrieval_tool）
│   ├── chat.py                  # 聊天编排（Redis 会话 + PG 持久化）
│   ├── document_service.py       # 文档 CRUD 服务层
│   ├── main.py                  # FastAPI 入口 + 所有端点
│   ├── cli.py                   # CLI
│   └── documents/              # PDF 存储目录
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

---

## 🛠️ 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 高性能 ASGI，托管前端静态文件 |
| 前端 | HTML/CSS/JS | 零依赖，原生前端 |
| LLM | LiteLLM | 统一调用 OpenAI/Anthropic/DeepSeek |
| Embedding | sentence-transformers | 本地 HuggingFace 向量化 |
| 向量数据库 | Milvus | 生产级向量检索 |
| Agent | Deep Agents | 多轮推理 + 子 Agent |
| 追踪 | LangSmith | 全链路调试 |
| 缓存 | Redis | 查询缓存 + 会话状态 |
| 存储 | PostgreSQL | 文档元数据 + 对话历史 |

---

## 📄 许可证

MIT 许可证 — 详见 [LICENSE](LICENSE) 文件。
