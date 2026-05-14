# RagMate

[English](README.md)

一个基于检索增强生成（RAG）的企业级知识管理系统。用户上传文档后，系统通过向量检索和大语言模型推理，从知识库中检索最相关内容并生成准确答案。

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/fastapi-0.115-green.svg)](https://fastapi.tiangolo.com/)

---

## 核心特性

- **混合检索** — Dense + Sparse 向量混合搜索（RRF 融合）+ 交叉编码器 Reranking，召回率和准确率显著优于纯向量检索
- **深度 Agent** — 基于 LangGraph 的多轮推理 Agent，支持工具调用、子 Agent 派生和任务规划
- **流式输出** — SSE 实时流式返回，用户可逐 token 看到生成过程
- **多格式文档** — 支持 PDF、DOCX、XLSX、TXT、Markdown
- **智能 Chunk** — Markdown 按标题层级切分，PDF 保留页码，所有 chunk 带序号元数据
- **多语言 Embedding** — BAAI/bge-m3（1024 维），原生支持 dense + sparse 双向量
- **灵活的 LLM 接入** — 通过 LangChain ChatOpenAI 接入任意兼容 OpenAI 格式的 API（OpenAI、Anthropic、DeepSeek、MiMo 等）
- **全链路可观测** — LangSmith 追踪 Agent 执行过程
- **本地部署** — 所有数据自托管，无外部依赖
- **检索评估** — 内置评估系统，可量化检索召回率和精确率

---

## 架构

```
用户提问
    │
    ▼
┌─────────────────────┐
│   FastAPI 服务       │
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
│   混合检索           │
│   Dense + Sparse     │
│   + RRF + Reranking  │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌─────────┐
│ Milvus │  │ Redis   │
│ 向量库 │  │ 会话缓存│
└────┬───┘  └────┬────┘
     │           │
     ▼           ▼
┌────────┐  ┌─────────────┐
│ 文档   │  │ PostgreSQL  │
│ 入库   │  │ 对话历史    │
└────────┘  └─────────────┘
```

### 检索流程

1. 用户查询通过 BGE-M3 编码为 dense + sparse 双向量
2. Milvus 并行执行 dense（AUTOINDEX, IP）+ sparse（SPARSE_INVERTED_INDEX, IP）搜索
3. RRF（Reciprocal Rank Fusion）融合两路结果
4. 交叉编码器（BAAI/bge-reranker-v2-m3）对候选重排序
5. 返回 top-k 结果，包含文本、来源、页码、片段序号、相关性分数

### 入库流程

1. 扫描文档目录，增量处理新文件
2. 按格式选择加载器（PyPDF / Docx2txt / UnstructuredExcel / TextLoader）
3. Markdown 用 `MarkdownHeaderTextSplitter`（保留 H1/H2/H3 层级），其他用 `RecursiveCharacterTextSplitter`
4. BGE-M3 编码为 dense + sparse 双向量
5. 存入 Milvus（含 source、page、chunk_index 元数据）
6. 同步 PostgreSQL 文档状态

---

## 快速开始

### 环境要求

- Python 3.12+
- Docker Desktop（用于 Milvus、PostgreSQL、Redis）

### 1. 启动基础设施

```bash
docker-compose up -d
```

| 服务 | 端口 | 用途 |
|------|------|------|
| Milvus | 19530 | 向量数据库（dense + sparse） |
| Attu | 8080 | Milvus Web 管理界面 |
| PostgreSQL | 5432 | 文档元数据、对话历史 |
| Redis | 6379 | 会话缓存、分布式锁 |
| MinIO | 9001 | Milvus 对象存储后端 |

### 2. 安装依赖

```bash
cd backend
pip install -e .
```

### 3. 配置

```bash
cp .env.example .env
```

编辑 `.env`，配置 LLM API：

```env
LLM_MODEL=gpt-4o
LLM_API_KEY=your_api_key
LLM_API_BASE_URL=https://api.openai.com/v1
```

支持任意 OpenAI 兼容 API，例如 DeepSeek、MiMo 等：

```env
LLM_MODEL=deepseek-chat
LLM_API_KEY=your_key
LLM_API_BASE_URL=https://api.deepseek.com/v1
```

### 4. 启动服务

```bash
uvicorn main:app --reload --port 8000
```

浏览器打开 http://localhost:8000

---

## 使用

### Web UI

- **对话** — 基于知识库的流式问答，支持多轮对话
- **文档** — 上传文档、管理文档、触发入库

### CLI

```bash
python backend/cli.py
```

| 选项 | 功能 |
|------|------|
| 1 | 摄入文档 |
| 2 | 检索文档 |
| 3 | 聊天问答 |
| 4 | 评估（检索质量测试） |
| 5 | 退出 |

### 评估系统

```bash
python backend/cli.py
# 选择 4
```

评估流程：加载 `eval/dataset.json` 测试数据集，对每个问题运行检索，计算召回率和精确率，输出格式化报告。扩充测试数据集：编辑 `eval/dataset.json` 添加更多 Q&A 用例。

---

## API 参考

### 聊天

```
POST /chat
Body: { "message": "...", "session_id": "可选" }
Response: { "response": "...", "session_id": "..." }
```

```
POST /chat/stream
Body: { "message": "...", "session_id": "可选" }
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

### 文档

```
GET /documents
Response: { "documents": [{ "filename": "...", "size_bytes": ..., "status": "...", "chunk_count": ... }] }
```

```
POST /documents/upload
Body: multipart/form-data，字段名 "file"（支持 PDF/DOCX/XLSX/TXT/MD，最大 50MB）
Response: { "filename": "...", "status": "uploaded" }
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
Response: { "status": "idle|running|success|failed", "document_count": ..., "chunk_count": ... }
```

### 系统

```
GET /health
Response: { "status": "ok" }

GET /ready
Response: { "status": "ready|degraded", "checks": { "milvus": ..., "postgresql": ..., "redis": ... } }
```

---

## 配置

所有配置通过 `.env` 文件或环境变量设置，使用 `pydantic-settings` 验证。

| 类别 | 变量 | 默认值 | 说明 |
|------|------|--------|------|
| **LLM** | `LLM_MODEL` | `gpt-4o` | 模型名称 |
| | `LLM_API_KEY` | | API Key |
| | `LLM_API_BASE_URL` | | 自定义 API 地址 |
| **Embedding** | `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` 或 `openai` |
| | `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding 模型 |
| | `EMBEDDING_DEVICE` | `cpu` | `cpu` 或 `cuda` |
| | `EMBEDDING_NORMALIZE` | `true` | 向量归一化 |
| | `HF_TOKEN` | | HuggingFace Token |
| **数据库** | `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL 连接 |
| | `REDIS_URL` | `redis://localhost:6379/0` | Redis 连接 |
| **Milvus** | `MILVUS_HOST` | `localhost` | 主机 |
| | `MILVUS_PORT` | `19530` | 端口 |
| | `MILVUS_COLLECTION` | `ragmate_docs` | Collection 名称 |
| **入库** | `CHUNK_SIZE` | `500` | 文本分块大小 |
| | `CHUNK_OVERLAP` | `50` | 分块重叠 |
| **检索** | `HYBRID_SEARCH_ENABLED` | `true` | 启用混合检索 |
| | `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker 模型 |
| | `RERANK_CANDIDATES` | `20` | rerank 候选池大小 |
| | `FINAL_CONTEXT_K` | `4` | 最终给 LLM 的片段数 |
| | `RERANK_SCORE_THRESHOLD` | `0.06` | 低于此分数的结果丢弃 |
| **LangSmith** | `LANGSMITH_TRACING` | `false` | 启用追踪 |
| | `LANGSMITH_API_KEY` | | LangSmith API Key |

---

## 项目结构

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
    ├── config.py              # 配置（pydantic-settings）
    ├── main.py                # FastAPI 入口 + 所有端点
    ├── agent.py               # Deep Agent（系统提示 + retrieval_tool）
    ├── chat.py                # 聊天编排（同步 + 流式）
    ├── retriever.py           # 混合检索 + Reranking
    ├── ingest.py              # 文档处理 + 向量入库
    ├── streaming_llm.py       # ChatOpenAI 工厂
    ├── model_factory.py       # LLM / Embedding 工厂
    ├── document_service.py    # 文档 CRUD
    ├── database.py            # SQLAlchemy 异步/同步引擎
    ├── models.py              # ORM 模型（Document, ChatHistory）
    ├── redis_client.py        # Redis 会话 / 锁 / 状态
    ├── errors.py              # 类型化错误层级
    ├── cli.py                 # CLI
    ├── eval/                  # 评估系统
    │   ├── dataset.json       # 测试 Q&A 数据集
    │   ├── metrics.py         # 召回率 / 精确率
    │   └── runner.py          # 评估运行器
    └── documents/             # 文档存储目录
```

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | ASGI，托管前端静态文件 |
| 前端 | HTML/CSS/JS | 零依赖原生前端 |
| LLM | LangChain ChatOpenAI | 接入任意 OpenAI 兼容 API |
| Embedding | BAAI/bge-m3 | 1024 维，多语言，dense + sparse 双向量 |
| 向量数据库 | Milvus 2.5 | 混合检索（dense + sparse + RRF） |
| Reranker | BAAI/bge-reranker-v2-m3 | 交叉编码器重排序 |
| Agent | LangGraph (Deep Agents) | 多轮推理 + 工具调用 |
| 追踪 | LangSmith | Agent 执行监控 |
| 缓存 | Redis | 会话状态 + 分布式锁 |
| 存储 | PostgreSQL | 文档元数据 + 对话历史 |

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。
