<div align="center">

# RagMate

**面向文档检索与引用式问答的自托管 RAG 知识管理系统。**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Milvus](https://img.shields.io/badge/Milvus-2.5-2A6FB0)](https://milvus.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-4C566A)](LICENSE)

[English](README.md) · [中文](README_zh.md)

</div>

---

RagMate 是一个自托管 RAG 应用，可将 PDF、DOCX、XLSX、TXT 和 Markdown 文件构建为可搜索的知识库。它结合混合向量检索、交叉编码器重排序、Agent 推理、流式响应和可选的忠实度校验，同时让数据与基础设施完全掌握在你自己手中。

## 目录

- [项目亮点](#项目亮点)
- [快速开始](#快速开始)
- [首次使用](#首次使用)
- [目录结构](#目录结构)
- [系统架构](#系统架构)
- [API 参考](#api-参考)
- [配置](#配置)
- [评测](#评测)
- [开发](#开发)
- [数据、备份与安全](#数据备份与安全)
- [已知约束](#已知约束)
- [常见问题](#常见问题)
- [许可证](#许可证)
- [致谢](#致谢)

---

## 项目亮点

- **混合检索** — BGE-M3 稠密 + 稀疏检索，在 Milvus 中用 RRF 融合。
- **重排序与过滤后的上下文** — 交叉编码器重排序、动态分数阈值、同源去重、分数断崖截断、句子级上下文压缩。
- **父子分块** — 用小片段检索，把更大的父片段交给 LLM。
- **Agent 对话** — Deep Agents 工作流，支持多轮会话、SSE 流式输出，检索以工具形式暴露给 Agent。
- **回答质量信号** — 检索置信度（`high` / `medium` / `low`），以及可选的忠实度校验，标记缺乏依据的声明。
- **增量入库** — 基于 mtime 的变更检测、content-hash 去重，以及"先插新向量再删旧向量"的崩溃安全策略。
- **内置评测** — 可选 RAGAS CLI，支持测试集生成、打分和 CI 质量门禁。
- **自托管设计** — PostgreSQL、Redis、Milvus、MinIO 全部通过 Docker Compose 运行。

---

## 快速开始

### 前置要求

- Python 3.12+
- Docker Desktop，或支持 Compose v2 的 Docker Engine
- 一个 OpenAI 兼容的 LLM 接口及 API Key
- CPU 模式建议至少 8 GB 内存；较大知识库建议使用 GPU

### 1. 启动基础设施

在项目根目录执行：

```bash
docker compose up -d
docker compose ps
```

服务端口：PostgreSQL `5432`、Redis `6379`、Milvus `19530`、MinIO `9000` / `9001`、Attu `8080`。开发时应用本身运行在宿主机上。

停止服务但保留数据：

```bash
docker compose stop
```

### 2. 安装后端依赖

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

如需使用评测 CLI：

```bash
pip install -e ".[eval]"
```

### 3. 配置环境变量

仍在 `backend` 目录时执行：

```bash
cp .env.example .env
```

Windows PowerShell 请使用 `Copy-Item .env.example .env`。

> [!IMPORTANT]
> 两个变量为必填项，缺失时应用拒绝启动：`DATABASE_URL` 和 `LLM_API_KEY`。

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ragmate
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o

# 可选：使用任意 OpenAI 兼容服务
LLM_API_BASE_URL=https://api.openai.com/v1
```

默认数据库、Redis 和 Milvus 配置与 Docker Compose 保持一致。完整带注释的配置项见 [`backend/.env.example`](backend/.env.example)。

### 4. 启动 RagMate

回到项目根目录，并确保虚拟环境仍处于激活状态：

```bash
uvicorn backend.app:app --reload --port 8000
```

首次启动会下载 BGE-M3 与重排序模型（数 GB），并在后台预热，因此第一次请求需要多等一会儿。

打开 [http://localhost:8000](http://localhost:8000) 使用 Web 界面；API 文档位于 [http://localhost:8000/docs](http://localhost:8000/docs)。

---

## 首次使用

1. 打开 **文档** 标签页，上传一个或多个支持的文件。
2. 触发入库，等待状态变为 `success`。
3. 打开 **对话**，针对已索引文档提问。
4. 通过返回的引用查看原始内容片段。

支持格式：`.pdf`、`.docx`、`.xlsx`、`.xls`、`.txt`、`.md`。默认单文件上传上限为 50 MB。

---

## 目录结构

```
RagMate/
├── docker-compose.yml      # PostgreSQL、Redis、Milvus、MinIO、etcd、Attu
├── backend/
│   ├── app.py              # 应用工厂：中间件、异常处理、静态文件挂载
│   ├── api/                # HTTP 路由：chat、documents、ingest、health
│   ├── application/        # 用例编排
│   │   └── ingest/         # pipeline（加载 → 切分 → 去重 → 编码）、loaders、db_sync
│   ├── core/               # RAG 内核：retriever、agent、prompts/researcher.md
│   ├── domain/             # ORM 模型、Pydantic schema、错误类型
│   ├── infrastructure/     # config、database、redis、milvus、encoding、model_factory
│   ├── eval/               # RAGAS 评测 CLI（可选依赖）
│   └── tests/
└── frontend/               # 零构建原生 HTML/CSS/JS，由 FastAPI 托管
```

---

## 系统架构

后端严格分层，每层只向下调用：

| 层 | 目录 | 职责 |
|---|---|---|
| 接口层 | `api/` | 只处理 HTTP：路由、请求校验、限流 |
| 应用层 | `application/` | 用例编排：对话流程、文档 CRUD、入库生命周期 |
| 核心层 | `core/` | RAG 领域逻辑：检索管线、Agent、系统提示词 |
| 领域层 | `domain/` | ORM 模型、Pydantic schema、`AppError` 体系 |
| 基础设施层 | `infrastructure/` | 外部系统：Milvus、PostgreSQL、Redis、模型、配置 |

```mermaid
flowchart LR
    D[文档] --> P[加载与分块]
    P --> E[BGE-M3 编码]
    E --> V[(Milvus 向量)]
    P --> M[(PostgreSQL 元数据)]
    Q[用户问题] --> R[混合检索 + RRF]
    R --> RR[重排、压缩、过滤]
    RR --> A[Agent 与 LLM]
    A --> O[带引用的回答]
```

**入库链路** — 基于 mtime 的文件变更检测 → 按格式选择加载器（PyPDF、Docx2txt、TextLoader、Excel 转 Markdown）→ 标题感知切分与父子分块 → 基于 content-hash 对 Milvus 去重 → BGE-M3 稠密 + 稀疏编码 → 写入新向量 → 删除该来源的旧向量 → 同步 `documents` 表。

**查询链路** — 可选的查询上下文化（把追问改写为独立 query）→ 稠密 + 稀疏 ANN 检索并用 RRF 融合 → 交叉编码器重排序（logits 经 sigmoid 转为概率）→ 句子级上下文压缩 → 动态阈值、同源去重、分数断崖截断 → Agent 与 LLM → 引用规整 → 可选忠实度校验。

简单的寒暄类问题会绕过 Agent，直接路由到 LLM。

---

## API 参考

| 方法 | 端点 | 用途 |
|---|---|---|
| POST | `/chat` | 非流式问答 |
| POST | `/chat/stream` | SSE 流式问答 |
| GET | `/chat/sessions` | 列出最近 50 个会话 |
| GET | `/chat/sessions/{session_id}` | 获取某会话的完整消息历史 |
| DELETE | `/chat/sessions/{session_id}` | 删除会话（Redis 缓存 + PostgreSQL） |
| GET | `/documents` | 列出文档及入库状态 |
| POST | `/documents/upload` | 上传单个文件（上限 50 MB） |
| DELETE | `/documents/{filename}` | 删除文档、磁盘文件及其向量 |
| POST | `/ingest` | 启动入库（可指定文件名） |
| GET | `/ingest/status` | 查询入库状态与进度 |
| GET | `/health` | 存活探针 |
| GET | `/ready` | 就绪探针（PostgreSQL、Redis、Milvus） |

### 对话请求体

```jsonc
{
  "message": "什么是 RAG？",   // 必填，最长 10 000 字符
  "session_id": "...",        // 可选 UUID；不传则新建
  "replace_last": false       // 重试 / 重新生成：先丢弃最后一轮
}
```

示例：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"什么是 RAG？"}'
```

### 流式协议

`/chat/stream` 返回 Server-Sent Events，每个 `data:` 行是一个 JSON 对象：

```jsonc
data: {"token": "检索增强生成..."}                     // 增量文本

data: {"done": true, "session_id": "...",
       "confidence": {"level": "high", "score": 0.87, "chunks": 6},
       "unsupported_claims": [{"claim": "...", "supported": false}]}

data: {"error": "..."}                               // 终止性错误
```

空闲连接每 15 秒会收到一个 `:heartbeat` 注释行。`confidence` 和 `unsupported_claims` 只出现在 `done` 事件中；`unsupported_claims` 仅在开启 `FAITHFULNESS_CHECK` 且存在未通过校验的声明时出现。

非流式的 `POST /chat` 会在 JSON 响应体中返回同样的 `confidence` 与 `unsupported_claims`；未发生检索时（例如寒暄类问题直连 LLM）这两个字段为 `null`。

---

## 配置

配置从 `backend/.env` 读取（见 [`backend/.env.example`](backend/.env.example)）。启动时即完成校验，非法组合会快速失败，而不是等到查询时才暴露。

### 核心配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DATABASE_URL` | **必填** | 元数据和聊天历史 |
| `LLM_API_KEY` | **必填** | LLM 服务凭证 |
| `LLM_MODEL` | `gpt-4o` | 服务商提供的模型名称 |
| `LLM_API_BASE_URL` | 空 | 自定义 OpenAI 兼容端点 |
| `LLM_TEMPERATURE` | `0.01` | 保持低温度以提升事实性 |
| `REDIS_URL` | `redis://localhost:6379/0` | 会话、入库锁、限流 |
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | 向量数据库地址 |
| `MILVUS_COLLECTION` | `ragmate_docs` | 集合名称 |
| `DOCUMENTS_DIR` | `backend/documents` | 上传目录 |
| `CORS_ORIGINS` | `http://localhost:8000,...` | 逗号分隔；生产环境务必收紧 |

### 检索与重排序

| 变量 | 默认值 | 说明 |
|---|---|---|
| `HYBRID_SEARCH_ENABLED` | `true` | 稠密 + 稀疏；关闭则只用稠密检索 |
| `RERANK_CANDIDATES` | `30` | 进入交叉编码器的候选数 |
| `RERANK_SCORE_THRESHOLD` | `0.3` | 片段保留的 sigmoid 概率下限 |
| `FINAL_CONTEXT_K` | `15` | 发送给 LLM 的片段硬上限 |
| `DYNAMIC_THRESHOLD_RATIO` | `0.5` | 阈值取 `最高分 × 该比例` 与固定阈值的较大者 |
| `MAX_PER_SOURCE` / `MIN_PER_SOURCE` | `4` / `2` | 单来源片段数量上下限 |
| `SCORE_GAP_THRESHOLD` | `0.15` | 遇到分数断崖即停止截断 |
| `CONTEXTUAL_COMPRESSION` | `true` | 只保留片段内的相关句子 |
| `QUERY_CONTEXTUALIZE` | `true` | 将追问改写为独立 query |
| `QUERY_ROUTING_ENABLED` | `true` | 寒暄类问题跳过 Agent |
| `FAITHFULNESS_CHECK` | `false` | 每次回答额外增加一次 LLM 调用 |

<details>
<summary><strong>完整配置项</strong> — 分块、嵌入、Agent 与链路追踪</summary>

**分块** — `CHUNK_SIZE` (`1000`)、`CHUNK_OVERLAP` (`200`)、`CHUNK_SIZE_PDF` (`600`)、`CHUNK_OVERLAP_PDF` (`100`)、`CHUNK_SIZE_DOCX` (`800`)、`CHUNK_OVERLAP_DOCX` (`150`)、`CHUNK_SIZE_TXT` (`1000`)、`CHUNK_OVERLAP_TXT` (`200`)、`CHUNK_SIZE_TABLE` (`1500`)、`CHUNK_SIZE_PARENT` (`2500`)、`CHUNK_OVERLAP_PARENT` (`300`)。

重叠长度必须小于对应分块大小，否则启动时会抛出校验错误。

**重排序调参** — `HIGH_SCORE_RATIO` (`0.6`)、`SOURCE_DOMINANCE_THRESHOLD` (`0.9`)、`SOURCE_DOMINANCE_BOOST` (`1.5`)、`DROP_RATIO_SEARCH` (`0.2`)、`COMPRESSION_SCORE_THRESHOLD` (`0.4`)、`COMPRESSION_MIN_CHARS` (`300`)。

**嵌入** — `EMBEDDING_PROVIDER` (`huggingface`)、`EMBEDDING_MODEL` (`BAAI/bge-m3`)、`EMBEDDING_NORMALIZE` (`true`)、`HF_TOKEN`、`EMBEDDING_API_KEY`、`EMBEDDING_API_BASE_URL`。

`EMBEDDING_DEVICE` (`cpu`) 只作用于评测 CLI 使用的 HuggingFace embeddings 路径；入库与检索链路通过 FlagEmbedding 加载 BGE-M3，目前不读取该设置。

**Agent** — `AGENT_RECURSION_LIMIT` (`30`)。

**链路追踪** — `LANGSMITH_TRACING` (`false`)、`LANGSMITH_API_KEY`、`LANGSMITH_PROJECT` (`default`)、`LANGSMITH_ENDPOINT`。

</details>

如需使用 Ollama、LM Studio 等本地模型，将 `LLM_API_BASE_URL` 指向其 OpenAI 兼容接口地址。

---

## 评测

安装 `pip install -e ".[eval]"` 后，运行交互式菜单：

```bash
cd backend
ragmate-eval
```

或在脚本中使用：

```bash
ragmate-eval generate --size 50 --output ../eval/testsets/testset.json

ragmate-eval evaluate --testset ../eval/testsets/testset.json \
  --report ../eval/reports/report.json --threshold 0.75
```

评测指标包括忠实度、答案相关性、上下文精确率、上下文召回率和事实正确性。配合 `--threshold`，总分低于阈值时命令以非零状态退出，适合作为 CI 门禁。测试集与报告输出在仓库根目录的 `eval/` 下，已被 git 忽略。

---

## 开发

```bash
cd backend
pip install -e ".[test]"
pytest -v
```

测试位于 `backend/tests/`，目前覆盖检索过滤与去重逻辑；`conftest.py` 提供了 `override_settings` fixture，用于临时改写配置。

> [!NOTE]
> `ruff` 与 `mypy` 尚未在 `pyproject.toml` 中配置。代码遵循 PEP 8、4 空格缩进，公共函数使用中文 docstring。

所有阻塞调用（模型推理、Milvus I/O、入库）都通过 `asyncio.to_thread` 丢到线程中执行，请勿在协程里直接加入阻塞逻辑。

---

## 数据、备份与安全

Docker 数据卷写入 `volumes/`，包含 PostgreSQL、Redis、Milvus 和 MinIO 的数据。为保证备份一致性，请先停止服务，再复制该目录，完成后重新启动服务。

如果要从 localhost 暴露到外部网络：

- 将 `CORS_ORIGINS` 设置为真实来源。
- 替换 `docker-compose.yml` 中的默认基础设施凭证。
- 在 API 前面增加一层认证。
- 检查上传与限流设置 —— 当前对话、上传、删除、入库均为每 IP 每分钟 30 次。

RagMate 提供了请求校验：文件名白名单、magic bytes 校验、路径穿越拒绝、Milvus 过滤表达式注入防护、50 MB 上传上限，但它本身不构成完整的公网安全边界。

---

## 已知约束

- **假设单进程运行。** 入库任务句柄与锁 token 存放在进程内存中，多 worker 部署（例如 `uvicorn --workers 4`）会导致入库状态与取消功能失效。请使用单 worker。
- **无认证与多租户。** 任何能访问 API 的人都可看到全部会话与文档。
- **Schema 变更需手动重建。** Milvus 集合 schema 不匹配时启动会抛出校验错误而不是自动迁移，需要删除集合后重建。
- **没有迁移工具。** 表结构变更通过 `create_all` 加上 `infrastructure/database.py` 中一份手写的补列清单完成。
- **CPU 推理较慢。** 嵌入与重排序在本地执行，纯 CPU 环境下单次查询需要数秒。

---

## 常见问题

| 现象 | 排查方向 |
|---|---|
| `/ready` 返回 `degraded` | `docker compose ps`，再执行 `docker compose logs <service>` |
| 启动报 "Required env vars not set" | 检查 `backend/.env` 中的 `DATABASE_URL` 与 `LLM_API_KEY` |
| 模型下载或 CPU 推理很慢 | 使用 CUDA，或改用托管嵌入服务 |
| 对话没有检索到内容 | 确认入库已完成，且 `MILVUS_HOST`、`DATABASE_URL`、`REDIS_URL` 指向正在运行的服务 |
| 端口冲突 | 修改 `docker-compose.yml` 的宿主机端口，并同步更新 `.env` |
| 入库卡在 `running` | 上次崩溃可能仍持有锁；锁 10 分钟后自动过期，也可直接重启应用 |

---

## 许可证

MIT License，详见 [LICENSE](LICENSE)。

---

## 致谢

[FastAPI](https://fastapi.tiangolo.com/) · [LangChain](https://python.langchain.com/) · [Deep Agents](https://github.com/langchain-ai/deepagents) · [Milvus](https://milvus.io/) · [BGE](https://github.com/FlagOpen/FlagEmbedding) · [RAGAS](https://docs.ragas.io/)
