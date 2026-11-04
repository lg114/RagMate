<div align="center">

# RagMate

**面向文档检索与引用式问答的自托管 RAG 系统。**

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

- **混合检索** — 稠密 + 稀疏检索，在 Milvus 中用 RRF 融合。嵌入默认走 OpenAI 兼容 API；进程内 BGE-M3 需显式打开。
- **重排序与过滤后的上下文** — 默认远程重排，加上动态分数阈值、同源去重、分数断崖截断、句子级上下文压缩。本地交叉编码器需显式打开。
- **标题路径分块** — Markdown 子块带 `H1 > H2 > H3` 路径便于检索；交给 LLM 的仍是更大的父片段。该策略已冻结。
- **接地对话** — 一次检索、一次回答，再附引用。没有留下片段就直接拒答，不调用回答模型。只有 `DEEP_AGENTS_ENABLED=true` 才走 Deep Agents。
- **回答质量信号** — 每次 RAG 回答都带检索置信度（`high` / `medium` / `low`）和真实引用；置信度低时拒答或明确保留；忠实度校验默认开启，标记缺乏依据的声明。
- **增量入库** — 基于 mtime 的变更检测、content-hash 去重，以及"先插新向量再删旧向量"的崩溃安全策略。
- **检索回归门禁** — `ragmate-eval retrieval-gate` 用确定性 BM25 重放 98 题样本。RAGAS 打分代码仍在仓库里，但不是 CI 门禁。
- **自托管设计** — PostgreSQL、Milvus、MinIO 通过 Docker Compose 运行。Redis 可选，只用于多进程共享限流。

---

## 快速开始

### 前置要求

- Python 3.12+
- Docker Desktop，或支持 Compose v2 的 Docker Engine
- 一个 OpenAI 兼容的 LLM 接口及 API Key
- 一个 OpenAI 兼容的嵌入接口，以及一个 `/rerank` 接口。Docker Compose 启动 Milvus、PostgreSQL 和 MinIO，不负责这两个模型服务。默认不启动 Redis。
- 本地 FlagEmbedding 是可选项。这条路径在 CPU 上大约需要 8 GB 内存；知识库较大时建议使用 GPU。

### 1. 启动基础设施

在项目根目录执行：

```bash
docker compose up -d
docker compose ps
```

服务端口：PostgreSQL `5432`、Milvus `19530`、MinIO `9000` / `9001`、Attu `8080`。这条默认路径不含 Redis。开发时应用本身运行在宿主机上。

入库状态、锁、取消标志和失败原因都在 PostgreSQL。会话历史也在 PostgreSQL。`REDIS_URL` 留空即可。限流这时用进程内计数，单进程够用。要在多个 worker 之间共享计数，设置 `REDIS_URL=redis://localhost:6379/0` 并启动可选服务：

```bash
docker compose --profile redis up -d
```

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

`ragmate-eval retrieval-gate` 随上面的安装一起提供。可选的 `.[eval]` 只拉入已搁置的 RAGAS 打分。测试和检索门禁不需要它。

### 3. 配置环境变量

仍在 `backend` 目录时执行：

```bash
cp .env.example .env
```

Windows PowerShell 请使用 `Copy-Item .env.example .env`。

> [!IMPORTANT]
> 以下变量缺失时进程拒绝启动：`DATABASE_URL`、`LLM_API_KEY`、`EMBEDDING_API_BASE_URL`、`RERANKER_API_BASE_URL`。
> 嵌入和重排的 API Key 在服务不校验时可以留空。
> 若要在 API 进程内加载 BGE-M3 和交叉编码器，设置 `EMBEDDING_PROVIDER=huggingface` 与 `RERANKER_PROVIDER=local`（说明见 `.env.example`）。`local` 和 `flagembedding` 与 `huggingface` 相同。

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ragmate
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o

# 可选：使用任意 OpenAI 兼容的对话服务
LLM_API_BASE_URL=https://api.openai.com/v1

# 默认远程提供方时必填。两边可以指向同一个侧车或托管 API。
EMBEDDING_PROVIDER=remote
EMBEDDING_API_BASE_URL=http://127.0.0.1:8081/v1
RERANKER_PROVIDER=api
RERANKER_API_BASE_URL=http://127.0.0.1:8081/v1
```

默认数据库和 Milvus 配置与 Docker Compose 保持一致。没有启用 Redis profile 时 `REDIS_URL` 留空。完整带注释的配置项见 [`backend/.env.example`](backend/.env.example)，其中包含本地 FlagEmbedding 的打开方式。

### 4. 启动 RagMate

回到项目根目录，并确保虚拟环境仍处于激活状态：

```bash
uvicorn backend.app:app --reload --port 8000
```

常规启动不会下载 BGE-M3 或重排序模型，也不会 import torch。第一次入库和第一次提问会请求你配置的嵌入与重排地址。若改成了本地模型，首次启动才会下载权重（数 GB）并在后台预热。

打开 [http://localhost:8000](http://localhost:8000) 使用 Web 界面；API 文档位于 [http://localhost:8000/docs](http://localhost:8000/docs)。

---

## 首次使用

1. 打开 **文档**，上传一个或多个支持的文件。
2. 入库未处理的文件，等页面上的进度条结束。
3. 打开 **对话**，针对已索引文档提问。
4. 点击引用，查看来源文件、页码、入库状态；打开检索调试时还可以看到命中的片段。
5. **设置** 里可以切换外观、同步「检索调试」开关、填写可选的 API 令牌，并查看 PostgreSQL 和 Milvus 是否就绪。只有设置了 `REDIS_URL` 时，列表里才会出现 Redis。

界面是一层薄演示壳：零构建的 `frontend/index.html`，由 FastAPI 挂在 `/`，不需要前端构建。检索调试是开发模式轨迹（`"debug": true`），正常演示保持关闭。检索参数不在页面上，它们是下面冻结的服务端默认值。

支持格式：`.pdf`、`.docx`、`.xlsx`、`.xls`、`.txt`、`.md`。默认单文件上传上限为 50 MB。

---

## 目录结构

```
RagMate/
├── docker-compose.yml      # PostgreSQL、Milvus、MinIO、etcd、Attu；Redis 在 profile "redis"
├── backend/
│   ├── app.py              # 应用工厂：中间件、异常处理、静态文件挂载
│   ├── api/                # HTTP 路由：chat、documents、ingest、health
│   ├── application/        # 用例编排
│   │   └── ingest/         # pipeline（加载 → 切分 → 去重 → 编码）、loaders、db_sync
│   ├── core/               # RAG 内核：retriever、agent、prompts/researcher.md
│   ├── domain/             # ORM 模型、Pydantic schema、错误类型
│   ├── infrastructure/     # config、database、可选 redis、milvus、encoding、model_factory
│   ├── eval/               # retrieval-gate，以及已搁置的 RAGAS 打分
│   └── tests/
└── frontend/               # 零构建界面（对话 / 文档 / 设置），由 FastAPI 挂在 /
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
| 基础设施层 | `infrastructure/` | 外部系统：Milvus、PostgreSQL、可选 Redis、模型、配置 |

```mermaid
flowchart LR
    D[文档] --> P[加载与分块]
    P --> E[嵌入 API]
    E --> V[(Milvus 向量)]
    P --> M[(PostgreSQL 元数据)]
    Q[用户问题] --> R[混合检索 + RRF]
    R --> RR[重排、压缩、过滤]
    RR --> A[Agent 与 LLM]
    A --> O[带引用的回答]
```

**入库链路** — 基于 mtime 的文件变更检测 → 按格式选择加载器（PyPDF、Docx2txt、TextLoader、Excel 转 Markdown）→ 标题感知切分（可选标题路径前缀）→ 基于 content-hash 对 Milvus 去重 → 经嵌入 API 做稠密 + 稀疏编码（只有显式打开时才在进程内用 FlagEmbedding）→ 写入新向量 → 删除该来源的旧向量 → 同步 `documents` 表。

**查询链路** — 可选的查询上下文化（把追问改写为独立 query）→ 稠密 + 稀疏 ANN 检索并用 RRF 融合 → 重排序（默认 `POST /rerank`；本地交叉编码器才把 logits 经 sigmoid 转为概率）→ 句子级上下文压缩 → 动态阈值、同源去重、分数断崖截断 → 一次回答 → 引用规整 → 忠实度校验（默认开启）。RAG 回答总会带上检索来源和置信度：一个片段都没留下时改为拒答；分数低但仍有片段时，在答案前加上明确的保留说明。

简单的寒暄类问题会绕过 Agent，直接路由到 LLM。

---

## API 参考

| 方法 | 端点 | 用途 |
|---|---|---|
| POST | `/chat` | 非流式问答 |
| POST | `/chat/stream` | SSE 流式问答 |
| GET | `/chat/sessions` | 列出最近 50 个会话 |
| GET | `/chat/sessions/{session_id}` | 获取某会话的完整消息历史 |
| DELETE | `/chat/sessions/{session_id}` | 删除会话（PostgreSQL） |
| GET | `/documents` | 列出文档及入库状态 |
| POST | `/documents/upload` | 上传单个文件（上限 50 MB） |
| DELETE | `/documents/{filename}` | 删除文档、磁盘文件及其向量 |
| POST | `/ingest` | 启动入库（可指定文件名） |
| POST | `/ingest/cancel` | 请求持锁 worker 停止入库 |
| GET | `/ingest/status` | 查询入库状态与进度（进程重启后仍然可读） |
| GET | `/health` | 存活探针 |
| GET | `/ready` | 就绪探针（PostgreSQL、Milvus；仅当设置了 `REDIS_URL` 时才检查 Redis） |

### 对话请求体

```jsonc
{
  "message": "什么是 RAG？",   // 必填，最长 10 000 字符
  "session_id": "...",        // 可选 UUID；不传则新建
  "replace_last": false,      // 重试 / 重新生成：先丢弃最后一轮
  "debug": false               // 为 true 时返回检索候选、分数，以及被过滤丢掉的原因
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
       "response": "落地后的回答…",
       "confidence": {"level": "high", "score": 0.87, "chunks": 6},
       "citations": [{"source": "guide.pdf", "page": 3}],
       "grounding": "grounded",
       "unsupported_claims": [{"claim": "...", "supported": false}],
       "retrieval_debug": [{"query": "...", "chunks": [{"rank": 1, "score": 0.91, "kept": true, "dropped_by": null, "preview": "..."}]}]}

data: {"error": "..."}                               // 终止性错误
```

空闲连接每 15 秒会收到一个 `:heartbeat` 注释行。`confidence`、`citations`、`grounding` 和 `unsupported_claims` 只出现在 `done` 事件中。该事件里的 `response` 是落地后的文本（拒答、低置信保留、以及真实来源行），客户端应用它替换已经流出的 token。检索没有留下片段时，流里不会先吐 token，拒答只出现在 `done`。`unsupported_claims` 在忠实度校验发现无依据声明时出现。`grounding` 取值为 `refused`、`hedged`、`grounded`；没有发生检索时省略。

非流式的 `POST /chat` 返回同样的字段。未发生检索时（例如寒暄类问题直连 LLM）`confidence` 与 `citations` 为 `null`。

`retrieval_debug` 是开发模式轨迹，只有请求带 `"debug": true` 时才返回。每次检索一条记录，里面是重排后的候选：分数、截断后的 preview，以及没被留下时的 `dropped_by`（`threshold`、`per_source`、`score_gap`、`top_k`）。输入框和 **设置** 页共用「检索调试」开关，选择会记在浏览器里。正常演示关掉；preview 是文档原文片段。

---

## 配置

配置从 `backend/.env` 读取（见 [`backend/.env.example`](backend/.env.example)）。启动时即完成校验，非法组合会快速失败，而不是等到查询时才暴露。

### 核心配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DATABASE_URL` | **必填** | 元数据、聊天历史和入库控制 |
| `LLM_API_KEY` | **必填** | LLM 服务凭证 |
| `EMBEDDING_API_BASE_URL` | 默认 **必填** | OpenAI 兼容的 `/embeddings`。`EMBEDDING_PROVIDER` 为 `huggingface`、`local` 或 `flagembedding` 时不需要 |
| `RERANKER_API_BASE_URL` | 默认 **必填** | `/rerank` 地址。`RERANKER_PROVIDER=local` 时不需要 |
| `LLM_MODEL` | `gpt-4o` | 服务商提供的模型名称 |
| `LLM_API_BASE_URL` | 空 | 自定义 OpenAI 兼容端点 |
| `LLM_TEMPERATURE` | `0.01` | 保持低温度以提升事实性 |
| `REDIS_URL` | 空 | 可选。只在多个 worker 共享限流时填写。入库和对话不用它 |
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | 向量数据库地址 |
| `MILVUS_COLLECTION` | `ragmate_docs` | 集合名称 |
| `DOCUMENTS_DIR` | `backend/documents` | 上传目录 |
| `CORS_ORIGINS` | `http://localhost:8000,...` | 逗号分隔；生产环境务必收紧 |

### 检索（已冻结）

只认一套经过验证的检索配置：稠密 + 稀疏混合检索，RRF 融合，再重排。fused 阶段的数字钉在 [`eval/reports/baseline/retrieval-baseline-98q-titlepath-rerun.json`](eval/reports/baseline/retrieval-baseline-98q-titlepath-rerun.json)。除非对那 98 题重跑 `ragmate-eval retrieval`，并改掉 [`eval/README.md`](eval/README.md) 和 `backend/tests/test_retrieval_gate.py` 里的门槛，否则保持下面的默认值。

| 部分 | 冻结值 |
|---|---|
| 稠密 + 稀疏混合 | `HYBRID_SEARCH_ENABLED=true` |
| 融合 | RRF，`RRF_K=60`（`FUSION_MODE=rrf`）。加权融合不是默认 |
| 分块 | `CHUNKING_STRATEGY=title_path`，`CHUNK_SIZE=1000`，重叠 `200`，父块 `2500` / `300` |
| 重排后过滤 | `RERANK_CANDIDATES=30`，`RERANK_SCORE_THRESHOLD=0.3`，`FINAL_CONTEXT_K=15` |
| 查询扩展 | 关闭（`MULTI_QUERY_ENABLED=false`） |

`FUSION_MODE=weighted`、`FUSION_DENSE_WEIGHT`、`MULTI_QUERY_COUNT`、`CHUNKING_STRATEGY=parent_child`，以及折叠区里的过滤比例，都只是环境变量。它们和基线一起冻结。不重跑确定性基线就改其中一项，不会变成受支持的配置。不要新增融合臂、不要把多查询改成默认开启、不要再加切块策略。

那份 98 题报告是用进程内 BGE-M3 和本地交叉编码器采的。冻结的是上面这套算法。通过 HTTP 调用这些模型（`EMBEDDING_PROVIDER=remote`、`RERANKER_PROVIDER=api`）是部署默认，不改变融合或分块。

`DEEP_AGENTS_ENABLED` 保持 `false`（一次检索、一次生成、再接地）。`FAITHFULNESS_CHECK` 保持 `true`。

<details>
<summary><strong>已冻结的环境变量</strong> — 列出来是为了部署时能看见，不是为了就地调参</summary>

**分块** — `CHUNK_SIZE` (`1000`)、`CHUNK_OVERLAP` (`200`)、`CHUNK_SIZE_PDF` (`600`)、`CHUNK_OVERLAP_PDF` (`100`)、`CHUNK_SIZE_DOCX` (`800`)、`CHUNK_OVERLAP_DOCX` (`150`)、`CHUNK_SIZE_TXT` (`1000`)、`CHUNK_OVERLAP_TXT` (`200`)、`CHUNK_SIZE_TABLE` (`1500`)、`CHUNK_SIZE_PARENT` (`2500`)、`CHUNK_OVERLAP_PARENT` (`300`)、`CHUNKING_STRATEGY`（`title_path`）。`parent_child` 切块边界相同，只是标题路径不进检索正文。它不是第二套受支持的策略。

重叠长度必须小于对应分块大小，否则启动时会抛出校验错误。

**过滤与压缩** — `DYNAMIC_THRESHOLD_RATIO` (`0.5`)、`MAX_PER_SOURCE` (`4`)、`MIN_PER_SOURCE` (`2`)、`SCORE_GAP_THRESHOLD` (`0.15`)、`CONTEXTUAL_COMPRESSION` (`true`)、`QUERY_CONTEXTUALIZE` (`true`，把追问改写成一条查询，不是多查询)、`QUERY_ROUTING_ENABLED` (`true`)、`HIGH_SCORE_RATIO` (`0.6`)、`SOURCE_DOMINANCE_THRESHOLD` (`0.9`)、`SOURCE_DOMINANCE_BOOST` (`1.5`)、`DROP_RATIO_SEARCH` (`0.2`)、`COMPRESSION_SCORE_THRESHOLD` (`0.4`)、`COMPRESSION_MIN_CHARS` (`300`)。

**嵌入** — `EMBEDDING_PROVIDER`（`remote`）、`EMBEDDING_MODEL`（`BAAI/bge-m3`）、`EMBEDDING_NORMALIZE`（`true`）、`HF_TOKEN`、`EMBEDDING_API_KEY`（可选）、`EMBEDDING_API_BASE_URL`（`remote` / `openai` 时必填）、`EMBEDDING_API_TIMEOUT`（`60`）。

`remote`（以及 `openai`）请求 `POST {EMBEDDING_API_BASE_URL}/embeddings`。启动时不 import torch，也不预热本地 BGE。`RERANKER_PROVIDER` 默认为 `api`，请求 `POST {RERANKER_API_BASE_URL}/rerank`（`RERANKER_API_KEY` 可选，`RERANKER_API_TIMEOUT` 为 `30`）。服务如果在每条结果里带了 `sparse`（或 `lexical_weights`）就用它；否则用稳定的词哈希补稀疏向量。这个哈希不是 BGE-M3 的 lexical weights，切换嵌入方式后必须重新入库。

**本地可选** — `EMBEDDING_PROVIDER=huggingface`（`local` 与 `flagembedding` 是同一条路径）会在 API 进程里用 FlagEmbedding 加载 BGE-M3，`EMBEDDING_DEVICE`（`cpu` 或 `cuda`）只在这时生效。`RERANKER_PROVIDER=local` 会在进程内加载交叉编码器。打开其中任意一项都会在首次使用时 import torch，并在启动时预热对应权重。写法见 [`backend/.env.example`](backend/.env.example) 里的注释。

CI 不需要 GPU，也不需要真的嵌入服务。测试 mock 了 HTTP 客户端，检索门禁走的是 BM25。工作流仍会导出占位 URL，因为配置在 import 时就会校验。

**Agent** — `AGENT_RECURSION_LIMIT` (`30`)。

**链路追踪** — `LANGSMITH_TRACING` (`false`)、`LANGSMITH_API_KEY`、`LANGSMITH_PROJECT` (`default`)、`LANGSMITH_ENDPOINT`。

</details>

如需使用 Ollama、LM Studio 等本地模型，将 `LLM_API_BASE_URL` 指向其 OpenAI 兼容接口地址。

---

## 评测

回归门禁是 `ragmate-eval retrieval-gate`，随基础安装提供。在 `backend/` 下执行：

```bash
ragmate-eval retrieval-gate
```

它对已入库的 98 题样本做确定性 BM25 重放（不连 Milvus、不加载 BGE、不用 GPU），Recall@k 或 nDCG 低于 `eval/reports/baseline/retrieval-baseline-98q-titlepath-rerun.json` fused 阶段钉住的门槛时失败。具体数字、0.015 的余量，以及有意更换基线的步骤见 [`eval/README.md`](eval/README.md)。GitHub Actions 在 pytest 之后运行这条命令。

RAGAS（`generate`、`evaluate`、`rescore`、`patch-raw`，以及 `ragmate-eval interactive`）仍留在仓库里。只有还想用这个 judge 时才安装 `pip install -e ".[eval]"`。它已搁置：不是 CI 门禁，也不用来调检索。judge 分数会跟着模型漂移。Python 3.14 上的 RAGAS 问题不在范围内。不带参数的 `ragmate-eval` 会打印这条门禁，不再进入 RAGAS 菜单。

---

## 开发

```bash
cd backend
pip install -e ".[test]"
pytest -v
```

测试位于 `backend/tests/`，目前覆盖检索过滤与去重逻辑；`conftest.py` 提供了 `override_settings` fixture，用于临时改写配置。没有 `backend/.env` 时，它会补上占位的 `DATABASE_URL`、`LLM_API_KEY` 和远程嵌入/重排 URL，因此 `pytest` 不需要 GPU 或正在运行的模型服务。仓库里如果已有 `.env`，则原样使用。`ragmate-eval retrieval-gate` 不经过 `conftest`；请像 CI 一样导出同样的占位 URL，或保留一份能通过校验的 `.env`。

> [!NOTE]
> `ruff` 与 `mypy` 尚未在 `pyproject.toml` 中配置。代码遵循 PEP 8、4 空格缩进，公共函数使用中文 docstring。

所有阻塞调用（模型推理、Milvus I/O、入库）都通过 `asyncio.to_thread` 丢到线程中执行，请勿在协程里直接加入阻塞逻辑。

---

## 数据、备份与安全

Docker 数据卷写入 `volumes/`，包含 PostgreSQL、Milvus 和 MinIO 的数据（若启用了 Redis profile，还有 Redis）。为保证备份一致性，请先停止服务，再复制该目录，完成后重新启动服务。

如果要从 localhost 暴露到外部网络：

- 将 `CORS_ORIGINS` 设置为真实来源。
- 替换 `docker-compose.yml` 中的默认基础设施凭证。
- 在 API 前面增加一层认证。
- 检查上传与限流设置 —— 当前对话、上传、删除、入库均为每 IP 每分钟 30 次。未设置 `REDIS_URL` 时，这个计数只在单个进程内。

RagMate 提供了请求校验：文件名白名单、magic bytes 校验、路径穿越拒绝、Milvus 过滤表达式注入防护、50 MB 上传上限，但它本身不构成完整的公网安全边界。

---

## 已知约束

- **同一时间只有一个 worker 在入库。** 状态、锁、进度、取消标志和失败原因都在 Postgres 的一行（`ingest_control`）里，任意 uvicorn worker 都能读进度并发起取消。只有持锁的 worker 写进度和终态，并按 `INGEST_HEARTBEAT_INTERVAL`（默认 5 秒）刷新心跳。该进程退出后，下一次状态查询会在 `INGEST_HEARTBEAT_STALE`（默认 30 秒）后把任务标成 `interrupted` 并释放锁。这一行没有 TTL，失败或中断的结果在重启之后还在。
- **无认证与多租户。** 任何能访问 API 的人都可看到全部会话与文档。
- **Schema 变更需手动重建。** Milvus 集合 schema 不匹配时启动会抛出校验错误而不是自动迁移，需要删除集合后重建。
- **没有迁移工具。** 表结构变更通过 `create_all` 加上 `infrastructure/database.py` 中一份手写的补列清单完成。
- **进程内嵌入和重排是可选项，CPU 推理较慢。** 默认请求远程嵌入和 `/rerank`，不加载 torch。只有希望本进程加载 BGE-M3 和交叉编码器时，才设置 `EMBEDDING_PROVIDER=huggingface` 和 `RERANKER_PROVIDER=local`。

---

## 常见问题

| 现象 | 排查方向 |
|---|---|
| `/ready` 返回 `degraded` | `docker compose ps`，再执行 `docker compose logs <service>` |
| 启动报 "Required env vars not set" | 检查 `backend/.env` 中的 `DATABASE_URL`、`LLM_API_KEY`，以及（未改成本地模型时）`EMBEDDING_API_BASE_URL` 与 `RERANKER_API_BASE_URL` |
| 本地模型下载或 CPU 推理很慢 | 保持默认的远程提供方，或在本地模式下设置 `EMBEDDING_DEVICE=cuda` |
| 对话没有检索到内容 | 确认入库已完成，且 `MILVUS_HOST`、`DATABASE_URL` 指向正在运行的服务 |
| 端口冲突 | 修改 `docker-compose.yml` 的宿主机端口，并同步更新 `.env` |
| 入库卡在 `running` | 状态由 Postgres 心跳对账。worker 退出后默认 30 秒（`INGEST_HEARTBEAT_STALE`）会变成 `interrupted`；`POST /ingest/cancel` 可请求仍在运行的 worker 停止 |

---

## 许可证

MIT License，详见 [LICENSE](LICENSE)。

---

## 致谢

[FastAPI](https://fastapi.tiangolo.com/) · [LangChain](https://python.langchain.com/) · [Deep Agents](https://github.com/langchain-ai/deepagents) · [Milvus](https://milvus.io/) · [BGE](https://github.com/FlagOpen/FlagEmbedding)
