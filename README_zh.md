<div align="center">

# RagMate

**轻量、高效的自托管 RAG 系统，提供结构化文档检索、交叉编码重排序与真实引用问答。**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.8.7-blue)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-4C566A)](LICENSE)

[English](README.md) · [中文](README_zh.md)

</div>

---

RagMate 能够将你的私有文档转化为支持精确引用核对的交互式知识库。它融合了稠密向量与 BM25 稀疏混合检索、互惠排序融合（RRF）、交叉编码器重排序与流式引用生成，全流程保持 100% 自托管，数据与基础设施完全由你掌控。

---

## 核心特性

- **工业级混合检索** — 稠密 HNSW 向量检索与词法 BM25 检索并行，通过互惠排序融合（RRF，$k=60$）无缝融合。
- **交叉编码器重排与上下文压缩** — 默认使用远程 `/rerank` 接口（亦支持进程内 CrossEncoder），辅以动态分数阈值过滤、单源截断、分数断崖过滤与句子级上下文压缩。
- **结构化标题路径分块** — Markdown 自动携带层级路径（`H1 > H2 > H3`）参与检索，最终将更连贯的父级大片段送入 LLM 进行总结。
- **真实引用与无依据拒答** — 回答直接附带真实文件名与页码跳转，输出置信度评分（`high` / `medium` / `low`）；无有效依据时直接拒答，坚决杜绝模型幻觉。
- **标准容器化一键部署** — 仅需一条 `docker compose up -d` 即可拉起带 pgvector 的 PostgreSQL 15 与前端应用，表结构和插件自动初始化。
- **本机免密与统一安全访问** — 本机访问（`127.0.0.1`）开箱即用无需输入密码；对外暴露时只需设置一个实例密码，全站统一受 HttpOnly Cookie 会话保护。
- **零构建极简双语界面** — 单页极简 Web 界面（中英文无缝切换），支持流式 SSE 问答、移动端抽屉导航、文档解析进度监控与模型连接管理。
- **确定性检索回归门禁** — 内置 `ragmate-eval retrieval-gate`，基于 130 题标注样本与确定性 BM25 基准线检验 Recall@k 与 nDCG，不依赖昂贵且不可控的 LLM Judge。

---

## 快速开始

标准自托管部署仅需安装 **Docker** 与 **Compose v2**，宿主机无需安装任何 Python 环境或编译工具。

```bash
# 1. （可选）复制配置文件模板
cp .env.example .env    # Windows 请使用: copy .env.example .env

# 2. 一键启动完整服务栈
docker compose up -d
```

启动完成后在浏览器打开 [http://localhost:8000](http://localhost:8000) 即可开始使用。

> **提示**：即便未预先配置 `.env`，RagMate 也会零摩擦就绪；你可以直接进入网页端右上角的 **Settings（设置）** 填写模型密钥与接口地址，并利用“测试连接”功能即时验证。

- **免手动建库**：PostgreSQL 15 容器、pgvector 向量扩展与数据库表结构在初次启动时全自动初始化。
- **数据持久化**：文档、向量索引与会话记录保存在 Docker 命名卷（`postgres_data`、`ragmate_docs`）中，容器销毁或镜像升级数据不丢失。
- **安全隔离**：Web 服务默认仅绑定 `127.0.0.1:8000`；数据库仅在容器内部网络互通，不对外暴露宿主机端口。

---

## 功能界面一览

RagMate 采用单页响应式界面，由 FastAPI 直接挂载托管，无需单独前端构建步骤：

1. **对话视图（Chat）**
   - 实时 Server-Sent Events (SSE) 流式打字机输出。
   - 引用来源卡片点击直达，标注具体来源文件与命中页码。
   - 可展开的**检索详情**（`"debug": true`），展示各候选片段的重排分数、命中原因及被过滤缘由。
   - 侧边栏历史会话管理，支持置顶、重命名与一键清理。

2. **文档视图（Documents）**
   - 拖拽批量上传支持 12+ 种常用格式。
   - 实时展示解析与入库进度，支持中途取消。
   - 基于文件修改时间（mtime）与内容哈希增量入库，自动跳过未修改文档。

3. **设置面板（Settings）**
   - **模型连接**：独立配置对话模型、嵌入模型与重排模型，支持独立连通性测试。具有索引锁定保护，避免切换不兼容模型维度导致向量库失效。
   - **服务状态与数据去向**：可视化展示数据库、向量引擎与各模型的连通状态，并明确说明问题、片段与候选数据的去向。
   - **高级架构规格**：折叠只读展示当前运行时的混合检索参数、切块配置与动态阈值。

---

## 支持文档格式

RagMate 内置多种主流格式的高质量抽取解析器：

| 分类 | 支持后缀 | 解析引擎 |
|---|---|---|
| 版式文档 | `.pdf` | PyPDF（保留页码信息） |
| 办公文档 | `.docx`、`.pptx`、`.xlsx`、`.csv` | docx2txt、python-pptx、openpyxl、标准 CSV 解析器 |
| 网页与结构化文本 | `.html`、`.htm`、`.rtf`、`.json`、`.xml` | BeautifulSoup4、striprtf、内置标准 JSON/XML 解析 |
| 电子书与纯文本 | `.epub`、`.txt`、`.md` | EbookLib、原生 UTF-8 流解析 |

*单文件上传大小默认上限为 50 MB（可通过 `MAX_UPLOAD_SIZE` 调整）。*

---

## 系统架构

后端遵循严格的单向分层架构：

```mermaid
flowchart LR
    subgraph Ingestion ["入库管道 Ingestion"]
        direction TB
        F["原始文档"] --> L["格式解析器"]
        L --> C["标题路径分块"]
        C --> D["哈希去重"]
        D --> E["嵌入模型 API"]
        E --> V[("pgvector")]
    end

    subgraph Retrieval ["混合检索管道 Retrieval"]
        direction TB
        Q["用户提问"] --> H["稠密 HNSW + BM25"]
        H --> RRF["RRF 融合 (k=60)"]
        RRF --> RERANK["交叉编码器重排序"]
        RERANK --> FILTER["动态阈值过滤"]
    end

    subgraph Generation ["真实回答生成 Generation"]
        direction TB
        FILTER --> LLM["对话大模型"]
        LLM --> ANS["附带引用的回答 (SSE)"]
    end

    Ingestion -.-> Retrieval
    Retrieval --> Generation
```

### 入库流程
文档扫描与修改时间检测  →  格式解析  →  层级分块（携带 `H1 > H2 > H3` 路径）  →  向量表内容哈希排重  →  调用 Embedding API 编码  →  原子替换向量（先插新向量再删旧向量）  →  同步元数据状态。

### 检索流程
用户提问  →  稠密向量内积检索与词法 BM25 并行  →  互惠排序融合（RRF，$k=60$）  →  CrossEncoder 重排序  →  分数断崖过滤与动态阈值筛选（0.5 × 最高分）  →  单次受控生成带来源标注的回答。

---

## 环境变量参考

RagMate 会自动读取项目根目录或 `backend/.env` 的配置。

### 模型连接配置

| 配置项 | 说明 | 默认值 |
|---|---|---|
| `LLM_MODEL` | 对话模型名称（如 `gpt-4o`、`deepseek-chat`、`qwen2.5:7b`） | `gpt-4o` |
| `LLM_API_KEY` | 对话模型 API 密钥（亦可在前端设置面板中直接填写） | *启动期可选* |
| `LLM_API_BASE_URL` | OpenAI 兼容接口地址 | `https://api.openai.com/v1` |
| `LLM_TEMPERATURE` | 采样温度（保持低值确保事实准确度） | `0.01` |
| `EMBEDDING_MODEL` | 向量嵌入模型标识 | `BAAI/bge-m3` |
| `EMBEDDING_API_BASE_URL` | 嵌入模型 API 端点 | `http://host.docker.internal:11434/v1` |
| `EMBEDDING_API_KEY` | 嵌入模型密钥（本地 Ollama 可留空） | *空* |
| `RERANKER_MODEL` | 重排序模型标识 | `BAAI/bge-reranker-v2-m3` |
| `RERANKER_API_BASE_URL` | 重排接口端点（如 SiliconFlow、Jina） | `http://host.docker.internal:11434/v1` |
| `RERANKER_API_KEY` | 重排服务密钥（本地模型可留空） | *空* |

### 安全与实例访问

| 配置项 | 说明 | 默认值 |
|---|---|---|
| `API_AUTH_TOKEN` | 实例访问密码。为空时本机免密；对外访问时需要此密码解锁 | *空* |
| `CORS_ORIGINS` | 允许的跨域来源 | `http://localhost:8000,http://127.0.0.1:8000` |
| `MAX_UPLOAD_SIZE` | 单文档上传上限（字节） | `52428800` (50 MB) |

### 数据库与存储

| 配置项 | 说明 | 默认值 |
|---|---|---|
| `DATABASE_URL` | PostgreSQL 连接串（Docker Compose 会自动注入） | `postgresql+asyncpg://postgres:postgres@localhost:5432/ragmate` |
| `VECTOR_TABLE` | pgvector 目标表名 | `ragmate_docs` |
| `DOCUMENTS_DIR` | 上传文档保存目录 | `/app/documents` |

---

## API 端点参考

| 请求方式 | 端点 | 说明 |
|---|---|---|
| `POST` | `/chat/stream` | 基于 Server-Sent Events (SSE) 流式返回带引用的回答 |
| `POST` | `/chat` | 非流式同步问答接口 |
| `GET` | `/chat/sessions` | 获取最近会话列表 |
| `GET` | `/chat/sessions/{id}` | 获取特定会话的全部历史对话记录 |
| `DELETE` | `/chat/sessions/{id}` | 彻底删除某个会话记录 |
| `GET` | `/documents` | 获取已上传与已入库的文档清单与状态 |
| `POST` | `/documents/upload` | 上传单份文档（multipart/form-data） |
| `DELETE` | `/documents/{name}` | 删除某篇文档、其原始文件及所有关联向量 |
| `POST` | `/ingest` | 触发增量后台入库任务 |
| `POST` | `/ingest/cancel` | 请求取消当前正在运行的入库任务 |
| `GET` | `/ingest/status` | 查询入库任务的执行状态与详细进度 |
| `GET` | `/settings` | 获取当前生效的运行时设置与索引锁状态 |
| `PUT` | `/settings` | 更新模型连接配置（支持脏字段更新与原子写入） |
| `POST` | `/settings/test/{type}` | 单独测试 `llm`、`embedding` 或 `reranker` 的连通性 |
| `GET` | `/health` | 存活探针（Liveness Probe） |
| `GET` | `/ready` | 就绪探针（Readiness Probe，检查数据库与向量表状态） |

---

## 质量评测与回归门禁

RagMate 提供确定性的检索评测工具，杜绝 LLM Judge 打分漂移：

```bash
# 运行确定性 130 题回归门禁
ragmate-eval retrieval-gate
```

- 使用确定性 BM25 算法重放 130 道标准测试集。
- 对照 `eval/reports/baseline/retrieval-baseline-130q-jina-rerun.json` 中锁定的基准线，严格核验 **Recall@k** 与 **nDCG**。
- 作为 GitHub Actions CI 的核心门禁，确保检索算法与分块逻辑修改不发生倒退。

---

## 本地开发与贡献

若你需要调试后端代码、编写测试或在宿主机本地运行 Python：

### 1. 启动开发数据库
使用开发覆盖配置将 PostgreSQL（5432）暴露给宿主机：
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres
```

### 2. 配置 Python 虚拟环境
```bash
cd backend
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\Activate.ps1   # Windows PowerShell

pip install -e ".[test]"
```

### 3. 运行开发服务器
在项目根目录下：
```bash
# 使用一键启动器（自动检测环境并拉起）
python run.py

# 或直接通过 Uvicorn 启动
uvicorn backend.app:app --reload --port 8000
```

### 4. 执行自动化测试
```bash
pytest backend/tests -v
```

---

## 数据持久化与安全建议

- **命名卷存储**：所有向量数据与会话信息存放在 Docker 命名卷 `postgres_data` 中，上传的原始文档存储在 `ragmate_docs` 中。升级镜像或重启容器绝不会丢失数据。
- **备份与迁移**：停止服务（`docker compose stop`）后，备份对应的 Docker 卷或使用 `pg_dump` 即可完成整库迁移。
- **网络隔离**：数据库容器默认不向宿主机映射端口，有效防止局域网直接嗅探数据库。
- **实例访问**：部署在公网或局域网环境时，请务必在 `.env` 中配置 `API_AUTH_TOKEN`。

---

## 常见问题解答 (FAQ)

| 问题表现 | 可能原因 | 解决办法 |
|---|---|---|
| 容器内连不上宿主机 Ollama | 容器内 `localhost` 指向容器本身 | 将 `LLM_API_BASE_URL` 或 `EMBEDDING_API_BASE_URL` 改为 `http://host.docker.internal:11434/v1`。 |
| 端口 8000 冲突 | 宿主机已有其他应用占用 8000 端口 | 在 `docker-compose.yml` 中将 `"127.0.0.1:8000:8000"` 改为 `"127.0.0.1:8080:8000"` 即可。 |
| 无法更改嵌入模型 | 向量表中已存在文档向量 | 设置界面为防止向量维度不兼容，在存在已入库文档时会锁定嵌入模型配置。先清空文档库即可切换。 |
| 宿主机运行 `pytest` 连不上数据库 | 默认 Compose 没有向宿主机暴露 5432 端口 | 执行 `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres` 即可开放端口。 |

---

## 开源协议

本项目采用 [MIT 许可证](LICENSE)。