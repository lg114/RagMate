# RagMate 检索实验能力改造方案

把「改切分策略 / 加 API 参数 / 对比检索配置」从**配置项存在但没法做实验**，变成**能跑、能对比、结论站得住**的实验能力。

分三阶段，每阶段独立可交付、可验证。第 0 阶段是前置修复，不做它，后面的实验结论不可信。

---

## 阶段 0（前置）：修掉父子分块的静默失效

**这是我在读代码时发现的 bug，不是设计决策。它必须排在第一位，因为所有"切分策略"实验都建立在它之上。**

### 证据链

| 位置 | 代码 | 说明 |
|---|---|---|
| `infrastructure/milvus.py:230` | `parent_texts = [chunk.metadata.pop("parent_text", ...)]` | 父片段从 metadata **弹出**，作为独立顶层字段存储 |
| `infrastructure/milvus.py:133` | `FieldSchema("parent_text", DataType.VARCHAR, ...)` | 确实是独立的 collection 字段，写入是成功的 |
| `core/retriever.py:163` | `output_fields=["text", "metadata"]` | 检索时**没有请求** `parent_text` |
| `core/retriever.py:190` | `"text": entity.get("parent_text") or entity["text"]` | 取不到 → 恒为 `None` → 静默回退到小片段 |

`parent_text` 被写进了 Milvus，但检索链路从来没有把它取出来。README 宣称的"用小片段检索，把更大的父片段交给 LLM"**在检索侧完全没生效**，实际交给 LLM 的和用来打分的都是小片段——父子分块只剩存储开销。

这是静默失效：不报错、不打日志，`or` 回退把问题吞掉了。

### 修法（不要只加一个 output_field）

天真的修法是给 `output_fields` 加 `"parent_text"`。但这样改会把**打分对象**也从小片段换成父片段，而 `RERANK_SCORE_THRESHOLD=0.3`、`DYNAMIC_THRESHOLD_RATIO=0.5`、`SCORE_GAP_THRESHOLD=0.15` 这一整套阈值都是在**小片段**上校准的——一次修复会把调参基线全部作废。

正确做法是把「打分文本」和「交付文本」拆开：

```python
# _extract_candidates：两个字段，各司其职
candidates.append({
    "text": entity["text"],                                   # 小片段：rerank / 压缩 / 阈值 都用它
    "context": entity.get("parent_text") or entity["text"],   # 父片段：最终交给 LLM
    "source": meta.get("source", ""),
    "page": meta.get("page"),
    "chunk_index": meta.get("chunk_index"),
})
```

`_do_search` 两处 `output_fields` 都加 `"parent_text"`（`hybrid_search` 分支和 `search` 分支）。rerank、`_compress_chunks`、`_filter_and_dedup` 保持用 `text` 不动，只在 `retrieve()` 返回前把 `context` 提升为输出文本。

### 验收

- **单元测试（已完成）** — `tests/test_parent_expansion.py`，13 项，覆盖：`_do_search` 必须请求
  `parent_text`（回归防线）、`_extract_candidates` 正确拆分 text/context、`_expand_to_parent` 只替换交付
  文本且保留 score/source、压缩收到的是父片段而非小片段。
- **Milvus 层对照验证（已完成）** — 调用项目自己的 `ensure_collection` / `insert_chunks` 写入一条带
  `parent_text` 的记录，再用两种 `output_fields` 查询同一份数据：

  | `output_fields` | `parent_text` | 长度 |
  |---|---|---|
  | `["text", "metadata"]`（修复前） | **未返回** | 0 |
  | `["text", "parent_text", "metadata"]`（修复后） | 返回 | 333 |

  同一行记录里 `text` 长度 15、`parent_text` 长度 333 —— **原缺陷与修复效果都被直接观测到**，
  small-to-big 两侧的粒度差异符合设计预期。

  > 方法备注：`infrastructure/milvus.py` 的 import 链只依赖 `pymilvus` + `pydantic-settings`，**不含 torch**。
  > 因此不必安装 `FlagEmbedding` / `sentence-transformers` 就能在 Milvus 层做验证。验证脚本临时改写
  > `settings.MILVUS_COLLECTION` 到独立集合名，跑完即 drop，不污染真实数据。
- **真实语料实测（已完成，并暴露出一个关键前提）**

  用 4 份中文技术文档（约 2.4 万字符）真实入库后实测，**结果与预期不符**：

  **第一轮 · 默认参数 `CHUNK_SIZE=1000`**

  | 文件 | 片段数 | text 平均 | parent 平均 | 比值 |
  |---|---|---|---|---|
  | 01-rag-overview.md | 7 | 324 | 359 | 1.11 |
  | 02-vector-index.md | 7 | 297 | 327 | 1.10 |
  | 03-chunking-strategies.md | 6 | 317 | 341 | 1.07 |
  | 04-hnsw-deep-dive.md | 3 | 694 | 721 | 1.04 |

  修复后交付文本只变长约 8%，远低于预期。

  **根因**：`.md` 的切分路径是「标题切分 → 章节内按 `CHUNK_SIZE` 递归切」，
  而**所有章节都短于 1000 字符**（最大 802），因此每个章节只切出**一个**子片段，
  `parent_text` 与 `text` 几乎相同——父子分块退化成「父子相同」，差的 30 余字符恰好是标题长度。

  **这不是修复失败，是语料不满足前提条件。**

  **第二轮 · `CHUNK_SIZE=300`**（独立 collection `ragmate_docs__c300`，不污染原数据）

  | 文件 | 片段数 | text 平均 | parent 平均 | 比值 |
  |---|---|---|---|---|
  | 01-rag-overview.md | 12 | 187 | 420 | **2.24** |
  | 02-vector-index.md | 10 | 210 | 357 | **1.70** |
  | 03-chunking-strategies.md | 9 | 214 | 405 | **1.89** |
  | 04-hnsw-deep-dive.md | 9 | 229 | 721 | **3.14** |

  单条记录里能直接观察到「多子一父」结构，例如 `04-hnsw-deep-dive.md`：
  子片段 627 / 627 / 627 共享同一个 627 字符父片段，708 / 708 / 708 共享 708，829 / 829 / 829 共享 829。

  **结论**：父子分块的收益**完全取决于「章节长度 ÷ `CHUNK_SIZE`」这个比值**。
  当章节短于 `CHUNK_SIZE` 时，这套机制只有存储开销、没有检索收益。
  **做任何切分参数调优之前，先确认文档结构能支撑两级切分**——否则测出来的是噪声。

- **RAGAS 对比（待做，非必需）** — 阶段 0 的核心结论已由上面两组数据回答。
  若要补 RAGAS，需要可用的 LLM（生成回答 + judge 打分）以及 `ragas` 包能正常 import。

### 风险

修好后交给 LLM 的上下文会显著变长（每个片段从约 600 字符变成约 2500 字符，`FINAL_CONTEXT_K=15` 时约 37k 字符 ≈ 10k token）。token 成本上升，需要确认没超出所选模型的 context window。

---

## 阶段一：检索参数运行时可覆盖

### 目标

不改 `.env`、不重启进程，按请求指定检索参数。

### 设计决策：走 `_contexts`，不新增机制

`core/agent.py` 已经有一个 per-session 的 `_contexts` 字典，存着 `messages` / `metrics` / `texts`，而 `retrieval_tool` **已经在读它**。把检索参数挂进去即可，零新增机制：

```python
# agent.py
def _prepare_session(session_id, messages, retrieval_options=None):
    _current_session_id.set(session_id)
    with _contexts_lock:
        _contexts[session_id] = {
            "messages": messages, "metrics": [], "texts": [],
            "retrieval_options": retrieval_options,
        }

# retrieval_tool 内
ctx = _contexts.get(sid)
opts = ctx.get("retrieval_options") if ctx else None
results = retrieve(query, k=settings.FINAL_CONTEXT_K, options=opts)
```

**为什么不用 contextvar：** 项目里已经有一个 `_current_session_id` contextvar，再加一个也能跑，但 `_contexts` 本来就是为"跨线程传会话状态"设计的，功能重叠。少一个机制就少一处会漏的地方。

**为什么不给 `retrieval_tool` 加参数：** 它是 `@tool` 装饰的函数，签名直接暴露给 LLM，加参数会让模型试图自己填检索参数。

### 配置解析：入口解析一次，内部显式传参

`core/retriever.py` 现在有 15 处 `settings.XXX` 散落在 `retrieve` / `_do_search` / `_filter_and_dedup` / `_compress_chunks` 里。加一个 resolver 把它们收敛成一份不可变配置：

```python
@dataclass(frozen=True)
class ResolvedRetrievalConfig:
    top_k: int
    hybrid: bool
    drop_ratio_search: float
    rerank_candidates: int
    score_threshold: float
    dynamic_threshold_ratio: float
    high_score_ratio: float
    min_per_source: int
    max_per_source: int
    score_gap_threshold: float
    source_dominance_threshold: float
    source_dominance_boost: float
    contextual_compression: bool
    compression_score_threshold: float
    compression_min_chars: int


def resolve_retrieval_config(k=None, options=None) -> ResolvedRetrievalConfig:
    """以 settings 为底，options 中显式给出的项逐项覆盖。"""
```

`retrieve(query, k=None, options=None)` 在入口调一次 resolver，然后把结果**显式传给**内部函数。`_filter_and_dedup(candidates, cfg)` 从"读全局单例"变成"收一个参数"——可测性完全不同，这是顺手拿到的收益。

### API 形态

`domain/schemas.py` 新增：

```python
class RetrievalOptions(BaseModel):
    top_k: int | None = Field(default=None, gt=0, le=50)
    hybrid: bool | None = None
    rerank_candidates: int | None = Field(default=None, gt=0, le=200)
    score_threshold: float | None = Field(default=None, gt=0.0, lt=1.0)
    dynamic_threshold_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    max_per_source: int | None = Field(default=None, gt=0, le=20)
    contextual_compression: bool | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=10000)
    session_id: str | None = None
    replace_last: bool = False
    retrieval: RetrievalOptions | None = None   # 新增
```

请求示例：

```jsonc
{
  "message": "什么是 RAG？",
  "retrieval": { "top_k": 8, "hybrid": false, "score_threshold": 0.4 }
}
```

**只暴露安全子集。** 不进 `RetrievalOptions` 的东西：`AGENT_RECURSION_LIMIT`、`AGENT_TIMEOUT`、`MILVUS_*`、`MAX_UPLOAD_SIZE`、限流参数。上限（`top_k ≤ 50`、`rerank_candidates ≤ 200`）是必需的——没有上限，一个 `top_k=100000` 就能把 LLM 上下文和 Milvus 查询打爆，这等于开了个 DoS 口子。

### 改动清单

| 文件 | 改动 |
|---|---|
| `domain/schemas.py` | 新增 `RetrievalOptions`；`ChatRequest` 加 `retrieval` 字段 |
| `api/chat.py` | 两个端点把 `body.retrieval` 传给 `chat` / `chat_stream` |
| `application/chat.py` | `chat()` / `chat_stream()` 加 `retrieval_options` 参数，透传到 `run_agent` / `run_agent_streaming` |
| `core/agent.py` | `_prepare_session` 存 options；`run_agent` / `run_agent_streaming` 加参数；`retrieval_tool` 从 ctx 取 |
| `core/retriever.py` | 新增 `ResolvedRetrievalConfig` + `resolve_retrieval_config`；`retrieve` 加 `options` 参数；内部函数改为收 cfg |
| `eval/retrieval.py` | `retrieve_contexts` 加 `options` 参数（评测侧也要能用） |

### 验收

- 单测：`resolve_retrieval_config` 的覆盖语义（options 部分给出时，未给出的项回落到 settings）。
- 单测：越界值被 Pydantic 拒绝（`top_k=0`、`top_k=51`、`score_threshold=1.5`）。
- 集成：同一 query 用 `hybrid: true` / `false` 各请求一次，断言走的是不同检索分支（mock `_do_search`）。

---

## 阶段二：切分策略可切换 + 实验隔离

### 先说那个坑

入库是**增量**的，两道闸门都会拦住重新切分：

1. `_detect_new_files` 按 **mtime** 判断文件是否变化；
2. `deduplicate_chunks` 按 **content_hash** 去重。

所以改完 `CHUNK_SIZE_PDF` 之类的参数后，**已入库的文件不会重新切分**。第一次跑切分对比实验，你会得到一份"新切分的部分 + 旧切分的部分"的混合语料，而结果看起来像是"切分参数没用"。

### 实验隔离：推荐子进程 + env

`settings` 是模块级单例，`milvus._collection_loaded` 是全局 bool，`get_milvus_client()` 也是单例——**在同一进程里切换 collection 名或切分参数，会踩到这三处全局状态**。

对"做对比实验"这个目标，最省事也最可靠的做法是**每个配置起一个子进程**：

```python
# eval/sweep.py 的核心
for cfg in grid:
    env = {**os.environ, **cfg.as_env()}      # MILVUS_COLLECTION / CHUNK_SIZE_PDF / ...
    subprocess.run(
        [sys.executable, "-m", "backend.eval.cli", "evaluate",
         "--testset", testset, "--report", f"eval/reports/{cfg.slug}.json"],
        env=env, check=True,
    )
```

子进程天然隔离全部单例和全局状态，零重构风险。代价是每个配置要重载一次 BGE-M3（数 GB 模型），慢，但实验本来就是离线的。

**用不同 collection 名做隔离**（`ragmate_docs__c600`、`ragmate_docs__c1000`）比清空重建更安全——可以并行保留多套结果，出问题时原始数据还在。

### 策略层：建议先不做

容易滑向"设计一个 `ChunkingStrategy` 基类 + registry + 语义分块插件"的过度设计。**没有必要。**

真实有价值的对比实验是**参数对比**（`CHUNK_SIZE_PDF` = 400 / 600 / 1000，`CHUNK_SIZE_PARENT` = 1500 / 2500 / 4000），不是策略对比。参数对比零重构风险，而且结论对使用者直接有用。

如果确实要显式化，最小改动是把 `_get_text_splitter` 里的隐式分支抽成一个 `CHUNK_STRATEGY` 枚举（`fixed` / `markdown_aware`），默认值等于当前行为——**默认行为必须逐字节不变**，否则你就无法区分"策略改动"和"默认值漂移"带来的差异。

### 验收

- 同一份文档、同一套 testset，用两个不同 `MILVUS_COLLECTION` 跑完，两边的 chunk 总数不同且各自自洽。
- 反向验证那个坑：改 `CHUNK_SIZE_PDF` 但**不换 collection**，观察 chunk 总数不变——确认自己理解了增量逻辑，而不是碰巧没踩到。

---

## 阶段三：多配置对比评测

### 先补全报告快照

`eval/cli.py:120` 的 `retriever_config` 只记 3 个字段：

```python
"retriever_config": {
    "hybrid_search": settings.HYBRID_SEARCH_ENABLED,
    "reranker": settings.RERANKER_MODEL,
    "score_threshold": settings.RERANK_SCORE_THRESHOLD,
},
```

**对比的前提是能证明两份报告确实跑的是不同配置。** 记不全，事后就无法归因。直接改成 `asdict(resolve_retrieval_config())` —— 复用阶段一产出的那个 dataclass，字段自然对齐。切分参数（`CHUNK_SIZE_*`）和 `MILVUS_COLLECTION` 也要记进去。

### 新增 compare / sweep 子命令

```
ragmate-eval sweep --testset eval/testsets/testset.json --grid eval/grid.yaml
ragmate-eval compare --reports eval/reports/c600.json eval/reports/c1000.json --output eval/reports/compare.json
```

`compare` 读多份报告，输出指标矩阵（每行一个指标，每列一个配置，附 delta）；`sweep` 读一个参数网格，对每个组合起子进程跑 `evaluate`，最后调 `compare` 汇总。

`eval/interactive.py` 的菜单加一项「对比历史报告」，复用同一套对比逻辑。

### 统计噪声——这是最容易被问倒的地方

RAGAS 的 `Faithfulness` / `AnswerRelevancy` 依赖 LLM judge，**同一份数据跑两次结果不同**。如果结论是"`chunk_size=600` 比 `1000` 好 0.03"，而重跑一次变成 -0.02，这个结论不成立。

所以对比功能不能只输出一张均值表，至少要：

1. **固定一切能固定的**：同一份 `testset.json`、同一个 `judge_model`、同一台机器、同一批文档；
2. **同一配置重复跑 2–3 次**，报告里带标准差，而不是只报单次；
3. **结论只认超过噪声带宽的差异**。先跑一遍"完全相同的配置两次"，量出噪声带宽是多少，再拿它当判据——这一步很关键，很多人跳过它，然后拿着 0.02 的差异写结论。

这部分是整份方案里最值钱的东西：**知道自己的结论在什么条件下才成立**，比多调对一个参数难得多。

---

## 交付物与验收

| 阶段 | 产出 | 验收判据 |
|---|---|---|
| 0 | `retriever.py` 修复 + 单测 | 父片段确实进入了 LLM 上下文；修复前后 `ContextRecall` 有对比数据 |
| 1 | `RetrievalOptions` + resolver + 单测 | 同一 query 用不同 `retrieval` 参数得到不同检索结果，无需重启 |
| 2 | 隔离后的多套 collection + `sweep.py` | 同一 testset 上多配置各自跑通，chunk 数互不污染 |
| 3 | `compare` / `sweep` + 噪声带宽测量 | 一份含均值、标准差、delta 的对比报告 |

### 对应到简历上的表述

- 阶段 0：「发现并修复父子分块在检索侧静默失效的问题——`parent_text` 已写入向量库但未被检索取出，`or` 回退把缺陷吞掉；拆分为打分文本与交付文本后……」
- 阶段 1：「为检索链路引入运行时可覆盖的参数层，收敛 15 处散落的全局配置读取，按请求粒度切换混合/稠密检索与阈值，无需重启」
- 阶段 2：「设计增量入库下的实验隔离方案，避免 mtime + content-hash 双重去重导致切分参数变更不生效」
- 阶段 3：「构建多配置对比评测，量化 LLM judge 噪声带宽并据此设定显著性判据」

---

## 明确不做的事

- **不暴露 `AGENT_*` / `MILVUS_*` / 限流参数到 API**——安全和稳定性参数不该由请求方决定。
- **不做 `ChunkingStrategy` 插件框架**——参数对比已经能回答问题，抽象层只是自娱自乐。
- **不在同进程内切换 collection**——三处全局单例的代价大于收益，子进程更简单。
- **不为了让数字好看而调 judge_model**——评测模型必须全程固定，否则对比无意义。
