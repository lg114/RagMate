# RAGAS Evaluation Script Design

## Context

RagMate 需要一个评估工具来量化当前 RAG 系统的表现。之前的 `eval/` 模块在 Prototype 11 被移除，计划用 RAGAS 替代。目标是先做一个独立评估脚本摸底，后续可升级为 API 服务。

## 文件结构

```
eval/
├── ragas_eval.py          # 主脚本（唯一代码文件）
├── testsets/              # 测试集版本管理
│   └── testset_v1.json    # 生成的测试集（人工审核后提交）
└── reports/               # 历史评估报告
    └── report_20260518.json
```

## CLI 接口

```bash
# 生成测试集
python eval/ragas_eval.py generate \
  --size 50 \
  --max-docs 20 \
  --output eval/testsets/testset_v1.json \
  --seed 42

# 运行评估
python eval/ragas_eval.py evaluate \
  --testset eval/testsets/testset_v1.json \
  --report eval/reports/report_20260518.json \
  --top-k 4 \
  --threshold 0.75 \
  --judge-model deepseek-v4-pro
```

## Generate 模式

### 流程

1. 用 `load_langchain_docs()` 加载 `backend/documents/` 下的文档，返回 `list[langchain.Document]`，metadata 包含 source、page
2. 用 RAGAS `TestsetGenerator.generate_with_langchain_docs()` 生成测试集
3. 输出到指定 JSON 文件

### load_langchain_docs() 实现要点

复用 `ingest.load_document()` 加载每个文件，确保 metadata 包含 `source`（文件名）和 `page`（页码）。与 ingest 流程的区别：不做切分，直接传原始 Document 给 RAGAS（RAGAS 内部会自己切分）。

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--size` | 50 | 生成测试用例数量 |
| `--max-docs` | 无限制 | 最多加载多少个文档（防 token 爆炸） |
| `--output` | `eval/testsets/testset.json` | 输出路径 |
| `--seed` | 42 | 随机种子（可重复性） |
| `--distributions` | RAGAS 默认 | simple/reasoning/multi-context 比例 |

### 输出格式

```json
[
  {
    "user_input": "问题文本",
    "reference": "ground truth 答案",
    "retrieved_contexts": [],
    "response": "",
    "source": "文件名.pdf"
  }
]
```

`retrieved_contexts` 和 `response` 生成时为空，evaluate 时填充。`source` 是自定义字段，用于追溯。

## Evaluate 模式

### 流程

1. 读取 testset JSON
2. 对每条问题：
   - 调用 `retriever.retrieve(query, k=top_k)` 获取 contexts（完整走 hybrid + rerank 流程）
   - 调用 LLM 生成 response（通过 `get_rag_response()` 包装函数）
3. 构建 RAGAS `EvaluationDataset`
4. 调用 `ragas.evaluate()` 批量计算指标
5. 输出报告

### get_rag_response() 包装函数

关键设计：必须使用和生产环境一致的生成逻辑。

```python
def get_rag_response(question: str, contexts: list[str]) -> str:
    """用 LLM 基于检索到的上下文生成答案。"""
    # 使用 ChatOpenAI + 固定 prompt，而非 Deep Agent
    # 原因：评估要测的是 RAG 基础能力，不是 agent 的规划能力
    # 后续可加 --mode agent 切换到完整 agent 管道
```

两种模式（`--mode` 参数，预留但初期只实现 single）：
- `single`（默认）：`retrieve()` + LLM 直接生成，测 RAG 基础能力
- `agent`（后续）：走完整 Deep Agent 管道，测 agent 端到端表现

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--testset` | 必填 | 测试集 JSON 路径 |
| `--report` | `eval/reports/report_{date}.json` | 报告输出路径 |
| `--top-k` | 4（复用配置） | 检索返回的 chunk 数量 |
| `--threshold` | 无 | overall score 低于此值时 exit code 非 0 |
| `--judge-model` | 复用 .env 配置 | RAGAS judge LLM |
| `--mode` | `single` | single 或 agent |

### 评估指标（5 个）

```python
from ragas.metrics import (
    Faithfulness,          # 答案是否忠实于上下文
    AnswerRelevancy,       # 答案是否回答了问题
    ContextPrecision,      # 相关 chunk 是否排在前面
    ContextRecall,         # 上下文是否覆盖 ground truth
    FactualCorrectness,    # 比 Faithfulness 更严格
)
```

### Judge LLM 配置

通过 `LangchainLLMWrapper` 包装现有的 ChatOpenAI 实例，复用 `.env` 中的 LLM 配置。`--judge-model` 参数允许切换到其他模型（如 gpt-4o）。

Embeddings 通过 `LangchainEmbeddingsWrapper` 包装现有的 BGE-M3。

### 报告输出

控制台输出：
```
=== RAGAS Evaluation Report ===
Test cases: 50
Mode: single
Top-k: 4
Judge model: deepseek-v4-pro

Metrics:
  Faithfulness:        0.82
  Answer Relevancy:    0.75
  Context Precision:   0.68
  Context Recall:      0.71
  Factual Correctness: 0.78
  Overall:             0.75

Detailed report saved to eval/reports/report_20260518.json
```

JSON 报告结构：
```json
{
  "meta": {
    "timestamp": "2026-05-18T10:30:00",
    "testset": "eval/testsets/testset_v1.json",
    "test_cases": 50,
    "mode": "single",
    "top_k": 4,
    "judge_model": "deepseek-v4-pro",
    "retriever_config": {
      "hybrid_search": true,
      "reranker": "BAAI/bge-reranker-v2-m3",
      "score_threshold": 0.06
    }
  },
  "summary": {
    "faithfulness": 0.82,
    "answer_relevancy": 0.75,
    "context_precision": 0.68,
    "context_recall": 0.71,
    "factual_correctness": 0.78,
    "overall": 0.75
  },
  "details": [
    {
      "user_input": "...",
      "reference": "...",
      "response": "...",
      "contexts": ["..."],
      "scores": {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 0.7,
        "context_recall": 0.65,
        "factual_correctness": 0.85
      }
    }
  ]
}
```

### 错误处理

- 单条问题失败（检索异常、LLM 超时等）不中断整体评估
- 失败的用例在报告中标记为 `"error": "..."`，不计入指标均值
- 用 tqdm 显示进度条

## 依赖

在 `backend/pyproject.toml` 添加：

```toml
[project.optional-dependencies]
eval = [
    "ragas>=0.2.0",
    "tqdm>=4.65.0",
]
```

## 复用的现有模块

| 模块 | 复用内容 |
|------|----------|
| `ingest.py` | `load_document()` — 文档加载 |
| `retriever.py` | `retrieve()` — 混合检索 + rerank |
| `config.py` | `settings` — LLM、Milvus、Embedding 配置 |
| `model_factory.py` | `get_llm()` — LLM 实例 |

## 后续升级路径

1. 加 `--mode agent` 支持评估完整 Deep Agent 管道
2. 在 `main.py` 加 `POST /eval` API 端点
3. 结果存入 PostgreSQL，支持版本对比
4. 集成到 CI/CD（`--threshold` 门禁）
