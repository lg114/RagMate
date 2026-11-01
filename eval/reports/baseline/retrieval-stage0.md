# 检索基线 · 阶段 0：父子分块修复前后

> 来源：`RETRIEVAL_EXPERIMENTS.md` 阶段 0 的实测记录（2026-10）。
> 背景：`parent_text` 写入了 Milvus，但检索时未请求该字段，父子分块在检索侧静默失效；
> 本基线即修复（打分文本与交付文本拆分）前后的对比。

## 环境

| 组件 | 配置 |
|---|---|
| 向量库 | Milvus 容器，collection `ragmate_docs__c300`（40 条） |
| 嵌入 / 重排 | 本地 BGE-M3 + 本地 bge-reranker-v2-m3 |
| LLM | Ollama `qwen3:8b`（须设 `LLM_REASONING_EFFORT=none`） |
| 测试集 | 10 道手写问答，参考答案逐条可从原文核对（不用 LLM 生成） |
| 语料 | 4 份中文技术文档，约 2.4 万字符 |

## 对比结果（RAGAS）

| 指标 | 修复前 | 修复后 | delta |
|---|---|---|---|
| faithfulness | 0.9833 | 0.9798 | −0.0035 |
| answer_relevancy | 0.8226 | 0.8284 | +0.0058 |
| context_precision | 0.9048 | 0.9762 | **+0.0714** |
| context_recall | 0.9667 | 1.0000 | **+0.0333** |
| factual_correctness (f1) | 0.8110 | 0.7760 | −0.0350 |

平均 context 长度 230 → 528 字符（2.3x），与父子比实测（1.70~3.14）一致，修复在检索侧确实生效。

## 噪声带宽（同配置重复 2 轮的极差）

| 指标 | 极差 = 噪声 |
|---|---|
| context_precision | 0.0000 |
| context_recall | 0.0000 |
| answer_relevancy | 0.0004 |
| faithfulness | 0.0125 |
| factual_correctness | 0.0160 |

## 判定

- context_precision +0.0714、context_recall +0.0333：噪声为 0，**真实提升**。
- answer_relevancy +0.0058：超出噪声但幅度极小。
- faithfulness −0.0035：落在噪声内，不可区分。
- factual_correctness −0.0350：约噪声 2.2 倍，**可能确实下降**，需更多轮次定论。

结论：修复在检索侧带来确凿提升；更长的上下文并不自动等于更好的答案。
样本量只有 10 题，本基线用于回归对比，不外推为普遍效果。
`answer_relevancy` 因 EvalLLM 兼容处理（pop 掉 `n`）可靠性打折，数值谨慎看待。
