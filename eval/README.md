# Evaluation assets

评测相关的文件分两类：本地生成的大文件不入仓，小而关键的基线入仓，
保证仓库里的每组数字都能被追溯和复现。

## 目录约定

| 路径 | 入仓 | 说明 |
|---|---|---|
| `eval/testsets/*.json` | 否 | 本地生成的测试集，体积和内容因人而异 |
| `eval/testsets/sample/` | 是 | 精选的小测试集，与基线报告配套，供复现和回归 |
| `eval/reports/*.json` | 否 | 本地评测的原始报告 |
| `eval/reports/baseline/` | 是 | 已确认的基线结果（附环境与样本说明），改动检索链路后应与之对比 |

## 基线

- [阶段 0 基线（父子分块修复前后对比）](reports/baseline/retrieval-stage0.md)：
  10 道手写问答、4 份中文技术文档、本地 BGE-M3 + bge-reranker-v2-m3 + Ollama qwen3:8b。
  结论与噪声带宽见该文件；完整实验过程见仓库根目录 `RETRIEVAL_EXPERIMENTS.md`。

## 注意

- 样本量小（10 题）的结论只适用于回归对比，不外推为普遍效果。
- 生成侧指标（faithfulness / answer relevancy / factual correctness）依赖 LLM judge，
  解读 delta 前先量噪声带宽：同配置重复跑两次，极差即噪声。
- 评测在算分前会把未打分的管线结果写成 `<report>.raw.json`（不入仓）；算分失败时
  用 `ragmate-eval rescore --report <raw 文件>` 只重算分，不重跑管线。跑评测前 CLI
  会先探一下 judge 能不能返回可解析的 JSON，探不过就直接停，不浪费一轮管线。
- Python 3.14 注意：ragas 0.4.3 的打分超时（`asyncio.wait_for`）在 3.14 下会让全部
  打分任务秒灭（`Timeout should be used inside a task`），`compute_metrics` 因此传
  `RunConfig(timeout=None)` 绕开；单次调用的超时由 HTTP 客户端默认兜底。3.12 不受影响。
