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

## 确定性检索评测（tier-1，不经过 LLM）

RAGAS 的 `context_precision` / `context_recall` 是 LLM judge 打分的指标，
不是确定性检索指标——名字像，不等于。真正的检索基线用 gold chunk ID
直接算，不经过 judge：

```bash
# 先校验标注：schema + chunk 清单；--live 再校验 live Milvus 索引
ragmate-eval check-gold --testset eval/testsets/sample/testset.json [--live]

# 分阶段跑检索并算 hit/recall/MRR/nDCG（默认四臂：dense / sparse / fused / reranked）
ragmate-eval retrieval --testset eval/testsets/sample/testset.json
```

`retrieval` 命令写三个文件（默认 `--report` 为
`eval/reports/baseline/retrieval-baseline-50q.json`）：指标 JSON 报告、
同名 `.md` 基线表（入仓）、同名 `.retrieval.json` 分阶段排序落盘（不入仓，
供重算指标时复用，不重跑检索）。reranked 阶段只取排序，不做阈值过滤、
去重、父展开——那些是生成侧的取舍，不应污染检索基线。unanswerable 题
不计入均值，单独计数。

## 实验纪律：manifest + 一次只改一个变量 + 噪声带宽

每次 `retrieval` 运行都在报告里写一份 manifest：题库文件 SHA-256、全部
检索相关配置、harness 参数、git commit。用 `--compare-manifest <上一份报告>`
在运行前打印与上次的配置差异；差异超过一个变量时，对比结论不成立。

噪声带宽分两层量：

- 确定性检索层：`--repeat 3` 重复跑，指标极差应当为 0（固定索引下的
  检索排序是确定的）。极差非 0 时先修不确定性来源，再比 A/B。
- judge 层：同配置重复跑，极差即噪声；50 题时 1 题占 2 分，<5 题的
  翻转不算赢（95% 区间约 ±11 分）。
