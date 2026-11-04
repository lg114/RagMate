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
  解读 delta 前先量噪声带宽：用 `--score-repeats 3`，报告 `meta.score_spread` 即极差。
- Python 3.14 注意：ragas 0.4.3 的打分超时（`asyncio.wait_for`）在 3.14 下会让全部
  打分任务秒灭（`Timeout should be used inside a task`），`compute_metrics` 因此传
  `RunConfig(timeout=None)` 绕开；单次调用的超时由 HTTP 客户端默认兜底。3.12 不受影响。

## 生成与打分分离（P1）

生成结果是不可变 artifact；打分只读它，不再跑检索/生成：

```bash
# 1) 只生成 → <report>.raw.json（已存在则拒绝覆盖，除非 --force-raw）
ragmate-eval evaluate --testset eval/testsets/sample/testset.json \
  --report eval/reports/run.json --generate-only

# 2) 只打分（默认 temperature=0；解析失败整表重试；可选 3 遍看噪声）
ragmate-eval rescore --report eval/reports/run.raw.json \
  --output eval/reports/run.json \
  --judge-temperature 0 --parse-retries 2 --score-repeats 3
```

- raw meta 钉死：`generator_model`、answer prompt sha、`judge_model` /
  `judge_temperature` / metric pack sha。
- 解析失败经 `--parse-retries` 仍空的格子 → `status: unscored`，不进均值；
  `meta.n_scored` / `n_partial` / `n_unscored` 可查。
- `rescore` 读 `.raw.json` 时默认写到同名 `.json`，**不会改写 raw**。
- 旧的一键 `evaluate`（不带 `--generate-only`）仍可用：先落 raw 再打分。
- 生成侧 `EVAL_LLM_TIMEOUT`（默认 60s）+ `EVAL_LLM_RETRIES`：503/429/超时有限重试，
  避免干等半小时。若仍有缺口：

```bash
ragmate-eval patch-raw --raw eval/reports/run.raw.json \
  --testset eval/testsets/sample/testset.json
```

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

## CI 检索质量门禁

GitHub Actions（`.github/workflows/ci.yml`）在 pytest 之后跑：

```bash
ragmate-eval retrieval-gate
```

每次 PR 都拉起 BGE-M3、Milvus 和 reranker 太重，也依赖本机模型和 GPU。
默认嵌入和重排已经是远程 HTTP，门禁仍然不调用它们；CI 只为通过启动校验而导出占位 URL。
门禁用生产切块把 `eval/testsets/sample/corpus` 复现成 chunk，再用确定性
BM25（汉字二字组 + 拉丁串，无网络、无随机）给 98 道题排序，指标仍是
`deterministic_metrics` 里的 Recall@k / nDCG。命令退出码非 0 即 CI 失败。

门槛钉在已入库报告
[`retrieval-baseline-98q-titlepath-rerun.json`](reports/baseline/retrieval-baseline-98q-titlepath-rerun.json)
的 **fused** 阶段（与当前默认 `CHUNKING_STRATEGY=title_path` 的混合检索一致；
reranked 那一臂要 cross-encoder，词法重放对不齐）。余量 `0.015`，约等于
91 道可答题里的 1 题（`1/91 ≈ 0.011`），用来容纳 BM25 和 BGE 混合检索的差距，
不是放宽线上数字。观察值低于「基线 − 0.015」就失败。

| 指标 | fused 基线 | CI 门槛 |
|---|---|---|
| recall@5 | 0.9963 | 0.9813 |
| nDCG@5 | 0.9839 | 0.9689 |
| recall@10 | 1.0000 | 0.9850 |
| nDCG@10 | 0.9855 | 0.9705 |
| recall@20 | 1.0000 | 0.9850 |
| nDCG@20 | 0.9855 | 0.9705 |

报告里记录的题库 sha256 是 `685eeb359d6a…`，与当前提交的
`testset.json` 字节不一致；题干顺序和报告里的 `per_question` 一致，门禁
按题干对齐，不按那条 sha 放行。

有意更新基线时：

1. 在有 Milvus 和模型的环境重跑 `ragmate-eval retrieval`，用新的报告替换
   `eval/reports/baseline/retrieval-baseline-98q-titlepath-rerun.json`
   （保留 `stages.fused.summary` 和按题顺序的 `per_question`）。
2. 把上表和 `backend/tests/test_retrieval_gate.py` 里的 `PINNED_FUSED`
   改成新的 fused 数字。门槛仍是这些数字减去 `0.015`。
3. 确认 `ragmate-eval retrieval-gate` 对当前语料退出码为 0。词法重放过不了
   新门槛时，先看切块或排序是不是真的退步了，不要把 `ALLOWANCE` 调大来换绿灯。

## 实验纪律：manifest + 一次只改一个变量 + 噪声带宽

每次 `retrieval` 运行都在报告里写一份 manifest：题库文件 SHA-256、全部
检索相关配置、harness 参数、git commit。用 `--compare-manifest <上一份报告>`
在运行前打印与上次的配置差异；差异超过一个变量时，对比结论不成立。

噪声带宽分两层量：

- 确定性检索层：`--repeat 3` 重复跑，指标极差应当为 0（固定索引下的
  检索排序是确定的）。极差非 0 时先修不确定性来源，再比 A/B。
- judge 层：同配置重复跑，极差即噪声；50 题时 1 题占 2 分，<5 题的
  翻转不算赢（95% 区间约 ±11 分）。

## 融合调参（P2，离线扫参）

用已有 `.retrieval.json` 的 dense/sparse 排序重算 RRF / 加权融合，不重跑检索：

```bash
python -m backend.eval.fusion_sweep \
  --checkpoint eval/reports/baseline/retrieval-baseline-98q-titlepath-rerun.retrieval.json
```

生产开关：`FUSION_MODE`（`rrf`|`weighted`）、`RRF_K`（默认 60）、`FUSION_DENSE_WEIGHT`。
扫完若相对 k=60 没有稳定题型收益，保持默认，避免 98 题过拟合。

## 切块策略（title_path 默认）

Markdown 默认 `CHUNKING_STRATEGY=title_path`：子块正文带 `标题路径: H1 > H2 > H3`，
`CHUNK_SIZE` 与 `parent_child` 对齐，98 题 gold 三元组可复用。对比数据见
`reports/baseline/retrieval-baseline-98q-titlepath-compare.md`。改策略后必须重建
Milvus 集合，否则检索正文与标注切块不一致。

## 两层打分：排序层 vs 生产交付（不重跑检索）

检索基线的 `reranked` 停在排序；生产还会做阈值、同源去重、`FINAL_CONTEXT_K`
截断，再父展开。用同一 checkpoint 对照两层，抓「排序有 gold、交付丢 gold」：

```bash
ragmate-eval delivery \
  --checkpoint eval/reports/baseline/retrieval-baseline-98q-titlepath-rerun.retrieval.json
```

若报告里 `parent_text == child` 接近 100%，说明当前 `CHUNK_SIZE` 下父展开是空操作；
要验「用 parent 正文重排是否稀释」需另建更小切块集合（如 c300）：

```bash
python -m backend.eval.run_parent_rerank_experiment
# → eval/reports/baseline/retrieval-baseline-98q-parent-rerank.md
```

## 最终 k 截断模拟（不重跑检索）

用已落盘的 `.retrieval.json`，把每阶段排序截到 `FINAL_CONTEXT_K` 候选值再算分：

```bash
ragmate-eval cutoff \
  --checkpoint eval/reports/baseline/retrieval-baseline-98q.retrieval.json \
  --final-ks 5 10 15
```

默认读 checkpoint manifest 里的题库路径；结果写 `<stem>-cutoff.json` / `.md`。
98 题样例结论见 `reports/baseline/retrieval-baseline-98q-cutoff.md`。
