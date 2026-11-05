# Evaluation assets

回归门禁是 `ragmate-eval retrieval-gate`。RAGAS 打分已删除；确定性检索指标是唯一的门禁口径。

检索只冻结一套配置，对上
[`retrieval-baseline-130q-jina-rerun.json`](reports/baseline/retrieval-baseline-130q-jina-rerun.json)
的 fused 阶段：混合检索（稠密 + 稀疏）、RRF（`RRF_K=60`）、
`CHUNKING_STRATEGY=title_path`（`CHUNK_SIZE=1000` / overlap `200`，父块 `2500` / `300`）、
重排候选 `30`、阈值 `0.3`、`FINAL_CONTEXT_K=15`。
加权融合、多查询、`parent_child` 只留在环境变量里，不是第二套默认。
不要新增融合臂、不要把多查询改成默认开、不要再加切块策略。
要改其中任何一项，先重跑 `ragmate-eval retrieval`，再按下面「有意更新基线」替换钉住的门槛。

评测相关的文件分两类：本地生成的大文件不入仓，小而关键的基线入仓，
保证仓库里的每组数字都能被追溯和复现。

## 目录约定

| 路径 | 入仓 | 说明 |
|---|---|---|
| `eval/testsets/*.json` | 否 | 本地生成的测试集，体积和内容因人而异 |
| `eval/testsets/sample/` | 是 | 精选的小测试集，与基线报告配套，供复现和回归 |
| `eval/testsets/scifact/` | 否 | BEIR SciFact 生成集（5,183 文档 / 10,416 块 / 300 题），脚本可复现 |
| `eval/reports/*.json` | 否 | 本地评测的原始报告 |
| `eval/reports/baseline/` | 是 | 已确认的基线结果（附环境与样本说明），改动检索链路后应与之对比 |

## 基线

- [阶段 0 基线（父子分块修复前后对比）](reports/baseline/retrieval-stage0.md)：
  10 道手写问答、4 份中文技术文档、本地 BGE-M3 + bge-reranker-v2-m3 + Ollama qwen3:8b。
  结论与噪声带宽见该文件；完整实验过程见仓库根目录 `RETRIEVAL_EXPERIMENTS.md`。

## 注意

- 样本量小（10 题）的结论只适用于回归对比，不外推为普遍效果。
- 诚实的回归数字是确定性检索指标（Recall@k / nDCG），由 `retrieval-gate` 守住。
  faithfulness / answer relevancy / factual correctness 依赖 LLM judge，已随 RAGAS 删除，不再是评测口径。

## 确定性检索评测（tier-1，不经过 LLM）

RAGAS 的 `context_precision` / `context_recall` 是 LLM judge 打分的指标，
不是确定性检索指标——名字像，不等于。真正的检索基线用 gold chunk ID
直接算，不经过 judge：

```bash
# 先校验标注：schema + chunk 清单；--live 再校验 live pgvector 表
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

## 大语料验证：BEIR SciFact

130 题手标集在 67 块语料上已经顶满（reranked 1.00），分不出检索臂的好坏。
SciFact 是第一个大语料验证集：BEIR 的英文科学文献摘要集，
**5,183 篇文档 → 10,416 块，300 道 test 题**。分数只在同集内前后对比，
不跟 130 题集比——两边的 gold 标准不同（见下）。

### 转换（BEIR zip → 语料 + 题集）

```bash
# 下载：https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
python -m backend.eval.convert_beir --zip scifact.zip --out-dir eval/testsets/scifact
```

输出 `eval/testsets/scifact/`：`corpus/<doc_id>.md`（一篇文档一个文件，
`# 标题` + 摘要）、`testset.json`（300 题）、`testset-10q.json`（前 10 题，
供先跑计时探针）。生成文件不入仓，脚本可复现。

转换规则：BEIR 只给文档级 gold（query → 相关文档），RagMate 要 chunk 级
gold——一篇 gold 文档的全部 chunk 都算 gold，标准比人工逐块标 gold 宽松。
BEIR 没有参考答案：`reference` 取 gold 文档标题、`expected_claims` 取原
claim 文本，仅为满足 schema，检索打分只看 `gold_chunks`。转换时用生产
切分复现 chunk 清单并校验 schema 与 gold 存在性，不通过即报错。

### 词法参考值（BM25 重放，VM 已验）

| 指标 | recall@5 | nDCG@5 | recall@10 | nDCG@10 | recall@20 | nDCG@20 | MRR | hit@20 |
|---|---|---|---|---|---|---|---|---|
| BM25 | 0.5435 | 0.5207 | 0.6251 | 0.5542 | 0.6841 | 0.5746 | 0.5938 | 0.8067 |

这是词法下限，不是门禁门槛：集子没有顶满，dense / fused / reranked
有上升空间。`ragmate-eval retrieval-gate` 的门槛仍只钉 130 题集，不要
拿这套数去当 SciFact 的门槛。

### 钉住基线（2026-11-05，300 题全量，Jina）

`eval/reports/baseline/retrieval-baseline-scifact-jina.json`（已入仓），
300/300 题，`--delay-seconds 6`，全程无 429 中断，总耗时 44.2 分钟。
配置：`jina-embeddings-v3` + `jina-reranker-v3.5`，RRF_K 60，git_commit
`69b9237`（#50 合并后）。单题端到端 mean 2.85s（encode 1.05s + rerank
1.48s，DB 检索可忽略）。

| 指标 | recall@5 | nDCG@5 | recall@10 | nDCG@10 | recall@20 | nDCG@20 | MRR | hit@20 |
|---|---|---|---|---|---|---|---|---|
| dense | 0.6235 | 0.5926 | 0.7201 | 0.6287 | 0.7880 | 0.6523 | 0.6346 | 0.7880 |
| sparse | 0.3440 | 0.3343 | 0.4058 | 0.3563 | 0.4623 | 0.3750 | 0.4325 | 0.4623 |
| fused | 0.5376 | 0.4962 | 0.6831 | 0.5534 | 0.7704 | 0.5837 | 0.5712 | 0.7704 |
| reranked | 0.7178 | 0.6964 | 0.7825 | 0.7189 | 0.8196 | 0.7324 | 0.7253 | 0.8196 |

注意 fused（0.5376）低于 dense（0.6235）：哈希回退的 sparse 臂拖了
融合后腿，而纯 BM25 词法重放是 0.5435，高于当时 sparse 臂（0.3440）。
sparse 臂已解耦为 BM25（与重放同一套打分）；上表是解耦前的钉住基线，
重跑并重钉后更新。

### 入库与评测（本机跑）

SciFact 必须进**单独的库**，别跟主库的 8 篇文档混在一起，否则以后
130 题的基线重跑会检索到 SciFact 的块：

```powershell
# 1. 建库（容器名 ragmate-postgres）
docker exec ragmate-postgres psql -U postgres -c "CREATE DATABASE ragmate_scifact;"
# 2. 入库（直调 pipeline，不用起 API；ORM 表 + 向量表/扩展由 ingest 入口自建，
#    新库不用手动 init_db。遇 Jina 429 自动按 Retry-After / 5/10/20/40 秒退避重试，
#    与 reranker 同口径；10,416 块约 163 个 batch，免费档 TPM 下会多次等限流，
#    总耗时以实测为准）
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@localhost:5432/ragmate_scifact'
.\backend\.venv\Scripts\python.exe -c "from backend.application.ingest.pipeline import ingest_documents; print(ingest_documents(directory='eval/testsets/scifact/corpus'))"
# 3. 先跑 10 题计时探针，确认单题耗时后再全量
.\backend\.venv\Scripts\ragmate-eval.exe retrieval --testset eval/testsets/scifact/testset-10q.json --report eval/reports/scifact-10q-probe.json --delay-seconds 6
# 4. 全量 300 题（--delay-seconds 6 是 Jina 限流下的经验值；300 题约 45 分钟，探针为准）
.\backend\.venv\Scripts\ragmate-eval.exe retrieval --testset eval/testsets/scifact/testset.json --report eval/reports/baseline/retrieval-baseline-scifact-jina.json --delay-seconds 6
```

跑完后把 `retrieval-baseline-scifact-jina.json` 当作 SciFact 的钉住基线
（可入仓），以后的改动（sparse 解耦、换 embedding）都在这套题上前后对比。

建库若报 `template1 collation version mismatch`：先
`docker exec ragmate-postgres psql -U postgres -c "ALTER DATABASE template1 REFRESH COLLATION VERSION;"`
再建库（只改记录的版本号，不动数据）。

## CI 检索质量门禁

这是仓库认的回归门禁。GitHub Actions（`.github/workflows/ci.yml`）在 pytest 之后只跑它，不跑任何 LLM judge：

```bash
ragmate-eval retrieval-gate
```

每次 PR 都拉起 BGE-M3、pgvector 和 reranker 太重，也依赖本机模型和 GPU。
默认嵌入和重排已经是远程 HTTP，门禁仍然不调用它们；CI 只为通过启动校验而导出占位 URL。
门禁用生产切块把 `eval/testsets/sample/corpus` 复现成 chunk，再用确定性
BM25（汉字二字组 + 拉丁串，无网络、无随机）给 130 道题排序，指标仍是
`deterministic_metrics` 里的 Recall@k / nDCG。命令退出码非 0 即 CI 失败。

门槛钉在已入库报告
[`retrieval-baseline-130q-jina-rerun.json`](reports/baseline/retrieval-baseline-130q-jina-rerun.json)
的 **fused** 阶段（与当前默认 `CHUNKING_STRATEGY=title_path` 的混合检索一致；
reranked 那一臂要 cross-encoder，词法重放对不齐）。余量 `0.015`，约等于
121 道可答题里的 1 题（`1/121 ≈ 0.0083`），用来容纳 BM25 和 Jina 混合检索的差距，
不是放宽线上数字。观察值低于「基线 − 0.015」就失败。

| 指标 | fused 基线 | CI 门槛 |
|---|---|---|
| recall@5 | 0.9669 | 0.9519 |
| nDCG@5 | 0.9505 | 0.9355 |
| recall@10 | 0.9945 | 0.9795 |
| nDCG@10 | 0.9606 | 0.9456 |
| recall@20 | 1.0000 | 0.9850 |
| nDCG@20 | 0.9625 | 0.9475 |

报告里记录的题库 sha256 是 `9f23cc477947…`，与当前提交的
`testset.json` 字节不一致；题干顺序和报告里的 `per_question` 一致，门禁
按题干对齐，不按那条 sha 放行。

有意更新基线时：

1. 在有 pgvector 和模型的环境重跑 `ragmate-eval retrieval`，用新的报告替换
   `eval/reports/baseline/retrieval-baseline-130q-jina-rerun.json`
   （保留 `stages.fused.summary` 和按题顺序的 `per_question`）。
2. 把上表和 `backend/tests/test_retrieval_gate.py` 里的 `PINNED_FUSED`
   改成新的 fused 数字。门槛仍是这些数字减去 `0.015`。
3. 确认 `ragmate-eval retrieval-gate` 对当前语料退出码为 0。词法重放过不了
   新门槛时，先看切块或排序是不是真的退步了，不要把 `ALLOWANCE` 调大来换绿灯。

## 标注流程：新增题目（人工定 gold）

gold 标注是人工判断，没有生成工具：读 chunk 文本，判断哪几块能回答这道题。
工具只校验和落盘。流程：

```bash
# 1. 新文档放进语料目录（chunk_index 按文件独立编号，不改已有 gold）
cp new-doc.md eval/testsets/sample/corpus/

# 2. 列出生产切分复现的 chunk 清单，读文本定 gold
ragmate-eval chunks --corpus eval/testsets/sample/corpus --source new-doc.md
ragmate-eval chunks --corpus eval/testsets/sample/corpus --source new-doc.md --full

# 3. 追加一道题（--gold 可重复；答不出题用 --type unanswerable 且不给 --gold）
ragmate-eval annotate --testset eval/testsets/sample/testset.json \
  --question "……" --reference "……" --type factoid \
  --gold new-doc.md:3 --claims "……" --claims "……"
```

`annotate` 写盘前校验四件事，任一不过就拒绝写盘：题干不重复、schema
完整（`reference` / `expected_claims` / 类型齐全，答不出题的 gold 必须为空、
其余题至少一块 gold）、每块 gold 都在复现的 chunk 清单里。`--dry-run`
只校验不写盘。写盘是原子的（临时文件 + 替换），不会写坏已有题集。

题集一改，门禁在基线更新前一定失败：门禁按题干对齐题集与基线报告，
题数或题干变了就报错。后续三步缺一不可：

1. `ragmate-eval check-gold --testset eval/testsets/sample/testset.json`
  （重建线上索引后再加 `--live` 验 pgvector 里也有这些 gold）。
2. 把语料重建进 pgvector，重跑 `ragmate-eval retrieval`，按上面「有意更新
   基线」替换钉住的报告文件与 `PINNED_FUSED`。
3. `ragmate-eval retrieval-gate` 退出码为 0。

不要手改基线报告凑门禁，不要把 `ALLOWANCE` 调大来换绿灯。一次扩题的量
以能分辨为准：130 题时 1 题约 0.8 分，3–5 分的改进需要 100–150 题的规模；
已扩到 130 题，再看分辨力够不够。

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

扫参脚本已删除；扫参结论见 [`retrieval-baseline-98q-failures.md`](reports/baseline/retrieval-baseline-98q-failures.md)（融合调参一点）与仓库根目录 `RETRIEVAL_EXPERIMENTS.md`。

生产融合已冻结为 RRF（`RRF_K=60`）；加权融合已删除。不要新增融合臂。若某次扫参真要替换默认，
按上面「有意更新基线」重跑确定性检索并改门槛，避免 130 题过拟合。

## 切块策略（冻结为 title_path）

Markdown 冻结为 `CHUNKING_STRATEGY=title_path`：子块正文带 `标题路径: H1 > H2 > H3`，
`CHUNK_SIZE` 与历史上的 `parent_child` 边界对齐，130 题 gold 三元组可复用。
`parent_child` 只是环境变量，不是第二套策略。对比数据见
`reports/baseline/retrieval-baseline-98q-titlepath-compare.md`。改策略后必须重建
向量表并重跑确定性基线，否则检索正文与标注切块不一致。

## 两层打分：排序层 vs 生产交付（不重跑检索）

检索基线的 `reranked` 停在排序；生产还会做阈值、同源去重、`FINAL_CONTEXT_K`
截断，再父展开。用同一 checkpoint 对照两层，抓「排序有 gold、交付丢 gold」：

```bash
ragmate-eval delivery \
  --checkpoint eval/reports/baseline/retrieval-baseline-98q-titlepath-rerun.retrieval.json
```

若报告里 `parent_text == child` 接近 100%，说明当前 `CHUNK_SIZE` 下父展开是空操作；
要验「用 parent 正文重排是否稀释」需另建更小切块集合（如 c300）：

对比数据见已入库报告 [`retrieval-baseline-98q-parent-rerank.md`](reports/baseline/retrieval-baseline-98q-parent-rerank.md)；该实验脚本已删除。

## 最终 k 截断模拟（不重跑检索）

用已落盘的 `.retrieval.json`，把每阶段排序截到 `FINAL_CONTEXT_K` 候选值再算分：

```bash
ragmate-eval cutoff \
  --checkpoint eval/reports/baseline/retrieval-baseline-98q.retrieval.json \
  --final-ks 5 10 15
```

默认读 checkpoint manifest 里的题库路径；结果写 `<stem>-cutoff.json` / `.md`。
98 题样例结论见 `reports/baseline/retrieval-baseline-98q-cutoff.md`。
