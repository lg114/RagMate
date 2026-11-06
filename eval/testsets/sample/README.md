# 样本评测集（重建版）

本目录是 2026-11-02 重建、2026-11-03 加难扩容后的精选小测试集：原始 10
题测试集与语料只在作者本地、已丢失，按原实验设计（`RETRIEVAL_EXPERIMENTS.md` 阶段 0）重做并扩容。参考答案
全部手写，逐条可从语料原文核对，没有用 LLM 生成。

## 内容

| 文件 | 说明 |
|---|---|
| `testset.json` | 130 道问答，v1.1 标注：`user_input` / `reference` / `gold_chunks` / `expected_claims` / `question_type` / `source_doc` |
| `corpus/*.md` | 8 篇中文技术文档，约 24,000 字：Docker Compose、PostgreSQL 索引、Redis 缓存、Python asyncio、HTTP 缓存与 CDN、JWT/OAuth、Linux 进程与信号、Git 内部原理 |

### 题型分布（130 题）

| 题型 | 数量 | 作用 |
|---|---|---|
| `factoid` | 65 | 单 chunk 事实题（含覆盖原先稀薄的章节） |
| `paraphrase` | 8 | 同义改写，少用原文关键词，压 sparse 捷径 |
| `ambiguous` | 8 | 近义概念对（击穿/雪崩、no-cache/no-store、zombie/orphan 等） |
| `multi-hop` | 10 | 同文档多 chunk 或跨文档（如 Compose Redis + 缓存 AOF） |
| `unanswerable` | 7 | 语料未覆盖（K8s、Terraform、Milvus、Kafka、Ceph、eBPF、Istio） |

加难目标：把原先 50 题上四臂 hit@5≈1.0 的天花板拉开。旧的 50 题基线数字不可与本集直接比。检索默认配置已冻结（hybrid + RRF k=60 + `title_path`，多查询关闭）。本集给 `ragmate-eval retrieval-gate` 做回归，不作为新增融合臂、多查询默认或新切块策略的依据。

## v1.1 标注：gold chunk ID（P0-1）

| 字段 | 说明 |
|---|---|
| `gold_chunks` | `{source, page, chunk_index}` 列表；chunk 身份用三元组，不用向量库自增 id |
| `expected_claims` | 参考答案拆出的可核查要点（1–3 条） |
| `question_type` | `factoid` / `paraphrase` / `ambiguous` / `multi-hop` / `unanswerable` |
| `source_doc` | 主来源文档名；跨文档题填首个主文档；unanswerable 为空 |

标注方法：用仓库真实入库切分逻辑（`_load_and_split`）从 `corpus/` 复现
chunk 清单（67 块：11/11/9/10/7/7/6/6），只有“只读该 chunk 就能支撑对应
claim”的才算 gold。用 `ragmate-eval check-gold`（可选 `--live`）校验。

## 复现方法

1. 把 `corpus/` 下的 8 篇文档放进 `backend/documents/`（或 `DOCUMENTS_DIR`），触发入库；
2. 确定性检索评测（不需要 LLM）：

```bash
ragmate-eval check-gold --testset eval/testsets/sample/testset.json --live
ragmate-eval retrieval --testset eval/testsets/sample/testset.json --repeat 3
```

3. 不同题集之间**不可直接比数字**；只做同集前后对比。

## 局限

- 130 题仍偏小，结论只适用于回归与配置间比较，不外推为普遍效果；
- 语料仍是人工干净技术文，难度低于真实杂乱文档；paraphrase/ambiguous/
  multi-hop 用来在**同一语料**上拉开检索差距，不是模拟脏数据。
