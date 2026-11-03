# title_path vs parent_child（98q）

同一题库、同一 `CHUNK_SIZE=1000`（块边界对齐，gold 三元组可复用）。  
实验时基线：`ragmate_docs`（当时为 parent_child）  
实验 collection：`ragmate_docs__titlepath`（正文带 `标题路径: H1 > H2 > H3`）  
**现状：** 主集合 `ragmate_docs` 已按 `title_path` 重建（67 块，gold 全命中）。

报告：`retrieval-baseline-98q-titlepath.json` / `-cutoff.md`

## 总表（answerable 均值）

| stage | 指标 | parent_child | title_path | delta |
|---|---|---|---|---|
| dense | MRR | 0.9643 | 0.9670 | +0.003 |
| dense | nDCG@5 | 0.9671 | 0.9706 | +0.004 |
| sparse | MRR | 0.9460 | 0.9670 | **+0.021** |
| sparse | nDCG@5 | 0.9529 | 0.9755 | **+0.023** |
| sparse | multi-hop recall@5 | 0.8833 | **1.0000** | **+0.117** |
| fused | MRR | 0.9802 | 0.9835 | +0.003 |
| fused | multi-hop recall@5 | 0.8833 | 0.9667 | **+0.083** |
| reranked | MRR | 0.9890 | 0.9872 | −0.002 |
| reranked | multi-hop recall@5 | 0.9667 | **1.0000** | **+0.033** |

## 题 67（重启策略，三 gold）— 失败清单主目标

gold chunk_index = 2(web) / 4(db) / 5(redis)

| stage | parent_child ranks | title_path ranks |
|---|---|---|
| dense | 11, 14, 4 | 8, 9, **2** |
| sparse | 4, 6, 7 | **3, 5, 2**（三者均进 top5） |
| fused | 6, 8, 5 | **5, 8, 2** |
| reranked | 2, 4, **6** | **3, 5, 4**（三者均 ≤5） |

`cutoff`：parent_child 下 `reranked@5` 仍丢题 67；**title_path 下 incomplete@5 = 0**。

## 结论

1. **标题路径对题 67 类「多证据 + 近义干扰」有效**：服务名进路径后，sparse/rerank 不再把 redis/db 挤出 top5。  
2. **sparse / multi-hop 收益最大**；整体 reranked MRR 几乎持平（−0.002，可当噪声）。  
3. 块大小与 parent_child 对齐（67 块、正文一致），对比公平。  
4. **已采纳为默认：** `CHUNKING_STRATEGY=title_path`（见 `backend/.env` / `config.py`）。切换回 `parent_child` 后需重建索引。

## 复现 / 重建主索引

```bash
# A/B 实验（独立 collection）
python -m backend.eval.run_titlepath_experiment

# 把主集合 ragmate_docs 重建为 title_path（与默认配置对齐）
python -c "from backend.infrastructure.config import settings; from backend.eval.corpus_ingest import ingest_corpus_to_collection; settings.CHUNKING_STRATEGY='title_path'; print(ingest_corpus_to_collection('eval/testsets/sample/corpus', 'ragmate_docs'))"
```
