# Parent vs child rerank (c300)

Collection `ragmate_docs__c300`, CHUNK_SIZE=300, strategy=title_path.  
Chunks=140, parent==child=24 (17%), median len(parent)/len(child)=1.49 —— 真分层成立。

| metric | value |
|---|---|
| mean Kendall τ | 0.7653 |
| mean overlap@5 | 0.7276 |
| mean overlap@15 | 0.7264 |
| top1 flip rate | 0.1429 |

## 怎么读

- 同一 fused 候选，用 **parent 正文**重排 vs **child 正文**重排：约 14% 题 top1 翻面，top5 Jaccard ≈ 0.73。  
  父段噪声**会**改序——所以生产必须继续「child 打分 → 再展开 parent」，不能改成对 parent 打分。
- 主索引 `CHUNK_SIZE=1000` 下 parent==child（见 delivery 报告），这层差异测不出来；c300 只作诊断集合，**不改生产默认切块**。
- 与 delivery 两层（filter/dedup）互补：那边查的是「排序有 gold、交付丢 gold」；这边查的是「若误用 parent 打分会否稀释」。

复现：`python -m backend.eval.run_parent_rerank_experiment`
