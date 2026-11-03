# Two-layer delivery scoring (child rank vs production filter)

Checkpoint: `C:\Users\19097\Documents\GitHub\RagMate\eval\reports\baseline\retrieval-baseline-98q-titlepath-rerun.retrieval.json`
Collection: `ragmate_docs`, CHUNK_SIZE=1000, strategy=title_path
Delivery: FINAL_CONTEXT_K=15, threshold=0.3, MAX_PER_SOURCE=4, expand=True

## Parent vs child text

- chunks seen in checkpoint: 67
- parent_text == child text: 67 (1.0)
- median/mean len(parent)/len(child): 1.0 / 1.0

Parent expansion is a **no-op** at this CHUNK_SIZE — sections rarely split, so parent_text equals the child. Re-ranking on parent text cannot diverge here. See `retrieval-baseline-98q-parent-rerank.md`（c300）for the parent-scoring dilution check.

## Metrics

| layer | hit@5 / recall@5 / nDCG@5 | hit@10 / recall@10 / nDCG@10 | MRR | failure@20 |
|---|---|---|---|---|
| child (reranked) | 1.00 / 1.00 / 0.99 | 1.00 / 1.00 / 0.99 | 0.99 | 0.00 |
| delivered (filter/dedup) | 1.00 / 1.00 / 0.99 | 1.00 / 1.00 / 0.99 | 0.99 | 0.00 |

Questions that lose ≥1 gold between child rank and delivery: **0**

No answerable question loses a gold chunk that was present in the reranked list — filter/dedup/cutoff did not create a silent gap on this checkpoint.

