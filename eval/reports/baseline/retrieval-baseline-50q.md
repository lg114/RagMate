# Retrieval baseline (deterministic, no LLM judge)

Test set: `C:\Users\19097\Documents\GitHub\RagMate\eval\testsets\sample\testset.json` (50 questions, sha256 `19da47f189c4`), over_fetch=50, repeat=3, deterministic across repeats: True.

Metrics are computed from gold chunk IDs only: hit/recall/MRR/nDCG need no LLM. Unanswerable questions are excluded from the means and counted separately.

| stage | hit@5 / recall@5 / nDCG@5 | hit@10 / recall@10 / nDCG@10 | hit@20 / recall@20 / nDCG@20 | MRR | failure_rate@20 |
|---|---|---|---|---|
| dense | 1.00 / 1.00 / 0.98 | 1.00 / 1.00 / 0.98 | 1.00 / 1.00 / 0.98 | 0.97 | 0.00 |
| sparse | 1.00 / 1.00 / 0.97 | 1.00 / 1.00 / 0.97 | 1.00 / 1.00 / 0.97 | 0.96 | 0.00 |
| fused | 1.00 / 1.00 / 0.98 | 1.00 / 1.00 / 0.98 | 1.00 / 1.00 / 0.98 | 0.98 | 0.00 |
| reranked | 1.00 / 1.00 / 0.99 | 1.00 / 1.00 / 0.99 | 1.00 / 1.00 / 0.99 | 0.99 | 0.00 |

## Failures (recall@20 < 1)

### dense (0 failing)

### sparse (0 failing)

### fused (0 failing)

### reranked (0 failing)
