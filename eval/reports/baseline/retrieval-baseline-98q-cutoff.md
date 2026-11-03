# Final-k cutoff simulation (from retrieval checkpoint)

Checkpoint: `C:\Users\19097\Documents\GitHub\RagMate\eval\reports\baseline\retrieval-baseline-98q.retrieval.json`
Questions: 98, final_ks=[5, 10, 15]

Each row truncates that stage's ranked list to final_k (as if `FINAL_CONTEXT_K=final_k`), then scores hit/recall/nDCG/MRR on the truncated list. incomplete = answerable questions with recall < 1.

## final_k = 5

| stage | hit | recall | nDCG | MRR | incomplete |
|---|---|---|---|---|---|
| dense | 1.0000 | 0.9927 | 0.9671 | 0.9643 | 1 |
| sparse | 1.0000 | 0.9872 | 0.9529 | 0.9460 | 2 |
| fused | 1.0000 | 0.9872 | 0.9769 | 0.9802 | 2 |
| reranked | 1.0000 | 0.9963 | 0.9904 | 0.9890 | 1 |

### Incomplete under `reranked` @ 5 (1)
- [67] (multi-hop) recall=0.67: Compose 里 db 与 redis 的重启策略分别是什么？和 web/worker 有何不同？

### multi-hop recall by stage @ 5

| stage | multi-hop recall | multi-hop nDCG |
|---|---|---|
| dense | 0.9333 | 0.8930 |
| sparse | 0.8833 | 0.8735 |
| fused | 0.8833 | 0.8714 |
| reranked | 0.9667 | 0.9498 |

## final_k = 10

| stage | hit | recall | nDCG | MRR | incomplete |
|---|---|---|---|---|---|
| dense | 1.0000 | 0.9927 | 0.9671 | 0.9643 | 1 |
| sparse | 1.0000 | 0.9945 | 0.9565 | 0.9460 | 1 |
| fused | 1.0000 | 1.0000 | 0.9824 | 0.9802 | 0 |
| reranked | 1.0000 | 1.0000 | 0.9923 | 0.9890 | 0 |

### Incomplete under `reranked` @ 10 (0)

_none_

### multi-hop recall by stage @ 10

| stage | multi-hop recall | multi-hop nDCG |
|---|---|---|
| dense | 0.9333 | 0.8930 |
| sparse | 0.9500 | 0.9059 |
| fused | 1.0000 | 0.9214 |
| reranked | 1.0000 | 0.9665 |

## final_k = 15

| stage | hit | recall | nDCG | MRR | incomplete |
|---|---|---|---|---|---|
| dense | 1.0000 | 1.0000 | 0.9698 | 0.9643 | 0 |
| sparse | 1.0000 | 0.9945 | 0.9565 | 0.9460 | 1 |
| fused | 1.0000 | 1.0000 | 0.9824 | 0.9802 | 0 |
| reranked | 1.0000 | 1.0000 | 0.9923 | 0.9890 | 0 |

### Incomplete under `reranked` @ 15 (0)

_none_

### multi-hop recall by stage @ 15

| stage | multi-hop recall | multi-hop nDCG |
|---|---|---|
| dense | 1.0000 | 0.9181 |
| sparse | 0.9500 | 0.9059 |
| fused | 1.0000 | 0.9214 |
| reranked | 1.0000 | 0.9665 |
