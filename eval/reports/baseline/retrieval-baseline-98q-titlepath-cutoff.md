# Final-k cutoff simulation (from retrieval checkpoint)

Checkpoint: `C:\Users\19097\Documents\GitHub\RagMate\eval\reports\baseline\retrieval-baseline-98q-titlepath.retrieval.json`
Questions: 98, final_ks=[5, 10, 15]

Each row truncates that stage's ranked list to final_k (as if `FINAL_CONTEXT_K=final_k`), then scores hit/recall/nDCG/MRR on the truncated list. incomplete = answerable questions with recall < 1.

## final_k = 5

| stage | hit | recall | nDCG | MRR | incomplete |
|---|---|---|---|---|---|
| dense | 1.0000 | 0.9927 | 0.9706 | 0.9670 | 1 |
| sparse | 1.0000 | 1.0000 | 0.9755 | 0.9670 | 0 |
| fused | 1.0000 | 0.9963 | 0.9839 | 0.9835 | 1 |
| reranked | 1.0000 | 1.0000 | 0.9917 | 0.9872 | 0 |

### Incomplete under `reranked` @ 5 (0)

_none_

### multi-hop recall by stage @ 5

| stage | multi-hop recall | multi-hop nDCG |
|---|---|---|
| dense | 0.9333 | 0.9173 |
| sparse | 1.0000 | 0.9552 |
| fused | 0.9667 | 0.9355 |
| reranked | 1.0000 | 0.9618 |

## final_k = 10

| stage | hit | recall | nDCG | MRR | incomplete |
|---|---|---|---|---|---|
| dense | 1.0000 | 1.0000 | 0.9738 | 0.9670 | 0 |
| sparse | 1.0000 | 1.0000 | 0.9755 | 0.9670 | 0 |
| fused | 1.0000 | 1.0000 | 0.9855 | 0.9835 | 0 |
| reranked | 1.0000 | 1.0000 | 0.9917 | 0.9872 | 0 |

### Incomplete under `reranked` @ 10 (0)

_none_

### multi-hop recall by stage @ 10

| stage | multi-hop recall | multi-hop nDCG |
|---|---|---|
| dense | 1.0000 | 0.9463 |
| sparse | 1.0000 | 0.9552 |
| fused | 1.0000 | 0.9503 |
| reranked | 1.0000 | 0.9618 |

## final_k = 15

| stage | hit | recall | nDCG | MRR | incomplete |
|---|---|---|---|---|---|
| dense | 1.0000 | 1.0000 | 0.9738 | 0.9670 | 0 |
| sparse | 1.0000 | 1.0000 | 0.9755 | 0.9670 | 0 |
| fused | 1.0000 | 1.0000 | 0.9855 | 0.9835 | 0 |
| reranked | 1.0000 | 1.0000 | 0.9917 | 0.9872 | 0 |

### Incomplete under `reranked` @ 15 (0)

_none_

### multi-hop recall by stage @ 15

| stage | multi-hop recall | multi-hop nDCG |
|---|---|---|
| dense | 1.0000 | 0.9463 |
| sparse | 1.0000 | 0.9552 |
| fused | 1.0000 | 0.9503 |
| reranked | 1.0000 | 0.9618 |
