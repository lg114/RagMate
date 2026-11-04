"""稠密/稀疏结果融合：RRF。

生产路径在 pgvector 上分别取稠密、稀疏的内积名次，再用 RRF 合并（k=60）。
加权融合已删除，不再是可选项。
"""
from __future__ import annotations


def rrf_contribution(rank: int, k: int) -> float:
    """0-based 名次 → 1/(k+rank+1)，与名次从 1 计的标准 RRF 一致。"""
    return 1.0 / (k + rank + 1)


def rrf_fuse_rank_lists(
    dense_ranked: list[tuple],
    sparse_ranked: list[tuple],
    *,
    k: int = 60,
    limit: int = 50,
) -> list[tuple]:
    """用两边的名次做 RRF，返回融合后的 key 列表（高分在前）。

    dense_ranked / sparse_ranked：已按检索分排序的 chunk key 序列。
    只在一侧出现的文档只累加该侧贡献（标准 RRF）。
    """
    scores: dict[tuple, float] = {}
    for rank, key in enumerate(dense_ranked):
        scores[key] = scores.get(key, 0.0) + rrf_contribution(rank, k)
    for rank, key in enumerate(sparse_ranked):
        scores[key] = scores.get(key, 0.0) + rrf_contribution(rank, k)
    ordered = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return ordered[:limit]


