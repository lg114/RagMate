"""稠密/稀疏结果融合：RRF 与加权。

生产路径在 pgvector 上分别取稠密、稀疏的内积名次，再用 RRF 合并（默认 k=60）。
`weighted_fuse_score_lists` 是离线扫参用的归一化加权，不是线上默认。
线上 `FUSION_MODE=weighted` 对原始内积加权，不走这里的 min-max。
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


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def weighted_fuse_score_lists(
    dense_items: list[tuple[tuple, float]],
    sparse_items: list[tuple[tuple, float]],
    *,
    dense_weight: float = 0.5,
    limit: int = 50,
) -> list[tuple]:
    """按归一化原始分加权融合。items = [(key, score), ...] 已按分排序亦可。"""
    w_d = min(1.0, max(0.0, float(dense_weight)))
    w_s = 1.0 - w_d
    d_keys = [k for k, _ in dense_items]
    s_keys = [k for k, _ in sparse_items]
    d_norm = dict(zip(d_keys, _minmax([s for _, s in dense_items])))
    s_norm = dict(zip(s_keys, _minmax([s for _, s in sparse_items])))
    scores: dict[tuple, float] = {}
    for key in set(d_keys) | set(s_keys):
        scores[key] = w_d * d_norm.get(key, 0.0) + w_s * s_norm.get(key, 0.0)
    ordered = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return ordered[:limit]
