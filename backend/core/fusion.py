"""稠密/稀疏结果融合：RRF。

生产路径在 pgvector 上分别取稠密、稀疏的内积名次，再用 RRF 合并（k=60）。
加权融合已删除，不再是可选项。
"""
from __future__ import annotations


def rrf_contribution(rank: int, k: int) -> float:
    """0-based 名次 → 1/(k+rank+1)，与名次从 1 计的标准 RRF 一致。"""
    return 1.0 / (k + rank + 1)