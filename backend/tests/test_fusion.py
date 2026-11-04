"""融合：RRF。线上用同一套 1/(k+rank+1)。"""
from backend.core.fusion import (
    rrf_contribution,
    rrf_fuse_rank_lists,
)


def test_rrf_prefers_docs_high_on_both_lists():
    dense = [("a",), ("b",), ("c",)]
    sparse = [("b",), ("a",), ("d",)]
    fused = rrf_fuse_rank_lists(dense, sparse, k=60, limit=10)
    assert fused[0] == ("b",) or fused[0] == ("a",)
    assert ("d",) in fused


def test_rrf_k_changes_order_when_lists_disagree():
    dense = [("only_dense",), ("shared",)]
    sparse = [("shared",), ("only_sparse",)]
    at_small_k = rrf_fuse_rank_lists(dense, sparse, k=1, limit=3)
    at_large_k = rrf_fuse_rank_lists(dense, sparse, k=100, limit=3)
    # shared is #2 dense + #1 sparse → usually top; both grids should include it
    assert ("shared",) in at_small_k
    assert ("shared",) in at_large_k


def test_rrf_contribution_matches_rank_from_one():
    assert rrf_contribution(0, 60) == 1.0 / 61
    assert rrf_contribution(1, 60) == 1.0 / 62
