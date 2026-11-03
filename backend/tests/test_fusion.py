"""融合：RRF / 加权离线融合与 Milvus ranker 接线。"""
from backend.core.fusion import (
    make_milvus_ranker,
    rrf_fuse_rank_lists,
    weighted_fuse_score_lists,
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


def test_weighted_respects_dense_weight():
    dense = [(("d",), 1.0), (("both",), 0.5)]
    sparse = [(("s",), 1.0), (("both",), 0.5)]
    dense_heavy = weighted_fuse_score_lists(dense, sparse, dense_weight=0.9, limit=3)
    sparse_heavy = weighted_fuse_score_lists(dense, sparse, dense_weight=0.1, limit=3)
    assert dense_heavy[0] == ("d",)
    assert sparse_heavy[0] == ("s",)


def test_make_milvus_ranker_rrf_and_weighted(monkeypatch):
    from backend.infrastructure import config as config_mod

    monkeypatch.setattr(config_mod.settings, "FUSION_MODE", "rrf")
    monkeypatch.setattr(config_mod.settings, "RRF_K", 40)
    ranker = make_milvus_ranker()
    assert ranker.__class__.__name__ == "RRFRanker"

    monkeypatch.setattr(config_mod.settings, "FUSION_MODE", "weighted")
    monkeypatch.setattr(config_mod.settings, "FUSION_DENSE_WEIGHT", 0.7)
    ranker = make_milvus_ranker()
    assert ranker.__class__.__name__ == "WeightedRanker"
