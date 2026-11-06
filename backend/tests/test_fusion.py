"""融合：RRF。线上用同一套 1/(k+rank+1)。"""
from backend.core.fusion import rrf_contribution


def test_rrf_contribution_matches_rank_from_one():
    assert rrf_contribution(0, 60) == 1.0 / 61
    assert rrf_contribution(1, 60) == 1.0 / 62
