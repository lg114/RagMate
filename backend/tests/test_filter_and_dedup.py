"""_filter_and_dedup 动态过滤逻辑测试。"""
import pytest

from backend.core.retriever import _filter_and_dedup


def _c(source: str, score: float) -> dict:
    """构造候选 chunk。"""
    return {"source": source, "score": score, "text": f"chunk from {source}"}


# ── 边界情况 ──────────────────────────────────────────────────────────────────


class TestEmpty:
    def test_empty_list(self):
        assert _filter_and_dedup([], 0.3, 10) == []

    def test_all_below_threshold(self):
        # threshold=0.9，所有 chunk 都低于
        candidates = [_c("a.pdf", 0.1), _c("b.pdf", 0.2)]
        assert _filter_and_dedup(candidates, 0.9, 10) == []


# ── 动态阈值 ──────────────────────────────────────────────────────────────────


class TestDynamicThreshold:
    def test_threshold_floor(self, override_settings):
        """当 top_score * ratio < threshold 时，使用 threshold 作为保底。"""
        override_settings(DYNAMIC_THRESHOLD_RATIO=0.5)
        candidates = [_c("a.pdf", 0.4), _c("b.pdf", 0.35), _c("c.pdf", 0.2)]
        # effective_threshold = max(0.4 * 0.5, 0.3) = max(0.2, 0.3) = 0.3
        result = _filter_and_dedup(candidates, 0.3, 10)
        scores = [c["score"] for c in result]
        assert all(s >= 0.3 for s in scores)
        assert 0.2 not in scores  # c.pdf 被过滤

    def test_dynamic_ratio_raises_floor(self, override_settings):
        """当 top_score * ratio > threshold 时，使用动态值。"""
        override_settings(
            DYNAMIC_THRESHOLD_RATIO=0.5,
            SCORE_GAP_THRESHOLD=0.99,  # 禁用断崖，只测阈值
            SOURCE_DOMINANCE_THRESHOLD=2.0,  # 禁用主导度放宽
        )
        candidates = [_c("a.pdf", 0.9), _c("b.pdf", 0.5), _c("c.pdf", 0.3)]
        # effective_threshold = max(0.9 * 0.5, 0.1) = 0.45
        result = _filter_and_dedup(candidates, 0.1, 10)
        scores = [c["score"] for c in result]
        assert 0.9 in scores
        assert 0.5 in scores
        assert 0.3 not in scores  # 低于 0.45，被过滤


# ── 源去重 ────────────────────────────────────────────────────────────────────


class TestSourceDedup:
    def test_per_source_limit(self, override_settings):
        """同源 chunk 数受 MAX_PER_SOURCE 限制（非主导来源）。"""
        override_settings(
            MAX_PER_SOURCE=3,
            MIN_PER_SOURCE=1,
            SOURCE_DOMINANCE_THRESHOLD=2.0,  # 禁用主导度放宽（阈值不可达）
            SCORE_GAP_THRESHOLD=0.99,  # 禁用断崖
        )
        candidates = [_c("a.pdf", 0.9 - i * 0.01) for i in range(6)]
        result = _filter_and_dedup(candidates, 0.1, 20)
        assert len(result) == 3

    def test_dominance_boost(self, override_settings):
        """主导来源的上限被放宽。"""
        override_settings(
            MAX_PER_SOURCE=2,
            MIN_PER_SOURCE=1,
            SOURCE_DOMINANCE_THRESHOLD=0.8,
            SOURCE_DOMINANCE_BOOST=2.0,
            HIGH_SCORE_RATIO=0.97,  # 只有前 3 个算高分
            SCORE_GAP_THRESHOLD=0.99,
        )
        # a.pdf 是唯一来源，dominance = 1.0 >= 0.8
        candidates = [_c("a.pdf", 0.9 - i * 0.01) for i in range(5)]
        result = _filter_and_dedup(candidates, 0.1, 20)
        # high_count=3 (>= 0.9*0.97=0.873), limit=max(3,1)=3
        # soft_limit=min(int(3*2.0),20)=6, limit=min(3,6)=3
        assert len(result) == 3

    def test_non_dominant_source_capped(self, override_settings):
        """非主导来源使用 MAX_PER_SOURCE 硬上限。"""
        override_settings(
            MAX_PER_SOURCE=2,
            MIN_PER_SOURCE=1,
            SOURCE_DOMINANCE_THRESHOLD=2.0,  # 禁用主导度放宽
            SCORE_GAP_THRESHOLD=0.99,  # 禁用断崖
        )
        candidates = [
            _c("a.pdf", 0.9),
            _c("a.pdf", 0.89),
            _c("a.pdf", 0.88),
            _c("b.pdf", 0.87),
            _c("b.pdf", 0.86),
        ]
        result = _filter_and_dedup(candidates, 0.1, 20)
        sources = [c["source"] for c in result]
        assert sources.count("a.pdf") <= 2
        assert sources.count("b.pdf") <= 2


# ── 断崖检测 ──────────────────────────────────────────────────────────────────


class TestScoreGap:
    def test_gap_truncates(self, override_settings):
        """分数断崖处截断。"""
        override_settings(
            SCORE_GAP_THRESHOLD=0.15,
            MAX_PER_SOURCE=10,
            MIN_PER_SOURCE=1,
            SOURCE_DOMINANCE_THRESHOLD=0.99,
        )
        candidates = [
            _c("a.pdf", 0.9),
            _c("b.pdf", 0.88),
            _c("c.pdf", 0.85),
            _c("d.pdf", 0.6),  # gap = 0.25 > 0.15，断崖
            _c("e.pdf", 0.55),
        ]
        result = _filter_and_dedup(candidates, 0.1, 10)
        scores = [c["score"] for c in result]
        assert 0.6 not in scores  # 断崖后被切掉
        assert len(result) == 3

    def test_smooth_scores_take_k(self, override_settings):
        """分数平滑递减时取满 k 个。"""
        override_settings(
            SCORE_GAP_THRESHOLD=0.15,
            MAX_PER_SOURCE=10,
            MIN_PER_SOURCE=1,
            SOURCE_DOMINANCE_THRESHOLD=0.99,
        )
        candidates = [_c(f"{i}.pdf", 0.9 - i * 0.02) for i in range(8)]
        result = _filter_and_dedup(candidates, 0.1, 5)
        assert len(result) == 5

    def test_k_equals_1(self, override_settings):
        """k=1 时只返回最高分。"""
        override_settings(
            SCORE_GAP_THRESHOLD=0.99,  # 禁用断崖
            MAX_PER_SOURCE=10,
            MIN_PER_SOURCE=1,
            SOURCE_DOMINANCE_THRESHOLD=0.99,
        )
        candidates = [_c("a.pdf", 0.9), _c("b.pdf", 0.8), _c("c.pdf", 0.7)]
        result = _filter_and_dedup(candidates, 0.1, 1)
        assert len(result) == 1
        assert result[0]["score"] == 0.9
