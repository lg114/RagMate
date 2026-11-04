"""_filter_and_dedup 动态过滤逻辑测试。

过滤常量已冻结为代码里的字面值（动态阈值 0.5、高分比 0.6、同源上限 4/
最少 2、断崖 0.15、主导度 0.9×1.5），不再是可调配置。本文件用输入分数
构造「断崖/主导度触发与否」的情形来隔离各行为，不再改配置。
"""
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
    def test_threshold_floor(self):
        """当 top_score * 0.5 < threshold 时，使用 threshold 作为保底。"""
        candidates = [_c("a.pdf", 0.4), _c("b.pdf", 0.35), _c("c.pdf", 0.2)]
        # effective_threshold = max(0.4 * 0.5, 0.3) = max(0.2, 0.3) = 0.3
        result = _filter_and_dedup(candidates, 0.3, 10)
        scores = [c["score"] for c in result]
        assert all(s >= 0.3 for s in scores)
        assert 0.2 not in scores  # c.pdf 被过滤

    def test_dynamic_ratio_raises_floor(self):
        """当 top_score * 0.5 > threshold 时，使用动态值。"""
        candidates = [_c("a.pdf", 0.9), _c("b.pdf", 0.8), _c("c.pdf", 0.3)]
        # effective_threshold = max(0.9 * 0.5, 0.1) = 0.45；0.9→0.8 间距 0.1，无断崖
        result = _filter_and_dedup(candidates, 0.1, 10)
        scores = [c["score"] for c in result]
        assert 0.9 in scores
        assert 0.8 in scores
        assert 0.3 not in scores  # 低于 0.45，被过滤


# ── 源去重 ────────────────────────────────────────────────────────────────────


class TestSourceDedup:
    def test_per_source_limit(self):
        """同源 chunk 数受上限 4 限制。"""
        candidates = [
            _c("a.pdf", 0.9),
            _c("a.pdf", 0.89),
            _c("a.pdf", 0.88),
            _c("a.pdf", 0.87),
            _c("a.pdf", 0.5),
            _c("a.pdf", 0.49),
        ]
        # 高分（≥0.9*0.6=0.54）4 个 → limit=4；主导度 1.0 放宽到 6，不 binding
        # 0.87→0.5 间距 0.37 触发断崖，同样停在 4 个
        result = _filter_and_dedup(candidates, 0.1, 20)
        assert len(result) == 4

    def test_dominance_boost(self):
        """主导来源的上限被放宽到 4 以上。"""
        candidates = [_c("a.pdf", 0.9 - i * 0.01) for i in range(5)]
        # 唯一来源，主导度 1.0 ≥ 0.9：limit=max(5, 2)=5，放宽上限 min(7, 20)=7 → 5 个全留
        result = _filter_and_dedup(candidates, 0.1, 20)
        assert len(result) == 5

    def test_non_dominant_source_capped(self):
        """非主导来源使用上限 4。"""
        candidates = [
            _c("x.pdf", 0.95),
            _c("a.pdf", 0.85),
            _c("a.pdf", 0.84),
            _c("a.pdf", 0.83),
            _c("a.pdf", 0.82),
            _c("a.pdf", 0.81),
            _c("b.pdf", 0.8),
            _c("b.pdf", 0.79),
        ]
        # a.pdf 主导度 0.85/0.95 ≈ 0.895 < 0.9，非主导 → 上限 4
        result = _filter_and_dedup(candidates, 0.1, 20)
        sources = [c["source"] for c in result]
        assert sources.count("a.pdf") == 4
        assert sources.count("x.pdf") == 1


# ── 断崖检测 ──────────────────────────────────────────────────────────────────


class TestScoreGap:
    def test_gap_truncates(self):
        """分数断崖处截断。"""
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

    def test_smooth_scores_take_k(self):
        """分数平滑递减时取满 k 个。"""
        candidates = [_c(f"{i}.pdf", 0.9 - i * 0.02) for i in range(8)]
        result = _filter_and_dedup(candidates, 0.1, 5)
        assert len(result) == 5

    def test_k_equals_1(self):
        """k=1 时只返回最高分。"""
        candidates = [_c("a.pdf", 0.9), _c("b.pdf", 0.8), _c("c.pdf", 0.7)]
        result = _filter_and_dedup(candidates, 0.1, 1)
        assert len(result) == 1
        assert result[0]["score"] == 0.9
