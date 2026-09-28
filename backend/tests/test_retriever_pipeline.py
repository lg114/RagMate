"""测试 retriever.py 的检索管线组件。"""
import pytest

from backend.core.retriever import calculate_confidence, reset_retrieval_metrics, get_retrieval_metrics


class TestCalculateConfidence:
    """测试 calculate_confidence 函数。"""

    def test_empty_metrics(self):
        """空指标列表应返回 None。"""
        assert calculate_confidence([]) is None

    def test_high_confidence(self):
        """高分且多结果应为高置信。"""
        metrics = [
            {"top_score": 0.8, "returned": 5},
            {"top_score": 0.7, "returned": 3},
        ]
        result = calculate_confidence(metrics)
        assert result is not None
        assert result["level"] == "high"
        assert result["score"] == 0.8
        assert result["chunks"] == 8

    def test_medium_confidence(self):
        """中等分且有一些结果应为中置信。"""
        metrics = [
            {"top_score": 0.5, "returned": 2},
            {"top_score": 0.4, "returned": 1},
        ]
        result = calculate_confidence(metrics)
        assert result is not None
        assert result["level"] == "medium"
        assert result["score"] == 0.5
        assert result["chunks"] == 3

    def test_low_confidence_low_score(self):
        """低分应为低置信。"""
        metrics = [
            {"top_score": 0.3, "returned": 2},
        ]
        result = calculate_confidence(metrics)
        assert result is not None
        assert result["level"] == "low"
        assert result["score"] == 0.3

    def test_low_confidence_few_results(self):
        """高分但结果太少应为中置信（非高置信）。"""
        metrics = [
            {"top_score": 0.8, "returned": 1},
        ]
        result = calculate_confidence(metrics)
        assert result is not None
        # 高置信需要 score >= 0.7 且 returned >= 3
        # 这里 returned=1，所以是中置信
        assert result["level"] == "medium"

    def test_low_confidence_very_few(self):
        """结果极少应为低置信。"""
        metrics = [
            {"top_score": 0.6, "returned": 0},
        ]
        result = calculate_confidence(metrics)
        assert result is not None
        assert result["level"] == "low"

    def test_best_score_used(self):
        """应使用所有指标中的最高分。"""
        metrics = [
            {"top_score": 0.3, "returned": 2},
            {"top_score": 0.9, "returned": 4},
            {"top_score": 0.5, "returned": 1},
        ]
        result = calculate_confidence(metrics)
        assert result is not None
        assert result["score"] == 0.9
        assert result["level"] == "high"

    def test_total_chunks_summed(self):
        """总 chunk 数应为所有指标的 returned 之和。"""
        metrics = [
            {"top_score": 0.6, "returned": 3},
            {"top_score": 0.5, "returned": 2},
            {"top_score": 0.4, "returned": 1},
        ]
        result = calculate_confidence(metrics)
        assert result is not None
        assert result["chunks"] == 6


class TestRetrievalMetrics:
    """测试检索指标的线程本地存储。"""

    def test_reset_metrics(self):
        """reset_retrieval_metrics 应清空当前线程的指标。"""
        from backend.core.retriever import _metrics
        _metrics.last = {"test": "data"}
        reset_retrieval_metrics()
        assert get_retrieval_metrics() is None

    def test_get_metrics_default(self):
        """新线程的指标应为 None。"""
        reset_retrieval_metrics()
        assert get_retrieval_metrics() is None
