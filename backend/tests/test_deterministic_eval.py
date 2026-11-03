"""确定性检索指标与实验 manifest 的测试：纯函数，不碰 Milvus/LLM。"""
import json

import pytest

from backend.eval.deterministic_metrics import (
    evaluate_questions,
    metric_spread,
    normalize_key,
    score_ranking,
)
from backend.eval.manifest import build_manifest, diff_manifests, sha256_file

GOLD = [("01-a.md", None, 2), ("01-a.md", None, 5)]


def test_normalize_key_accepts_dict_and_tuple():
    assert normalize_key({"source": "a.md", "chunk_index": 3}) == ("a.md", None, 3)
    assert normalize_key(("a.md", 1, 3)) == ("a.md", 1, 3)


def test_score_ranking_perfect_top():
    ranked = [("01-a.md", None, 2), ("01-a.md", None, 5), ("01-a.md", None, 7)]
    scores = score_ranking(ranked, GOLD, ks=(5, 10, 20))
    assert scores["hit@5"] == 1.0
    assert scores["recall@5"] == 1.0
    assert scores["precision@5"] == 2 / 5
    assert scores["ndcg@5"] == 1.0
    assert scores["mrr"] == 1.0


def test_score_ranking_gold_beyond_k_counts_only_within_k():
    ranked = [("x.md", None, i) for i in range(10)] + [("01-a.md", None, 2)]
    scores = score_ranking(ranked, GOLD, ks=(5, 10, 20))
    assert scores["hit@5"] == 0.0
    assert scores["recall@5"] == 0.0
    assert scores["hit@20"] == 1.0
    assert scores["recall@20"] == 0.5
    assert scores["mrr"] == pytest.approx(1 / 11)


def test_score_ranking_empty_ranked_is_all_zero():
    scores = score_ranking([], GOLD, ks=(5,))
    assert scores == {"hit@5": 0.0, "recall@5": 0.0, "precision@5": 0.0, "ndcg@5": 0.0, "mrr": 0.0}


def test_score_ranking_partial_ndcg_is_between_zero_and_one():
    ranked = [("x.md", None, 0), ("01-a.md", None, 2), ("x.md", None, 1)]
    scores = score_ranking(ranked, GOLD, ks=(5,))
    assert 0.0 < scores["ndcg@5"] < 1.0
    assert scores["recall@5"] == 0.5


def test_evaluate_questions_excludes_unanswerable_from_means():
    items = [
        {"ranked": [("01-a.md", None, 2)], "gold": GOLD[:1],
         "source_doc": "01-a.md", "question_type": "factoid", "user_input": "q1"},
        {"ranked": [("x.md", None, 0)], "gold": [],
         "source_doc": "02-b.md", "question_type": "unanswerable", "user_input": "q2"},
    ]
    result = evaluate_questions(items, ks=(5,))
    assert result["n_questions"] == 2
    assert result["n_answerable"] == 1
    assert result["n_unanswerable"] == 1
    assert result["summary"]["hit@5"] == 1.0
    assert "unanswerable" not in result["by_question_type"]


def test_evaluate_questions_breakdowns_and_failure_rate():
    items = [
        {"ranked": [("01-a.md", None, 2)], "gold": GOLD[:1],
         "source_doc": "01-a.md", "question_type": "factoid", "user_input": "q1"},
        {"ranked": [("x.md", None, 0)], "gold": GOLD[:1],
         "source_doc": "01-a.md", "question_type": "factoid", "user_input": "q2"},
    ]
    result = evaluate_questions(items, ks=(20,))
    assert result["summary"]["recall@20"] == 0.5
    assert result["summary"]["failure_rate@20"] == 0.5
    assert result["by_source_doc"]["01-a.md"]["hit@20"] == 0.5


def test_metric_spread_zero_for_identical_runs():
    summaries = [{"hit@5": 0.8, "mrr": 0.7}, {"hit@5": 0.8, "mrr": 0.7}]
    assert metric_spread(summaries) == {"hit@5": 0.0, "mrr": 0.0}


def test_metric_spread_reports_max_min_gap():
    summaries = [{"hit@5": 0.8}, {"hit@5": 0.84}, {"hit@5": 0.78}]
    assert metric_spread(summaries) == {"hit@5": 0.06}


def test_manifest_records_testset_hash_and_settings(tmp_path, override_settings):
    testset = tmp_path / "testset.json"
    testset.write_text(json.dumps([{"user_input": "q"}]), encoding="utf-8")
    override_settings(RERANK_CANDIDATES=30, HYBRID_SEARCH_ENABLED=True)
    manifest = build_manifest(
        testset_path=testset, stages=["fused"], over_fetch=50,
        ks=[5, 10, 20], repeat=1, n_questions=1, git_commit="abc",
    )
    assert manifest["testset"]["sha256"] == sha256_file(testset)
    assert manifest["testset"]["n_questions"] == 1
    assert manifest["retrieval_config"]["RERANK_CANDIDATES"] == 30
    assert manifest["harness"] == {"stages": ["fused"], "over_fetch": 50, "ks": [5, 10, 20], "repeat": 1}
    assert manifest["git_commit"] == "abc"


def test_diff_manifests_flags_changed_settings_and_testset():
    old = {
        "retrieval_config": {"RERANK_CANDIDATES": 30, "FINAL_CONTEXT_K": 15},
        "harness": {"stages": ["fused"]},
        "testset": {"sha256": "aaa"},
    }
    new = {
        "retrieval_config": {"RERANK_CANDIDATES": 50, "FINAL_CONTEXT_K": 15},
        "harness": {"stages": ["fused"]},
        "testset": {"sha256": "bbb"},
    }
    changes = diff_manifests(old, new)
    assert changes["retrieval_config.RERANK_CANDIDATES"] == {"old": 30, "new": 50}
    assert "retrieval_config.FINAL_CONTEXT_K" not in changes
    assert changes["testset_changed"] == {"old": "aaa", "new": "bbb"}
