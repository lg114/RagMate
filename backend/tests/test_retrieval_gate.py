"""检索质量门禁：门槛来自 130 题 fused 基线，词法重放低于门槛必须失败。"""
import json

import pytest

from backend.eval.lexical_retrieval import LexicalIndex, tokenize
from backend.eval.retrieval_gate import (
    ALLOWANCE,
    GATED_METRICS,
    check_baseline_consistency,
    check_question_alignment,
    compare_to_floors,
    floor_metric,
    floors_from_published,
    load_baseline_report,
    published_summary,
    run_retrieval_gate,
)

# 数字抄自 retrieval-baseline-130q-jina-rerun.json 的 stages.fused.summary。
# 换基线报告时要连这张表和 eval/README.md 一起改。
PINNED_FUSED = {
    "recall@5": 0.9669,
    "ndcg@5": 0.9505,
    "recall@10": 0.9945,
    "ndcg@10": 0.9606,
    "recall@20": 1.0,
    "ndcg@20": 0.9625,
}


def test_tokenize_keeps_latin_and_cjk_bigrams():
    tokens = tokenize("访问 localhost:8000，端口映射")
    assert "L:localhost:8000" in tokens
    assert "C:端口" in tokens
    assert "C:口映" in tokens
    assert "C:映射" in tokens


def test_bm25_ranks_matching_chunk_first():
    chunks = [
        {"key": ("a.md", None, 0), "text": "完全无关的园艺记录，只谈浇水。"},
        {"key": ("b.md", None, 1), "text": "work_mem 默认 4MB，排序超出后走 external merge。"},
    ]
    ranked = LexicalIndex(chunks).rank("work_mem 默认是多少？")
    assert ranked[0] == ("b.md", None, 1)


def test_floor_is_baseline_minus_allowance_at_4_decimals():
    assert floor_metric(0.9669) == pytest.approx(0.9519)
    assert floor_metric(1.0) == pytest.approx(0.985)
    floors = floors_from_published(PINNED_FUSED)
    assert floors == {
        "recall@5": 0.9519,
        "ndcg@5": 0.9355,
        "recall@10": 0.9795,
        "ndcg@10": 0.9456,
        "recall@20": 0.985,
        "ndcg@20": 0.9475,
    }


def test_compare_passes_on_floor_and_fails_just_below():
    floors = floors_from_published(PINNED_FUSED)
    assert compare_to_floors(floors, floors) == []
    dropped = dict(floors)
    dropped["recall@10"] = round(floors["recall@10"] - 0.0001, 4)
    failures = compare_to_floors(dropped, floors)
    assert len(failures) == 1
    assert failures[0].startswith("recall@10 ")


def test_committed_baseline_matches_pinned_fused_numbers():
    report = load_baseline_report()
    summary = published_summary(report)
    for name, expected in PINNED_FUSED.items():
        assert summary[name] == expected
    assert check_baseline_consistency(report) == []
    assert set(GATED_METRICS) <= set(summary)


def test_question_alignment_flags_an_edited_question():
    report = {
        "stages": {
            "fused": {
                "per_question": [
                    {"user_input": "原题", "unanswerable": False, "scores": {}},
                ],
            },
        },
    }
    errors = check_question_alignment([{"user_input": "改过的题"}], report)
    assert errors
    assert "regenerate the baseline" in errors[0]


def test_sample_corpus_meets_pinned_floors():
    result = run_retrieval_gate()
    assert result.failures == [], result.render()
    assert result.ok
    assert result.n_questions == 130
    assert result.n_answerable == 121
    for name, floor in result.floors.items():
        assert result.summary[name] >= floor
        assert result.published[name] - result.summary[name] <= ALLOWANCE + 1e-9


def test_reversed_ranking_fails_the_gate():
    def reverse_rank(_query, index):
        return [chunk["key"] for chunk in reversed(index.chunks)]

    result = run_retrieval_gate(ranker=reverse_rank)
    assert not result.ok
    assert any(item.startswith("recall@") or item.startswith("ndcg@") for item in result.failures)


def test_consistency_check_catches_a_tampered_summary(tmp_path):
    report = load_baseline_report()
    report["stages"]["fused"]["summary"]["recall@5"] = 0.5
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    errors = check_baseline_consistency(json.loads(path.read_text(encoding="utf-8")))
    assert any("recall@5" in item for item in errors)
