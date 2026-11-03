"""确定性检索评测的测试：分阶段执行、gold 校验、报告落盘（假 Milvus）。"""
import json

import pytest

from backend.eval.retrieval_eval import (
    STAGES,
    build_retrieval_report,
    load_golden_testset,
    run_retrieval_eval,
    run_stages,
    validate_gold_schema,
    verify_gold_in_inventory,
    write_retrieval_checkpoint,
    write_retrieval_markdown,
    _extract_stage_records,
)
from backend.eval.utils import resolve_path


def _hit(distance, source, chunk_index, text):
    return {
        "distance": distance,
        "entity": {
            "text": text,
            "parent_text": text,
            "metadata": {"source": source, "page": None, "chunk_index": chunk_index},
        },
    }


class _FakeClient:
    def __init__(self, hits):
        self._hits = hits
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return [self._hits]

    def hybrid_search(self, **kwargs):
        self.calls.append(("hybrid_search", kwargs))
        return [self._hits]


HITS = [
    _hit(0.9, "a.md", 0, "first chunk text"),
    _hit(0.8, "a.md", 1, "second chunk text"),
    _hit(0.7, "b.md", 0, "third chunk text"),
]


def test_extract_stage_records_keeps_triple_rank_score():
    records = _extract_stage_records([HITS])
    assert [(r["rank"], r["source"], r["chunk_index"], r["score"]) for r in records] == [
        (1, "a.md", 0, 0.9), (2, "a.md", 1, 0.8), (3, "b.md", 0, 0.7),
    ]
    assert records[0]["text_preview"] == "first chunk text"


def test_run_stages_dense_sparse_fused_use_same_vectors(monkeypatch):
    encoded = []
    monkeypatch.setattr("backend.eval.retrieval_eval.encode_query",
                        lambda q: encoded.append(q) or ([0.1], "sparse-vec"))
    client = _FakeClient(HITS)
    out = run_stages(client, "q", ["dense", "sparse", "fused"], over_fetch=10)
    assert set(out) == {"dense", "sparse", "fused"}
    assert encoded == ["q"]
    search_calls = [kw for kind, kw in client.calls if kind == "search"]
    assert [kw["anns_field"] for kw in search_calls] == ["dense", "sparse"]
    assert any(kind == "hybrid_search" for kind, _ in client.calls)


def test_search_calls_use_search_params_not_param(monkeypatch):
    """Regression: MilvusClient.search takes `search_params`; a `param=`
    kwarg is forwarded twice by the client and raises TypeError."""
    monkeypatch.setattr("backend.eval.retrieval_eval.encode_query", lambda q: ([0.1], "s"))
    client = _FakeClient(HITS)
    run_stages(client, "q", ["dense", "sparse"], over_fetch=10)
    search_calls = [kw for kind, kw in client.calls if kind == "search"]
    assert len(search_calls) == 2
    for kw in search_calls:
        assert "param" not in kw
        assert kw["search_params"]["metric_type"] == "IP"
    assert search_calls[0]["search_params"] == {"metric_type": "IP", "params": {}}
    assert search_calls[1]["search_params"]["params"]["drop_ratio_search"] == 0.2


def test_do_search_dense_branch_uses_search_params(override_settings):
    """Same latent bug in the production dense-only path (hybrid off):
    it must call client.search with `search_params`, not `param=`."""
    from backend.core.retriever import _do_search

    override_settings(HYBRID_SEARCH_ENABLED=False)
    client = _FakeClient(HITS)
    results = _do_search(client, [0.1], "sparse-vec", over_fetch=10)
    assert results == [HITS]
    assert len(client.calls) == 1
    _, kw = client.calls[0]
    assert "param" not in kw
    assert kw["search_params"] == {"metric_type": "IP", "params": {}}


def test_run_stages_reranked_reorders_by_score_pairs(monkeypatch):
    monkeypatch.setattr("backend.eval.retrieval_eval.encode_query", lambda q: ([0.1], "s"))
    monkeypatch.setattr("backend.eval.retrieval_eval.score_pairs",
                        lambda q, texts: [0.1, 0.9, 0.5])
    client = _FakeClient(HITS)
    out = run_stages(client, "q", ["reranked"], over_fetch=10)
    ranked = out["reranked"]
    assert [r["chunk_index"] for r in ranked] == [1, 0, 0]
    assert [r["source"] for r in ranked] == ["a.md", "b.md", "a.md"]
    assert [r["rank"] for r in ranked] == [1, 2, 3]
    assert ranked[0]["score"] == pytest.approx(0.9)


def test_run_stages_rejects_unknown_stage(monkeypatch):
    monkeypatch.setattr("backend.eval.retrieval_eval.encode_query", lambda q: ([0.1], "s"))
    with pytest.raises(ValueError, match="Unknown stage"):
        run_stages(_FakeClient(HITS), "q", ["nope"], over_fetch=10)


def _valid_case(**overrides):
    case = {
        "user_input": "q?",
        "reference": "r",
        "gold_chunks": [{"source": "a.md", "page": None, "chunk_index": 0}],
        "expected_claims": ["claim"],
        "question_type": "factoid",
        "source_doc": "a.md",
    }
    case.update(overrides)
    return case


def test_validate_gold_schema_accepts_valid_cases():
    assert validate_gold_schema([_valid_case()]) == []
    assert validate_gold_schema([_valid_case(
        gold_chunks=[], question_type="unanswerable", source_doc="")]) == []


def test_validate_gold_schema_reports_problems():
    errors = validate_gold_schema([
        _valid_case(gold_chunks=[], question_type="factoid"),
        _valid_case(question_type="weird"),
        _valid_case(gold_chunks=[{"source": "a.md"}]),
        _valid_case(expected_claims=[]),
    ])
    assert any("at least one gold chunk" in e for e in errors)
    assert any("question_type" in e for e in errors)
    assert any("source and chunk_index" in e for e in errors)
    assert any("expected_claims" in e for e in errors)


def test_validate_gold_schema_unanswerable_must_have_empty_gold():
    errors = validate_gold_schema([_valid_case(question_type="unanswerable")])
    assert any("unanswerable" in e for e in errors)


def test_verify_gold_in_inventory_reports_missing():
    inventory = {("a.md", 0): "text"}
    missing = verify_gold_in_inventory([_valid_case()], inventory)
    assert missing == []
    missing = verify_gold_in_inventory(
        [_valid_case(gold_chunks=[{"source": "a.md", "page": None, "chunk_index": 9}])],
        inventory,
    )
    assert len(missing) == 1 and "('a.md', 9)" in missing[0]


def test_run_retrieval_eval_end_to_end_with_fake_client(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.eval.retrieval_eval.encode_query", lambda q: ([0.1], "s"))
    monkeypatch.setattr("backend.eval.retrieval_eval.score_pairs",
                        lambda q, texts: [0.9] * len(texts))
    monkeypatch.setattr("backend.eval.retrieval_eval.init_milvus", lambda: _FakeClient(HITS))
    cases = [_valid_case(user_input="q1"), _valid_case(user_input="q2", gold_chunks=[])]
    cases[1]["question_type"] = "unanswerable"
    cases[1]["source_doc"] = ""
    testset_file = tmp_path / "testset.json"
    testset_file.write_text(json.dumps(cases), encoding="utf-8")
    from backend.eval.manifest import build_manifest
    manifest = build_manifest(
        testset_path=testset_file, stages=list(STAGES), over_fetch=10,
        ks=[5, 10, 20], repeat=2, n_questions=len(cases), git_commit="test",
    )
    result = run_retrieval_eval(cases, list(STAGES), over_fetch=10, repeat=2,
                                ks=(5, 10, 20), manifest=manifest)
    assert len(result["evaluations"]) == 2
    first = result["evaluations"][0]
    assert first["dense"]["n_answerable"] == 1
    assert first["dense"]["n_unanswerable"] == 1
    assert first["dense"]["summary"]["hit@5"] == 1.0
    assert set(result["checkpoint"]["stages"]["0"]) == set(STAGES)
    assert "text" not in result["checkpoint"]["stages"]["0"]["dense"][0]
    report = build_retrieval_report(manifest, result["evaluations"], tmp_path / "c.json", list(STAGES))
    assert report["meta"]["deterministic_across_repeats"] is True
    md_path = write_retrieval_markdown(report, tmp_path / "r.json")
    md = md_path.read_text(encoding="utf-8")
    assert "dense" in md and "reranked" in md
    cp_path = write_retrieval_checkpoint(result["checkpoint"], tmp_path / "r.json")
    assert json.loads(cp_path.read_text(encoding="utf-8"))["manifest"] == manifest


def test_run_retrieval_eval_requires_existing_collection(monkeypatch):
    monkeypatch.setattr("backend.eval.retrieval_eval.init_milvus", lambda: None)
    with pytest.raises(RuntimeError, match="does not exist"):
        run_retrieval_eval([_valid_case()], ["dense"], 10, 1, (5,), {})


def test_shipped_golden_set_passes_schema_and_inventory():
    testset_path = resolve_path("eval/testsets/sample/testset.json")
    cases = load_golden_testset(testset_path)
    assert len(cases) == 98
    assert sum(1 for c in cases if c["gold_chunks"]) == 91
    assert sum(1 for c in cases if not c["gold_chunks"]) == 7
    from backend.eval.retrieval_eval import build_chunk_inventory, verify_gold_in_inventory
    inventory = build_chunk_inventory(resolve_path("eval/testsets/sample/corpus"))
    assert len(inventory) == 67
    assert verify_gold_in_inventory(cases, inventory) == []


def test_simulate_final_ks_truncation_exposes_missed_gold(tmp_path):
    from backend.eval.retrieval_eval import simulate_final_ks, write_cutoff_report

    # Three golds at ranks 2, 4, 6 — final_k=5 misses one; final_k=10 catches all.
    records = [
        {"rank": i, "source": "a.md", "page": None, "chunk_index": i,
         "score": 1.0 - i * 0.01, "text_preview": f"c{i}"}
        for i in range(1, 11)
    ]
    # Place golds at chunk_index 2, 4, 6 which sit at ranks 2, 4, 6
    checkpoint = {
        "manifest": {"harness": {"stages": ["reranked"]}},
        "cases": [{"index": 0, "user_input": "q"}],
        "stages": {"0": {"reranked": records}},
    }
    cases = [{
        "user_input": "q",
        "reference": "r",
        "gold_chunks": [
            {"source": "a.md", "page": None, "chunk_index": 2},
            {"source": "a.md", "page": None, "chunk_index": 4},
            {"source": "a.md", "page": None, "chunk_index": 6},
        ],
        "expected_claims": ["c"],
        "question_type": "multi-hop",
        "source_doc": "a.md",
    }]
    result = simulate_final_ks(checkpoint, cases, final_ks=(5, 10), stages=["reranked"])
    at5 = result["by_final_k"]["5"]["reranked"]
    at10 = result["by_final_k"]["10"]["reranked"]
    assert at5["summary"]["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert at5["n_incomplete"] == 1
    assert at10["summary"]["recall"] == pytest.approx(1.0)
    assert at10["n_incomplete"] == 0
    out = write_cutoff_report(result, tmp_path / "x.retrieval.json", tmp_path / "x-cutoff.json")
    assert out.exists()
    assert out.with_suffix(".md").exists()
