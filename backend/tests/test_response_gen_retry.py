"""评测生成：瞬时错误识别与 raw 补洞合并。"""
import pytest

from backend.eval.response_gen import is_transient_llm_error, generate_response
from backend.eval.report import (
    merge_into_raw,
    questions_missing_from_raw,
    write_raw_checkpoint,
)


def test_is_transient_detects_503_and_timeout():
    assert is_transient_llm_error(RuntimeError("Error code: 503"))
    assert is_transient_llm_error(TimeoutError("Request timed out"))
    assert is_transient_llm_error(RuntimeError("Service Unavailable"))


def test_is_transient_rejects_auth_errors():
    assert not is_transient_llm_error(RuntimeError("Error code: 401"))
    assert not is_transient_llm_error(ValueError("bad prompt"))


def test_generate_response_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _LLM:
        def invoke(self, messages):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Error code: 503")
            class R:
                content = "ok answer"
            return R()

    monkeypatch.setattr("backend.eval.response_gen._eval_llm", lambda: _LLM())
    monkeypatch.setattr("backend.infrastructure.config.settings.EVAL_LLM_RETRIES", 2)
    monkeypatch.setattr("backend.infrastructure.config.settings.EVAL_LLM_RETRY_BACKOFF", 0.01)
    assert generate_response("q", ["ctx"]) == "ok answer"
    assert calls["n"] == 3


def test_generate_response_does_not_retry_non_transient(monkeypatch):
    class _LLM:
        def invoke(self, messages):
            raise RuntimeError("Error code: 401")

    monkeypatch.setattr("backend.eval.response_gen._eval_llm", lambda: _LLM())
    monkeypatch.setattr("backend.infrastructure.config.settings.EVAL_LLM_RETRIES", 3)
    with pytest.raises(RuntimeError, match="401"):
        generate_response("q", ["ctx"])


def test_questions_missing_and_merge_into_raw(tmp_path):
    testset = [
        {"user_input": "q1", "reference": "r1"},
        {"user_input": "q2", "reference": "r2"},
        {"user_input": "q3", "reference": "r3"},
    ]
    raw = {
        "meta": {"failed_questions": [{"user_input": "q2", "error": "503"}]},
        "details": [
            {"user_input": "q1", "reference": "r1", "response": "a1", "contexts": ["c"]},
        ],
    }
    missing = questions_missing_from_raw(testset, raw)
    assert [m["user_input"] for m in missing] == ["q2", "q3"]

    patched = [
        {
            "user_input": "q2",
            "reference": "r2",
            "response": "a2",
            "retrieved_contexts": ["c2"],
        },
        {
            "user_input": "q3",
            "reference": "r3",
            "response": "",
            "retrieved_contexts": [],
            "error": "still 503",
        },
    ]
    merged = merge_into_raw(raw, patched)
    qs = {d["user_input"] for d in merged["details"]}
    assert qs == {"q1", "q2"}
    assert merged["meta"]["valid_cases"] == 2
    assert merged["meta"]["failed_cases"] == 1
    assert merged["meta"]["failed_questions"][0]["user_input"] == "q3"


def test_write_raw_records_failed_questions(tmp_path):
    cases = [
        {"user_input": "ok", "reference": "r", "response": "a", "retrieved_contexts": ["c"]},
        {"user_input": "bad", "reference": "r", "response": "", "retrieved_contexts": [], "error": "503"},
    ]
    path = write_raw_checkpoint(tmp_path / "r.json", {"testset": "t"}, cases, force=True)
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["details"]) == 1
    assert data["meta"]["failed_cases"] == 1
    assert data["meta"]["failed_questions"][0]["user_input"] == "bad"
