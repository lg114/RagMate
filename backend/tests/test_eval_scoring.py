"""评测打分链路的防护：judge 探针、全空分数拦截、落盘与 rescore 映射。"""
import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from backend.eval.metrics import probe_judge, validate_scores
from backend.eval.report import cases_from_report, write_raw_checkpoint


class _StubWrapper:
    """Stand-in for the RAGAS LLM wrapper: generate_text returns fixed text."""

    def __init__(self, text: str):
        self._text = text

    def generate_text(self, prompt, **kwargs):
        return LLMResult(generations=[[ChatGeneration(message=AIMessage(content=self._text))]])


def test_validate_scores_all_missing_raises():
    scores = [
        {"faithfulness": float("nan"), "context_precision": None},
        {"faithfulness": float("inf"), "context_precision": float("nan")},
    ]
    with pytest.raises(RuntimeError, match="no valid scores"):
        validate_scores(scores)


def test_validate_scores_partial_missing_only_warns(caplog):
    scores = [
        {"faithfulness": 0.9, "context_precision": float("nan")},
        {"faithfulness": 0.8, "context_precision": 0.7},
    ]
    with caplog.at_level("WARNING", logger="ragmate"):
        validate_scores(scores)
    assert "missing" in caplog.text


def test_validate_scores_all_valid_passes():
    validate_scores([{"faithfulness": 0.9}, {"faithfulness": 0.8}])


def test_probe_judge_accepts_plain_json():
    probe_judge(_StubWrapper('{"verdict": "yes"}'), model_name="test")


def test_probe_judge_accepts_fenced_json():
    probe_judge(_StubWrapper('```json\n{"verdict": "yes"}\n```'), model_name="test")


def test_probe_judge_rejects_garbage_and_shows_raw_output():
    with pytest.raises(RuntimeError, match="did not return parseable JSON") as exc_info:
        probe_judge(_StubWrapper("I cannot answer that."), model_name="test")
    assert "I cannot answer that." in str(exc_info.value)


def test_probe_judge_rejects_wrong_verdict():
    with pytest.raises(RuntimeError, match="without the expected verdict"):
        probe_judge(_StubWrapper('{"verdict": "no"}'), model_name="test")


def test_write_raw_checkpoint_and_cases_round_trip(tmp_path):
    meta = {"judge_model": "qwen3:8b", "report_path": str(tmp_path / "r.json")}
    cases = [
        {"user_input": "q1", "reference": "ref1", "response": "a1", "retrieved_contexts": ["c1"]},
        {"user_input": "q2", "reference": "ref2", "response": "", "retrieved_contexts": []},
    ]
    raw_path = write_raw_checkpoint(tmp_path / "r.json", meta, cases)
    assert raw_path.name == "r.raw.json"
    saved = json.loads(raw_path.read_text(encoding="utf-8"))
    assert saved["meta"]["raw_checkpoint"] is True
    assert len(saved["details"]) == 1  # only the valid case is checkpointed
    assert cases_from_report(saved) == [cases[0]]


def test_cases_from_report_skips_empty_contexts():
    report = {"details": [
        {"user_input": "q1", "reference": "r", "response": "a", "contexts": ["c"]},
        {"user_input": "q2", "reference": "r", "response": "a", "contexts": []},
        {"user_input": "q3", "reference": "r", "response": "", "contexts": ["c"]},
    ]}
    cases = cases_from_report(report)
    assert len(cases) == 1
    assert cases[0]["retrieved_contexts"] == ["c"]
