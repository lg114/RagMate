"""评测打分链路的防护：judge 探针、全空分数拦截、落盘与 rescore 映射。"""
import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from backend.eval.metrics import (
    ScoreMatrix,
    build_judge_pin,
    compute_metrics,
    merge_score_runs,
    probe_judge,
    score_with_repeats,
    summary_spread,
    validate_scores,
)
from backend.eval.report import build_report, cases_from_report, write_raw_checkpoint


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


def test_write_raw_checkpoint_refuses_overwrite(tmp_path):
    meta = {"judge_model": "x"}
    cases = [
        {"user_input": "q", "reference": "r", "response": "a", "retrieved_contexts": ["c"]},
    ]
    write_raw_checkpoint(tmp_path / "r.json", meta, cases)
    with pytest.raises(FileExistsError, match="already exists"):
        write_raw_checkpoint(tmp_path / "r.json", meta, cases)
    write_raw_checkpoint(tmp_path / "r.json", meta, cases, force=True)


def test_cases_from_report_skips_empty_contexts():
    report = {"details": [
        {"user_input": "q1", "reference": "r", "response": "a", "contexts": ["c"]},
        {"user_input": "q2", "reference": "r", "response": "a", "contexts": []},
        {"user_input": "q3", "reference": "r", "response": "", "contexts": ["c"]},
    ]}
    cases = cases_from_report(report)
    assert len(cases) == 1
    assert cases[0]["retrieved_contexts"] == ["c"]


def test_merge_score_runs_prefers_first_non_null():
    runs = [
        [{"faithfulness": None, "context_recall": 0.5}],
        [{"faithfulness": 0.9, "context_recall": 0.7}],
    ]
    merged = merge_score_runs(runs)
    assert merged == [{"faithfulness": 0.9, "context_recall": 0.5}]


def test_build_report_marks_unscored_and_spread():
    cases = [
        {"user_input": "q1", "reference": "r", "response": "a", "retrieved_contexts": ["c"]},
        {"user_input": "q2", "reference": "r", "response": "a", "retrieved_contexts": ["c"]},
    ]
    matrix = ScoreMatrix(
        runs=[
            [{"faithfulness": 0.8}, {"faithfulness": None}],
            [{"faithfulness": 0.9}, {"faithfulness": None}],
        ],
        pin=build_judge_pin("stub-model", 0.0),
    )
    report = build_report({"testset": "t"}, cases, matrix)
    assert report["details"][0]["status"] == "scored"
    assert report["details"][0]["scores"]["faithfulness"] == 0.85  # mean of repeats
    assert report["details"][1]["status"] == "unscored"
    assert report["meta"]["n_unscored"] == 1
    assert report["meta"]["score_repeats"] == 2
    assert "faithfulness" in report["meta"]["score_spread"]
    assert report["meta"]["judge_temperature"] == 0.0
    assert report["details"][0]["scores_runs"]


def test_summary_spread_two_runs():
    runs = [
        [{"m": 0.5}, {"m": 0.7}],
        [{"m": 0.9}, {"m": 0.8}],
    ]
    # means: (0.6) and (0.85) → spread 0.25
    spread = summary_spread(runs)
    assert spread["m"] == 0.25


def test_compute_metrics_disables_ragas_wait_for_timeout(monkeypatch):
    """打分必须绕开 ragas 的 wait_for 超时：3.14 下它让全部任务秒灭。

    只验证接线（run_config.timeout is None），不跑真实 judge：把
    ragas.evaluate 换成捕获参数的假实现，judge/向量换成不联网的桩。
    """
    ragas = pytest.importorskip("ragas")
    from langchain_core.embeddings.fake import DeterministicFakeEmbedding
    from langchain_openai import ChatOpenAI
    from ragas.llms.base import LangchainLLMWrapper

    captured = {}

    class _FakeResult:
        scores = [{"faithfulness": 0.9, "answer_relevancy": 0.8}]

    def fake_evaluate(dataset=None, metrics=None, **kwargs):
        captured.update(kwargs)
        return _FakeResult()

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        "backend.eval.metrics.build_judge_llm",
        lambda judge_model=None, temperature=None: LangchainLLMWrapper(
            ChatOpenAI(model="stub", api_key="stub", base_url="http://127.0.0.1:9/v1")
        ),
    )
    monkeypatch.setattr(
        "backend.infrastructure.model_factory.get_embeddings",
        lambda: DeterministicFakeEmbedding(size=8),
    )

    result = compute_metrics(
        [{"user_input": "q", "response": "a", "retrieved_contexts": ["c"], "reference": "r"}],
        parse_retries=0,
    )
    assert isinstance(result, ScoreMatrix)
    assert result.primary[0]["faithfulness"] == 0.9
    assert captured["run_config"].timeout is None


def test_score_with_repeats_runs_n_times(monkeypatch):
    calls = {"n": 0}

    def fake_compute(test_cases, judge_model=None, temperature=None, parse_retries=2):
        calls["n"] += 1
        return ScoreMatrix(
            runs=[[{"faithfulness": 0.5 + calls["n"] * 0.1}]],
            pin=build_judge_pin("stub", 0.0),
        )

    monkeypatch.setattr("backend.eval.metrics.compute_metrics", fake_compute)
    matrix = score_with_repeats(
        [{"user_input": "q", "response": "a", "retrieved_contexts": ["c"], "reference": "r"}],
        score_repeats=3,
        parse_retries=0,
    )
    assert calls["n"] == 3
    assert len(matrix.runs) == 3


def test_build_judge_llm_passes_temperature(monkeypatch):
    import backend.eval.metrics as metrics_mod

    captured_create = {}
    captured_eval = {}

    class _FakeLLM:
        model_name = "m"
        openai_api_key = "k"
        openai_api_base = "http://x"

    def fake_create_llm(**kwargs):
        captured_create.update(kwargs)
        return _FakeLLM()

    class _CaptureEval:
        def __init__(self, **kwargs):
            captured_eval.update(kwargs)

    monkeypatch.setattr(
        "backend.infrastructure.streaming_llm.create_llm", fake_create_llm,
    )
    monkeypatch.setattr(metrics_mod, "EvalLLM", _CaptureEval)
    monkeypatch.setattr(
        "ragas.llms.base.LangchainLLMWrapper",
        lambda llm: llm,
    )

    metrics_mod.build_judge_llm("my-judge", temperature=0.0)
    assert captured_create.get("temperature") == 0.0
    assert captured_eval.get("temperature") == 0.0


def test_build_judge_pin_defaults_temp_zero():
    pin = build_judge_pin("m", None)
    assert pin.temperature == 0.0
    assert "Faithfulness" in pin.metric_pack
    assert pin.as_meta()["judge_temperature"] == 0.0
