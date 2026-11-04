"""冻结的检索默认值，以及无参数 CLI 指向 retrieval-gate。"""
import sys

import pytest

from backend.infrastructure.config import Settings


def _field_default(name: str):
    return Settings.model_fields[name].default


def test_retrieval_defaults_stay_frozen():
    assert _field_default("RRF_K") == 60
    assert _field_default("CHUNKING_STRATEGY") == "title_path"
    assert _field_default("RERANK_CANDIDATES") == 30
    assert _field_default("FINAL_CONTEXT_K") == 15
    assert _field_default("RERANK_SCORE_THRESHOLD") == 0.3


def test_removed_knobs_are_not_settings():
    """收敛掉的旋钮不得以 Settings 字段形式复活。"""
    removed = [
        "HYBRID_SEARCH_ENABLED", "FUSION_MODE", "FUSION_DENSE_WEIGHT",
        "MULTI_QUERY_ENABLED", "MULTI_QUERY_COUNT",
        "CONTEXTUAL_COMPRESSION", "COMPRESSION_SCORE_THRESHOLD",
        "COMPRESSION_MIN_CHARS", "DYNAMIC_THRESHOLD_RATIO",
        "HIGH_SCORE_RATIO", "MAX_PER_SOURCE", "MIN_PER_SOURCE",
        "SCORE_GAP_THRESHOLD", "SOURCE_DOMINANCE_THRESHOLD",
        "SOURCE_DOMINANCE_BOOST",
    ]
    for name in removed:
        assert name not in Settings.model_fields


def test_bare_eval_cli_points_at_retrieval_gate(capsys, monkeypatch):
    from backend.eval import cli

    monkeypatch.setattr(sys, "argv", ["ragmate-eval"])
    cli.main()
    out = capsys.readouterr().out
    assert "retrieval-gate" in out
    assert "ragas" not in out.lower()
    assert "shelved" not in out.lower()


def test_help_lists_only_deterministic_commands(capsys, monkeypatch):
    from backend.eval import cli

    monkeypatch.setattr(sys, "argv", ["ragmate-eval", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "{retrieval,check-gold,retrieval-gate,delivery,cutoff}" in out
    assert "[shelved]" not in out
    assert "ragas" not in out.lower()
