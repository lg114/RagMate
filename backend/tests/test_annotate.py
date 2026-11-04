"""标注流程测试：解析、组装、校验、原子写盘。全部离线，不碰数据库。"""
import json

import pytest

from backend.eval.annotate import (
    append_case,
    build_case,
    list_chunk_rows,
    parse_gold_spec,
    validate_new_case,
)
from backend.eval.retrieval_eval import build_chunk_inventory


@pytest.fixture()
def corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc.md").write_text(
        "# Cats\n\nAlpha paragraph about cats and naps.\n\n"
        "## Dogs\n\nBeta paragraph about dogs and walks.\n",
        encoding="utf-8",
    )
    return corpus_dir


@pytest.fixture()
def inventory(corpus):
    inv = build_chunk_inventory(corpus)
    assert len(inv) >= 2
    return inv


@pytest.fixture()
def testset(tmp_path, inventory):
    first_key = sorted(inventory)[0]
    cases = [{
        "user_input": "What do cats like?",
        "reference": "Cats like naps.",
        "gold_chunks": [{"source": first_key[0], "page": None,
                         "chunk_index": first_key[1]}],
        "expected_claims": ["Cats like naps"],
        "question_type": "factoid",
        "source_doc": first_key[0],
    }]
    path = tmp_path / "testset.json"
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


class TestParseGoldSpec:
    def test_valid(self):
        assert parse_gold_spec("01-docker-compose.md:2") == {
            "source": "01-docker-compose.md", "page": None, "chunk_index": 2}

    @pytest.mark.parametrize("spec", [
        "doc.md", "doc.md:x", "doc.md:-1", "doc.md:1:2", ":0", "",
        "sub/doc.md:0", "doc.md:  ",
    ])
    def test_invalid(self, spec):
        with pytest.raises(ValueError):
            parse_gold_spec(spec)


class TestBuildCase:
    def test_unanswerable_source_doc_defaults_empty(self):
        case = build_case(question="Q?", reference="No such content.",
                          question_type="unanswerable", gold=[],
                          claims=["No such content"])
        assert case["source_doc"] == ""
        assert case["gold_chunks"] == []

    def test_single_source_gold_defaults_source_doc(self):
        case = build_case(question="Q?", reference="R.", question_type="factoid",
                          gold=[{"source": "a.md", "page": None, "chunk_index": 0}],
                          claims=["c"])
        assert case["source_doc"] == "a.md"

    def test_multi_source_gold_requires_source_doc(self):
        with pytest.raises(ValueError):
            build_case(question="Q?", reference="R.", question_type="multi-hop",
                       gold=[{"source": "a.md", "page": None, "chunk_index": 0},
                             {"source": "b.md", "page": None, "chunk_index": 1}],
                       claims=["c"])

    def test_bad_type_and_empty_claims_rejected(self):
        with pytest.raises(ValueError):
            build_case(question="Q?", reference="R.", question_type="nope",
                       gold=[], claims=["c"])
        with pytest.raises(ValueError):
            build_case(question="Q?", reference="R.", question_type="factoid",
                       gold=[{"source": "a.md", "page": None, "chunk_index": 0}],
                       claims=["  "])


class TestValidateNewCase:
    def test_good_case_passes(self, testset, inventory):
        import json as _json
        existing = _json.loads(testset.read_text(encoding="utf-8"))
        key = sorted(inventory)[-1]
        case = build_case(question="What do dogs like?", reference="Dogs like walks.",
                          question_type="factoid",
                          gold=[{"source": key[0], "page": None,
                                 "chunk_index": key[1]}],
                          claims=["Dogs like walks"])
        assert validate_new_case(case, existing, inventory) == []

    def test_duplicate_question_rejected(self, testset, inventory):
        existing = json.loads(testset.read_text(encoding="utf-8"))
        case = dict(existing[0])
        assert any("duplicate" in e for e in validate_new_case(case, existing, inventory))

    def test_missing_gold_rejected(self, testset, inventory):
        existing = json.loads(testset.read_text(encoding="utf-8"))
        case = build_case(question="New?", reference="R.", question_type="factoid",
                          gold=[{"source": "doc.md", "page": None, "chunk_index": 99}],
                          claims=["c"])
        errors = validate_new_case(case, existing, inventory)
        assert any("not in chunk inventory" in e for e in errors)

    def test_unanswerable_with_gold_rejected(self, testset, inventory):
        existing = json.loads(testset.read_text(encoding="utf-8"))
        key = sorted(inventory)[0]
        case = {
            "user_input": "New unanswerable?",
            "reference": "Nothing here.",
            "gold_chunks": [{"source": key[0], "page": None, "chunk_index": key[1]}],
            "expected_claims": ["Nothing here."],
            "question_type": "unanswerable",
            "source_doc": "",
        }
        errors = validate_new_case(case, existing, inventory)
        assert any("unanswerable" in e for e in errors)


class TestAppendCase:
    def test_append_writes_and_preserves(self, testset, corpus, inventory):
        key = sorted(inventory)[-1]
        case = build_case(question="What do dogs like?", reference="Dogs like walks.",
                          question_type="factoid",
                          gold=[{"source": key[0], "page": None,
                                 "chunk_index": key[1]}],
                          claims=["Dogs like walks"])
        updated = append_case(testset, case, corpus)
        assert len(updated) == 2
        on_disk = json.loads(testset.read_text(encoding="utf-8"))
        assert on_disk == updated
        assert on_disk[0]["user_input"] == "What do cats like?"

    def test_dry_run_writes_nothing(self, testset, corpus, inventory):
        before = testset.read_text(encoding="utf-8")
        key = sorted(inventory)[-1]
        case = build_case(question="Dry?", reference="R.", question_type="factoid",
                          gold=[{"source": key[0], "page": None,
                                 "chunk_index": key[1]}],
                          claims=["c"])
        updated = append_case(testset, case, corpus, dry_run=True)
        assert len(updated) == 2
        assert testset.read_text(encoding="utf-8") == before

    def test_rejected_append_leaves_file_untouched(self, testset, corpus):
        before = testset.read_text(encoding="utf-8")
        case = build_case(question="Bad?", reference="R.", question_type="factoid",
                          gold=[{"source": "doc.md", "page": None, "chunk_index": 99}],
                          claims=["c"])
        with pytest.raises(ValueError):
            append_case(testset, case, corpus)
        assert testset.read_text(encoding="utf-8") == before


class TestListChunkRows:
    def test_sorted_and_preview_truncated(self, inventory):
        rows = list_chunk_rows(inventory)
        assert [(r["source"], r["chunk_index"]) for r in rows] == sorted(inventory)
        assert all(len(r["preview"]) <= 120 for r in rows)
        assert all(r["chars"] > 0 for r in rows)

    def test_source_filter(self, inventory):
        only = list_chunk_rows(inventory, source="doc.md")
        assert only and all(r["source"] == "doc.md" for r in only)
        assert list_chunk_rows(inventory, source="nope.md") == []


class TestAnnotateCli:
    def test_chunks_command_lists_inventory(self, corpus, capsys):
        from backend.eval import cli
        import argparse

        args = argparse.Namespace(corpus=str(corpus), source=None, full=False)
        cli.cmd_chunks(args)
        out = capsys.readouterr().out
        assert "doc.md:0" in out

    def test_annotate_command_end_to_end(self, testset, corpus, inventory, capsys):
        from backend.eval import cli
        import argparse

        key = sorted(inventory)[-1]
        args = argparse.Namespace(
            testset=str(testset), corpus=str(corpus),
            question="What do dogs like?", reference="Dogs like walks.",
            question_type="factoid",
            gold=[f"{key[0]}:{key[1]}"], claims=["Dogs like walks"],
            source_doc=None, dry_run=False,
        )
        cli.cmd_annotate(args)
        out = capsys.readouterr().out
        assert "2 cases total" in out
        assert "check-gold" in out
        assert len(json.loads(testset.read_text(encoding="utf-8"))) == 2
