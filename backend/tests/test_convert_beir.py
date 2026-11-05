"""BEIR 转换：doc 级 gold 映射到 chunk，不联网、不碰数据库。"""
import json
import zipfile
from pathlib import Path

import pytest

from backend.eval.convert_beir import build_cases, convert_beir, load_beir_source, write_corpus
from backend.eval.retrieval_eval import build_chunk_inventory


def _write_beir_zip(tmp_path: Path, docs, queries, qrels_rows) -> Path:
    corpus_lines = [json.dumps(d, ensure_ascii=False) for d in docs]
    query_lines = [json.dumps(q, ensure_ascii=False) for q in queries]
    qrels = "query-id\tcorpus-id\tscore\n" + "".join(
        f"{q}\t{d}\t{s}\n" for q, d, s in qrels_rows
    )
    zip_path = tmp_path / "mini.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("mini/corpus.jsonl", "\n".join(corpus_lines))
        zf.writestr("mini/queries.jsonl", "\n".join(query_lines))
        zf.writestr("mini/qrels/test.tsv", qrels)
    return zip_path


def _mini_docs():
    long_text = " ".join(["The enzyme catalyzes the reaction."] * 120)
    return [
        {"_id": "10", "title": "Cats", "text": "Cats sleep a lot. Cats like fish."},
        {"_id": "20", "title": "Dogs", "text": long_text},
        {"_id": "30", "title": "Birds", "text": "Birds migrate in autumn."},
    ]


def _mini_queries():
    return [
        {"_id": "1", "text": "Do cats like fish?"},
        {"_id": "2", "text": "What do dogs do?"},
    ]


def test_convert_maps_doc_gold_to_all_its_chunks(tmp_path):
    zip_path = _write_beir_zip(
        tmp_path, _mini_docs(), _mini_queries(),
        [("1", "10", 1), ("1", "20", 1), ("2", "30", 1)],
    )
    out_dir = tmp_path / "out"
    stats = convert_beir(zip_path=zip_path, data_dir=None, split="test",
                         out_dir=out_dir, probe_size=1)
    assert stats["docs"] == 3
    assert stats["cases"] == 2
    assert (out_dir / "corpus" / "10.md").read_text(encoding="utf-8").startswith("# Cats\n")
    assert (out_dir / "testset-1q.json").exists()

    cases = json.loads((out_dir / "testset.json").read_text(encoding="utf-8"))
    inventory = build_chunk_inventory(out_dir / "corpus")
    chunks_of = {}
    for (source, chunk_index) in inventory:
        chunks_of.setdefault(source, []).append(chunk_index)
    # doc 20 的长文本应切出多块；gold 必须覆盖 gold 文档的全部 chunk
    assert len(chunks_of["20.md"]) > 1
    case1 = cases[0]
    assert case1["user_input"] == "Do cats like fish?"
    assert case1["question_type"] == "factoid"
    assert case1["source_doc"] == "10.md"
    assert "Cats" in case1["reference"] and "Dogs" in case1["reference"]
    expected_gold = sorted(
        [{"source": f, "page": None, "chunk_index": i}
         for f in ("10.md", "20.md") for i in sorted(chunks_of[f])],
        key=lambda g: (g["source"], g["chunk_index"]),
    )
    got_gold = sorted(case1["gold_chunks"],
                      key=lambda g: (g["source"], g["chunk_index"]))
    assert got_gold == expected_gold


def test_convert_rejects_duplicate_question(tmp_path):
    zip_path = _write_beir_zip(
        tmp_path, _mini_docs(),
        [{"_id": "1", "text": "Same question?"}, {"_id": "2", "text": "Same question?"}],
        [("1", "10", 1), ("2", "30", 1)],
    )
    with pytest.raises(ValueError, match="duplicate question"):
        convert_beir(zip_path=zip_path, data_dir=None, split="test",
                     out_dir=tmp_path / "out")


def test_convert_rejects_question_without_gold(tmp_path):
    zip_path = _write_beir_zip(
        tmp_path, _mini_docs(), _mini_queries(),
        [("1", "10", 1), ("2", "30", 0)],
    )
    with pytest.raises(ValueError, match="no relevant doc"):
        convert_beir(zip_path=zip_path, data_dir=None, split="test",
                     out_dir=tmp_path / "out")


def test_convert_rejects_gold_doc_missing_from_corpus(tmp_path):
    zip_path = _write_beir_zip(
        tmp_path, _mini_docs(), _mini_queries(),
        [("1", "10", 1), ("1", "99", 1)],
    )
    with pytest.raises(ValueError, match="not in corpus"):
        convert_beir(zip_path=zip_path, data_dir=None, split="test",
                     out_dir=tmp_path / "out")


def test_load_beir_source_requires_one_input(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        load_beir_source(zip_path=None, data_dir=None, split="test")


def test_write_corpus_skips_unchanged_files(tmp_path):
    corpus_dir = tmp_path / "corpus"
    docs = _mini_docs()
    write_corpus(docs, corpus_dir)
    target = corpus_dir / "10.md"
    mtime = target.stat().st_mtime_ns
    write_corpus(docs, corpus_dir)
    assert target.stat().st_mtime_ns == mtime
