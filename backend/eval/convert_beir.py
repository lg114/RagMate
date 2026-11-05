"""BEIR 评测集 → RagMate 题集转换（SciFact 首个）。

BEIR 只给文档级 gold（query → 相关文档），RagMate 的题集要 chunk 级 gold。
做法：文档写成单篇 .md，用生产切分复现 chunk 清单，一篇 gold 文档的
全部 chunk 都算 gold——标准比人工逐块标 gold 宽松，所以这套题的分数
只能同集前后对比，不能跟 130 题手标集比。

BEIR 没有参考答案：reference 取 gold 文档标题、expected_claims 取原
claim 文本，仅为满足 schema；检索打分只看 gold_chunks。

用法（仓库根目录，venv 已激活）：
    python -m backend.eval.convert_beir --zip <scifact.zip> --out-dir eval/testsets/scifact
    python -m backend.eval.convert_beir --data-dir <解压目录> --split test

输出：
    <out-dir>/corpus/<doc_id>.md   一篇文档一个文件（# 标题 + 摘要）
    <out-dir>/testset.json         全部题目（test: 300 题）
    <out-dir>/testset-10q.json     前 10 题，供先跑计时探针

生成文件不入仓（见 .gitignore），用脚本可复现。入库与评测见 eval/README.md。
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

from .retrieval_eval import (
    build_chunk_inventory,
    validate_gold_schema,
    verify_gold_in_inventory,
)

BEIR_ZIP_MEMBERS = {
    "corpus": "corpus.jsonl",
    "queries": "queries.jsonl",
    "qrels": "qrels/{split}.tsv",
}

DEFAULT_PROBE_SIZE = 10


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def load_beir_source(*, zip_path: Path | None, data_dir: Path | None, split: str):
    """读 BEIR 数据。返回 (corpus[{_id, title, text}], queries{id: text}, qrels[(qid, docid, score)])。"""
    if bool(zip_path) == bool(data_dir):
        raise ValueError("exactly one of --zip / --data-dir is required")
    if split not in ("test", "train"):
        raise ValueError(f"--split must be test or train, got {split!r}")

    def _read_member(name: str) -> list[str]:
        if zip_path is not None:
            with zipfile.ZipFile(zip_path) as zf:
                # BEIR zip 内是 scifact/corpus.jsonl 这样的带目录路径
                match = [n for n in zf.namelist() if n.endswith("/" + name) or n == name]
                if not match:
                    raise ValueError(f"{name} not found in {zip_path}")
                return zf.read(match[0]).decode("utf-8").splitlines()
        return _read_lines(data_dir / name)

    corpus = [json.loads(line) for line in _read_member(BEIR_ZIP_MEMBERS["corpus"]) if line.strip()]
    queries = {}
    for line in _read_member(BEIR_ZIP_MEMBERS["queries"]):
        if not line.strip():
            continue
        row = json.loads(line)
        queries[str(row["_id"])] = (row.get("text") or "").strip()
    qrels = []
    for line in _read_member(BEIR_ZIP_MEMBERS["qrels"].format(split=split)):
        if not line.strip() or line.startswith("query-id"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"bad qrels line: {line!r}")
        qid, docid, score = parts
        qrels.append((str(qid), str(docid), int(score)))
    return corpus, queries, qrels


def write_corpus(corpus: list[dict], out_corpus_dir: Path) -> dict[str, str]:
    """一篇文档写一个 .md（# 标题 + 摘要）。内容不变不重写，省得改 mtime。

    返回 {doc_id: 文件名}。
    """
    out_corpus_dir.mkdir(parents=True, exist_ok=True)
    doc_files = {}
    for doc in corpus:
        doc_id = str(doc["_id"])
        title = (doc.get("title") or "").strip()
        text = (doc.get("text") or "").strip()
        if not title or not text:
            raise ValueError(f"doc {doc_id} has empty title or text")
        filename = f"{doc_id}.md"
        content = f"# {title}\n\n{text}\n"
        path = out_corpus_dir / filename
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
        doc_files[doc_id] = filename
    return doc_files


def build_cases(queries, qrels, doc_files, doc_titles, inventory) -> list[dict]:
    """按 qrels 组题：gold 文档的全部 chunk 都算 gold。

    inventory 是 build_chunk_inventory 的 {(basename, chunk_index): text}。
    """
    chunks_by_source: dict[str, list[int]] = {}
    for (source, chunk_index) in inventory:
        chunks_by_source.setdefault(source, []).append(chunk_index)
    for source in chunks_by_source:
        chunks_by_source[source].sort()

    gold_docs: dict[str, list[str]] = {}
    qrels_qids: set[str] = set()
    for qid, docid, score in qrels:
        qrels_qids.add(qid)
        if score < 1:
            continue
        if docid not in doc_files:
            raise ValueError(f"qrels references doc {docid} not in corpus (query {qid})")
        gold_docs.setdefault(qid, []).append(docid)
    no_gold = sorted(qrels_qids - set(gold_docs), key=int)
    if no_gold:
        raise ValueError(f"queries with no relevant doc (score>=1): {no_gold[:5]}")

    cases = []
    seen_questions = set()
    for qid in sorted(gold_docs, key=lambda q: int(q)):
        if qid not in queries:
            raise ValueError(f"query {qid} not in queries.jsonl")
        question = queries[qid]
        if question in seen_questions:
            raise ValueError(f"duplicate question text: {question[:60]!r}")
        seen_questions.add(question)
        docs = gold_docs[qid]
        gold_chunks = []
        for docid in docs:
            filename = doc_files[docid]
            if filename not in chunks_by_source:
                raise ValueError(f"gold doc {docid} ({filename}) produced no chunks")
            for chunk_index in chunks_by_source[filename]:
                gold_chunks.append({"source": filename, "page": None, "chunk_index": chunk_index})
        cases.append({
            "user_input": question,
            "reference": "、".join(doc_titles[d] for d in docs),
            "gold_chunks": gold_chunks,
            "expected_claims": [question],
            "question_type": "factoid",
            "source_doc": doc_files[docs[0]],
        })
    return cases


def convert_beir(*, zip_path: Path | None, data_dir: Path | None, split: str,
                 out_dir: Path, probe_size: int = DEFAULT_PROBE_SIZE) -> dict:
    """完整转换：写 corpus + testset.json + testset-10q.json，并校验。返回统计。"""
    corpus, queries, qrels = load_beir_source(zip_path=zip_path, data_dir=data_dir, split=split)
    corpus_dir = out_dir / "corpus"
    doc_files = write_corpus(corpus, corpus_dir)
    doc_titles = {str(d["_id"]): (d.get("title") or "").strip() for d in corpus}

    inventory = build_chunk_inventory(corpus_dir)
    cases = build_cases(queries, qrels, doc_files, doc_titles, inventory)

    errors = validate_gold_schema(cases)
    if errors:
        raise ValueError("converted testset failed schema validation:\n" + "\n".join(errors))
    missing = verify_gold_in_inventory(cases, inventory)
    if missing:
        raise ValueError("gold chunks missing from chunk inventory:\n" + "\n".join(missing[:10]))

    out_dir.mkdir(parents=True, exist_ok=True)
    testset_path = out_dir / "testset.json"
    testset_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    probe_path = out_dir / f"testset-{probe_size}q.json"
    probe_path.write_text(
        json.dumps(cases[:probe_size], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gold_per_case = [len(c["gold_chunks"]) for c in cases]
    return {
        "docs": len(corpus),
        "chunks": len(inventory),
        "cases": len(cases),
        "gold_chunks_total": sum(gold_per_case),
        "gold_per_case_min": min(gold_per_case),
        "gold_per_case_max": max(gold_per_case),
        "testset": str(testset_path),
        "probe": str(probe_path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert a BEIR dataset to a RagMate testset.")
    parser.add_argument("--zip", dest="zip", type=Path, default=None,
                        help="BEIR dataset zip, e.g. scifact.zip")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Extracted BEIR directory (corpus.jsonl, queries.jsonl, qrels/)")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--out-dir", type=Path, default=Path("eval/testsets/scifact"))
    parser.add_argument("--probe-size", type=int, default=DEFAULT_PROBE_SIZE,
                        help="How many leading questions go into testset-<n>q.json")
    args = parser.parse_args(argv)
    try:
        stats = convert_beir(zip_path=args.zip, data_dir=args.data_dir,
                             split=args.split, out_dir=args.out_dir,
                             probe_size=args.probe_size)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"docs: {stats['docs']}, chunks: {stats['chunks']}, cases: {stats['cases']}")
    print(f"gold chunks: {stats['gold_chunks_total']} total, "
          f"{stats['gold_per_case_min']}–{stats['gold_per_case_max']} per case")
    print(f"wrote {stats['testset']} and {stats['probe']}")
    print("Schema + chunk inventory checks passed.")
    print("Note: BEIR gold is document-level; scores are only comparable within this set,")
    print("not against the 130-question hand-labeled set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
