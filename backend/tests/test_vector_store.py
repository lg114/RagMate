"""pgvector 混合检索的纯函数：RRF、表名；BM25 sparse 臂。"""
from backend.core.lexical import LexicalIndex
from backend.infrastructure.config import resolve_vector_table
from backend.infrastructure.vector_store import (
    dense_search_sql,
    format_dense,
    fuse_hits,
    quote_ident,
)


def _hit(item_id, score, text="t"):
    return {
        "id": item_id,
        "distance": score,
        "entity": {"text": text, "parent_text": text, "metadata": {}},
    }


def test_resolve_vector_table_prefers_explicit_name_then_legacy():
    assert resolve_vector_table("", "") == "ragmate_docs"
    assert resolve_vector_table("", "legacy_docs") == "legacy_docs"
    assert resolve_vector_table("custom_docs", "legacy_docs") == "custom_docs"
    try:
        resolve_vector_table("bad name", "")
    except ValueError as exc:
        assert "VECTOR_TABLE" in str(exc)
    else:
        raise AssertionError("invalid table name was accepted")


def test_format_dense_rejects_short_vector():
    try:
        format_dense([0.1, 0.2])
    except Exception as exc:
        assert "1024" in str(exc)
    else:
        raise AssertionError("short dense vector was accepted")


def test_fuse_hits_rrf_k60_prefers_the_shared_chunk():
    dense = [_hit(1, 0.2), _hit(2, 0.9)]
    sparse = [_hit(3, 0.2), _hit(2, 0.9)]
    fused = fuse_hits(dense, sparse, limit=3, rrf_k=60)
    assert [hit["id"] for hit in fused][0] == 2
    # id 2 is rank 1 on both lists (0-based) → 1/62 + 1/62
    assert fused[0]["distance"] == 1.0 / 62 + 1.0 / 62


def test_search_sql_uses_inner_product():
    dense_sql = dense_search_sql("ragmate_docs")
    assert "<#>" in dense_sql and "vector" in dense_sql
    assert "<=>" not in dense_sql
    assert quote_ident("ragmate_docs") == '"ragmate_docs"'
    try:
        quote_ident('docs"; drop table documents;--')
    except Exception:
        pass
    else:
        raise AssertionError("unsafe identifier was quoted")


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    """execute() 按 SQL 分流：COUNT(*) 走指纹，其余走全表快照。"""

    def __init__(self, rows, calls):
        self._rows = rows
        self._calls = calls

    def execute(self, stmt, params=None):
        if "COUNT(*)" in str(stmt):
            self._calls.append("fingerprint")
            max_id = max((r["id"] for r in self._rows), default=0)
            return _FakeResult([(len(self._rows), max_id)])
        self._calls.append("fetch")
        return _FakeResult([dict(r) for r in self._rows])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _lexical_rows():
    def row(item_id, text, source):
        return {
            "id": item_id, "text": text, "parent_text": text,
            "source": source, "page": None, "chunk_index": 0,
            "content_hash": f"h{item_id}", "ingest_batch_id": "b1",
        }

    return [
        row(1, "work_mem 默认是 4MB", "a.md"),
        row(2, "work_mem 太大可能 OOM", "b.md"),
        row(3, "完全不相关的内容", "c.md"),
    ]


def _store_with_rows(monkeypatch, rows, calls):
    from backend.infrastructure import vector_store as vs_module
    from backend.infrastructure.vector_store import PgVectorStore

    store = PgVectorStore("ragmate_docs")
    monkeypatch.setattr(store, "_connect", lambda: _FakeConn(rows, calls))
    monkeypatch.setattr(vs_module, "_lexical_indexes", {})
    return store


def test_search_sparse_bm25_matches_lexical_replay(monkeypatch):
    """生产 sparse 臂与门禁词法重放同一套排序：同 chunk、同问题，结果一致。"""
    rows = _lexical_rows()
    store = _store_with_rows(monkeypatch, rows, [])
    query = "work_mem 默认是多少？"
    hits = store.search_sparse(query, 3)
    chunks = [
        {"key": (r["source"], r["page"], r["chunk_index"]), "text": r["text"]}
        for r in rows
    ]
    expected = LexicalIndex(chunks).rank(query)
    assert [(h["entity"]["metadata"]["source"], h["entity"]["metadata"]["page"],
             h["entity"]["metadata"]["chunk_index"]) for h in hits] == expected
    assert hits[0]["distance"] > hits[1]["distance"] > 0
    assert hits[2]["distance"] == 0.0
    assert hits[0]["id"] == 1
    assert hits[0]["entity"]["text"] == "work_mem 默认是 4MB"
    assert hits[0]["entity"]["parent_text"] == "work_mem 默认是 4MB"


def test_search_sparse_caches_index_until_fingerprint_moves(monkeypatch):
    rows = _lexical_rows()
    calls = []
    store = _store_with_rows(monkeypatch, rows, calls)
    store.search_sparse("work_mem", 2)
    assert calls == ["fingerprint", "fetch"]
    store.search_sparse("work_mem", 2)
    assert calls == ["fingerprint", "fetch", "fingerprint"]
    rows.append({
        "id": 4, "text": "work_mem 新文档", "parent_text": "work_mem 新文档",
        "source": "d.md", "page": None, "chunk_index": 0,
        "content_hash": "h4", "ingest_batch_id": "b2",
    })
    hits = store.search_sparse("work_mem", 4)
    assert calls == ["fingerprint", "fetch", "fingerprint", "fingerprint", "fetch"]
    assert len(hits) == 4


def test_search_sparse_empty_table_returns_empty(monkeypatch):
    calls = []
    store = _store_with_rows(monkeypatch, [], calls)
    assert store.search_sparse("work_mem", 5) == []
    assert store.search_sparse("work_mem", 0) == []
    assert calls == ["fingerprint", "fetch"]


def test_search_hybrid_fuses_dense_and_bm25(monkeypatch):
    rows = _lexical_rows()
    store = _store_with_rows(monkeypatch, rows, [])
    monkeypatch.setattr(store, "search_dense", lambda dense_vec, limit: [_hit(3, 0.9)])
    fused = store.search_hybrid([0.1], "work_mem 默认是多少？", 3)
    # id 3：dense 第 1 名 + BM25 第 3 名；id 1：BM25 第 1 名；id 2：BM25 第 2 名
    assert [hit["id"] for hit in fused] == [3, 1, 2]

