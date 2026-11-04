"""pgvector 混合检索的纯函数：稀疏格式、RRF、表名，以及 Milvus 迁移去重。"""
from backend.infrastructure.config import resolve_vector_table, settings
from backend.infrastructure.migrate_from_milvus import copy_rows, identity_key, rows_to_copy
from backend.infrastructure.vector_store import (
    SPARSE_DIM,
    apply_drop_ratio,
    dense_search_sql,
    format_dense,
    format_sparse,
    fuse_hits,
    quote_ident,
    sparse_search_sql,
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


def test_format_sparse_is_one_based_and_dense_keeps_bge_width():
    assert format_sparse({0: 1.0, 2: 0.5}) == "{1:1,3:0.5}/" + str(SPARSE_DIM)
    assert format_sparse({}) == "{1:1}/" + str(SPARSE_DIM)
    try:
        format_sparse({SPARSE_DIM: 1.0})
    except Exception as exc:
        assert "sparse index" in str(exc)
    else:
        raise AssertionError("out-of-range sparse index was accepted")
    try:
        format_dense([0.1, 0.2])
    except Exception as exc:
        assert "1024" in str(exc)
    else:
        raise AssertionError("short dense vector was accepted")


def test_drop_ratio_removes_smallest_weights_and_keeps_one():
    kept = apply_drop_ratio({0: 0.1, 1: 0.9, 2: 0.2, 3: 0.05}, 0.5)
    assert kept == {1: 0.9, 2: 0.2}
    assert apply_drop_ratio({4: 0.3}, 0.9) == {4: 0.3}
    assert apply_drop_ratio({4: 0.3}, 0.0) == {4: 0.3}


def test_fuse_hits_rrf_k60_prefers_the_shared_chunk():
    dense = [_hit(1, 0.2), _hit(2, 0.9)]
    sparse = [_hit(3, 0.2), _hit(2, 0.9)]
    fused = fuse_hits(dense, sparse, limit=3, rrf_k=60)
    assert [hit["id"] for hit in fused][0] == 2
    # id 2 is rank 1 on both lists (0-based) → 1/62 + 1/62
    assert fused[0]["distance"] == 1.0 / 62 + 1.0 / 62


def test_search_sql_uses_inner_product():
    dense_sql = dense_search_sql("ragmate_docs")
    sparse_sql = sparse_search_sql("ragmate_docs")
    assert "<#>" in dense_sql and "vector" in dense_sql
    assert "<=>" not in dense_sql
    assert "<#>" in sparse_sql and "sparsevec" in sparse_sql
    assert quote_ident("ragmate_docs") == '"ragmate_docs"'
    try:
        quote_ident('docs"; drop table documents;--')
    except Exception:
        pass
    else:
        raise AssertionError("unsafe identifier was quoted")


def test_search_sparse_applies_configured_drop_ratio(monkeypatch):
    from backend.infrastructure.vector_store import PgVectorStore

    seen = {}
    store = PgVectorStore("ragmate_docs")

    def fake_query(sql, params, ef_search=None):
        seen["q"] = params["q"]
        return []

    monkeypatch.setattr(store, "_ann_query", fake_query)
    monkeypatch.setattr(settings, "DROP_RATIO_SEARCH", 0.5)
    store.search_sparse({0: 0.1, 1: 0.2, 2: 0.9, 3: 0.05}, 5)
    assert seen["q"] == "{2:0.2,3:0.9}/" + str(SPARSE_DIM)


def test_milvus_copy_skips_rows_already_stored():
    row = {
        "dense": [0.0] * 1024,
        "sparse": {0: 1.0},
        "text": "hello",
        "parent_text": "hello parent",
        "metadata": {"source": "notes.txt", "chunk_index": 1, "content_hash": "abc"},
    }
    assert identity_key(row) == ("notes.txt", 1, "abc")
    assert rows_to_copy({("notes.txt", 1, "abc")}, [row]) == []

    class Store:
        def __init__(self):
            self.inserted = []

        def existing_identity_keys(self, hashes):
            return {("notes.txt", 1, "abc")}

        def insert_rows(self, rows):
            self.inserted.extend(rows)
            return list(range(1, len(rows) + 1))

    fresh = dict(row)
    fresh["metadata"] = {"source": "other.txt", "chunk_index": 0, "content_hash": "def"}
    store = Store()
    stats = copy_rows(store, [row, fresh])
    assert stats["inserted"] == 1
    assert stats["skipped"] == 1
    assert store.inserted[0]["source"] == "other.txt"
    assert store.inserted[0]["parent_text"] == "hello parent"
