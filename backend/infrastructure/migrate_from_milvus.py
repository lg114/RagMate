"""把还在跑的 Milvus collection 拷进 pgvector。可重复执行。

pymilvus 不是运行时依赖。只在这一次迁移时安装：

    pip install 'pymilvus>=2.5'
    python -m backend.infrastructure.migrate_from_milvus

已经存在相同 (source, chunk_index, content_hash) 的行会跳过。
Milvus 已经拆掉时不要用这个脚本，把 documents.status 改成 uploaded 再重新入库。
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _metadata(row: dict) -> dict:
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        parsed = json.loads(meta)
        return parsed if isinstance(parsed, dict) else {}
    return meta if isinstance(meta, dict) else {}


def identity_key(row: dict) -> tuple:
    """迁移去重键：(文件名, chunk_index, content_hash)。"""
    meta = _metadata(row)
    source = os.path.basename(str(meta.get("source") or "unknown"))
    return (source, meta.get("chunk_index"), str(meta.get("content_hash") or ""))


def rows_to_copy(existing: set[tuple], rows: list[dict]) -> list[dict]:
    """丢掉已经在 pgvector 里、或本批已经选过的行。"""
    chosen = []
    seen = set(existing)
    for row in rows:
        key = identity_key(row)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(row)
    return chosen


def milvus_row_to_insert(row: dict) -> dict:
    meta = _metadata(row)
    source = os.path.basename(str(meta.get("source") or "unknown"))
    page = meta.get("page")
    if isinstance(page, bool) or not isinstance(page, int):
        page = None
    chunk_index = meta.get("chunk_index")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
        chunk_index = 0
    text = row.get("text") or ""
    return {
        "dense": row.get("dense"),
        "sparse": row.get("sparse") or {},
        "text": text,
        "parent_text": row.get("parent_text") or text,
        "source": source,
        "page": page,
        "chunk_index": chunk_index,
        "content_hash": str(meta.get("content_hash") or ""),
        "ingest_batch_id": str(meta.get("ingest_batch_id") or ""),
    }


def copy_rows(store, rows: list[dict]) -> dict:
    """把一批 Milvus 行写入 store。返回 scanned / inserted / skipped。"""
    pending = rows_to_copy(set(), rows)
    if not pending:
        return {"scanned": len(rows), "inserted": 0, "skipped": len(rows)}
    hashes = [str(_metadata(row).get("content_hash") or "") for row in pending]
    hashes = [item for item in hashes if item]
    existing = store.existing_identity_keys(hashes) if hashes else set()
    fresh = rows_to_copy(existing, pending)
    inserted = 0
    if fresh:
        inserted = len(store.insert_rows([milvus_row_to_insert(row) for row in fresh]))
    return {
        "scanned": len(rows),
        "inserted": inserted,
        "skipped": len(rows) - inserted,
    }


def _iter_milvus(client, collection: str, page: int):
    last_id = -1
    while True:
        batch = client.query(
            collection_name=collection,
            filter=f"id > {last_id}",
            output_fields=["id", "dense", "sparse", "text", "parent_text", "metadata"],
            limit=page,
        )
        if not batch:
            return
        ids = []
        for row in batch:
            value = row.get("id")
            if isinstance(value, int) and not isinstance(value, bool):
                ids.append(value)
        if not ids or max(ids) <= last_id:
            return
        last_id = max(ids)
        yield batch
        if len(batch) < page:
            return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy a Milvus collection into pgvector")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=19530)
    parser.add_argument("--collection", default="")
    parser.add_argument("--page", type=int, default=500)
    args = parser.parse_args(argv)

    try:
        from pymilvus import MilvusClient
    except ImportError:
        print(
            "pymilvus is not installed. For a one-time copy:\n"
            "  pip install 'pymilvus>=2.5'\n"
            "  python -m backend.infrastructure.migrate_from_milvus\n"
            "If Milvus is already gone, re-ingest instead:\n"
            "  UPDATE documents SET status = 'uploaded';\n"
            "then run ingest again. Files must still be on disk.",
            file=sys.stderr,
        )
        return 2

    from backend.infrastructure.config import settings
    from backend.infrastructure.vector_store import PgVectorStore

    collection = args.collection.strip() or settings.VECTOR_TABLE
    store = PgVectorStore()
    store.ensure_schema()
    client = MilvusClient(uri=f"http://{args.host}:{args.port}")
    if not client.has_collection(collection):
        print(f"Milvus collection {collection!r} does not exist.", file=sys.stderr)
        return 2

    scanned = inserted = skipped = 0
    for batch in _iter_milvus(client, collection, max(1, args.page)):
        stats = copy_rows(store, batch)
        scanned += stats["scanned"]
        inserted += stats["inserted"]
        skipped += stats["skipped"]
    print(
        f"Copied {inserted} chunk(s) into {store.table} "
        f"(scanned {scanned}, skipped {skipped} already present)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
