"""评测用语料入库：切分 → 编码 → 写入指定 Milvus collection，不碰 PostgreSQL。

用于切块策略 A/B（独立 collection），避免增量 mtime / 文档表状态干扰实验。
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.application.ingest import pipeline as ingest_pipeline
from backend.infrastructure.config import settings
from backend.infrastructure.encoding import encode_documents
from backend.infrastructure.milvus import (
    ensure_collection,
    get_milvus_client,
    insert_chunks,
    reset_collection_loaded_flag,
)

logger = logging.getLogger("ragmate")


def ingest_corpus_to_collection(corpus_dir, collection_name: str | None = None) -> dict:
    """把 corpus_dir 下全部文件按当前 CHUNKING_STRATEGY 写入 collection。

    若目标 collection 已存在则先 drop 再建，保证实验可复现。
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise ValueError(f"Corpus directory not found: {corpus_dir}")
    files = sorted(f.name for f in corpus_dir.iterdir() if f.is_file())
    if not files:
        raise ValueError(f"No files in corpus directory: {corpus_dir}")

    target = collection_name or settings.MILVUS_COLLECTION
    previous = settings.MILVUS_COLLECTION
    settings.MILVUS_COLLECTION = target
    reset_collection_loaded_flag()
    try:
        client = get_milvus_client()
        if client.has_collection(target):
            logger.info("Dropping existing collection %s for clean ingest", target)
            client.drop_collection(target)
        ensure_collection(client)

        original_reporter = ingest_pipeline.set_ingest_status_sync
        ingest_pipeline.set_ingest_status_sync = lambda data: None
        try:
            chunks, chunk_counter, failed = ingest_pipeline._load_and_split(
                str(corpus_dir), files
            )
        finally:
            ingest_pipeline.set_ingest_status_sync = original_reporter

        if failed:
            raise RuntimeError(f"Failed to load some corpus files: {failed}")
        if not chunks:
            raise RuntimeError("No chunks produced from corpus")

        texts = [c.page_content for c in chunks]
        dense_vecs, sparse_vecs = encode_documents(texts)
        insert_chunks(client, chunks, dense_vecs, sparse_vecs)
        # standalone Milvus 上 get_collection_stats 有时短期仍报 0；flush + 点查确认可检索。
        try:
            client.flush(collection_name=target)
        except Exception:
            logger.debug("flush after ingest failed", exc_info=True)
        client.load_collection(target)
        probe = client.query(
            collection_name=target,
            filter="",
            output_fields=["metadata"],
            limit=16384,
        )
        return {
            "collection": target,
            "strategy": settings.CHUNKING_STRATEGY,
            "files": len(files),
            "chunks": len(chunks),
            "chunk_counts": dict(chunk_counter),
            "row_count": len(probe),
        }
    finally:
        settings.MILVUS_COLLECTION = previous
