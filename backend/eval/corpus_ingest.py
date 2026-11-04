"""评测用语料入库：切分 → 编码 → 写入指定 pgvector 表，不碰 documents 表。

用于切块策略实验（独立表），避免增量 mtime / 文档表状态干扰。
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.application.ingest import pipeline as ingest_pipeline
from backend.infrastructure.config import settings
from backend.infrastructure.encoding import encode_documents
from backend.infrastructure.vector_store import (
    PgVectorStore,
    insert_chunks,
    reset_vector_store_cache,
)

logger = logging.getLogger("ragmate")


def ingest_corpus_to_collection(corpus_dir, collection_name: str | None = None) -> dict:
    """把 corpus_dir 下全部文件按当前 CHUNKING_STRATEGY 写入向量表。

    目标表已存在则先 drop 再建，保证实验可复现。collection_name 是表名。
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise ValueError(f"Corpus directory not found: {corpus_dir}")
    files = sorted(f.name for f in corpus_dir.iterdir() if f.is_file())
    if not files:
        raise ValueError(f"No files in corpus directory: {corpus_dir}")

    target = collection_name or settings.VECTOR_TABLE
    previous = settings.VECTOR_TABLE
    settings.VECTOR_TABLE = target
    reset_vector_store_cache()
    try:
        store = PgVectorStore(target)
        logger.info("Dropping vector table %s for clean ingest", target)
        store.drop_table()
        store.ensure_schema()

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
        insert_chunks(store, chunks, dense_vecs, sparse_vecs)
        return {
            "collection": target,
            "strategy": settings.CHUNKING_STRATEGY,
            "files": len(files),
            "chunks": len(chunks),
            "chunk_counts": dict(chunk_counter),
            "row_count": store.count_rows(),
        }
    finally:
        settings.VECTOR_TABLE = previous
        reset_vector_store_cache()
