"""入库主流程：文件发现 → 加载 → 切分 → 去重 → 编码 → 写入 Milvus → 同步 PostgreSQL。"""
import hashlib
import logging
import os
from collections import Counter

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from sqlalchemy import select

from config import settings
from database import SyncSession
from errors import ServiceUnavailableError
from models import Document
from redis_client import set_ingest_status_sync
from retriever import _check_milvus_available, get_milvus_client

from .db_sync import sync_documents_table
from .encoding import encode_documents
from .loaders import SUPPORTED_EXTENSIONS, load_document
from .milvus_ops import (
    build_source_filter,
    deduplicate_chunks,
    delete_old_chunks,
    ensure_collection,
    insert_chunks,
)

logger = logging.getLogger("ragmate")


def _discover_files(docs_dir, filenames):
    """发现目录下所有支持的文件，可选过滤指定文件名。"""
    all_files = sorted(f for f in os.listdir(docs_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS)
    if filenames is not None:
        all_files = [f for f in all_files if f in filenames]
    return all_files


def _detect_new_files(docs_dir, all_files):
    """检测新文件或已修改的文件（基于 mtime）。"""
    with SyncSession() as session:
        result = session.execute(
            select(Document.filename, Document.file_mtime).where(
                Document.filename.in_(all_files),
                Document.status == "ingested",
            )
        )
        ingested_info = {row[0]: row[1] for row in result.fetchall()}

    new_files = []
    for f in all_files:
        if f not in ingested_info:
            new_files.append(f)
        else:
            filepath = os.path.join(docs_dir, f)
            current_mtime = os.path.getmtime(filepath)
            stored_mtime = ingested_info[f]
            if stored_mtime is None:
                new_files.append(f)
            elif abs(current_mtime - stored_mtime) > 1:
                logger.info(f"File modified: {f}, will re-ingest")
                new_files.append(f)

    logger.info(f"Found {len(all_files)} files, {len(ingested_info)} already ingested, {len(new_files)} new")
    return new_files, ingested_info


def _load_and_split(docs_dir, new_files):
    """逐文件加载 + 切分，实时更新进度。返回 (chunks, chunk_counter)。"""
    chunks = []
    md_headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=md_headers)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )

    failed_files = []
    for i, filename in enumerate(new_files):
        try:
            set_ingest_status_sync({
                "status": "running", "stage": "loading",
                "current_file": filename, "progress": i, "total": len(new_files),
            })
            filepath = os.path.join(docs_dir, filename)
            pages = load_document(filepath)
            logger.info(f"{filename}: {len(pages)} pages, {sum(len(p.page_content) for p in pages)} chars")

            if not pages:
                continue

            set_ingest_status_sync({
                "status": "running", "stage": "splitting",
                "current_file": filename, "progress": i, "total": len(new_files),
            })
            for doc in pages:
                ext = os.path.splitext(doc.metadata.get("source", ""))[1].lower()
                if ext in (".md", ".markdown"):
                    md_chunks = md_splitter.split_text(doc.page_content)
                    for c in md_chunks:
                        c.metadata.update({k: v for k, v in doc.metadata.items() if k not in c.metadata})
                    chunks.extend(text_splitter.split_documents(md_chunks))
                else:
                    chunks.extend(text_splitter.split_documents([doc]))
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            failed_files.append({"filename": filename, "error": str(e)})

    # 添加 chunk_index 和 content_hash 元数据
    chunk_counter = Counter()
    file_chunk_index: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        fname = os.path.basename(source)
        chunk_counter[fname] += 1
        file_chunk_index[fname] = file_chunk_index.get(fname, 0)
        chunk.metadata["chunk_index"] = file_chunk_index[fname]
        file_chunk_index[fname] += 1
        chunk.metadata["content_hash"] = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()

    logger.info(f"Split {len(new_files)} files into {len(chunks)} chunks")
    return chunks, chunk_counter


def ingest_documents(directory: str = None, filenames: list[str] = None) -> dict:
    """读取文档，切分，向量入库到 Milvus。增量模式：只处理新文件。"""
    docs_dir = directory or settings.DOCUMENTS_DIR

    if not os.path.exists(docs_dir):
        return {"status": "failed", "error": f"Documents directory not found: {docs_dir}"}

    # 1. 文件发现
    all_files = _discover_files(docs_dir, filenames)
    if not all_files:
        return {"status": "failed", "error": f"No supported files found in {docs_dir}"}

    if not _check_milvus_available():
        raise ServiceUnavailableError(
            f"Milvus 服务不可达 ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})",
        )

    # 2. 增量检测
    new_files, _ = _detect_new_files(docs_dir, all_files)
    if not new_files:
        return {
            "status": "success", "document_count": 0, "chunk_count": 0,
            "filenames": [], "chunk_counts": {}, "collection": settings.MILVUS_COLLECTION,
            "message": "All documents already ingested",
        }

    # 3. 加载 + 切分
    chunks, chunk_counter = _load_and_split(docs_dir, new_files)
    if not chunks:
        return {"status": "failed", "error": "No text extracted from any file"}

    # 4. 内容去重
    client = get_milvus_client()
    chunks, skipped_files = deduplicate_chunks(client, chunks, chunk_counter)

    # 5. 确保 collection 存在
    ensure_collection(client)

    # 6. 删除旧向量
    delete_old_chunks(client, new_files)

    # 7. 如果全部去重，提前返回
    if not chunks:
        skipped_list = [{"filename": f, "reason": r} for f, r in skipped_files.items()]
        result = {
            "status": "success", "document_count": 0, "chunk_count": 0,
            "skipped": skipped_list, "message": "All chunks already exist",
        }
        set_ingest_status_sync(result)
        return result

    # 8. 编码
    texts = [chunk.page_content for chunk in chunks]
    set_ingest_status_sync({
        "status": "running", "stage": "encoding",
        "current_file": "", "progress": len(new_files), "total": len(new_files),
        "chunk_count": len(chunks),
    })
    dense_vecs, sparse_vecs = encode_documents(texts)

    # 9. 写入 Milvus
    set_ingest_status_sync({
        "status": "running", "stage": "storing",
        "current_file": "", "progress": len(new_files), "total": len(new_files),
        "chunk_count": len(chunks),
    })
    insert_chunks(client, chunks, dense_vecs, sparse_vecs)

    # 10. 同步 PostgreSQL
    ingested_files = [f for f in new_files if f not in skipped_files]
    try:
        sync_documents_table(docs_dir, ingested_files, dict(chunk_counter))
    except Exception as e:
        logger.warning(f"Failed to sync documents table: {e}")

    skipped_list = [{"filename": f, "reason": r} for f, r in skipped_files.items()]
    result = {
        "status": "success",
        "document_count": len(ingested_files),
        "chunk_count": len(chunks),
        "filenames": ingested_files,
        "chunk_counts": dict(chunk_counter),
        "skipped": skipped_list,
        "failed": failed_files,
        "collection": settings.MILVUS_COLLECTION,
    }
    set_ingest_status_sync(result)
    return result
