"""入库主流程：文件发现 → 加载 → 切分 → 去重 → 编码 → 写入 Milvus → 同步 PostgreSQL。"""
import hashlib
import logging
import os
import uuid
from collections import Counter

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from sqlalchemy import select

from backend.infrastructure.config import settings
from backend.infrastructure.database import SyncSession
from backend.domain.errors import ServiceUnavailableError
from backend.domain.models import Document
from backend.infrastructure.redis_client import set_ingest_status_sync
from backend.infrastructure.milvus import (
    check_milvus_available,
    deduplicate_chunks,
    delete_old_chunks,
    ensure_collection,
    get_milvus_client,
    insert_chunks,
)
from backend.infrastructure.encoding import encode_documents

from .db_sync import sync_documents_table
from .loaders import SUPPORTED_EXTENSIONS, load_document

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
            stored_mtime = ingested_info[f]
            try:
                current_mtime = os.path.getmtime(filepath)
            except OSError:
                continue  # 文件仍不可访问，跳过
            if stored_mtime is None or abs(current_mtime - stored_mtime) > 1:
                logger.info(f"File modified: {f}, will re-ingest")
                new_files.append(f)

    logger.info(f"Found {len(all_files)} files, {len(ingested_info)} already ingested, {len(new_files)} new")
    return new_files, ingested_info


def _load_and_split(docs_dir, new_files, cancel_event=None):
    """逐文件加载 + 切分，实时更新进度。返回 (chunks, chunk_counter, failed_files)。"""
    chunks = []
    md_headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=md_headers)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )

    failed_files = []
    for i, filename in enumerate(new_files):
        if cancel_event and cancel_event.is_set():
            break
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
    return chunks, chunk_counter, failed_files


def ingest_documents(directory: str = None, filenames: list[str] = None, cancel_event=None) -> dict:
    """读取文档，切分，向量入库到 Milvus。增量模式：只处理新文件。

    Args:
        cancel_event: threading.Event，设置后尽早终止入库流程。
    """
    docs_dir = directory or settings.DOCUMENTS_DIR

    if not os.path.exists(docs_dir):
        return {"status": "failed", "error": f"Documents directory not found: {docs_dir}"}

    # 1. 文件发现
    all_files = _discover_files(docs_dir, filenames)
    if not all_files:
        return {"status": "failed", "error": f"No supported files found in {docs_dir}"}

    if not check_milvus_available():
        raise ServiceUnavailableError(
            f"Milvus 服务不可达 ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})",
        )

    # 2. 增量检测
    new_files, ingested_info = _detect_new_files(docs_dir, all_files)
    if not new_files:
        return {
            "status": "success", "document_count": 0, "chunk_count": 0,
            "filenames": [], "chunk_counts": {}, "collection": settings.MILVUS_COLLECTION,
            "message": "All documents already ingested",
        }

    if cancel_event and cancel_event.is_set():
        return {"status": "idle", "message": "Cancelled"}

    # 3. 加载 + 切分
    chunks, chunk_counter, failed_files = _load_and_split(docs_dir, new_files, cancel_event)
    if cancel_event and cancel_event.is_set():
        return {"status": "idle", "message": "Cancelled"}
    if not chunks:
        return {"status": "failed", "error": "No text extracted from any file", "failed": failed_files}

    ingest_batch_id = uuid.uuid4().hex
    for chunk in chunks:
        chunk.metadata["ingest_batch_id"] = ingest_batch_id

    # 4. 内容去重
    client = get_milvus_client()
    # 同名文件重新入库时，即使部分 chunk 内容未变化，也必须重新插入到本批次。
    # 否则后续清理旧向量会把这些“未重新插入”的 chunk 一并删掉。
    reingested_files = set(new_files).intersection(ingested_info)
    chunks, skipped_files = deduplicate_chunks(
        client,
        chunks,
        chunk_counter,
        force_include_sources=reingested_files,
    )

    # 5. 确保 collection 存在
    ensure_collection(client)

    # 6. 如果全部去重，提前返回（不删除任何旧向量）
    if not chunks:
        skipped_list = [{"filename": f, "reason": r} for f, r in skipped_files.items()]
        result = {
            "status": "success", "document_count": 0, "chunk_count": 0,
            "skipped": skipped_list, "failed": failed_files,
            "message": "All chunks already exist",
        }
        set_ingest_status_sync(result)
        return result

    if cancel_event and cancel_event.is_set():
        return {"status": "idle", "message": "Cancelled"}

    # 7. 编码
    files_with_new_chunks = set()
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        files_with_new_chunks.add(os.path.basename(source))

    texts = [chunk.page_content for chunk in chunks]
    set_ingest_status_sync({
        "status": "running", "stage": "encoding",
        "current_file": "", "progress": len(new_files), "total": len(new_files),
        "chunk_count": len(chunks),
    })
    dense_vecs, sparse_vecs = encode_documents(texts)

    # 8. 写入新向量（先插入后删除，失败则旧数据不丢）
    set_ingest_status_sync({
        "status": "running", "stage": "storing",
        "current_file": "", "progress": len(new_files), "total": len(new_files),
        "chunk_count": len(chunks),
    })
    keep_ids_by_source = insert_chunks(client, chunks, dense_vecs, sparse_vecs)

    # 9. 新向量写入成功后，再删除旧向量（仅限确实有新数据的文件）
    delete_old_chunks(client, {source: keep_ids_by_source[source] for source in files_with_new_chunks})

    # 10. 同步 PostgreSQL
    failed_names = {f["filename"] for f in failed_files}
    ingested_files = [f for f in new_files if f not in skipped_files and f not in failed_names]
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
        "ingest_batch_id": ingest_batch_id,
    }
    set_ingest_status_sync(result)
    return result
