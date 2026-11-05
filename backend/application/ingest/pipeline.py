"""入库主流程：文件发现 → 加载 → 切分 → 去重 → 编码 → 写入 pgvector → 同步文档表。"""
import hashlib
import logging
import os
import uuid
from collections import Counter

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from sqlalchemy import select

from backend.infrastructure.config import settings
from backend.infrastructure.database import SyncSession, ensure_ingest_schema_sync
from backend.domain.errors import ServiceUnavailableError
from backend.domain.models import Document
from backend.infrastructure.encoding import EncodingCancelled, encode_documents
from backend.infrastructure.ingest_store import ingest_cancel_requested_sync, set_ingest_status_sync
from backend.infrastructure.vector_store import (
    check_vector_store_available,
    deduplicate_chunks,
    delete_old_chunks,
    get_vector_store,
    insert_chunks,
)
from .db_sync import sync_documents_table
from .loaders import SUPPORTED_EXTENSIONS, load_document

logger = logging.getLogger("ragmate")


def _stop_requested(cancel_event) -> bool:
    """本地事件或数据库取消标志任一出现即停止，并把标志同步回本地事件。"""
    if cancel_event is not None and cancel_event.is_set():
        return True
    try:
        if ingest_cancel_requested_sync():
            if cancel_event is not None:
                cancel_event.set()
            return True
    except Exception:
        logger.debug("Failed to read ingest cancel flag", exc_info=True)
    return False


def _discover_files(docs_dir: str, filenames: list[str] | None) -> list[str]:
    """发现目录下所有支持的文件，可选过滤指定文件名。
    
    Args:
        docs_dir: 文档目录路径
        filenames: 要过滤的文件名列表，None 表示返回所有文件
        
    Returns:
        排序后的文件名列表
    """
    all_files = sorted(f for f in os.listdir(docs_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS)
    if filenames is not None:
        all_files = [f for f in all_files if f in filenames]
    return all_files


def _detect_new_files(docs_dir: str, all_files: list[str]) -> tuple[list[str], dict]:
    """检测新文件或已修改的文件（基于 mtime）。
    
    Args:
        docs_dir: 文档目录路径
        all_files: 所有待检查的文件名列表
        
    Returns:
        (新文件列表, 已索引文件信息字典 {filename: mtime})
    """
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


def _reconstruct_with_headers(doc):
    """从 MarkdownHeaderTextSplitter 的 metadata 中还原带标题的完整文本。"""
    headers = []
    for key in ["Header 1", "Header 2", "Header 3"]:
        if key in doc.metadata:
            level = int(key.split()[-1])
            headers.append("#" * level + " " + doc.metadata[key])
    if headers:
        return "\n".join(headers) + "\n\n" + doc.page_content
    return doc.page_content


def _title_path_from_metadata(metadata: dict) -> str:
    """从 Header 1/2/3 metadata 拼出「H1 > H2 > H3」路径；无标题则空串。"""
    parts = [metadata[k] for k in ("Header 1", "Header 2", "Header 3") if metadata.get(k)]
    return " > ".join(parts)


def _apply_title_path(body: str, title_path: str) -> str:
    """把标题路径贴进块正文，供向量与 rerank 直接看见章节语境。"""
    if not title_path:
        return body
    return f"标题路径: {title_path}\n\n{body}"


def strip_title_path_prefix(text: str) -> str:
    """去掉 title_path 策略写入的前缀，便于与旧 gold 正文做重叠比对。"""
    if text.startswith("标题路径: "):
        parts = text.split("\n\n", 1)
        return parts[1] if len(parts) > 1 else text
    return text


def _get_text_splitter(ext: str):
    """根据文件扩展名返回对应的 text splitter。"""
    size, overlap = {
        ".md": (settings.CHUNK_SIZE, settings.CHUNK_OVERLAP),
        ".markdown": (settings.CHUNK_SIZE, settings.CHUNK_OVERLAP),
        ".pdf": (settings.CHUNK_SIZE_PDF, settings.CHUNK_OVERLAP_PDF),
        ".docx": (settings.CHUNK_SIZE_DOCX, settings.CHUNK_OVERLAP_DOCX),
        ".txt": (settings.CHUNK_SIZE_TXT, settings.CHUNK_OVERLAP_TXT),
    }.get(ext, (settings.CHUNK_SIZE, settings.CHUNK_OVERLAP))
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)


def _split_markdown_section(section, ext: str) -> list:
    """按 CHUNKING_STRATEGY 切一个 markdown 章节，CHUNK_SIZE 两边对齐。"""
    splitter = _get_text_splitter(ext)
    strategy = (settings.CHUNKING_STRATEGY or "title_path").strip().lower()
    if strategy == "title_path":
        title_path = _title_path_from_metadata(section.metadata)
        # 先按与 parent_child 相同的 CHUNK_SIZE 切正文，再给每个子块贴路径，
        # 避免路径字符挤占额度导致两边切块边界对不齐。
        small_chunks = splitter.split_documents([section])
        parent_body = section.page_content
        parent_text = _apply_title_path(parent_body, title_path)
        for c in small_chunks:
            c.page_content = _apply_title_path(c.page_content, title_path)
            c.metadata["parent_text"] = parent_text
            c.metadata["title_path"] = title_path
            c.metadata["chunking_strategy"] = "title_path"
        return small_chunks

    if strategy != "parent_child":
        logger.warning(
            "Unknown CHUNKING_STRATEGY=%r, falling back to parent_child",
            settings.CHUNKING_STRATEGY,
        )
    small_chunks = splitter.split_documents([section])
    parent_text = _reconstruct_with_headers(section)
    for c in small_chunks:
        c.metadata["parent_text"] = parent_text
        c.metadata["chunking_strategy"] = "parent_child"
    return small_chunks


def _load_and_split(docs_dir, new_files, cancel_event=None):
    """逐文件加载 + 切分，实时更新进度。返回 (chunks, chunk_counter, failed_files)。"""
    chunks = []
    md_headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=md_headers)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE_PARENT, chunk_overlap=settings.CHUNK_OVERLAP_PARENT
    )
    failed_files = []
    for i, filename in enumerate(new_files):
        if _stop_requested(cancel_event):
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
                    sections = md_splitter.split_text(doc.page_content)
                    for section in sections:
                        section.metadata.update({k: v for k, v in doc.metadata.items() if k not in section.metadata})
                        chunks.extend(_split_markdown_section(section, ext))
                else:
                    splitter = _get_text_splitter(ext)
                    if splitter is not None:
                        parents = parent_splitter.split_documents([doc])
                        for parent in parents:
                            small_chunks = splitter.split_documents([parent])
                            for c in small_chunks:
                                c.metadata["parent_text"] = parent.page_content
                            chunks.extend(small_chunks)
                    else:
                        chunks.append(doc)
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

    logger.info(
        f"Split {len(new_files)} files into {len(chunks)} chunks "
        f"(strategy={settings.CHUNKING_STRATEGY})"
    )
    return chunks, chunk_counter, failed_files


def ingest_documents(directory: str = None, filenames: list[str] = None, cancel_event=None) -> dict:
    """读取文档，切分，向量入库到 pgvector。增量模式：只处理新文件。

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

    # 新库直接入库：先建 ORM 表 + 向量表/扩展，否则下面的可用性检查
    # （要求 vector 扩展已存在）和 _detect_new_files（查 documents 表）
    # 会在新库上直接失败。API 已起过的库上重复调用无害。
    ensure_ingest_schema_sync()

    if not check_vector_store_available():
        raise ServiceUnavailableError("向量库不可用（PostgreSQL 未就绪或未安装 pgvector）")

    # 2. 增量检测
    new_files, ingested_info = _detect_new_files(docs_dir, all_files)
    if not new_files:
        return {
            "status": "success", "document_count": 0, "chunk_count": 0,
            "filenames": [], "chunk_counts": {}, "collection": settings.VECTOR_TABLE,
            "message": "All documents already ingested",
        }

    if _stop_requested(cancel_event):
        return {"status": "idle", "message": "Cancelled"}

    # 3. 加载 + 切分
    chunks, chunk_counter, failed_files = _load_and_split(docs_dir, new_files, cancel_event)
    if _stop_requested(cancel_event):
        return {"status": "idle", "message": "Cancelled"}
    if not chunks:
        return {"status": "failed", "error": "No text extracted from any file", "failed": failed_files}

    ingest_batch_id = uuid.uuid4().hex
    for chunk in chunks:
        chunk.metadata["ingest_batch_id"] = ingest_batch_id

    # 4. 内容去重
    client = get_vector_store()
    # 同名文件重新入库时，即使部分 chunk 内容未变化，也必须重新插入到本批次。
    # 否则后续清理旧向量会把这些“未重新插入”的 chunk 一并删掉。
    reingested_files = set(new_files).intersection(ingested_info)
    chunks, skipped_files = deduplicate_chunks(
        client,
        chunks,
        chunk_counter,
        force_include_sources=reingested_files,
    )

    # 5. 确保向量表存在
    client.ensure_schema()

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

    if _stop_requested(cancel_event):
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
    try:
        dense_vecs = encode_documents(texts, cancel_event)
    except EncodingCancelled:
        return {"status": "idle", "message": "Cancelled"}

    # 8. 写入新向量（先插入后删除，失败则旧数据不丢）
    set_ingest_status_sync({
        "status": "running", "stage": "storing",
        "current_file": "", "progress": len(new_files), "total": len(new_files),
        "chunk_count": len(chunks),
    })
    keep_ids_by_source = insert_chunks(client, chunks, dense_vecs)

    # 9. 新向量写入成功后，再删除旧向量（仅限确实有新数据的文件）。
    # 删除失败必须让本次入库失败：不能在旧向量还在时把状态写成 success。
    try:
        delete_old_chunks(
            client,
            {source: keep_ids_by_source.get(source, []) for source in files_with_new_chunks},
            ingest_batch_id=ingest_batch_id,
        )
    except Exception as e:
        logger.error("Failed to delete old vectors; ingest will not be marked success", exc_info=True)
        result = {
            "status": "failed",
            "error": f"删除旧向量失败，本次入库未标记为成功: {e}",
            "ingest_batch_id": ingest_batch_id,
            "chunk_count": len(chunks),
            "failed": failed_files,
            "collection": settings.VECTOR_TABLE,
        }
        set_ingest_status_sync(result)
        return result

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
        "collection": settings.VECTOR_TABLE,
        "ingest_batch_id": ingest_batch_id,
    }
    set_ingest_status_sync(result)
    return result
