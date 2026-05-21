import datetime
import hashlib
import logging
import os
import uuid
from collections import Counter
from functools import lru_cache

from FlagEmbedding import BGEM3FlagModel
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pymilvus import DataType, CollectionSchema, FieldSchema
from sqlalchemy import select

from config import settings
from database import SyncSession
from errors import ServiceUnavailableError, ValidationError
from models import Document
from redis_client import set_ingest_status_sync
from retriever import _check_milvus_available, get_milvus_client

logger = logging.getLogger("ragmate")


def build_source_filter(filename: str) -> str:
    """构建安全的 Milvus metadata source 过滤表达式。

    拒绝包含反斜杠或双引号的文件名（不可能是合法文件名），
    从根本上杜绝过滤器注入。
    """
    if '\\' in filename or '"' in filename:
        raise ValidationError(f"Filename contains invalid characters: {filename}")
    return f'metadata["source"] == "{filename}"'

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md"}


@lru_cache(maxsize=1)
def get_bge_m3():
    """加载 bge-m3 模型（单例），用于提取 dense + sparse 向量。"""
    return BGEM3FlagModel(settings.EMBEDDING_MODEL, use_fp16=False)


ENCODE_BATCH_SIZE = 64


def encode_documents(texts: list[str]) -> tuple[list, list]:
    """用 bge-m3 提取 dense 和 sparse 向量（分批处理避免内存溢出）。返回 (dense_vecs, sparse_vecs)。"""
    model = get_bge_m3()
    all_dense = []
    all_sparse = []

    for i in range(0, len(texts), ENCODE_BATCH_SIZE):
        batch = texts[i:i + ENCODE_BATCH_SIZE]
        output = model.encode(batch, return_dense=True, return_sparse=True)
        all_dense.extend(output["dense_vecs"].tolist())
        for lexical_weight in output["lexical_weights"]:
            all_sparse.append({int(k): float(v) for k, v in lexical_weight.items()})

    return all_dense, all_sparse


def encode_query(query: str) -> tuple[list, dict]:
    """用 bge-m3 提取 query 的 dense 和 sparse 向量。"""
    model = get_bge_m3()
    output = model.encode([query], return_dense=True, return_sparse=True)
    dense_vec = output["dense_vecs"][0].tolist()
    sparse_vec = {int(k): float(v) for k, v in output["lexical_weights"][0].items()}
    return dense_vec, sparse_vec


def load_document(filepath: str):
    """根据文件扩展名选择合适的 loader 加载文档。"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(filepath).load()
    elif ext == ".docx":
        return Docx2txtLoader(filepath).load()
    elif ext in (".xlsx", ".xls"):
        return UnstructuredExcelLoader(filepath).load()
    elif ext in (".txt", ".md"):
        return TextLoader(filepath, encoding="utf-8").load()
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return []


def _sync_documents_table(directory: str, filenames: list[str], chunk_counts: dict[str, int]):
    """同步 PostgreSQL documents 表：将本次入库的文件标记为 ingested。已有记录则更新，没有则自动创建"""
    now = datetime.datetime.now(datetime.timezone.utc)
    with SyncSession() as session:
        result = session.execute(
            select(Document).where(Document.filename.in_(filenames))
        )
        existing = {doc.filename: doc for doc in result.scalars().all()}

        for filename in filenames:
            doc = existing.get(filename)
            filepath = os.path.join(directory, filename)
            try:
                st = os.stat(filepath)
                size_bytes, mtime = st.st_size, st.st_mtime
            except OSError:
                size_bytes, mtime = 0, None
            if doc:
                doc.status = "ingested"
                doc.chunk_count = chunk_counts.get(filename, 0)
                doc.size_bytes = size_bytes
                doc.file_mtime = mtime
                doc.ingested_at = now
            else:
                doc = Document(
                    filename=filename,
                    size_bytes=size_bytes,
                    file_mtime=mtime,
                    status="ingested",
                    chunk_count=chunk_counts.get(filename, 0),
                    uploaded_at=now,
                    ingested_at=now,
                )
                session.add(doc)
        session.commit()


def ingest_documents(directory: str = None, filenames: list[str] = None, verbose: bool = False) -> dict:
    """读取文档，切分，向量入库到 Milvus。增量模式：只处理新文件。返回结构化结果 dict。

    Args:
        filenames: 指定要入库的文件列表。为 None 时处理目录下所有新文件。
    """
    docs_dir = directory or settings.DOCUMENTS_DIR

    if not os.path.exists(docs_dir):
        return {"status": "failed", "error": f"Documents directory not found: {docs_dir}"}

    all_files = sorted(f for f in os.listdir(docs_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS)
    if not all_files:
        return {"status": "failed", "error": f"No supported files found in {docs_dir}"}

    # 如果指定了文件列表，只处理指定的文件
    if filenames is not None:
        all_files = [f for f in all_files if f in filenames]
        if not all_files:
            return {"status": "failed", "error": "None of the specified files found in documents directory"}

    if not _check_milvus_available():
        raise ServiceUnavailableError(
            f"Milvus 服务不可达 ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})",
        )

    with SyncSession() as session:
        result = session.execute(
            select(Document.filename, Document.file_mtime).where(
                Document.filename.in_(all_files),
                Document.status == "ingested",
            )
        )
        ingested_info = {row[0]: row[1] for row in result.fetchall()}

    # 检查新文件或文件已修改（mtime 变化）
    new_files = []
    for f in all_files:
        if f not in ingested_info:
            new_files.append(f)
        else:
            filepath = os.path.join(docs_dir, f)
            current_mtime = os.path.getmtime(filepath)
            stored_mtime = ingested_info[f]
            if stored_mtime is None:
                # 旧记录没有 mtime，重新入库以补上
                new_files.append(f)
            elif abs(current_mtime - stored_mtime) > 1:
                logger.info(f"File modified: {f}, will re-ingest")
                new_files.append(f)
    logger.info(f"Found {len(all_files)} files, {len(ingested_info)} already ingested, {len(new_files)} new")
    if not new_files:
        return {
            "status": "success",
            "document_count": 0,
            "chunk_count": 0,
            "filenames": [],
            "chunk_counts": {},
            "collection": settings.MILVUS_COLLECTION,
            "message": "All documents already ingested",
        }

    # 逐文件加载 + 切分，实时更新进度
    chunks = []
    md_headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=md_headers)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )

    for i, filename in enumerate(new_files):
        # 阶段 1：加载
        set_ingest_status_sync({
            "status": "running", "stage": "loading",
            "current_file": filename, "progress": i, "total": len(new_files),
        })
        filepath = os.path.join(docs_dir, filename)
        pages = load_document(filepath)
        logger.info(f"{filename}: {len(pages)} pages, {sum(len(p.page_content) for p in pages)} chars")

        if not pages:
            continue

        # 阶段 2：切分
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

    if not chunks:
        return {"status": "failed", "error": "No text extracted from any file"}

    # 添加 chunk_index 元数据
    chunk_counter = Counter()
    file_chunk_index: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        filename = os.path.basename(source)
        chunk_counter[filename] += 1
        file_chunk_index[filename] = file_chunk_index.get(filename, 0)
        chunk.metadata["chunk_index"] = file_chunk_index[filename]
        file_chunk_index[filename] += 1

    # 计算 chunk 内容 hash，用于去重
    for chunk in chunks:
        chunk.metadata["content_hash"] = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()

    logger.info(f"Split {len(new_files)} files into {len(chunks)} chunks")

    # 内容去重：查询 Milvus 中已存在的 hash，跳过重复 chunk
    skipped_files = {}  # filename -> reason
    client = get_milvus_client()
    collection_exists = client.has_collection(settings.MILVUS_COLLECTION)

    if collection_exists and chunks:
        new_hashes = list({c.metadata["content_hash"] for c in chunks})
        try:
            # 分批查询（Milvus filter 表达式长度限制）
            existing_hashes = set()
            batch_size = 100
            for i in range(0, len(new_hashes), batch_size):
                batch = new_hashes[i:i + batch_size]
                hash_list = ", ".join(f'"{h}"' for h in batch)
                expr = f'metadata["content_hash"] in [{hash_list}]'
                result = client.query(
                    collection_name=settings.MILVUS_COLLECTION,
                    filter=expr,
                    output_fields=["metadata"],
                )
                for r in result:
                    h = r.get("metadata", {}).get("content_hash")
                    if h:
                        existing_hashes.add(h)

            if existing_hashes:
                original_count = len(chunks)
                # 按文件统计跳过情况
                file_skip_count = Counter()
                filtered_chunks = []
                for chunk in chunks:
                    if chunk.metadata["content_hash"] in existing_hashes:
                        src = os.path.basename(chunk.metadata.get("source", ""))
                        file_skip_count[src] += 1
                    else:
                        filtered_chunks.append(chunk)

                # 如果某个文件的全部 chunk 都被跳过，记录为 skipped
                for src, skip_cnt in file_skip_count.items():
                    total_in_file = chunk_counter.get(src, 0)
                    if skip_cnt >= total_in_file:
                        skipped_files[src] = f"内容与其他文档完全重复"
                        logger.info(f"Skip {src}: all {skip_cnt} chunks already exist")
                    else:
                        logger.info(f"Dedup {src}: {skip_cnt}/{total_in_file} chunks already exist")

                chunks = filtered_chunks
                logger.info(f"Content dedup: {original_count} -> {len(chunks)} chunks ({original_count - len(chunks)} skipped)")
        except Exception as e:
            logger.warning(f"Content dedup query failed, proceeding without dedup: {e}")

    # 检查旧 collection 是否有 sparse 字段和索引，没有则重建
    if collection_exists:
        try:
            info = client.describe_collection(settings.MILVUS_COLLECTION)
            field_names = [f["name"] for f in info.get("fields", [])]
            indexes = client.list_indexes(settings.MILVUS_COLLECTION)
            has_sparse_field = "sparse" in field_names
            has_sparse_index = any(idx.get("field") == "sparse" for idx in indexes)
            if not has_sparse_field or not has_sparse_index:
                logger.info("Schema/index mismatch, dropping old collection...")
                client.drop_collection(settings.MILVUS_COLLECTION)
                collection_exists = False
        except Exception:
            logger.debug("Failed to check/drop collection schema", exc_info=True)

    if not collection_exists:
        dim = 1024  # BGE-M3 固定输出维度
        schema = CollectionSchema(fields=[
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("dense", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("metadata", DataType.JSON),
        ])
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="IP")
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"drop_ratio_build": 0.2})
        client.create_collection(
            collection_name=settings.MILVUS_COLLECTION,
            schema=schema,
            index_params=index_params,
        )
        client.load_collection(settings.MILVUS_COLLECTION)

    for filename in new_files:
        try:
            client.delete(
                collection_name=settings.MILVUS_COLLECTION,
                filter=build_source_filter(filename),
            )
        except Exception:
            logger.debug(f"Failed to delete old chunks for {filename}", exc_info=True)

    # 如果所有 chunk 都被去重跳过
    if not chunks:
        skipped_list = [{"filename": f, "reason": r} for f, r in skipped_files.items()]
        result = {
            "status": "success",
            "document_count": 0,
            "chunk_count": 0,
            "skipped": skipped_list,
            "message": "All chunks already exist",
        }
        set_ingest_status_sync(result)
        return result

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{
        "source": os.path.basename(chunk.metadata.get("source", "unknown")),
        "page": chunk.metadata.get("page"),
        "chunk_index": chunk.metadata.get("chunk_index", 0),
        "content_hash": chunk.metadata.get("content_hash", ""),
    } for chunk in chunks]

    # 阶段 3：编码
    set_ingest_status_sync({
        "status": "running", "stage": "encoding",
        "current_file": "", "progress": len(new_files), "total": len(new_files),
        "chunk_count": len(chunks),
    })
    dense_vecs, sparse_vecs = encode_documents(texts)

    # 阶段 4：写入向量库
    set_ingest_status_sync({
        "status": "running", "stage": "storing",
        "current_file": "", "progress": len(new_files), "total": len(new_files),
        "chunk_count": len(chunks),
    })

    data = [
        {"id": uuid.uuid4().int & ((1 << 63) - 1), "dense": d_vec, "sparse": s_vec, "text": text, "metadata": meta}
        for d_vec, s_vec, text, meta in zip(dense_vecs, sparse_vecs, texts, metadatas)
    ]
    client.insert(
        collection_name=settings.MILVUS_COLLECTION,
        data=data,
    )
    client.load_collection(settings.MILVUS_COLLECTION)

    # 从 new_files 中移除被跳过的文件
    ingested_files = [f for f in new_files if f not in skipped_files]

    try:
        _sync_documents_table(docs_dir, ingested_files, dict(chunk_counter))
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
        "collection": settings.MILVUS_COLLECTION,
    }
    set_ingest_status_sync(result)
    return result


if __name__ == "__main__":
    result = ingest_documents(verbose=True)
    logger.info(result)
