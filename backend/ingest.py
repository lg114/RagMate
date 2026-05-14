import datetime
import logging
import os
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
from errors import ServiceUnavailableError
from model_factory import get_embeddings
from models import Document
from redis_client import set_ingest_status_sync
from retriever import _check_milvus_available, get_milvus_client

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".txt", ".md"}


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
    elif ext in (".docx", ".doc"):
        return Docx2txtLoader(filepath).load()
    elif ext in (".xlsx", ".xls"):
        return UnstructuredExcelLoader(filepath).load()
    elif ext in (".txt", ".md"):
        return TextLoader(filepath, encoding="utf-8").load()
    else:
        logging.getLogger("ragmate").warning(f"Unsupported file type: {ext}")
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
            size_bytes = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            mtime = os.path.getmtime(filepath) if os.path.exists(filepath) else None
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


def ingest_documents(directory: str = None, verbose: bool = False) -> dict:
    """读取文档，切分，向量入库到 Milvus。增量模式：只处理新文件。返回结构化结果 dict。"""
    docs_dir = directory or settings.DOCUMENTS_DIR

    if not os.path.exists(docs_dir):
        return {"status": "failed", "error": f"Documents directory not found: {docs_dir}"}

    all_files = [f for f in os.listdir(docs_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
    if not all_files:
        return {"status": "failed", "error": f"No supported files found in {docs_dir}"}

    if not _check_milvus_available():
        raise ServiceUnavailableError(
            f"Milvus 服务不可达 ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})"
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
                logging.getLogger("ragmate").info(f"File modified: {f}, will re-ingest")
                new_files.append(f)
    logging.getLogger("ragmate").info(f"Found {len(all_files)} files, {len(ingested_info)} already ingested, {len(new_files)} new")
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

    documents = []
    for filename in new_files:
        filepath = os.path.join(docs_dir, filename)
        pages = load_document(filepath)
        logging.getLogger("ragmate").info(f"{filename}: {len(pages)} pages, {sum(len(p.page_content) for p in pages)} chars")
        documents.extend(pages)

    if not documents:
        return {"status": "failed", "error": "No text extracted from any file"}

    # 按文件类型分组切分
    chunks = []
    md_headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=md_headers)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )

    for doc in documents:
        ext = os.path.splitext(doc.metadata.get("source", ""))[1].lower()
        if ext in (".md", ".markdown"):
            md_chunks = md_splitter.split_text(doc.page_content)
            for c in md_chunks:
                c.metadata.update({k: v for k, v in doc.metadata.items() if k not in c.metadata})
            chunks.extend(md_chunks)
        else:
            chunks.extend(text_splitter.split_documents([doc]))

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

    logging.getLogger("ragmate").info(f"Split {len(documents)} pages into {len(chunks)} chunks")

    client = get_milvus_client()

    collection_exists = client.has_collection(settings.MILVUS_COLLECTION)

    # 检查旧 collection 是否有 sparse 字段和索引，没有则重建
    if collection_exists:
        try:
            info = client.describe_collection(settings.MILVUS_COLLECTION)
            field_names = [f["name"] for f in info.get("fields", [])]
            indexes = client.list_indexes(settings.MILVUS_COLLECTION)
            has_sparse_field = "sparse" in field_names
            has_sparse_index = any(idx.get("field") == "sparse" for idx in indexes)
            if not has_sparse_field or not has_sparse_index:
                logging.getLogger("ragmate").info("Schema/index mismatch, dropping old collection...")
                client.drop_collection(settings.MILVUS_COLLECTION)
                collection_exists = False
        except Exception:
            logging.getLogger("ragmate").debug("Failed to check/drop collection schema", exc_info=True)

    if not collection_exists:
        dim = len(get_bge_m3().encode(["dim"])["dense_vecs"][0])
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
            escaped = filename.replace('\\', '\\\\').replace('"', '\\"')
            client.delete(
                collection_name=settings.MILVUS_COLLECTION,
                filter=f'metadata["source"] == "{escaped}"',
            )
        except Exception:
            logging.getLogger("ragmate").debug(f"Failed to delete old chunks for {filename}", exc_info=True)

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{
        "source": os.path.basename(chunk.metadata.get("source", "unknown")),
        "page": chunk.metadata.get("page"),
        "chunk_index": chunk.metadata.get("chunk_index", 0),
    } for chunk in chunks]

    dense_vecs, sparse_vecs = encode_documents(texts)

    import uuid
    data = [
        {"id": uuid.uuid4().int >> 64, "dense": d_vec, "sparse": s_vec, "text": text, "metadata": meta}
        for d_vec, s_vec, text, meta in zip(dense_vecs, sparse_vecs, texts, metadatas)
    ]
    client.insert(
        collection_name=settings.MILVUS_COLLECTION,
        data=data,
    )
    client.load_collection(settings.MILVUS_COLLECTION)

    try:
        _sync_documents_table(docs_dir, new_files, dict(chunk_counter))
    except Exception as e:
        logging.getLogger("ragmate").warning(f"Failed to sync documents table: {e}")

    result = {
        "status": "success",
        "document_count": len(new_files),
        "chunk_count": len(chunks),
        "filenames": new_files,
        "chunk_counts": dict(chunk_counter),
        "collection": settings.MILVUS_COLLECTION,
    }
    set_ingest_status_sync(result)
    return result


if __name__ == "__main__":
    result = ingest_documents(verbose=True)
    logging.getLogger("ragmate").info(result)
