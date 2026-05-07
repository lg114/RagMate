import datetime
import logging
import os
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select

from config import settings
from database import SyncSession
from errors import ServiceUnavailableError
from model_factory import get_embeddings
from models import Document
from redis_client import set_ingest_status_sync
from retriever import _check_milvus_available, get_milvus_client


def _sync_documents_table(directory: str, filenames: list[str], chunk_counts: dict[str, int]):
    """同步 PostgreSQL documents 表：将本次入库的文件标记为 ingested。已有记录则更新，没有则自动创建"""
    now = datetime.datetime.now(datetime.timezone.utc)
    with SyncSession() as session:
        # 批量查询，一次 DB 往返，避免 N+1
        result = session.execute(
            select(Document).where(Document.filename.in_(filenames))
        )
        existing = {doc.filename: doc for doc in result.scalars().all()}

        for filename in filenames:
            doc = existing.get(filename)
            filepath = os.path.join(directory, filename)
            size_bytes = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            if doc:
                doc.status = "ingested"
                doc.chunk_count = chunk_counts.get(filename, 0)
                doc.size_bytes = size_bytes
                doc.ingested_at = now
            else:
                doc = Document(
                    filename=filename,
                    size_bytes=size_bytes,
                    status="ingested",
                    chunk_count=chunk_counts.get(filename, 0),
                    uploaded_at=now,
                    ingested_at=now,
                )
                session.add(doc)
        session.commit()


def ingest_documents(directory: str = None, verbose: bool = False) -> dict:
    """读取 PDF 文档，切分，向量入库到 Milvus。增量模式：只处理新文件。返回结构化结果 dict。"""
    docs_dir = directory or settings.DOCUMENTS_DIR

    if not os.path.exists(docs_dir):
        return {"status": "failed", "error": f"Documents directory not found: {docs_dir}"}

    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]
    if not pdf_files:
        return {"status": "failed", "error": f"No PDF files found in {docs_dir}"}

    if not _check_milvus_available():
        raise ServiceUnavailableError(
            f"Milvus 服务不可达 ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})"
        )

    # 查找已入库的文件，只处理新文件
    with SyncSession() as session:
        result = session.execute(
            select(Document.filename).where(
                Document.filename.in_(pdf_files),
                Document.status == "ingested",
            )
        )
        ingested_filenames = {row[0] for row in result.fetchall()}

    new_files = [f for f in pdf_files if f not in ingested_filenames]
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

    if verbose:
        logging.getLogger(__name__).info(
            f"Skipping {len(ingested_filenames)} already ingested, "
            f"processing {len(new_files)} new file(s)"
        )

    documents = []
    for pdf_file in new_files:
        pdf_path = os.path.join(docs_dir, pdf_file)
        if verbose:
            logging.getLogger(__name__).debug(f"Loading {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    if verbose:
        logging.getLogger(__name__).debug(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    if verbose:
        logging.getLogger(__name__).debug(f"Split into {len(chunks)} chunks")

    chunk_counter = Counter()
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        filename = os.path.basename(source)
        chunk_counter[filename] += 1

    embeddings = get_embeddings()

    client = get_milvus_client()

    collection_exists = client.has_collection(settings.MILVUS_COLLECTION)

    if not collection_exists:
        dim = len(embeddings.embed_query("dim"))
        client.create_collection(
            collection_name=settings.MILVUS_COLLECTION,
            dimension=dim,
            metric_type="IP",
            vector_field_name="dense",
        )
        if verbose:
            logging.getLogger(__name__).info(f"Created collection: {settings.MILVUS_COLLECTION}")

    # 计算 ID 偏移量，避免与已有数据主键冲突
    id_offset = 0
    if collection_exists:
        try:
            stats = client.get_collection_stats(settings.MILVUS_COLLECTION)
            id_offset = stats.get("row_count", 0)
        except Exception:
            pass

    # 删除待处理文件已有的 chunks（处理重新入库场景）
    for filename in new_files:
        try:
            escaped = filename.replace('"', '\\"')
            client.delete(
                collection_name=settings.MILVUS_COLLECTION,
                filter=f'metadata["source"] == "{escaped}"',
            )
        except Exception:
            pass

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{"source": os.path.basename(chunk.metadata.get("source", "unknown"))} for chunk in chunks]

    vecs = embeddings.embed_documents(texts)

    data = [
        {"id": id_offset + i, "dense": vec, "text": text, "metadata": meta}
        for i, (vec, text, meta) in enumerate(zip(vecs, texts, metadatas))
    ]
    client.insert(
        collection_name=settings.MILVUS_COLLECTION,
        data=data,
    )
    if verbose:
        logging.getLogger(__name__).info(f"Inserted {len(texts)} chunks into Milvus")

    # 同步 PostgreSQL
    try:
        _sync_documents_table(docs_dir, new_files, dict(chunk_counter))
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to sync documents table: {e}")

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
    logging.getLogger(__name__).info(result)
