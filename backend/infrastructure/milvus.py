"""Milvus 向量库基础设施：客户端管理、collection 操作、去重、插入。"""
import logging
import os
import socket
import threading
import uuid
from collections import Counter

from pymilvus import AnnSearchRequest, CollectionSchema, DataType, FieldSchema, MilvusClient, RRFRanker

from backend.domain.errors import ServiceUnavailableError, ValidationError
from backend.infrastructure.config import settings

logger = logging.getLogger("ragmate")

# ── 客户端管理 ──────────────────────────────────────────────────────────────

_milvus_client: MilvusClient | None = None
_milvus_lock = threading.Lock()
_collection_loaded: bool = False


def get_milvus_client() -> MilvusClient:
    """获取复用的 MilvusClient 实例（线程安全）"""
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client
    with _milvus_lock:
        if _milvus_client is None:
            _milvus_client = MilvusClient(
                uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
                timeout=settings.MILVUS_TIMEOUT,
            )
        return _milvus_client


def check_milvus_available() -> bool:
    """快速检查 Milvus 服务是否可达"""
    try:
        with socket.create_connection(
            (settings.MILVUS_HOST, settings.MILVUS_PORT),
            timeout=3,
        ):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def init_milvus():
    """确保 Milvus collection 已加载（线程安全）。collection 不存在时返回 None。"""
    global _collection_loaded
    client = get_milvus_client()
    with _milvus_lock:
        if _collection_loaded:
            return client
        try:
            if not client.has_collection(settings.MILVUS_COLLECTION):
                return None
            client.load_collection(settings.MILVUS_COLLECTION)
            _collection_loaded = True
        except Exception as e:
            raise ServiceUnavailableError("检索服务异常，请稍后重试") from e
    return client


def canonical_source(source: str) -> str:
    """归一化来源文件名，用于检索去重。"""
    if not source:
        return ""
    base, ext = os.path.splitext(source)
    for suffix in ["_副本", " (副本)", "_copy", " (copy)"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.endswith(")") and "(" in base:
        idx = base.rfind("(")
        if base[idx + 1 : -1].isdigit():
            base = base[:idx]
    return (base + ext).lower()


# ── Collection 操作 ──────────────────────────────────────────────────────────

def build_source_filter(filename: str) -> str:
    """构建安全的 Milvus metadata source 过滤表达式。

    拒绝包含反斜杠或双引号的文件名（不可能是合法文件名），
    从根本上杜绝过滤器注入。
    """
    if '\\' in filename or '"' in filename:
        raise ValidationError(f"Filename contains invalid characters: {filename}")
    return f'metadata["source"] == "{filename}"'


def ensure_collection(client):
    """确保 Milvus collection 存在。不存在则创建；schema 不匹配则报错要求手动处理。"""
    collection_exists = client.has_collection(settings.MILVUS_COLLECTION)

    if collection_exists:
        try:
            info = client.describe_collection(settings.MILVUS_COLLECTION)
            field_names = [f["name"] for f in info.get("fields", [])]
            indexes = client.list_indexes(settings.MILVUS_COLLECTION)
            has_sparse_field = "sparse" in field_names
            has_sparse_index = any(idx.get("field") == "sparse" for idx in indexes)
            if not has_sparse_field or not has_sparse_index:
                missing = []
                if not has_sparse_field:
                    missing.append("sparse field")
                if not has_sparse_index:
                    missing.append("sparse index")
                raise ValidationError(
                    f"Milvus collection '{settings.MILVUS_COLLECTION}' schema 不匹配"
                    f"（缺少: {', '.join(missing)}）。"
                    f"请手动确认后删除旧集合重建："
                    f"  from pymilvus import utility; utility.drop_collection('{settings.MILVUS_COLLECTION}')"
                )
        except ValidationError:
            raise
        except Exception:
            logger.warning("Failed to describe collection schema, will attempt to use as-is", exc_info=True)

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

    return True


def deduplicate_chunks(client, chunks, chunk_counter, force_include_sources: set[str] | None = None):
    """查询 Milvus 中已存在的 content_hash，跳过重复 chunk。返回 (filtered_chunks, skipped_files)。"""
    skipped_files = {}
    force_include_sources = force_include_sources or set()
    collection_exists = client.has_collection(settings.MILVUS_COLLECTION)

    if not collection_exists or not chunks:
        return chunks, skipped_files

    new_hashes = list({c.metadata["content_hash"] for c in chunks})
    try:
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
            file_skip_count = Counter()
            filtered_chunks = []
            for chunk in chunks:
                src = os.path.basename(chunk.metadata.get("source", ""))
                if src not in force_include_sources and chunk.metadata["content_hash"] in existing_hashes:
                    file_skip_count[src] += 1
                else:
                    filtered_chunks.append(chunk)

            for src, skip_cnt in file_skip_count.items():
                total_in_file = chunk_counter.get(src, 0)
                if skip_cnt >= total_in_file:
                    skipped_files[src] = "内容与其他文档完全重复"
                    logger.info(f"Skip {src}: all {skip_cnt} chunks already exist")
                else:
                    logger.info(f"Dedup {src}: {skip_cnt}/{total_in_file} chunks already exist")

            chunks = filtered_chunks
            logger.info(f"Content dedup: {original_count} -> {len(chunks)} chunks ({original_count - len(chunks)} skipped)")
    except Exception as e:
        logger.warning(f"Content dedup query failed, proceeding without dedup: {e}")

    return chunks, skipped_files


def build_stale_source_filter(filename: str, keep_ids: list[int]) -> str:
    """构建删除同源旧向量的过滤表达式，保留本批次刚插入的 id。"""
    if not keep_ids:
        raise ValidationError(f"No inserted ids provided for source cleanup: {filename}")
    if any(not isinstance(item_id, int) for item_id in keep_ids):
        raise ValidationError(f"Invalid Milvus ids for source cleanup: {filename}")
    keep_list = ", ".join(str(item_id) for item_id in keep_ids)
    return f'{build_source_filter(filename)} and id not in [{keep_list}]'


def delete_old_chunks(client, keep_ids_by_source: dict[str, list[int]]):
    """删除指定文件的旧向量，保留本批次刚插入的新向量。"""
    if not keep_ids_by_source:
        return
    for filename, keep_ids in keep_ids_by_source.items():
        try:
            client.delete(
                collection_name=settings.MILVUS_COLLECTION,
                filter=build_stale_source_filter(filename, keep_ids),
            )
        except Exception:
            logger.debug(f"Failed to delete old chunks for {filename}", exc_info=True)


def insert_chunks(client, chunks, dense_vecs, sparse_vecs) -> dict[str, list[int]]:
    """将编码后的 chunk 写入 Milvus，并返回按来源文件分组的新 id。"""
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{
        "source": os.path.basename(chunk.metadata.get("source", "unknown")),
        "page": chunk.metadata.get("page"),
        "chunk_index": chunk.metadata.get("chunk_index", 0),
        "content_hash": chunk.metadata.get("content_hash", ""),
        "ingest_batch_id": chunk.metadata.get("ingest_batch_id", ""),
    } for chunk in chunks]

    ids = [uuid.uuid4().int & ((1 << 63) - 1) for _ in chunks]
    data = [
        {"id": item_id, "dense": d_vec, "sparse": s_vec, "text": text, "metadata": meta}
        for item_id, d_vec, s_vec, text, meta in zip(ids, dense_vecs, sparse_vecs, texts, metadatas)
    ]
    client.insert(
        collection_name=settings.MILVUS_COLLECTION,
        data=data,
    )
    client.load_collection(settings.MILVUS_COLLECTION)

    ids_by_source: dict[str, list[int]] = {}
    for item_id, meta in zip(ids, metadatas):
        ids_by_source.setdefault(meta["source"], []).append(item_id)
    return ids_by_source
