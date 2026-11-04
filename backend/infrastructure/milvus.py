"""Milvus 向量库基础设施：客户端管理、collection 操作、去重、插入。"""
import logging
import os
import socket
import threading
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


def reset_collection_loaded_flag():
    """切换 MILVUS_COLLECTION 后调用，迫使下次 init_milvus 重新 load。"""
    global _collection_loaded
    with _milvus_lock:
        _collection_loaded = False


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
    # 仅匹配 "文件名 (数字).ext" 这种典型的副本后缀格式（空格+括号+纯数字）
    if base.endswith(")") and " (" in base:
        idx = base.rfind(" (")
        if base[idx + 2 : -1].isdigit():
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
            has_parent_text = "parent_text" in field_names
            if not has_sparse_field or not has_sparse_index or not has_parent_text:
                missing = []
                if not has_sparse_field:
                    missing.append("sparse field")
                if not has_sparse_index:
                    missing.append("sparse index")
                if not has_parent_text:
                    missing.append("parent_text field")
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
            FieldSchema("parent_text", DataType.VARCHAR, max_length=65535),
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


_QUERY_PAGE = 1000
_QUERY_HARD_CAP = 100_000
_DELETE_BATCH = 500


def _as_milvus_id(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _row_batch_id(row: dict) -> str:
    """行上的 ingest_batch_id。字段不存在或为空时返回空串。"""
    meta = row.get("metadata") or {}
    if not isinstance(meta, dict):
        return ""
    batch = meta.get("ingest_batch_id")
    if not isinstance(batch, str):
        return ""
    return batch.strip()


def _query_rows_by_filter(
    client,
    filter_expr: str,
    output_fields: list[str],
    *,
    page_limit: int | None = None,
) -> list[dict]:
    """按过滤条件读出向量。page_limit 给定时只取一页，用于删除后的存在性确认。"""
    def _once(**extra):
        batch = client.query(
            collection_name=settings.MILVUS_COLLECTION,
            filter=filter_expr,
            output_fields=output_fields,
            consistency_level="Strong",
            **extra,
        )
        if not isinstance(batch, list):
            raise ServiceUnavailableError("向量查询结果无法确认")
        return batch

    if page_limit is not None:
        return _once(limit=page_limit)

    rows: list[dict] = []
    offset = 0
    while True:
        batch = _once(limit=_QUERY_PAGE, offset=offset)
        rows.extend(batch)
        if len(batch) < _QUERY_PAGE:
            return rows
        offset += len(batch)
        if offset > _QUERY_HARD_CAP:
            raise ServiceUnavailableError("向量数量异常，无法确认旧向量已删除")


def _stale_vector_ids(
    rows: list[dict],
    ingest_batch_id: str | None,
    keep_ids: set[int],
) -> list[int]:
    """挑出要删除的主键。

    行上有 ingest_batch_id 时，只保留最新批次。没有该字段（或为空）时，
    退回「不在本批插入 id 里」的规则，用来清掉更早写入的向量。
    """
    stale: list[int] = []
    for row in rows:
        row_id = _as_milvus_id(row.get("id"))
        batch = _row_batch_id(row)
        if ingest_batch_id and batch:
            if batch == ingest_batch_id:
                continue
            if row_id is None:
                raise ServiceUnavailableError("旧向量缺少主键，无法确认删除")
            stale.append(row_id)
            continue
        if row_id is not None and row_id in keep_ids:
            continue
        if row_id is None:
            raise ServiceUnavailableError("旧向量缺少主键，无法确认删除")
        stale.append(row_id)
    return stale


def _delete_ids(client, ids: list[int]) -> None:
    for start in range(0, len(ids), _DELETE_BATCH):
        client.delete(
            collection_name=settings.MILVUS_COLLECTION,
            ids=ids[start:start + _DELETE_BATCH],
        )


def delete_source_vectors(client, filename: str) -> None:
    """删除某个文件的全部向量，并确认已经不存在。

    失败抛 ServiceUnavailableError。调用方必须在此成功之后才能提交数据库删除。
    """
    if client is None:
        raise ServiceUnavailableError("向量库不可用，已保留文档记录")
    try:
        exists = client.has_collection(settings.MILVUS_COLLECTION)
    except Exception as e:
        logger.error("Milvus unavailable during document delete for %s", filename, exc_info=True)
        raise ServiceUnavailableError("向量库不可用，已保留文档记录") from e
    if not exists:
        return

    source_filter = build_source_filter(filename)
    try:
        client.delete(
            collection_name=settings.MILVUS_COLLECTION,
            filter=source_filter,
        )
        leftover = _query_rows_by_filter(client, source_filter, ["id"], page_limit=1)
    except ServiceUnavailableError:
        raise
    except Exception as e:
        logger.error("Milvus delete failed for %s", filename, exc_info=True)
        raise ServiceUnavailableError("向量删除失败，已保留文档记录") from e
    if leftover:
        logger.error("Milvus delete left vectors for %s", filename)
        raise ServiceUnavailableError("向量删除未完成，已保留文档记录")


def delete_old_chunks(
    client,
    keep_ids_by_source: dict[str, list[int]],
    *,
    ingest_batch_id: str | None = None,
) -> None:
    """删除指定文件的旧向量。

    优先按 ingest_batch_id 去掉非最新批次。行上没有该字段时，保留本批插入的 id。
    删除没有确认完成就抛 ServiceUnavailableError，调用方不得把入库记为成功。
    """
    if not keep_ids_by_source:
        return
    batch_id = None
    if ingest_batch_id is not None:
        if not isinstance(ingest_batch_id, str) or not ingest_batch_id.strip():
            raise ServiceUnavailableError("入库批次号无效，无法删除旧向量")
        batch_id = ingest_batch_id.strip()

    for filename, keep_ids in keep_ids_by_source.items():
        keep = {item_id for item_id in (keep_ids or []) if isinstance(item_id, int)}
        source_filter = build_source_filter(filename)
        try:
            rows = _query_rows_by_filter(client, source_filter, ["id", "metadata"])
            stale = _stale_vector_ids(rows, batch_id, keep)
            if not stale:
                continue
            _delete_ids(client, stale)
            remaining = _query_rows_by_filter(client, source_filter, ["id", "metadata"])
            left = _stale_vector_ids(remaining, batch_id, keep)
            if left:
                raise ServiceUnavailableError(f"旧向量删除未完成: {filename}")
        except ServiceUnavailableError:
            logger.error("Failed to delete old chunks for %s", filename, exc_info=True)
            raise
        except Exception as e:
            logger.error("Failed to delete old chunks for %s", filename, exc_info=True)
            raise ServiceUnavailableError(f"删除旧向量失败: {filename}") from e


def insert_chunks(client, chunks, dense_vecs, sparse_vecs) -> dict[str, list[int]]:
    """将编码后的 chunk 写入 Milvus，并返回按来源文件分组的新 id。"""
    texts = [chunk.page_content for chunk in chunks]
    parent_texts = [chunk.metadata.pop("parent_text", chunk.page_content) for chunk in chunks]
    metadatas = [{
        "source": os.path.basename(chunk.metadata.get("source", "unknown")),
        "page": chunk.metadata.get("page"),
        "chunk_index": chunk.metadata.get("chunk_index", 0),
        "content_hash": chunk.metadata.get("content_hash", ""),
        "ingest_batch_id": chunk.metadata.get("ingest_batch_id", ""),
    } for chunk in chunks]

    data = [
        {"dense": d_vec, "sparse": s_vec, "text": text, "parent_text": parent_text, "metadata": meta}
        for d_vec, s_vec, text, parent_text, meta in zip(dense_vecs, sparse_vecs, texts, parent_texts, metadatas)
    ]
    result = client.insert(
        collection_name=settings.MILVUS_COLLECTION,
        data=data,
    )
    client.load_collection(settings.MILVUS_COLLECTION)

    # auto_id=True 时 Milvus 自动分配 id，从返回值中获取
    ids = result.get("ids", [])
    ids_by_source: dict[str, list[int]] = {}
    for item_id, meta in zip(ids, metadatas):
        ids_by_source.setdefault(meta["source"], []).append(item_id)
    return ids_by_source
