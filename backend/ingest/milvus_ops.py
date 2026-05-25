"""Milvus 向量库操作：collection 管理、去重、插入。"""
import logging
import os
import uuid
from collections import Counter

from pymilvus import DataType, CollectionSchema, FieldSchema

from config import settings
from errors import ValidationError

logger = logging.getLogger("ragmate")


def build_source_filter(filename: str) -> str:
    """构建安全的 Milvus metadata source 过滤表达式。

    拒绝包含反斜杠或双引号的文件名（不可能是合法文件名），
    从根本上杜绝过滤器注入。
    """
    if '\\' in filename or '"' in filename:
        raise ValidationError(f"Filename contains invalid characters: {filename}")
    return f'metadata["source"] == "{filename}"'


def ensure_collection(client):
    """确保 Milvus collection 存在且 schema 正确，否则重建。"""
    collection_exists = client.has_collection(settings.MILVUS_COLLECTION)

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

    return True


def deduplicate_chunks(client, chunks, chunk_counter):
    """查询 Milvus 中已存在的 content_hash，跳过重复 chunk。返回 (filtered_chunks, skipped_files)。"""
    skipped_files = {}
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
                if chunk.metadata["content_hash"] in existing_hashes:
                    src = os.path.basename(chunk.metadata.get("source", ""))
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


def delete_old_chunks(client, filenames):
    """删除指定文件的旧向量。"""
    for filename in filenames:
        try:
            client.delete(
                collection_name=settings.MILVUS_COLLECTION,
                filter=build_source_filter(filename),
            )
        except Exception:
            logger.debug(f"Failed to delete old chunks for {filename}", exc_info=True)


def insert_chunks(client, chunks, dense_vecs, sparse_vecs):
    """将编码后的 chunk 写入 Milvus。"""
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{
        "source": os.path.basename(chunk.metadata.get("source", "unknown")),
        "page": chunk.metadata.get("page"),
        "chunk_index": chunk.metadata.get("chunk_index", 0),
        "content_hash": chunk.metadata.get("content_hash", ""),
    } for chunk in chunks]

    data = [
        {"id": uuid.uuid4().int & ((1 << 63) - 1), "dense": d_vec, "sparse": s_vec, "text": text, "metadata": meta}
        for d_vec, s_vec, text, meta in zip(dense_vecs, sparse_vecs, texts, metadatas)
    ]
    client.insert(
        collection_name=settings.MILVUS_COLLECTION,
        data=data,
    )
    client.load_collection(settings.MILVUS_COLLECTION)
