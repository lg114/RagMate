import socket
from typing import List

from pymilvus import MilvusClient

from config import settings
from model_factory import get_embeddings

_milvus_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    """获取复用的 MilvusClient 实例"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            timeout=settings.MILVUS_TIMEOUT,
        )
    return _milvus_client


def _check_milvus_available() -> bool:
    """快速检查 Milvus 服务是否可达"""
    try:
        sock = socket.create_connection(
            (settings.MILVUS_HOST, settings.MILVUS_PORT),
            timeout=3,
        )
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def retrieve(query: str, k: int = 5) -> List[str]:
    """从 Milvus 向量数据库检索最相关的 k 个文档片段"""
    embeddings = get_embeddings()
    client = get_milvus_client()
    query_vec = embeddings.embed_query(query)

    results = client.search(
        collection_name=settings.MILVUS_COLLECTION,
        data=[query_vec],
        limit=k,
        output_fields=["text"],
    )
    if not results or not results[0]:
        return []
    return [hit["entity"]["text"] for hit in results[0]]


if __name__ == "__main__":
    result = retrieve("test query", k=2)
    print(f"Retrieved {len(result)} docs" if result else "No docs found")
