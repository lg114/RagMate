import socket
from functools import lru_cache
from typing import List

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker
from sentence_transformers import CrossEncoder

from config import settings

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


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """加载 Reranker 模型（单例）。"""
    return CrossEncoder(settings.RERANKER_MODEL)


def retrieve(query: str, k: int = None) -> List[dict]:
    """混合检索 + Reranking。返回 [{text, source, page}, ...]。空列表表示无匹配。"""
    import logging
    from errors import RetrievalError
    from ingest import encode_query

    if k is None:
        k = settings.RERANKER_TOP_K

    try:
        client = get_milvus_client()
        client.load_collection(settings.MILVUS_COLLECTION)
        dense_vec, sparse_vec = encode_query(query)

        # 混合检索：dense + sparse，多取一些给 reranker
        over_fetch = k * 3

        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense",
            param={"metric_type": "IP", "params": {}},
            limit=over_fetch,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=over_fetch,
        )

        results = client.hybrid_search(
            collection_name=settings.MILVUS_COLLECTION,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=over_fetch,
            output_fields=["text", "metadata"],
        )

        if not results or not results[0]:
            return []

        # 提取候选
        candidates = []
        for hit in results[0]:
            entity = hit["entity"]
            meta = entity.get("metadata", {})
            candidates.append({
                "text": entity["text"],
                "source": meta.get("source", ""),
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
            })

        # Reranking
        reranker = get_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)

        for c, score in zip(candidates, scores):
            c["score"] = float(score)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k]

    except RetrievalError:
        raise
    except Exception as e:
        logging.getLogger("ragmate").error(f"Retrieval failed: {e}", exc_info=True)
        raise RetrievalError() from e


if __name__ == "__main__":
    results = retrieve("test query", k=2)
    print(f"Retrieved {len(results)} docs" if results else "No docs found")
    for r in results:
        print(f"  [{r.get('score', 0):.3f}] {r['source']} p{r.get('page', '?')}: {r['text'][:80]}...")
