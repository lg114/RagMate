import logging
import socket
import threading
from functools import lru_cache
from typing import List

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker
from sentence_transformers import CrossEncoder

from config import settings
from source_utils import canonical_source

logger = logging.getLogger("ragmate")

_milvus_client: MilvusClient | None = None
_milvus_lock = threading.Lock()
_collection_loaded: bool = False


# ── Milvus ──────────────────────────────────────────────────────────────────

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


def _init_milvus():
    """确保 Milvus collection 已加载。"""
    from errors import RetrievalError

    global _collection_loaded
    try:
        client = get_milvus_client()
        if not _collection_loaded:
            client.load_collection(settings.MILVUS_COLLECTION)
            _collection_loaded = True
    except Exception as e:
        if "collection" in str(e).lower() or "not loaded" in str(e).lower():
            _collection_loaded = False
            try:
                client.load_collection(settings.MILVUS_COLLECTION)
                _collection_loaded = True
            except Exception:
                raise RetrievalError() from e
        else:
            raise RetrievalError() from e
    return client


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """加载 Reranker 模型（单例）。"""
    return CrossEncoder(settings.RERANKER_MODEL)


# ── Search ──────────────────────────────────────────────────────────────────

def _do_search(client, dense_vec, sparse_vec, over_fetch: int):
    """执行一次混合检索或 dense 检索。"""
    if settings.HYBRID_SEARCH_ENABLED:
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
        return client.hybrid_search(
            collection_name=settings.MILVUS_COLLECTION,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=over_fetch,
            output_fields=["text", "metadata"],
        )
    else:
        return client.search(
            collection_name=settings.MILVUS_COLLECTION,
            data=[dense_vec],
            anns_field="dense",
            param={"metric_type": "IP", "params": {}},
            limit=over_fetch,
            output_fields=["text", "metadata"],
        )


def _extract_candidates(results) -> list[dict]:
    """从 Milvus 结果中提取候选。"""
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
    return candidates


# ── Reranking + Filtering ──────────────────────────────────────────────────

def _rerank_and_score(query: str, candidates: list[dict]) -> tuple[list[dict], float]:
    """Rerank 候选并返回 (scored_list, top_score)。"""
    if not candidates:
        return [], 0.0
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["score"] = float(s)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates, candidates[0]["score"]


def _filter_and_dedup(candidates: list[dict], threshold: float, k: int) -> list[dict]:
    """过滤低分 + source 去重 + 截取 top-k。"""
    filtered_by_score = [c for c in candidates if c["score"] >= threshold]
    source_count: dict[str, int] = {}
    deduped = []
    for c in filtered_by_score:
        canonical = canonical_source(c["source"])
        if source_count.get(canonical, 0) < 2:
            deduped.append(c)
            source_count[canonical] = source_count.get(canonical, 0) + 1
    return deduped[:k]


# ── Public API ─────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = None) -> List[dict]:
    """混合检索 + Reranking。返回 [{text, source, page, score}, ...]。"""
    from errors import RetrievalError
    from ingest import encode_query

    if k is None:
        k = settings.FINAL_CONTEXT_K

    try:
        client = _init_milvus()
        over_fetch = settings.RERANK_CANDIDATES

        dense_vec, sparse_vec = encode_query(query)
        candidates = _extract_candidates(_do_search(client, dense_vec, sparse_vec, over_fetch))

        if not candidates:
            logger.info(f"Retrieval: query='{query[:50]}' → 0 candidates")
            return []

        candidates, top_score = _rerank_and_score(query, candidates)

        threshold = settings.RERANK_SCORE_THRESHOLD
        result = _filter_and_dedup(candidates, threshold, k)

        if not result:
            logger.info(f"Retrieval: query='{query[:50]}' → top_score={top_score:.3f} < {threshold}")
            return []

        logger.info(
            f"Retrieval: query='{query[:50]}' → top_score={top_score:.3f}, "
            f"candidates={len(candidates)}, returned={len(result)}"
        )
        return result

    except RetrievalError:
        raise
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        raise RetrievalError() from e


if __name__ == "__main__":
    results = retrieve("test query", k=2)
    print(f"Retrieved {len(results)} docs" if results else "No docs found")
    for r in results:
        print(f"  [{r.get('score', 0):.3f}] {r['source']} p{r.get('page', '?')}: {r['text'][:80]}...")
