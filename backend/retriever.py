import logging
import math
import socket
import threading
from functools import lru_cache

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker
from sentence_transformers import CrossEncoder

from config import settings


def canonical_source(source: str) -> str:
    """归一化来源文件名，用于检索去重。"""
    if not source:
        return ""
    import os
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
        with socket.create_connection(
            (settings.MILVUS_HOST, settings.MILVUS_PORT),
            timeout=3,
        ):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def _init_milvus():
    """确保 Milvus collection 已加载（线程安全）。"""
    from errors import ServiceUnavailableError

    global _collection_loaded
    client = get_milvus_client()
    with _milvus_lock:
        if _collection_loaded:
            return client
        try:
            client.load_collection(settings.MILVUS_COLLECTION)
            _collection_loaded = True
        except Exception as e:
            raise ServiceUnavailableError("检索服务异常，请稍后重试") from e
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

def _sigmoid(x: float) -> float:
    """将 raw logit 转换为 [0, 1] 概率。"""
    return 1.0 / (1.0 + math.exp(-x))


def _rerank_and_score(query: str, candidates: list[dict]) -> tuple[list[dict], float]:
    """Rerank 候选，logits 经 sigmoid 转为概率后返回 (scored_list, top_score)。"""
    if not candidates:
        return [], 0.0
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    logits = reranker.predict(pairs)
    for c, logit in zip(candidates, logits):
        c["score"] = _sigmoid(float(logit))
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates, candidates[0]["score"]


def _filter_and_dedup(candidates: list[dict], threshold: float, k: int) -> list[dict]:
    """基于 sigmoid 概率的动态过滤、去重、截断。"""
    if not candidates:
        return []

    top_score = candidates[0]["score"]  # 已是 sigmoid 概率

    # 1. 动态阈值：top_score 的 50%，最低保底 threshold
    effective_threshold = max(top_score * 0.5, threshold)
    scored = [c for c in candidates if c["score"] >= effective_threshold]

    if not scored:
        return []

    # 2. 动态源文件去重：同源高分 chunk 多则放宽
    source_groups: dict[str, list] = {}
    for c in scored:
        canonical = canonical_source(c["source"])
        source_groups.setdefault(canonical, []).append(c)

    deduped = []
    for canonical, chunks in source_groups.items():
        chunks.sort(key=lambda x: x["score"], reverse=True)
        high_count = sum(1 for c in chunks if c["score"] >= top_score * 0.6)
        limit = min(max(high_count, 2), 4)
        deduped.extend(chunks[:limit])

    deduped.sort(key=lambda x: x["score"], reverse=True)

    # 3. 动态 top-k：分数断崖检测
    result = [deduped[0]]
    for i in range(1, len(deduped)):
        gap = deduped[i - 1]["score"] - deduped[i]["score"]
        if gap > 0.15:
            break
        if len(result) >= k:
            break
        result.append(deduped[i])

    return result


# ── Public API ─────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = None) -> list[dict]:
    """混合检索 + Reranking。返回 [{text, source, page, score}, ...]。"""
    from errors import AppError, ServiceUnavailableError, ValidationError
    from ingest.encoding import encode_query

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

    except AppError:
        raise
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        raise ServiceUnavailableError("检索服务异常，请稍后重试") from e


if __name__ == "__main__":
    results = retrieve("test query", k=2)
    print(f"Retrieved {len(results)} docs" if results else "No docs found")
    for r in results:
        print(f"  [{r.get('score', 0):.3f}] {r['source']} p{r.get('page', '?')}: {r['text'][:80]}...")
