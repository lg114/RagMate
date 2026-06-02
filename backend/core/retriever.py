"""检索管线：Milvus 混合检索 + Reranking + 动态过滤。"""
import logging
import math
import threading

from pymilvus import AnnSearchRequest, RRFRanker
from sentence_transformers import CrossEncoder

from backend.domain.errors import AppError, ServiceUnavailableError
from backend.infrastructure.config import settings
from backend.infrastructure.encoding import encode_query
from backend.infrastructure.milvus import check_milvus_available, canonical_source, get_milvus_client, init_milvus

logger = logging.getLogger("ragmate")

_reranker: CrossEncoder | None = None
_reranker_lock = threading.Lock()


def get_reranker() -> CrossEncoder:
    """加载 Reranker 模型（单例，失败后可重试）。"""
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is None:
            _reranker = CrossEncoder(settings.RERANKER_MODEL)
        return _reranker


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
            param={"metric_type": "IP", "params": {"drop_ratio_search": settings.DROP_RATIO_SEARCH}},
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

    # 1. 动态阈值：top_score 的一定比例，最低保底 threshold
    effective_threshold = max(top_score * settings.DYNAMIC_THRESHOLD_RATIO, threshold)
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
        high_count = sum(1 for c in chunks if c["score"] >= top_score * settings.HIGH_SCORE_RATIO)
        limit = min(max(high_count, settings.MIN_PER_SOURCE), settings.MAX_PER_SOURCE)
        deduped.extend(chunks[:limit])

    deduped.sort(key=lambda x: x["score"], reverse=True)

    # 3. 动态 top-k：分数断崖检测
    result = [deduped[0]]
    for i in range(1, len(deduped)):
        gap = deduped[i - 1]["score"] - deduped[i]["score"]
        if gap > settings.SCORE_GAP_THRESHOLD:
            break
        if len(result) >= k:
            break
        result.append(deduped[i])

    return result


# ── Public API ─────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = None) -> list[dict]:
    """混合检索 + Reranking。返回 [{text, source, page, score}, ...]。"""
    if k is None:
        k = settings.FINAL_CONTEXT_K

    try:
        client = init_milvus()
        if client is None:
            logger.info("Retrieval: collection does not exist, returning empty")
            return []
        over_fetch = settings.RERANK_CANDIDATES

        dense_vec, sparse_vec = encode_query(query)
        candidates = _extract_candidates(_do_search(client, dense_vec, sparse_vec, over_fetch))

        if not candidates:
            logger.info(f"Retrieval: len={len(query)} → 0 candidates")
            return []

        candidates, top_score = _rerank_and_score(query, candidates)

        threshold = settings.RERANK_SCORE_THRESHOLD
        result = _filter_and_dedup(candidates, threshold, k)

        if not result:
            logger.info(f"Retrieval: len={len(query)} → top_score={top_score:.3f} < {threshold}")
            return []

        logger.info(
            f"Retrieval: len={len(query)} → top_score={top_score:.3f}, "
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
