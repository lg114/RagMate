import contextvars
import logging
import re
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
_rewrite_queries_var: contextvars.ContextVar[list[str]] = contextvars.ContextVar("_rewrite_queries", default=[])


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


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """加载 Reranker 模型（单例）。"""
    return CrossEncoder(settings.RERANKER_MODEL)


# ── Query Rewrite ──

_REWRITE_PROMPT = """你是检索优化器。把用户问题翻译成英文技术搜索词，用于检索英文技术文档。

要求：
- 翻译为英文，保留技术术语（如 HTTP、API、SDK、error code）
- 保留错误码、模型名、文件名等实体不变
- 补全 1-2 个关键技术词
- 最多 8 个词，只输出一行英文搜索 query

用户问题：{query}"""


def _should_skip_rewrite(query: str) -> bool:
    """精确查找类 query 跳过 rewrite，避免伤害检索。"""
    # 错误码：429、404、500 等（独立数字）
    if re.search(r'\b\d{3}\b', query) and len(query) < 20:
        return True
    # 文件扩展名
    if re.search(r'\.\w{2,4}\b', query):
        return True
    # 全大写短标识符（如 HTTP、API、SDK）
    if query.isupper() and len(query) < 15:
        return True
    # CamelCase 标识符（如 UserService、getConnection）
    if re.search(r'[a-z][A-Z][a-z]', query):
        return True
    # 纯英文技术短语（如 "HTTP 429"、"rate limit"）
    if re.match(r'^[a-zA-Z0-9\s\-_\.]+$', query) and len(query) < 40:
        return True
    return False


def _rewrite_query_raw(query: str) -> str | None:
    """调用 LLM 改写 query，失败返回 None。"""
    try:
        from model_factory import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm()
        # 用低 temperature 保证一致性
        resp = llm.invoke(
            [HumanMessage(content=_REWRITE_PROMPT.format(query=query))],
            temperature=0,
            max_tokens=100,
        )
        rewritten = resp.content.strip()
        # 去掉可能的引号和多余内容
        rewritten = rewritten.strip('"\'')
        if rewritten and rewritten != query and len(rewritten) > 3:
            return rewritten
        return None
    except Exception as e:
        logger.debug(f"Query rewrite failed: {e}")
        return None


@lru_cache(maxsize=256)
def _cached_rewrite(query: str) -> str | None:
    """缓存 rewrite 结果，避免重复调用 LLM。"""
    return _rewrite_query_raw(query)


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


def get_rewrite_queries() -> list[str]:
    """获取当前请求中所有 rewrite 后的 query。"""
    return _rewrite_queries_var.get()


def clear_rewrite_queries():
    """清除当前请求的 rewrite queries（每次请求开始时调用）。

    设置一个新 list，asyncio.to_thread 会将当前 context 复制到工作线程，
    工作线程对同一个 list 的 append 对主线程可见。
    """
    _rewrite_queries_var.set([])


def record_retrieval_query(query: str):
    """记录发送给 retrieval_tool 的 query（用于过滤流式输出中的泄漏文本）。"""
    _rewrite_queries_var.get().append(query)


def retrieve(query: str, k: int = None) -> List[dict]:
    """混合检索 + Query Rewrite + Reranking。返回 [{text, source, page, score}, ...]。"""
    from errors import RetrievalError
    from ingest import encode_query

    if k is None:
        k = settings.FINAL_CONTEXT_K

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

    try:
        over_fetch = settings.RERANK_CANDIDATES

        # ── Query Rewrite ──
        rewritten = None
        rewrite_skipped = False
        if settings.QUERY_REWRITE_ENABLED:
            if _should_skip_rewrite(query):
                rewrite_skipped = True
            else:
                rewritten = _cached_rewrite(query)
                if rewritten:
                    _rewrite_queries_var.get().append(rewritten)

        # ── 双 Query 检索 ──
        dense_vec, sparse_vec = encode_query(query)
        candidates = _extract_candidates(_do_search(client, dense_vec, sparse_vec, over_fetch))

        if rewritten:
            rw_dense, rw_sparse = encode_query(rewritten)
            rw_candidates = _extract_candidates(_do_search(client, rw_dense, rw_sparse, over_fetch))
            # 合并去重（按 text 去重）
            seen_texts = {c["text"][:100] for c in candidates}
            for c in rw_candidates:
                if c["text"][:100] not in seen_texts:
                    candidates.append(c)
                    seen_texts.add(c["text"][:100])

        if not candidates:
            logger.info(f"Retrieval: query='{query[:50]}' → 0 candidates")
            return []

        # ── Reranking ──
        reranker = get_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)

        for c, score in zip(candidates, scores):
            c["score"] = float(score)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_score = candidates[0]["score"]

        # ── 过滤低分 ──
        threshold = settings.RERANK_SCORE_THRESHOLD
        candidates = [c for c in candidates if c["score"] >= threshold]
        if not candidates:
            logger.info(
                f"Retrieval: query='{query[:50]}' rewritten='{rewritten}' "
                f"bypass={rewrite_skipped} → top_score={top_score:.3f} < {threshold}"
            )
            return []

        # ── Source 去重 ──
        source_count: dict[str, int] = {}
        filtered = []
        for c in candidates:
            canonical = canonical_source(c["source"])
            if source_count.get(canonical, 0) < 2:
                filtered.append(c)
                source_count[canonical] = source_count.get(canonical, 0) + 1
        result = filtered[:k]

        logger.info(
            f"Retrieval: query='{query[:50]}' rewritten='{rewritten}' "
            f"bypass={rewrite_skipped} → top_score={top_score:.3f}, "
            f"candidates={len(candidates)}, after_dedup={len(filtered)}, returned={len(result)}"
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
