"""检索管线：Milvus 混合检索 + Reranking + 动态过滤。"""
import logging
import math
import re
import threading

from pymilvus import AnnSearchRequest, RRFRanker
from sentence_transformers import CrossEncoder

from backend.domain.errors import AppError, ServiceUnavailableError
from backend.infrastructure.config import settings
from backend.infrastructure.encoding import encode_query
from backend.infrastructure.milvus import check_milvus_available, canonical_source, get_milvus_client, init_milvus

logger = logging.getLogger("ragmate")

# ── 检索质量指标（线程本地，供 confidence 计算） ──────────────────────────────
_metrics = threading.local()


def get_retrieval_metrics() -> dict | None:
    """获取最近一次检索的质量指标。"""
    return getattr(_metrics, "last", None)

_CONTEXTUALIZE_PROMPT = """根据以下对话历史，将用户的最新问题改写为一个独立、完整的搜索查询。
要求：
- 补全省略的指代（如"它"、"这个"、"那个"）为具体实体
- 补全省略的上下文，使查询可以独立理解
- 保持原意，不要添加原问题没有的信息
- 只输出改写后的查询，不要解释

对话历史：
{history}

用户最新问题：{query}"""


def contextualize_query(query: str, messages: list[dict]) -> str:
    """用轻量 LLM 调用把追问改写为自包含的检索 query。失败时返回原 query。"""
    from backend.infrastructure.config import settings
    if not settings.QUERY_CONTEXTUALIZE:
        return query

    # 取最近 2-3 轮（最多 6 条 message）
    recent = messages[-6:-1]  # 排除最后一条（当前用户消息）
    if not recent:
        return query

    history_lines = []
    for msg in recent:
        role = "用户" if msg.get("role") == "user" else "助手"
        content = msg.get("content", "")
        # 截断过长的回复，只保留前 300 字符
        if len(content) > 300:
            content = content[:300] + "..."
        history_lines.append(f"{role}：{content}")
    history_text = "\n".join(history_lines)

    try:
        from backend.infrastructure.model_factory import get_llm
        llm = get_llm()
        prompt = _CONTEXTUALIZE_PROMPT.format(history=history_text, query=query)
        response = llm.invoke(prompt)
        rewritten = response.content.strip().strip('"').strip("'")
        if rewritten and rewritten != query:
            logger.info(f"Query contextualized: '{query}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"Query contextualization failed, using original: {e}")

    return query

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
            "text": entity.get("parent_text") or entity["text"],
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


# ── Contextual Compression ──────────────────────────────────────────────────

_SENT_RE = re.compile(r"(?<=[。！？.!?\n])\s*")


def _compress_chunks(query: str, candidates: list[dict]) -> list[dict]:
    """用 reranker 对 chunk 内句子级评分，只保留相关句子。减少传给 LLM 的噪声。"""
    from backend.infrastructure.config import settings
    if not settings.CONTEXTUAL_COMPRESSION:
        return candidates

    reranker = get_reranker()

    for c in candidates:
        text = c["text"]
        # 短 chunk 不压缩
        if len(text) < settings.COMPRESSION_MIN_CHARS:
            continue

        sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
        if len(sentences) <= 2:
            continue

        # 对每个句子做 reranker 评分
        pairs = [(query, s) for s in sentences]
        logits = reranker.predict(pairs)
        scored_sentences = list(zip(sentences, [_sigmoid(float(l)) for l in logits]))

        # 保留超过阈值的句子，至少保留 top-1
        threshold = settings.COMPRESSION_SCORE_THRESHOLD
        kept = [(s, sc) for s, sc in scored_sentences if sc >= threshold]
        if not kept:
            kept = [max(scored_sentences, key=lambda x: x[1])]

        compressed = "".join(s for s, _ in kept)
        # 只有压缩后明显更短才替换（避免微小变化的开销）
        if len(compressed) < len(text) * 0.8:
            c["text"] = compressed
            c["compressed"] = True

    return candidates


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

    # 2. 动态源文件去重：同源高分 chunk 多则放宽；核心来源软上限
    source_groups: dict[str, list] = {}
    for c in scored:
        canonical = canonical_source(c["source"])
        source_groups.setdefault(canonical, []).append(c)

    # 计算每个来源的"主导度"（该来源最高分 / 全局最高分）
    source_dominance: dict[str, float] = {}
    for canonical, chunks in source_groups.items():
        best = max(chunks, key=lambda x: x["score"])["score"]
        source_dominance[canonical] = best / top_score if top_score > 0 else 0

    deduped = []
    for canonical, chunks in source_groups.items():
        chunks.sort(key=lambda x: x["score"], reverse=True)
        high_count = sum(1 for c in chunks if c["score"] >= top_score * settings.HIGH_SCORE_RATIO)
        limit = max(high_count, settings.MIN_PER_SOURCE)

        # 软限制：当该来源主导度很高时，放宽上限
        dominance = source_dominance[canonical]
        if dominance >= settings.SOURCE_DOMINANCE_THRESHOLD:
            soft_limit = min(int(limit * settings.SOURCE_DOMINANCE_BOOST), k)
        else:
            soft_limit = settings.MAX_PER_SOURCE
        limit = min(limit, soft_limit)

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


# ── Faithfulness Check ───────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """判断以下"声明"是否被"证据"支持。逐条输出 JSON 数组，每条格式：
{{"claim": "声明内容", "supported": true/false}}

只输出 JSON 数组，不要解释。

声明：
{claims}

证据：
{evidence}"""


def check_faithfulness(response: str, source_texts: list[str]) -> list[dict] | None:
    """检查生成的 response 中的声明是否被 source_texts 支持。

    返回 [{"claim": "...", "supported": true/false}, ...]，失败时返回 None。
    """
    from backend.infrastructure.config import settings
    if not settings.FAITHFULNESS_CHECK:
        return None
    if not source_texts or not response:
        return None

    # 简单按句号拆分声明（取前 10 条避免 prompt 过长）
    sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?\n])", response) if s.strip() and len(s.strip()) > 10]
    if not sentences:
        return None
    sentences = sentences[:10]

    evidence = "\n---\n".join(source_texts[:10])

    try:
        from backend.infrastructure.model_factory import get_llm
        llm = get_llm()
        prompt = _FAITHFULNESS_PROMPT.format(
            claims="\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences)),
            evidence=evidence,
        )
        result = llm.invoke(prompt)
        import json
        # 提取 JSON 数组
        text = result.content.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as e:
        logger.warning(f"Faithfulness check failed: {e}")

    return None


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

        # 上下文压缩：rerank 后、过滤前，压缩 chunk 内无关句子
        candidates = _compress_chunks(query, candidates)

        threshold = settings.RERANK_SCORE_THRESHOLD
        result = _filter_and_dedup(candidates, threshold, k)

        if not result:
            logger.info(f"Retrieval: len={len(query)} → top_score={top_score:.3f} < {threshold}")
            return []

        logger.info(
            f"Retrieval: len={len(query)} → top_score={top_score:.3f}, "
            f"candidates={len(candidates)}, returned={len(result)}"
        )

        # 存储检索质量指标
        _metrics.last = {
            "top_score": round(top_score, 3),
            "candidates": len(candidates),
            "returned": len(result),
            "threshold": round(threshold, 3),
        }

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
