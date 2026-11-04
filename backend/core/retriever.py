"""检索管线：pgvector 混合检索 + Reranking + 动态过滤。"""
import contextvars
import json
import logging
import math
import os
import re
import threading
import urllib.request

from backend.domain.errors import AppError, ServiceUnavailableError
from backend.infrastructure.config import settings
from backend.infrastructure.encoding import encode_query
from backend.infrastructure.vector_store import init_vector_store

logger = logging.getLogger("ragmate")


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

# ── 检索质量指标（线程本地，供 confidence 计算） ──────────────────────────────
class _MetricsLocal(threading.local):
    """带默认值的线程本地存储，新线程无需先赋值即可读取。"""

    last: dict | None = None


_metrics = _MetricsLocal()

# 仅在请求显式打开 debug 时记录候选轨迹。ContextVar 会随线程池复制，
# 所以 agent 工具线程里也能看到调用方在 run_agent 上设置的开关。
_retrieval_debug_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "retrieval_debug_enabled", default=False,
)
_last_debug = threading.local()


def set_retrieval_debug(enabled: bool):
    """打开或关闭当前上下文的检索调试。返回 reset 用的 token。"""
    return _retrieval_debug_enabled.set(bool(enabled))


def reset_retrieval_debug(token) -> None:
    """恢复 set_retrieval_debug 之前的开关。"""
    _retrieval_debug_enabled.reset(token)


def retrieval_debug_enabled() -> bool:
    """当前上下文是否在记录检索调试信息。"""
    return bool(_retrieval_debug_enabled.get())


def take_retrieval_debug() -> dict | None:
    """取出 retrieve() 刚写下的调试记录。调试关闭时丢掉记录并返回 None。"""
    entry = getattr(_last_debug, "entry", None)
    _last_debug.entry = None
    if not retrieval_debug_enabled():
        return None
    return entry


def calculate_confidence(metrics_list: list[dict]) -> dict | None:
    """根据检索指标计算置信度。
    
    Args:
        metrics_list: 检索指标列表，每个元素包含 top_score 和 returned 字段
        
    Returns:
        置信度字典 {"level": "high|medium|low", "score": float, "chunks": int}
        或 None（如果输入为空）
    """
    if not metrics_list:
        return None
    
    best_score = max(m["top_score"] for m in metrics_list)
    total_returned = sum(m["returned"] for m in metrics_list)
    
    if best_score >= 0.7 and total_returned >= 3:
        level = "high"
    elif best_score >= 0.4 and total_returned >= 1:
        level = "medium"
    else:
        level = "low"
    
    return {"level": level, "score": best_score, "chunks": total_returned}


def get_retrieval_metrics() -> dict | None:
    """获取最近一次检索的质量指标。"""
    return _metrics.last


def reset_retrieval_metrics():
    """清空当前线程的检索指标。

    必须在每次 retrieve() 开始前调用：asyncio.to_thread 会复用线程池中的线程，
    若本次检索提前返回而未覆盖 last，读取到的会是上一次（可能是别的请求）的指标，
    导致 confidence 虚高。
    """
    _metrics.last = None


def _store_retrieval_metrics(top_score: float, candidates: int, returned: int, threshold: float | None = None):
    """写入本次检索的质量指标，供置信度计算读取。空结果也要写，避免被当成「没检索」。"""
    if threshold is None:
        threshold = settings.RERANK_SCORE_THRESHOLD
    _metrics.last = {
        "top_score": round(float(top_score), 3),
        "candidates": int(candidates),
        "returned": int(returned),
        "threshold": round(float(threshold), 3),
    }


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

_reranker = None
_reranker_lock = threading.Lock()


def get_reranker():
    """加载本地 CrossEncoder（单例，失败后可重试）。

    只在 RERANKER_PROVIDER=local 时调用。默认的 api 路径不会 import sentence-transformers。
    """
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(
                settings.RERANKER_MODEL,
                device=settings.EMBEDDING_DEVICE,
            )
        return _reranker


_MULTI_QUERY_PROMPT = """为同一个问题生成 {count} 个语义相同但表述不同的检索查询变体，用于从文档库中多角度召回。
要求：
- 每行一个变体，不要编号、不要引号、不要任何解释
- 变体之间尽量使用不同的关键词与句式
- 保持与原始问题相同的语言

原始问题：{query}"""

_NUM_PREFIX_RE = re.compile(r"^\s*\d+[.、)）]\s*")


def expand_queries(query: str) -> list[str]:
    """多查询扩展：返回 [原 query] + LLM 生成的改写变体（去重、限量）。

    MULTI_QUERY_ENABLED 关闭、或 LLM 调用失败/无有效变体时，只返回 [query]，
    检索行为与未扩展时完全一致。
    """
    if not settings.MULTI_QUERY_ENABLED:
        return [query]
    try:
        from backend.infrastructure.model_factory import get_llm

        llm = get_llm()
        prompt = _MULTI_QUERY_PROMPT.format(count=settings.MULTI_QUERY_COUNT, query=query)
        response = llm.invoke(prompt)
        variants: list[str] = []
        for line in response.content.splitlines():
            v = _NUM_PREFIX_RE.sub("", line).strip().strip('"').strip("\u201c\u201d").strip("'")
            if v and v != query and v not in variants:
                variants.append(v)
        variants = variants[: settings.MULTI_QUERY_COUNT]
        if variants:
            logger.info(f"Multi-query: '{query}' → {variants}")
            return [query] + variants
    except Exception as e:
        logger.warning(f"Multi-query expansion failed, using original only: {e}")
    return [query]


# ── Reranker 打分（本地 / 远程 API 两种 provider） ─────────────────────────────

def _api_rerank_scores(query: str, texts: list[str]) -> list[float]:
    """调用远程 /rerank 接口（Jina、硅基流动等兼容格式），返回与 texts 对齐的概率分。

    请求：POST {RERANKER_API_BASE_URL}/rerank，体为 {model, query, documents, top_n}；
    响应：{results: [{index, relevance_score}, ...]}，relevance_score 已是 [0, 1] 概率。
    响应缺失的下标按 0.0 计（等同于被阈值过滤）。
    """
    url = settings.RERANKER_API_BASE_URL.rstrip("/") + "/rerank"
    payload = json.dumps({
        "model": settings.RERANKER_MODEL,
        "query": query,
        "documents": texts,
        "top_n": len(texts),
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if settings.RERANKER_API_KEY:
        req.add_header("Authorization", f"Bearer {settings.RERANKER_API_KEY}")
    with urllib.request.urlopen(req, timeout=settings.RERANKER_API_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    scores = [0.0] * len(texts)
    for item in data.get("results", []):
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(texts):
            scores[idx] = float(item.get("relevance_score", 0.0))
    return scores


def score_pairs(query: str, texts: list[str]) -> list[float]:
    """对 (query, text) 逐对打分，统一返回 [0, 1] 概率分（与 texts 等长、顺序对齐）。

    local：本地 CrossEncoder 的 logits 经 sigmoid 转换（与历史行为一致）；
    api：远程接口直接返回的概率分。调用方无需关心 provider 差异。
    """
    if not texts:
        return []
    if settings.RERANKER_PROVIDER == "api":
        return _api_rerank_scores(query, texts)
    reranker = get_reranker()
    logits = reranker.predict([(query, t) for t in texts])
    return [_sigmoid(float(l)) for l in logits]


# ── Search ──────────────────────────────────────────────────────────────────

def _do_search(store, dense_vec: list[float], sparse_vec, over_fetch: int):
    """执行一次混合检索或 dense 检索。返回与旧结果相同的 [hits] 包装。

    混合检索：稠密 HNSW 内积 + 稀疏精确内积，再按 FUSION_MODE 融合（默认 RRF k=60）。
    HYBRID_SEARCH_ENABLED 关闭时只走稠密。
    """
    if settings.HYBRID_SEARCH_ENABLED:
        hits = store.search_hybrid(dense_vec, sparse_vec, over_fetch)
    else:
        hits = store.search_dense(dense_vec, over_fetch)
    return [hits]


def _extract_candidates(results) -> list[dict]:
    """从向量检索结果中提取候选。

    每个候选保留两份文本，对应 small-to-big 的两侧：
      - text：小片段，用于 rerank 打分、阈值判断与来源去重
      - context：父片段，最终交付给 LLM

    父片段必须在 _do_search 的 output_fields 中显式请求，否则取不到。

    Args:
        results: 检索结果，形状为 [hits]

    Returns:
        候选文档列表，每个包含 text, context, source, page, chunk_index
    """
    candidates = []
    for hit in results[0]:
        entity = hit["entity"]
        meta = entity.get("metadata", {})
        small = entity["text"]
        candidates.append({
            "text": small,
            "context": entity.get("parent_text") or small,
            "source": meta.get("source", ""),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
        })
    return candidates


def _search_multi(store, queries: list[str], over_fetch: int) -> list[dict]:
    """多路检索 + RRF 融合。

    每个 query 变体独立做一次混合检索，候选按 (source, page, chunk_index) 去重；
    同一片段在越多路结果中出现、排名越靠前，融合分越高。融合后截断到
    over_fetch 个再交给 reranker 精排，因此 reranker 的调用量与单路时相同。
    """
    rrf_k = settings.RRF_K
    fused: dict[tuple, dict] = {}
    fused_scores: dict[tuple, float] = {}
    for q in queries:
        dense_vec, sparse_vec = encode_query(q)
        results = _do_search(store, dense_vec, sparse_vec, over_fetch)
        for rank, cand in enumerate(_extract_candidates(results)):
            key = (cand["source"], cand.get("page"), cand.get("chunk_index"))
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            fused.setdefault(key, cand)
    ordered = sorted(
        fused.values(),
        key=lambda c: fused_scores[(c["source"], c.get("page"), c.get("chunk_index"))],
        reverse=True,
    )
    return ordered[:over_fetch]


# ── Reranking + Filtering ──────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """将 raw logit 转换为 [0, 1] 概率。"""
    return 1.0 / (1.0 + math.exp(-x))


def _rerank_and_score(query: str, candidates: list[dict]) -> tuple[list[dict], float]:
    """Rerank 候选，logits 经 sigmoid 转为概率后返回 (scored_list, top_score)。"""
    if not candidates:
        return [], 0.0
    scores = score_pairs(query, [c["text"] for c in candidates])
    for c, sc in zip(candidates, scores):
        c["score"] = sc
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates, candidates[0]["score"]


# ── Parent Expansion ────────────────────────────────────────────────────────

def _expand_to_parent(candidates: list[dict]) -> list[dict]:
    """把交付文本从小片段替换为父片段（small-to-big 的 "big" 一侧）。

    `_extract_candidates` 为每个候选保留两份文本：用于打分的小片段（text）与
    用于交付的父片段（context）。这里只替换交付内容，score / source / page /
    chunk_index 原样保留，因此不影响已经完成的阈值判断与来源去重。

    无父片段的场景（如 Excel 表格未设 parent_text）保持原文本不变。
    """
    for c in candidates:
        parent = c.get("context")
        if parent and parent != c["text"]:
            c["text"] = parent
            c["expanded"] = True
    return candidates


# ── Contextual Compression ──────────────────────────────────────────────────

_SENT_RE = re.compile(r"(?<=[。！？.!?\n])\s*")


def _compress_chunks(query: str, candidates: list[dict]) -> list[dict]:
    """用 reranker 对 chunk 内句子级评分，只保留相关句子。减少传给 LLM 的噪声。"""
    from backend.infrastructure.config import settings
    if not settings.CONTEXTUAL_COMPRESSION:
        return candidates

    for c in candidates:
        text = c["text"]
        # 短 chunk 不压缩
        if len(text) < settings.COMPRESSION_MIN_CHARS:
            continue

        sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
        if len(sentences) <= 2:
            continue

        # 对每个句子做 reranker 评分；打分失败时保留原 chunk（压缩只是优化，不值得让整次检索失败）
        try:
            probs = score_pairs(query, sentences)
        except Exception as e:
            logger.warning(f"Compression scoring failed, keeping chunk as-is: {e}")
            continue
        scored_sentences = list(zip(sentences, probs))

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


def _debug_source(source: str | None) -> str:
    text = str(source or "").replace("\\", "/")
    return os.path.basename(text) or text or "unknown"


def _debug_page(page) -> int | None:
    if isinstance(page, int) and not isinstance(page, bool):
        return page + 1
    return None


def _scrub_debug_fields(candidates: list[dict]) -> None:
    """去掉只给调试用的临时字段，避免它们跟着片段进入模型上下文。"""
    for cand in candidates:
        cand.pop("_debug_preview", None)
        cand.pop("dropped_by", None)


def build_retrieval_debug(query: str, candidates: list[dict], kept: list[dict], threshold: float, top_score: float) -> dict:
    """把重排后的候选整理成调试结构：分数、是否留下、被哪一步丢掉。

    只应在请求打开 debug 时调用。preview 截断，避免把整篇父片段塞进响应。
    """
    kept_ids = {id(c) for c in kept}
    chunks = []
    for rank, cand in enumerate(candidates, start=1):
        dropped = cand.get("dropped_by")
        if id(cand) in kept_ids:
            dropped = None
        preview = cand.get("_debug_preview")
        if preview is None:
            preview = (cand.get("text") or "")[:240]
        row = {
            "rank": rank,
            "source": _debug_source(cand.get("source")),
            "chunk_index": cand.get("chunk_index"),
            "score": round(float(cand.get("score") or 0.0), 4),
            "kept": id(cand) in kept_ids,
            "dropped_by": dropped,
            "preview": preview,
        }
        page = _debug_page(cand.get("page"))
        if page is not None:
            row["page"] = page
        if cand.get("compressed"):
            row["compressed"] = True
        chunks.append(row)
    return {
        "query": query,
        "top_score": round(float(top_score), 4),
        "threshold": round(float(threshold), 4),
        "candidates": len(candidates),
        "returned": len(kept),
        "chunks": chunks,
    }


def _annotate_drop_reasons(candidates: list[dict], threshold: float, k: int, kept: list[dict]) -> None:
    """给候选打上 dropped_by。仅调试路径调用，不改变过滤结果。"""
    if not candidates:
        return
    top_score = candidates[0]["score"]
    effective_threshold = max(top_score * settings.DYNAMIC_THRESHOLD_RATIO, threshold)
    scored = [c for c in candidates if c["score"] >= effective_threshold]
    scored_ids = {id(c) for c in scored}

    source_groups: dict[str, list] = {}
    for cand in scored:
        source_groups.setdefault(canonical_source(cand["source"]), []).append(cand)
    source_dominance = {}
    for canonical, group in source_groups.items():
        best = max(group, key=lambda x: x["score"])["score"]
        source_dominance[canonical] = best / top_score if top_score > 0 else 0

    deduped = []
    per_source_ids = set()
    for canonical, group in source_groups.items():
        group.sort(key=lambda x: x["score"], reverse=True)
        high_count = sum(1 for c in group if c["score"] >= top_score * settings.HIGH_SCORE_RATIO)
        limit = max(high_count, settings.MIN_PER_SOURCE)
        dominance = source_dominance[canonical]
        if dominance >= settings.SOURCE_DOMINANCE_THRESHOLD:
            soft_limit = min(int(limit * settings.SOURCE_DOMINANCE_BOOST), k)
        else:
            soft_limit = settings.MAX_PER_SOURCE
        limit = min(limit, soft_limit)
        deduped.extend(group[:limit])
        for cand in group[limit:]:
            per_source_ids.add(id(cand))
    deduped.sort(key=lambda x: x["score"], reverse=True)

    gap_ids = set()
    topk_ids = set()
    if deduped:
        stopped = False
        count = 1
        for i in range(1, len(deduped)):
            if stopped:
                break
            gap = deduped[i - 1]["score"] - deduped[i]["score"]
            if gap > settings.SCORE_GAP_THRESHOLD:
                for cand in deduped[i:]:
                    gap_ids.add(id(cand))
                stopped = True
                break
            if count >= k:
                for cand in deduped[i:]:
                    topk_ids.add(id(cand))
                stopped = True
                break
            count += 1

    kept_ids = {id(c) for c in kept}
    for cand in candidates:
        cid = id(cand)
        if cid in kept_ids:
            cand["dropped_by"] = None
        elif cid in gap_ids:
            cand["dropped_by"] = "score_gap"
        elif cid in topk_ids:
            cand["dropped_by"] = "top_k"
        elif cid in per_source_ids:
            cand["dropped_by"] = "per_source"
        elif cid not in scored_ids:
            cand["dropped_by"] = "threshold"
        else:
            cand["dropped_by"] = "filtered"


def _filter_and_dedup(candidates: list[dict], threshold: float, k: int) -> list[dict]:
    """基于 sigmoid 概率的动态过滤、去重、截断。"""
    if not candidates:
        return []

    top_score = candidates[0]["score"]  # 已是 sigmoid 概率

    # 1. 动态阈值：top_score 的一定比例，最低保底 threshold
    effective_threshold = max(top_score * settings.DYNAMIC_THRESHOLD_RATIO, threshold)
    scored = [c for c in candidates if c["score"] >= effective_threshold]

    if not scored:
        if retrieval_debug_enabled():
            for cand in candidates:
                cand["dropped_by"] = "threshold"
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

    if retrieval_debug_enabled():
        _annotate_drop_reasons(candidates, threshold, k, result)

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
    """混合检索 + Reranking。返回 [{text, source, page, score}, ...]。

    返回的 text 是父片段（small-to-big 的 "big" 一侧）；打分、阈值判断与来源
    去重用的都是小片段，两者在 _extract_candidates 中分别存于 text / context。
    """
    if k is None:
        k = settings.FINAL_CONTEXT_K

    # 先清空本线程的指标，保证后续所有 return / raise 路径都不会残留旧数据
    reset_retrieval_metrics()

    try:
        client = init_vector_store()
        if client is None:
            logger.info("Retrieval: vector table does not exist, returning empty")
            _store_retrieval_metrics(0.0, 0, 0)
            if retrieval_debug_enabled():
                _last_debug.entry = build_retrieval_debug(query, [], [], settings.RERANK_SCORE_THRESHOLD, 0.0)
            return []
        over_fetch = settings.RERANK_CANDIDATES

        queries = expand_queries(query)
        if len(queries) == 1:
            dense_vec, sparse_vec = encode_query(query)
            candidates = _extract_candidates(_do_search(client, dense_vec, sparse_vec, over_fetch))
        else:
            candidates = _search_multi(client, queries, over_fetch)

        if not candidates:
            logger.info(f"Retrieval: len={len(query)} → 0 candidates")
            _store_retrieval_metrics(0.0, 0, 0)
            if retrieval_debug_enabled():
                _last_debug.entry = build_retrieval_debug(query, [], [], settings.RERANK_SCORE_THRESHOLD, 0.0)
            return []

        # 打分用小片段：阈值、断崖、压缩门槛都是在片段粒度上校准的
        candidates, top_score = _rerank_and_score(query, candidates)
        if retrieval_debug_enabled():
            for cand in candidates:
                cand["_debug_preview"] = (cand.get("text") or "")[:240]

        threshold = settings.RERANK_SCORE_THRESHOLD
        result = _filter_and_dedup(candidates, threshold, k)

        if not result:
            logger.info(f"Retrieval: len={len(query)} → top_score={top_score:.3f} < {threshold}")
            _store_retrieval_metrics(top_score, len(candidates), 0, threshold)
            if retrieval_debug_enabled():
                _last_debug.entry = build_retrieval_debug(query, candidates, [], threshold, top_score)
                _scrub_debug_fields(candidates)
            return []

        # 先裁到最终要交付的片段，再换成父片段，最后压缩。
        # 压缩的目标是"减少传给 LLM 的噪声"，因此必须作用在交付内容上；
        # 放在过滤之后还省掉了对必然被丢弃的候选的 reranker 调用。
        result = _compress_chunks(query, _expand_to_parent(result))

        logger.info(
            f"Retrieval: len={len(query)} → top_score={top_score:.3f}, "
            f"candidates={len(candidates)}, returned={len(result)}"
        )

        _store_retrieval_metrics(top_score, len(candidates), len(result), threshold)
        if retrieval_debug_enabled():
            _last_debug.entry = build_retrieval_debug(query, candidates, result, threshold, top_score)
            _scrub_debug_fields(candidates)

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
