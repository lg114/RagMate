"""接地编排：弹出检索信号、算置信度、跑忠实度校验、改写回答。"""

import asyncio
import logging

from backend.application.chat_citations import (
    apply_grounding,
    citations_for_answer,
    extract_model_citations,
    note_unsupported_claims,
    public_citations,
    question_language,
    strip_citation_markup,
)

logger = logging.getLogger("ragmate")

async def _pop_retrieval_signals(
    session_id: str, response_text: str, is_error: bool, *, debug: bool = False,
    rdata: dict | None = None,
) -> tuple[dict | None, list[dict] | None, list[dict], list[dict] | None]:
    """取出本次问答的检索数据，计算置信度、引用与（可选的）无依据声明。

    rdata 由单次检索链路直接传入（见 _retrieval_rdata）。

    Returns:
        (confidence, unsupported_claims, citations)。
        confidence 为 None 表示本次没有发生检索；
        unsupported_claims 为 None 表示未做校验或全部通过；
        citations 为检索到的原始引用（可能为空列表）。
        第四项是检索调试轨迹，仅当本次请求打开了 debug 时才返回。
    """
    from backend.core.retriever import calculate_confidence, check_faithfulness

    if rdata is None:
        return None, None, [], None
    if not rdata:
        return None, None, [], None

    confidence = calculate_confidence(rdata.get("metrics", []))
    citations = list(rdata.get("citations") or [])

    unsupported = None
    # 没有任何留下的片段时回答会被拒答替换，不必再花一次 LLM 做忠实度校验
    no_evidence = bool(confidence) and confidence.get("level") == "low" and int(confidence.get("chunks") or 0) <= 0
    # 忠实度校验只在低置信度时跑：证据扎实（medium/high）时跳过这次 LLM 调用。
    # 置信度未知（None）时按低置信处理，保留原有行为。
    low_confidence = confidence is None or confidence.get("level") == "low"
    if not is_error and not no_evidence and low_confidence and rdata.get("texts"):
        # check_faithfulness 内部是同步 LLM 调用，必须挪到线程里，
        # 否则会阻塞整个事件循环
        claims = await asyncio.to_thread(check_faithfulness, response_text, rdata["texts"])
        if claims:
            unsupported = [c for c in claims if not c.get("supported")] or None
            if unsupported:
                logger.warning(
                    f"Faithfulness: {len(unsupported)} unsupported claims in session={session_id[:8]}"
                )

    debug_trace = list(rdata.get("debug") or []) if debug else None
    return confidence, unsupported, citations, debug_trace or None


async def _ground_response(
    session_id: str, response_text: str, is_error: bool, *, debug: bool = False,
    rdata: dict | None = None, question: str | None = None,
) -> tuple[str, dict]:
    """按检索置信度改写回答。返回 (最终文本, 要并入 API 的附加字段)。

    rdata 为 None 时表示本次没有检索信号；单次检索链路直接传入。
    question 用来决定引用格式的语言；缺省时保持中文。
    对外 citations 只保留模型写过且能对上检索片段的页，按首次出现排序。
    模型一条都没写出可解析的引用、但回答仍然接地时，改回取得分最高的 3 条。
    拒答不带引用。
    """
    confidence, unsupported, citations, debug_trace = await _pop_retrieval_signals(
        session_id, response_text, is_error, debug=debug, rdata=rdata,
    )
    extra: dict = {}
    if debug and debug_trace:
        extra["retrieval_debug"] = debug_trace
    if is_error:
        if confidence:
            extra["confidence"] = confidence
        return response_text, extra

    lang = question_language(question) if question else "zh"
    publishable = bool(confidence) and int(confidence.get("chunks") or 0) > 0
    if publishable:
        parsed = extract_model_citations(response_text)
        chosen = citations_for_answer(response_text, citations)
        # 写了引用但一条都对不上检索结果：丢掉这些标记，也不要用别的页充数。
        if parsed and not chosen:
            response_text = strip_citation_markup(response_text)
        attach = chosen
    else:
        chosen = []
        attach = citations

    response_text, grounding = apply_grounding(response_text, confidence, attach, lang)
    if grounding == "refused":
        unsupported = None
    elif unsupported:
        response_text = note_unsupported_claims(response_text, unsupported)
    if confidence:
        extra["confidence"] = confidence
    if grounding:
        extra["grounding"] = grounding
    if grounding == "refused":
        public = []
    elif publishable:
        public = public_citations(chosen)
    else:
        public = public_citations(citations)
    if public:
        extra["citations"] = public
    if unsupported:
        extra["unsupported_claims"] = unsupported
    return response_text, extra
