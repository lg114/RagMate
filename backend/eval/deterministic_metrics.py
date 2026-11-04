"""确定性检索指标：只用 gold chunk ID 与排序算分，不经过 LLM。

Tier-1（检索层）基线专用。与 RAGAS 的 context_precision / context_recall
不同——那两个是 judge 打分的指标（见 eval/README.md）——这里的每个数都由
gold chunk ID 集合与实际返回的排序直接算出，同一索引、同一问题，跑多少
遍结果都一样。

Chunk 的身份用 (source, page, chunk_index) 三元组：Milvus 的自增 id 不可
复现，而三元组是入库切分时写进 metadata 的确定性标识，检索链路内部也用它认 chunk。
"""
import math

DEFAULT_KS = (5, 10, 20)


def normalize_key(entry) -> tuple:
    """把一个 chunk 标识归一化为 (source, page, chunk_index) 三元组。

    接受 dict（{"source", "page", "chunk_index"}）或三元组本身；page 缺失
    记为 None（markdown 入库时没有 page 字段）。
    """
    if isinstance(entry, dict):
        return (
            entry.get("source", ""),
            entry.get("page"),
            entry.get("chunk_index"),
        )
    source, page, chunk_index = entry
    return (source, page, chunk_index)


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def score_ranking(ranked_keys, gold_keys, ks=DEFAULT_KS) -> dict:
    """算一道题在给定排序下的全部指标。

    ranked_keys：按排名顺序的 chunk 三元组（已归一化或可归一化）；
    gold_keys：该题的 gold chunk 三元组集合。返回的 key 形如 "hit@5"、
    "recall@10"、"ndcg@20"、"mrr"。排序为空时各项为 0；gold 为空时由
    调用方排除在均值之外（unanswerable 题没有可命中的 gold）。
    """
    ranked = [normalize_key(k) for k in ranked_keys]
    gold = {normalize_key(k) for k in gold_keys}
    scores = {}
    for k in ks:
        top = ranked[:k]
        hits = sum(1 for key in top if key in gold)
        scores[f"hit@{k}"] = 1.0 if hits > 0 else 0.0
        scores[f"recall@{k}"] = hits / len(gold) if gold else 0.0
        scores[f"precision@{k}"] = hits / k if k else 0.0
        relevances = [1 if key in gold else 0 for key in top]
        ideal = _dcg([1] * min(len(gold), k))
        scores[f"ndcg@{k}"] = _dcg(relevances) / ideal if ideal else 0.0
    first_hit = next((i for i, key in enumerate(ranked) if key in gold), None)
    scores["mrr"] = 1.0 / (first_hit + 1) if first_hit is not None else 0.0
    return scores


def evaluate_questions(items, ks=DEFAULT_KS) -> dict:
    """汇总一批题的确定性指标。

    items：每题 {"ranked": [...], "gold": [...], "source_doc": str,
    "question_type": str}。gold 为空的题（unanswerable）不计入均值，单独
    计数返回。返回 per_question 明细、summary 均值、按文档与题型的分组
    均值，以及 failure_rate@20（= 1 - recall@20，Anthropic 的失败率口径）。
    """
    per_question = []
    answerable = []
    n_unanswerable = 0
    for idx, item in enumerate(items):
        gold = [normalize_key(k) for k in item.get("gold", [])]
        if not gold:
            n_unanswerable += 1
            per_question.append({
                "index": idx,
                "user_input": item.get("user_input", ""),
                "source_doc": item.get("source_doc", ""),
                "question_type": item.get("question_type", ""),
                "unanswerable": True,
                "scores": {},
            })
            continue
        scores = score_ranking(item.get("ranked", []), gold, ks)
        per_question.append({
            "index": idx,
            "user_input": item.get("user_input", ""),
            "source_doc": item.get("source_doc", ""),
            "question_type": item.get("question_type", ""),
            "unanswerable": False,
            "scores": scores,
        })
        answerable.append((item, scores))

    def _mean(pairs, key):
        values = [scores[key] for _, scores in pairs]
        return round(sum(values) / len(values), 4) if values else None

    metric_names = []
    if answerable:
        metric_names = list(answerable[0][1].keys())
    summary = {name: _mean(answerable, name) for name in metric_names}
    if summary.get("recall@20") is not None:
        summary["failure_rate@20"] = round(1.0 - summary["recall@20"], 4)

    def _breakdown(field):
        groups: dict[str, list] = {}
        for item, scores in answerable:
            groups.setdefault(item.get(field, ""), []).append((item, scores))
        return {
            name: {metric: _mean(pairs, metric) for metric in metric_names}
            for name, pairs in sorted(groups.items())
        }

    return {
        "n_questions": len(items),
        "n_answerable": len(answerable),
        "n_unanswerable": n_unanswerable,
        "summary": summary,
        "by_source_doc": _breakdown("source_doc"),
        "by_question_type": _breakdown("question_type"),
        "per_question": per_question,
    }


def metric_spread(summaries: list[dict]) -> dict:
    """量重复跑的极差：同一组 summary 跑多次，每个指标的最大值减最小值。

    确定性检索层在固定索引下极差应为 0；非 0 即说明索引、排序或
    ANN 近似里有不确定性，先修它再比 A/B。
    """
    spread = {}
    if not summaries:
        return spread
    names = set().union(*(s.keys() for s in summaries))
    for name in sorted(names):
        values = [s[name] for s in summaries if s.get(name) is not None]
        if len(values) > 1:
            spread[name] = round(max(values) - min(values), 4)
        elif values:
            spread[name] = 0.0
    return spread
