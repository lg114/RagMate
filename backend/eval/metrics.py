"""评估指标计算。"""

from source_utils import canonical_source


def retrieval_recall(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    """检索召回率：期望来源中有多少被检索到了。"""
    if not expected_sources:
        return 1.0
    expected_set = {canonical_source(s) for s in expected_sources}
    retrieved_set = {canonical_source(s) for s in retrieved_sources}
    hits = expected_set & retrieved_set
    return len(hits) / len(expected_set)


def retrieval_precision(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    """检索精确率：检索到的来源中有多少是期望的。"""
    if not retrieved_sources:
        return 0.0
    expected_set = {canonical_source(s) for s in expected_sources}
    retrieved_set = {canonical_source(s) for s in retrieved_sources}
    hits = expected_set & retrieved_set
    return len(hits) / len(retrieved_set)


def contains_check(expected_contains: list[str], retrieved_texts: list[str]) -> float:
    """内容检查：期望的关键词在检索结果中出现了多少。"""
    if not expected_contains:
        return 1.0
    all_text = " ".join(retrieved_texts).lower()
    hits = sum(1 for kw in expected_contains if kw.lower() in all_text)
    return hits / len(expected_contains)


def must_not_check(must_not_sources: list[str], retrieved_sources: list[str]) -> float:
    """负面检查：不应出现的来源是否被过滤掉了。1.0 表示全部过滤成功。"""
    if not must_not_sources:
        return 1.0
    banned_set = {canonical_source(s) for s in must_not_sources}
    retrieved_set = {canonical_source(s) for s in retrieved_sources}
    leaks = banned_set & retrieved_set
    return 1.0 - len(leaks) / len(banned_set)


def format_report(results: list[dict]) -> str:
    """格式化评估报告。"""
    lines = ["=" * 60, "评估报告", "=" * 60, ""]

    total_recall = 0
    total_precision = 0
    total_contains = 0
    total_must_not = 0
    count = len(results)

    for i, r in enumerate(results, 1):
        recall = r.get("recall", 0)
        precision = r.get("precision", 0)
        contains = r.get("contains", 1.0)
        must_not = r.get("must_not", 1.0)
        total_recall += recall
        total_precision += precision
        total_contains += contains
        total_must_not += must_not

        status = "✓" if recall >= 1.0 else "✗"
        lines.append(f"[{status}] Q{i}: {r['question']}")
        lines.append(f"    期望来源: {r['expected_sources']}")
        lines.append(f"    检索来源: {r.get('retrieved_sources', [])}")
        lines.append(f"    召回率: {recall:.2f}  精确率: {precision:.2f}  内容匹配: {contains:.2f}  无泄漏: {must_not:.2f}")
        lines.append("")

    avg_recall = total_recall / count if count else 0
    avg_precision = total_precision / count if count else 0
    avg_contains = total_contains / count if count else 0
    avg_must_not = total_must_not / count if count else 0

    lines.append("-" * 60)
    lines.append(f"总计: {count} 条")
    lines.append(f"平均召回率: {avg_recall:.2f}")
    lines.append(f"平均精确率: {avg_precision:.2f}")
    lines.append(f"平均内容匹配: {avg_contains:.2f}")
    lines.append(f"平均无泄漏: {avg_must_not:.2f}")
    lines.append("=" * 60)

    return "\n".join(lines)
