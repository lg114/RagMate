"""评估指标计算。"""


def retrieval_recall(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    """检索召回率：期望来源中有多少被检索到了。"""
    if not expected_sources:
        return 1.0
    expected_set = {s.lower() for s in expected_sources}
    retrieved_set = {s.lower() for s in retrieved_sources}
    hits = expected_set & retrieved_set
    return len(hits) / len(expected_set)


def retrieval_precision(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    """检索精确率：检索到的来源中有多少是期望的。"""
    if not retrieved_sources:
        return 0.0
    expected_set = {s.lower() for s in expected_sources}
    retrieved_set = {s.lower() for s in retrieved_sources}
    hits = expected_set & retrieved_set
    return len(hits) / len(retrieved_set)


def format_report(results: list[dict]) -> str:
    """格式化评估报告。"""
    lines = ["=" * 60, "评估报告", "=" * 60, ""]

    total_recall = 0
    total_precision = 0
    count = len(results)

    for i, r in enumerate(results, 1):
        recall = r.get("recall", 0)
        precision = r.get("precision", 0)
        total_recall += recall
        total_precision += precision

        status = "✓" if recall >= 1.0 else "✗"
        lines.append(f"[{status}] Q{i}: {r['question']}")
        lines.append(f"    期望来源: {r['expected_sources']}")
        lines.append(f"    检索来源: {r.get('retrieved_sources', [])}")
        lines.append(f"    召回率: {recall:.2f}  精确率: {precision:.2f}")
        lines.append("")

    avg_recall = total_recall / count if count else 0
    avg_precision = total_precision / count if count else 0

    lines.append("-" * 60)
    lines.append(f"总计: {count} 条")
    lines.append(f"平均召回率: {avg_recall:.2f}")
    lines.append(f"平均精确率: {avg_precision:.2f}")
    lines.append("=" * 60)

    return "\n".join(lines)
