"""评估运行器：加载测试数据集，运行检索，计算指标。"""

import json
import os
import sys
import time
from collections import defaultdict

_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import settings
from eval.metrics import contains_check, format_report, must_not_check, retrieval_precision, retrieval_recall
from retriever import retrieve

REFUSAL_INDICATORS = ["没有找到", "无法确认", "不在知识库", "没有相关资料", "无法回答", "没有找到相关"]


def load_dataset(path: str = None) -> list[dict]:
    """加载评估数据集。"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval(dataset_path: str = None, k: int | None = None) -> dict:
    """运行评估，返回结果和报告。"""
    if k is None:
        k = settings.FINAL_CONTEXT_K
    dataset = load_dataset(dataset_path)
    results = []

    # 按 difficulty 分组统计
    category_stats = defaultdict(lambda: {"count": 0, "recall_sum": 0, "precision_sum": 0, "contains_sum": 0, "must_not_sum": 0, "refuse_correct": 0, "refuse_total": 0})

    print(f"运行评估: {len(dataset)} 条测试用例, top-{k}")
    print("-" * 40)

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        expected = item.get("expected_sources", [])
        difficulty = item.get("difficulty", "normal")
        should_refuse = item.get("should_refuse", False)

        print(f"[{i}/{len(dataset)}] [{difficulty}] {question}")

        start = time.time()
        retrieved = retrieve(question, k=k)
        latency = time.time() - start

        expected_contains = item.get("expected_contains", [])
        must_not = item.get("must_not_sources", [])

        retrieved_sources = list(dict.fromkeys(r.get("source", "") for r in retrieved if r.get("source")))
        retrieved_texts = [r["text"] for r in retrieved]

        recall = retrieval_recall(expected, retrieved_sources)
        precision = retrieval_precision(expected, retrieved_sources)
        contains = contains_check(expected_contains, retrieved_texts)
        must_not_score = must_not_check(must_not, retrieved_sources)

        # 拒答检测
        refusal_correct = 0
        if should_refuse:
            # 应该拒答：检查是否没有返回结果或结果中包含拒答指示
            if not retrieved or any(indicator in " ".join(retrieved_texts) for indicator in REFUSAL_INDICATORS):
                refusal_correct = 1

        # 详细检索结果
        details = []
        for r in retrieved:
            details.append({
                "source": r.get("source", ""),
                "page": r.get("page"),
                "chunk_index": r.get("chunk_index"),
                "score": round(r.get("score", 0), 4),
                "preview": r["text"][:80].replace("\n", " "),
            })

        results.append({
            "question": question,
            "expected_sources": expected,
            "retrieved_sources": retrieved_sources,
            "recall": recall,
            "precision": precision,
            "contains": contains,
            "must_not": must_not_score,
            "should_refuse": should_refuse,
            "refusal_correct": refusal_correct,
            "difficulty": difficulty,
            "latency_ms": round(latency * 1000),
            "details": details,
        })

        # 更新分类统计
        cat = category_stats[difficulty]
        cat["count"] += 1
        if should_refuse:
            cat["refuse_total"] += 1
            cat["refuse_correct"] += refusal_correct
        else:
            cat["recall_sum"] += recall
            cat["precision_sum"] += precision
            cat["contains_sum"] += contains
            cat["must_not_sum"] += must_not_score

        if should_refuse:
            status = "✓" if refusal_correct else "✗"
            print(f"  {status} refuse_correct={refusal_correct} latency={latency*1000:.0f}ms")
        else:
            status = "✓" if recall >= 1.0 else "✗"
            print(f"  {status} recall={recall:.2f} precision={precision:.2f} contains={contains:.2f} no_leak={must_not_score:.2f} latency={latency*1000:.0f}ms")
            for d in details:
                page_display = d["page"] + 1 if d["page"] is not None else "?"
                print(f"    [{d['score']:.4f}] {d['source']} p{page_display} c{d['chunk_index']}: {d['preview']}...")

    report = format_report(results)
    print()
    print(report)

    # 分类汇总
    print()
    print("=" * 60)
    print("分类汇总")
    print("=" * 60)

    for difficulty in ["normal", "fuzzy", "typo", "reject", "multi"]:
        cat = category_stats[difficulty]
        if cat["count"] == 0:
            continue

        print(f"\n{difficulty.upper()} ({cat['count']} cases):")

        if cat["refuse_total"] > 0:
            refuse_rate = cat["refuse_correct"] / cat["refuse_total"] * 100
            print(f"  拒答正确率: {refuse_rate:.1f}% ({cat['refuse_correct']}/{cat['refuse_total']})")

        non_refuse = cat["count"] - cat["refuse_total"]
        if non_refuse > 0:
            avg_recall = cat["recall_sum"] / non_refuse
            avg_precision = cat["precision_sum"] / non_refuse
            avg_contains = cat["contains_sum"] / non_refuse
            print(f"  平均召回率: {avg_recall:.2f}")
            print(f"  平均精确率: {avg_precision:.2f}")
            print(f"  平均内容匹配: {avg_contains:.2f}")

    print()
    print("=" * 60)

    return {"results": results, "report": report, "category_stats": dict(category_stats)}


if __name__ == "__main__":
    run_eval()
