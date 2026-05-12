"""评估运行器：加载测试数据集，运行检索，计算指标。"""

import json
import os
import sys
import time

_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import settings
from eval.metrics import contains_check, format_report, must_not_check, retrieval_precision, retrieval_recall
from retriever import retrieve


def load_dataset(path: str = None) -> list[dict]:
    """加载评估数据集。"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval(dataset_path: str = None, k: int | None = None) -> dict:
    """运行评估，返回结果和报告。"""
    if k is None:
        k = settings.RERANKER_TOP_K
    dataset = load_dataset(dataset_path)
    results = []

    print(f"运行评估: {len(dataset)} 条测试用例, top-{k}")
    print("-" * 40)

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        expected = item.get("expected_sources", [])

        print(f"[{i}/{len(dataset)}] {question}")

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
            "latency_ms": round(latency * 1000),
            "details": details,
        })

        status = "✓" if recall >= 1.0 else "✗"
        print(f"  {status} recall={recall:.2f} precision={precision:.2f} contains={contains:.2f} no_leak={must_not_score:.2f} latency={latency*1000:.0f}ms")
        for d in details:
            page_display = d["page"] + 1 if d["page"] is not None else "?"
            print(f"    [{d['score']:.4f}] {d['source']} p{page_display} c{d['chunk_index']}: {d['preview']}...")

    report = format_report(results)
    print()
    print(report)

    return {"results": results, "report": report}


if __name__ == "__main__":
    run_eval()
