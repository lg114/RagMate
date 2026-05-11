"""评估运行器：加载测试数据集，运行检索，计算指标。"""

import json
import os
import time

from eval.metrics import format_report, retrieval_precision, retrieval_recall
from retriever import retrieve


def load_dataset(path: str = None) -> list[dict]:
    """加载评估数据集。"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval(dataset_path: str = None, k: int = 3) -> dict:
    """运行评估，返回结果和报告。"""
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

        retrieved_sources = list({r.get("source", "") for r in retrieved if r.get("source")})

        recall = retrieval_recall(expected, retrieved_sources)
        precision = retrieval_precision(expected, retrieved_sources)

        results.append({
            "question": question,
            "expected_sources": expected,
            "retrieved_sources": retrieved_sources,
            "recall": recall,
            "precision": precision,
            "latency_ms": round(latency * 1000),
        })

        status = "✓" if recall >= 1.0 else "✗"
        print(f"  {status} recall={recall:.2f} precision={precision:.2f} latency={latency*1000:.0f}ms")

    report = format_report(results)
    print()
    print(report)

    return {"results": results, "report": report}


if __name__ == "__main__":
    run_eval()
