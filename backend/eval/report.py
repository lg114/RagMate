"""评估报告构建和打印。"""
from .utils import clean_score


def build_report(meta: dict, test_cases: list[dict], ragas_result) -> dict:
    """Build a structured evaluation report."""
    details = []
    scores_list = ragas_result.scores if hasattr(ragas_result, "scores") else []

    for i, tc in enumerate(test_cases):
        scores = {}
        if i < len(scores_list):
            for metric_name, value in scores_list[i].items():
                scores[metric_name] = clean_score(value)

        detail = {
            "user_input": tc["user_input"],
            "reference": tc["reference"],
            "response": tc["response"],
            "contexts": tc["retrieved_contexts"],
            "scores": scores,
        }
        if tc.get("error"):
            detail["error"] = tc["error"]
        details.append(detail)

    summary = {}
    if scores_list:
        metric_names = list(scores_list[0].keys())
        for metric_name in metric_names:
            values = [
                clean_score(s[metric_name])
                for s in scores_list
                if clean_score(s.get(metric_name)) is not None
            ]
            summary[metric_name] = round(sum(values) / len(values), 4) if values else None

    valid_scores = [v for v in summary.values() if v is not None]
    summary["overall"] = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None

    return {"meta": meta, "summary": summary, "details": details}


def print_report(report: dict):
    """Print evaluation report to console."""
    meta = report["meta"]
    summary = report["summary"]

    print("\n=== RAGAS Evaluation Report ===")
    print(f"Test cases: {meta['test_cases']}")
    print(f"Mode: {meta['mode']}")
    print(f"Top-k: {meta['top_k']}")
    print(f"Judge model: {meta['judge_model']}")
    print()

    if summary:
        print("Metrics:")
        for name, value in summary.items():
            label = name.replace("_", " ").title()
            if value is not None:
                print(f"  {label:<25} {value:.2f}")
            else:
                print(f"  {label:<25} N/A (failed)")
    else:
        print("No metrics computed (all cases may have errored).")

    print(f"\nDetailed report saved to {meta.get('report_path', 'N/A')}")
