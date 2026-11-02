"""评估报告构建和打印。"""
import json
from pathlib import Path

from .utils import clean_score


def write_raw_checkpoint(report_path, meta: dict, test_cases: list[dict]) -> Path:
    """Write unscored pipeline results next to the report, before scoring.

    Scoring can fail after a long pipeline run (RAGAS swallows per-row
    failures as NaN); the checkpoint preserves the retrieval+generation work
    so only scoring has to be redone (see `rescore`). Checkpoints are
    git-ignored (`*.raw.json`).
    """
    raw_path = Path(str(report_path)).with_suffix(".raw.json")
    checkpoint_meta = dict(meta)
    checkpoint_meta["raw_checkpoint"] = True
    details = [
        {
            "user_input": tc["user_input"],
            "reference": tc.get("reference", ""),
            "response": tc.get("response", ""),
            "contexts": tc.get("retrieved_contexts", []),
        }
        for tc in test_cases
        if tc.get("response") and tc.get("retrieved_contexts")
    ]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"meta": checkpoint_meta, "details": details}, f, ensure_ascii=False, indent=2)
    return raw_path


def cases_from_report(report: dict) -> list[dict]:
    """Rebuild scoring inputs from a saved report (or raw checkpoint)."""
    cases = []
    for detail in report.get("details", []):
        if detail.get("response") and detail.get("contexts"):
            cases.append({
                "user_input": detail["user_input"],
                "reference": detail.get("reference", ""),
                "response": detail["response"],
                "retrieved_contexts": detail["contexts"],
            })
    return cases


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
    print(f"Test cases: {meta.get('test_cases', len(report.get('details', [])))}")
    print(f"Mode: {meta.get('mode', '-')}")
    print(f"Top-k: {meta.get('top_k', '-')}")
    print(f"Judge model: {meta.get('judge_model', '-')}")
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
