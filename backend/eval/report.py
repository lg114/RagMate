"""评估报告构建和打印。"""
import json
from pathlib import Path

from .metrics import ScoreMatrix, summary_spread
from .utils import clean_score


def write_raw_checkpoint(
    report_path,
    meta: dict,
    test_cases: list[dict],
    *,
    force: bool = False,
) -> Path:
    """Write unscored pipeline results next to the report, before scoring.

    Scoring can fail after a long pipeline run (RAGAS swallows per-row
    failures as NaN); the checkpoint preserves the retrieval+generation work
    so only scoring has to be redone (see `rescore`). Checkpoints are
    git-ignored (`*.raw.json`).

    By default refuses to overwrite an existing raw file (immutable contract);
    pass force=True to replace.
    """
    raw_path = Path(str(report_path)).with_suffix(".raw.json")
    if raw_path.exists() and not force:
        raise FileExistsError(
            f"Raw checkpoint already exists: {raw_path}. "
            "Scoring should read it via `ragmate-eval rescore`; "
            "pass --force-raw to overwrite generation results."
        )
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
    failed = [
        {
            "user_input": tc["user_input"],
            "reference": tc.get("reference", ""),
            "error": tc.get("error") or "unknown",
        }
        for tc in test_cases
        if not (tc.get("response") and tc.get("retrieved_contexts"))
    ]
    if failed:
        checkpoint_meta["failed_questions"] = failed
        checkpoint_meta["failed_cases"] = len(failed)
    else:
        checkpoint_meta.pop("failed_questions", None)
        checkpoint_meta["failed_cases"] = 0
    checkpoint_meta["valid_cases"] = len(details)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"meta": checkpoint_meta, "details": details}, f, ensure_ascii=False, indent=2)
    return raw_path


def load_raw_checkpoint(raw_path) -> dict:
    path = Path(raw_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "details" not in data:
        raise ValueError(f"Not a raw/report JSON with details: {path}")
    return data


def questions_missing_from_raw(testset: list[dict], raw: dict) -> list[dict]:
    """题库里有、raw 里没有有效 response+contexts 的题（含 failed_questions）。"""
    present = {
        d["user_input"]
        for d in raw.get("details", [])
        if d.get("response") and d.get("contexts")
    }
    return [tc for tc in testset if tc.get("user_input") and tc["user_input"] not in present]


def merge_into_raw(raw: dict, new_cases: list[dict]) -> dict:
    """把补跑成功的题合并进 raw；同题覆盖。更新 meta 计数。"""
    by_q = {
        d["user_input"]: d
        for d in raw.get("details", [])
        if d.get("user_input")
    }
    for tc in new_cases:
        if not (tc.get("response") and tc.get("retrieved_contexts")):
            continue
        by_q[tc["user_input"]] = {
            "user_input": tc["user_input"],
            "reference": tc.get("reference", ""),
            "response": tc["response"],
            "contexts": tc["retrieved_contexts"],
        }
    details = list(by_q.values())
    meta = dict(raw.get("meta") or {})
    meta["valid_cases"] = len(details)
    # 仍缺失的失败题：保留旧 failed 里未补上的 + 本次仍失败的
    still_failed = []
    patched_ok = {tc["user_input"] for tc in new_cases if tc.get("response") and tc.get("retrieved_contexts")}
    for item in meta.get("failed_questions") or []:
        q = item.get("user_input")
        if q and q not in patched_ok and q not in by_q:
            still_failed.append(item)
    for tc in new_cases:
        if tc.get("response") and tc.get("retrieved_contexts"):
            continue
        still_failed.append({
            "user_input": tc["user_input"],
            "reference": tc.get("reference", ""),
            "error": tc.get("error") or "unknown",
        })
    # de-dup by question
    failed_by_q = {f["user_input"]: f for f in still_failed if f.get("user_input")}
    # drop any that are now in details
    for q in list(failed_by_q):
        if q in by_q:
            del failed_by_q[q]
    meta["failed_questions"] = list(failed_by_q.values())
    meta["failed_cases"] = len(meta["failed_questions"])
    meta["raw_checkpoint"] = True
    return {"meta": meta, "details": details}


def save_raw_checkpoint(raw_path, raw: dict) -> Path:
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    return path


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


def _scores_from_result(ragas_result) -> list[dict]:
    if isinstance(ragas_result, ScoreMatrix):
        return ragas_result.primary
    if hasattr(ragas_result, "scores"):
        return [dict(row) if isinstance(row, dict) else {} for row in (ragas_result.scores or [])]
    if isinstance(ragas_result, list):
        return ragas_result
    return []


def _per_case_status(scores: dict) -> tuple[str, str | None]:
    if not scores:
        return "unscored", "no metric values"
    if all(v is None for v in scores.values()):
        return "unscored", "all metrics null after parse retries"
    if any(v is None for v in scores.values()):
        missing = [k for k, v in scores.items() if v is None]
        return "partial", f"missing metrics: {', '.join(missing)}"
    return "scored", None


def build_report(meta: dict, test_cases: list[dict], ragas_result) -> dict:
    """Build a structured evaluation report.

    Accepts a legacy RAGAS result (``.scores``), a ``ScoreMatrix`` (repeats),
    or a plain list of per-case score dicts. Null cells after parse retries
    are marked ``unscored`` / ``partial`` and excluded from summary means.
    """
    matrix = ragas_result if isinstance(ragas_result, ScoreMatrix) else None
    scores_list = _scores_from_result(ragas_result)
    runs = matrix.runs if matrix and len(matrix.runs) > 1 else None

    details = []
    n_scored = n_partial = n_unscored = 0
    for i, tc in enumerate(test_cases):
        scores = {}
        if i < len(scores_list):
            for metric_name, value in scores_list[i].items():
                scores[metric_name] = clean_score(value)

        status, reason = _per_case_status(scores)
        if status == "scored":
            n_scored += 1
        elif status == "partial":
            n_partial += 1
        else:
            n_unscored += 1

        detail = {
            "user_input": tc["user_input"],
            "reference": tc["reference"],
            "response": tc["response"],
            "contexts": tc["retrieved_contexts"],
            "scores": scores,
            "status": status,
        }
        if reason:
            detail["unscored_reason"] = reason
        if runs is not None:
            detail["scores_runs"] = [
                {k: clean_score(v) for k, v in run[i].items()}
                if i < len(run) else {}
                for run in runs
            ]
        if tc.get("error"):
            detail["error"] = tc["error"]
        details.append(detail)

    summary = {}
    if scores_list:
        metric_names = list(scores_list[0].keys()) if scores_list[0] else []
        # Prefer union of keys across rows
        metric_names = sorted({k for row in scores_list for k in row.keys()})
        for metric_name in metric_names:
            values = [
                clean_score(s.get(metric_name))
                for s in scores_list
                if clean_score(s.get(metric_name)) is not None
            ]
            summary[metric_name] = round(sum(values) / len(values), 4) if values else None
            summary[f"{metric_name}__n"] = len(values)

    valid_scores = [
        v for k, v in summary.items()
        if v is not None and not str(k).endswith("__n")
    ]
    summary["overall"] = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None

    report_meta = dict(meta)
    report_meta["n_scored"] = n_scored
    report_meta["n_partial"] = n_partial
    report_meta["n_unscored"] = n_unscored
    if matrix and matrix.pin:
        report_meta.update(matrix.pin.as_meta())
    if runs is not None:
        report_meta["score_repeats"] = len(runs)
        report_meta["score_spread"] = summary_spread(runs)

    return {"meta": report_meta, "summary": summary, "details": details}


def print_report(report: dict):
    """Print evaluation report to console."""
    meta = report["meta"]
    summary = report["summary"]

    print("\n=== RAGAS Evaluation Report ===")
    print(f"Test cases: {meta.get('test_cases', len(report.get('details', [])))}")
    print(f"Mode: {meta.get('mode', '-')}")
    print(f"Top-k: {meta.get('top_k', '-')}")
    print(f"Judge model: {meta.get('judge_model', '-')}")
    if meta.get("judge_temperature") is not None:
        print(f"Judge temperature: {meta.get('judge_temperature')}")
    if meta.get("judge_metric_pack"):
        print(f"Judge metric pack: {meta.get('judge_metric_pack')} "
              f"(sha {meta.get('judge_metric_pack_sha256', '-')})")
    if meta.get("score_repeats"):
        print(f"Score repeats: {meta.get('score_repeats')}")
    if meta.get("n_unscored") or meta.get("n_partial"):
        print(
            f"Scored/partial/unscored: "
            f"{meta.get('n_scored', 0)}/{meta.get('n_partial', 0)}/{meta.get('n_unscored', 0)}"
        )
    print()

    if summary:
        print("Metrics:")
        for name, value in summary.items():
            if str(name).endswith("__n"):
                continue
            label = name.replace("_", " ").title()
            n = summary.get(f"{name}__n")
            n_s = f" (n={n})" if n is not None else ""
            if value is not None:
                print(f"  {label:<25} {value:.2f}{n_s}")
            else:
                print(f"  {label:<25} N/A (failed){n_s}")
        spread = meta.get("score_spread") or {}
        if spread:
            print("\nScore spread (max-min across repeats):")
            for name, value in spread.items():
                print(f"  {name:<25} {value:.4f}")
    else:
        print("No metrics computed (all cases may have errored).")

    print(f"\nDetailed report saved to {meta.get('report_path', 'N/A')}")
