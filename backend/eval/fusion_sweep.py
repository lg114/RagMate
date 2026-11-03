"""P2：融合调参扫参——用已有 retrieval checkpoint 的 dense/sparse 排序离线重融合。

一次只动一个变量（RRF_k 或 dense_weight），不重跑编码/检索。重点看 fused
臂的 paraphrase / multi-hop；默认 k=60 作对照。

用法（仓库根目录）：
  python -m backend.eval.fusion_sweep
  python -m backend.eval.fusion_sweep --checkpoint path/to/*.retrieval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CHECKPOINT = (
    ROOT / "eval" / "reports" / "baseline"
    / "retrieval-baseline-98q-titlepath-rerun.retrieval.json"
)
DEFAULT_TESTSET = ROOT / "eval" / "testsets" / "sample" / "testset.json"
DEFAULT_OUT = (
    ROOT / "eval" / "reports" / "baseline" / "retrieval-baseline-98q-fusion-sweep.json"
)

RRF_GRID = (10, 20, 40, 60, 80, 100)
WEIGHT_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
KS = (5, 10, 20)
FOCUS_TYPES = ("paraphrase", "multi-hop", "factoid", "ambiguous")


def _key(rec: dict) -> tuple:
    return (rec.get("source", ""), rec.get("page"), rec.get("chunk_index"))


def _eval_fused(cases, fused_rankings, ks=KS) -> dict:
    from backend.eval.deterministic_metrics import evaluate_questions, normalize_key

    items = []
    for tc, ranked in zip(cases, fused_rankings):
        items.append({
            "user_input": tc.get("user_input", ""),
            "source_doc": tc.get("source_doc", ""),
            "question_type": tc.get("question_type", ""),
            "ranked": [normalize_key(k) for k in ranked],
            "gold": [normalize_key(g) for g in tc.get("gold_chunks", [])],
        })
    return evaluate_questions(items, ks)


def _slice(evaluation: dict) -> dict:
    summary = evaluation["summary"]
    by_type = {}
    for qt in FOCUS_TYPES:
        metrics = evaluation["by_question_type"].get(qt) or {}
        by_type[qt] = {
            "mrr": metrics.get("mrr"),
            "recall@5": metrics.get("recall@5"),
            "ndcg@5": metrics.get("ndcg@5"),
        }
    return {
        "mrr": summary.get("mrr"),
        "recall@5": summary.get("recall@5"),
        "ndcg@5": summary.get("ndcg@5"),
        "hit@5": summary.get("hit@5"),
        "by_question_type": by_type,
    }


def sweep_from_checkpoint(checkpoint: dict, cases: list[dict], limit: int) -> dict:
    from backend.core.fusion import rrf_fuse_rank_lists, weighted_fuse_score_lists

    stage_lists = checkpoint["stages"]
    results = {"rrf": [], "weighted": []}

    # baseline fused from checkpoint (Milvus RRF k=60 historically)
    baseline_rankings = []
    for idx in range(len(cases)):
        baseline_rankings.append([_key(r) for r in stage_lists[str(idx)]["fused"]])
    baseline_eval = _eval_fused(cases, baseline_rankings)
    results["checkpoint_fused"] = _slice(baseline_eval)

    for k in RRF_GRID:
        rankings = []
        for idx in range(len(cases)):
            dense = [_key(r) for r in stage_lists[str(idx)]["dense"]]
            sparse = [_key(r) for r in stage_lists[str(idx)]["sparse"]]
            rankings.append(rrf_fuse_rank_lists(dense, sparse, k=k, limit=limit))
        evaluation = _eval_fused(cases, rankings)
        results["rrf"].append({"RRF_K": k, **_slice(evaluation)})

    for w in WEIGHT_GRID:
        rankings = []
        for idx in range(len(cases)):
            dense_items = [
                (_key(r), float(r["score"])) for r in stage_lists[str(idx)]["dense"]
            ]
            sparse_items = [
                (_key(r), float(r["score"])) for r in stage_lists[str(idx)]["sparse"]
            ]
            rankings.append(
                weighted_fuse_score_lists(
                    dense_items, sparse_items, dense_weight=w, limit=limit,
                )
            )
        evaluation = _eval_fused(cases, rankings)
        results["weighted"].append({
            "FUSION_DENSE_WEIGHT": w,
            "FUSION_SPARSE_WEIGHT": round(1.0 - w, 4),
            **_slice(evaluation),
        })

    return results


def _pick_best_rrf(rows: list[dict], baseline_k: int = 60) -> dict:
    """优先 multi-hop recall@5，其次 paraphrase MRR，再次整体 MRR；持平则靠近 60。"""
    def key(row):
        mh = (row.get("by_question_type") or {}).get("multi-hop") or {}
        para = (row.get("by_question_type") or {}).get("paraphrase") or {}
        return (
            mh.get("recall@5") or 0.0,
            para.get("mrr") or 0.0,
            row.get("mrr") or 0.0,
            -abs(row["RRF_K"] - baseline_k),
        )
    return max(rows, key=key)


def _pick_best_weight(rows: list[dict]) -> dict:
    def key(row):
        mh = (row.get("by_question_type") or {}).get("multi-hop") or {}
        para = (row.get("by_question_type") or {}).get("paraphrase") or {}
        return (
            mh.get("recall@5") or 0.0,
            para.get("mrr") or 0.0,
            row.get("mrr") or 0.0,
            -abs(row["FUSION_DENSE_WEIGHT"] - 0.5),
        )
    return max(rows, key=key)


def write_reports(payload: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = out_path.with_suffix(".md")
    lines = [
        "# Fusion sweep (offline from dense/sparse checkpoint)",
        "",
        f"Checkpoint: `{payload['meta']['checkpoint']}`",
        f"Limit (fused top-k): {payload['meta']['limit']}",
        "",
        "Re-fuses each question from saved dense/sparse ranks only — "
        "no re-encode, no Milvus. Compare fused-arm metrics; reranked not re-run.",
        "",
        "## Checkpoint fused (as stored)",
        "",
        f"- MRR={payload['results']['checkpoint_fused']['mrr']} "
        f"recall@5={payload['results']['checkpoint_fused']['recall@5']} "
        f"nDCG@5={payload['results']['checkpoint_fused']['ndcg@5']}",
        "",
        "## RRF_K grid",
        "",
        "| RRF_K | MRR | recall@5 | nDCG@5 | mh recall@5 | para MRR |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["results"]["rrf"]:
        mh = row["by_question_type"]["multi-hop"]
        para = row["by_question_type"]["paraphrase"]
        lines.append(
            f"| {row['RRF_K']} | {row['mrr']} | {row['recall@5']} | {row['ndcg@5']} | "
            f"{mh.get('recall@5')} | {para.get('mrr')} |"
        )
    best_rrf = payload["recommendation"]["best_rrf"]
    lines.extend([
        "",
        f"**Best RRF_K (heuristic): {best_rrf['RRF_K']}** — "
        f"MRR={best_rrf['mrr']}, mh recall@5="
        f"{best_rrf['by_question_type']['multi-hop'].get('recall@5')}",
        "",
        "## Weighted dense_weight grid",
        "",
        "| dense_w | sparse_w | MRR | recall@5 | mh recall@5 | para MRR |",
        "|---|---|---|---|---|---|",
    ])
    for row in payload["results"]["weighted"]:
        mh = row["by_question_type"]["multi-hop"]
        para = row["by_question_type"]["paraphrase"]
        lines.append(
            f"| {row['FUSION_DENSE_WEIGHT']} | {row['FUSION_SPARSE_WEIGHT']} | "
            f"{row['mrr']} | {row['recall@5']} | "
            f"{mh.get('recall@5')} | {para.get('mrr')} |"
        )
    best_w = payload["recommendation"]["best_weighted"]
    lines.extend([
        "",
        f"**Best dense_weight (heuristic): {best_w['FUSION_DENSE_WEIGHT']}** — "
        f"MRR={best_w['mrr']}",
        "",
        "## 结论口径",
        "",
        "- 与 k=60 / weight=0.5 持平或差在小数点后三位 → **保持默认**，避免过拟合 98 题。",
        "- 仅当 multi-hop / paraphrase 有稳定、同向提升时才改 `RRF_K` / `FUSION_MODE`。",
        "- 改生产前应对 winner 再跑一次完整 `ragmate-eval retrieval`（含 reranked）。",
        "",
    ])
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Offline fusion parameter sweep")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--testset", type=str, default=str(DEFAULT_TESTSET))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=50, help="Fused list length")
    args = parser.parse_args(argv)

    from backend.eval.retrieval_eval import load_golden_testset, load_retrieval_checkpoint

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    testset_path = Path(args.testset)
    if not testset_path.is_absolute():
        testset_path = ROOT / testset_path

    checkpoint = load_retrieval_checkpoint(checkpoint_path)
    cases = load_golden_testset(testset_path)
    n_cp = len(checkpoint.get("cases", []))
    if n_cp and n_cp != len(cases):
        raise SystemExit(
            f"Test set has {len(cases)} cases but checkpoint has {n_cp}"
        )

    results = sweep_from_checkpoint(checkpoint, cases, limit=args.limit)
    best_rrf = _pick_best_rrf(results["rrf"])
    best_w = _pick_best_weight(results["weighted"])
    baseline_rrf = next(r for r in results["rrf"] if r["RRF_K"] == 60)

    def delta(a, b, key="mrr"):
        if a.get(key) is None or b.get(key) is None:
            return None
        return round(a[key] - b[key], 4)

    mh_best = best_rrf["by_question_type"]["multi-hop"].get("recall@5") or 0
    mh_60 = baseline_rrf["by_question_type"]["multi-hop"].get("recall@5") or 0
    para_best = best_rrf["by_question_type"]["paraphrase"].get("mrr") or 0
    para_60 = baseline_rrf["by_question_type"]["paraphrase"].get("mrr") or 0
    mrr_d = delta(best_rrf, baseline_rrf, "mrr") or 0
    # 只有明显、同向的题型收益才建议改默认，避免 98 题过拟合
    worth_changing = best_rrf["RRF_K"] != 60 and (
        mrr_d >= 0.01
        or (mh_best - mh_60) >= 0.05
        or (para_best - para_60) >= 0.05
    )
    recommendation = {
        "best_rrf": best_rrf,
        "best_weighted": best_w,
        "vs_rrf60": {
            "mrr_delta": mrr_d,
            "recall@5_delta": delta(best_rrf, baseline_rrf, "recall@5"),
            "multi_hop_recall@5_delta": round(mh_best - mh_60, 4),
            "paraphrase_mrr_delta": round(para_best - para_60, 4),
            "keep_default_rrf60": not worth_changing,
        },
        "note": (
            "Prefer keep RRF_K=60 / FUSION_MODE=rrf unless multi-hop or paraphrase "
            "improves beyond tiny float noise on this 98q set."
        ),
    }

    payload = {
        "meta": {
            "checkpoint": str(checkpoint_path),
            "testset": str(testset_path),
            "limit": args.limit,
            "rrf_grid": list(RRF_GRID),
            "weight_grid": list(WEIGHT_GRID),
            "method": "offline re-fuse from dense/sparse ranks+scores in checkpoint",
        },
        "results": results,
        "recommendation": recommendation,
    }
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    md = write_reports(payload, out_path)
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
