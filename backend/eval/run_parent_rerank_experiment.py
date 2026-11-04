"""P1：在 parent≫child 的集合上，对比 child 正文重排 vs parent 正文重排。

当前主索引 CHUNK_SIZE=1000 时 parent_text==text，展开与「用 parent 打分」
都是空操作。本脚本临时 CHUNK_SIZE=300 写入独立 collection，再对同一批
fused 候选分别用 child / parent 文本调 reranker，看排序是否被父段噪声稀释。

不依赖 gold chunk_index（切块边界已变）；比的是 top-k 重叠与名次相关。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "eval" / "testsets" / "sample" / "corpus"
TESTSET = ROOT / "eval" / "testsets" / "sample" / "testset.json"
COLLECTION = "ragmate_docs__c300"
REPORT = ROOT / "eval" / "reports" / "baseline" / "retrieval-baseline-98q-parent-rerank.json"
CHILD_CHUNK = 300


def _kendall_tau(rank_a: list[tuple], rank_b: list[tuple]) -> float:
    """在共有 key 上算 Kendall τ（无 tie 修正的简易版）。"""
    pos_a = {k: i for i, k in enumerate(rank_a)}
    pos_b = {k: i for i, k in enumerate(rank_b)}
    common = [k for k in rank_a if k in pos_b]
    n = len(common)
    if n < 2:
        return 1.0
    concord = discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            ki, kj = common[i], common[j]
            sign_a = (pos_a[ki] - pos_a[kj])
            sign_b = (pos_b[ki] - pos_b[kj])
            if sign_a == 0 or sign_b == 0:
                continue
            if (sign_a > 0) == (sign_b > 0):
                concord += 1
            else:
                discord += 1
    total = concord + discord
    return (concord - discord) / total if total else 1.0


def _topk_overlap(a: list[tuple], b: list[tuple], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    from backend.infrastructure.config import settings
    from backend.infrastructure.vector_store import PgVectorStore, init_vector_store, reset_vector_store_cache
    from backend.eval.corpus_ingest import ingest_corpus_to_collection
    from backend.eval.retrieval_eval import _hit_entity, _search_fused, load_golden_testset
    from backend.core.retriever import score_pairs
    from backend.infrastructure.encoding import encode_query

    cases = load_golden_testset(TESTSET)
    answerable = [c for c in cases if c.get("gold_chunks")]

    original = {
        "CHUNK_SIZE": settings.CHUNK_SIZE,
        "CHUNKING_STRATEGY": settings.CHUNKING_STRATEGY,
        "VECTOR_TABLE": settings.VECTOR_TABLE,
    }
    settings.CHUNK_SIZE = CHILD_CHUNK
    settings.CHUNKING_STRATEGY = "title_path"
    settings.VECTOR_TABLE = COLLECTION

    print(f"=== 1) Ingest corpus CHUNK_SIZE={CHILD_CHUNK} → {COLLECTION}")
    info = ingest_corpus_to_collection(str(CORPUS), COLLECTION)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    reset_vector_store_cache()
    settings.VECTOR_TABLE = COLLECTION

    init_vector_store()
    store = PgVectorStore(COLLECTION)
    rows = store.fetch_chunks(limit=16384)
    ratios = []
    identical = 0
    for row in rows:
        child = row.get("text") or ""
        parent = row.get("parent_text") or child
        if parent.strip() == child.strip():
            identical += 1
        ratios.append((len(parent) + 1) / (len(child) + 1))
    ratios.sort()
    pc = {
        "n_chunks": len(rows),
        "identical_parent_child": identical,
        "identical_ratio": round(identical / max(len(rows), 1), 4),
        "median_len_ratio": round(ratios[len(ratios) // 2], 4) if ratios else 1.0,
        "mean_len_ratio": round(sum(ratios) / max(len(ratios), 1), 4),
    }
    print("parent/child:", pc)

    over_fetch = 50
    per_q = []
    tau_vals = []
    ov5 = []
    ov15 = []
    top1_flip = 0

    print(f"=== 2) Child vs parent rerank on {len(answerable)} answerable questions")
    from tqdm import tqdm

    for tc in tqdm(answerable, desc="parent-rerank", unit="q"):
        q = tc["user_input"]
        dense, sparse = encode_query(q)
        hits = _search_fused(store, dense, sparse, over_fetch)[0]

        cands = []
        for hit in hits:
            entity = _hit_entity(hit)
            meta = entity.get("metadata", {})
            text = entity.get("text") or ""
            parent = entity.get("parent_text") or text
            key = (meta.get("source", ""), meta.get("page"), meta.get("chunk_index"))
            cands.append({"key": key, "text": text, "parent": parent})

        if not cands:
            continue

        child_scores = score_pairs(q, [c["text"] for c in cands])
        parent_scores = score_pairs(q, [c["parent"] for c in cands])
        by_child = sorted(
            zip(cands, child_scores), key=lambda x: x[1], reverse=True,
        )
        by_parent = sorted(
            zip(cands, parent_scores), key=lambda x: x[1], reverse=True,
        )
        rank_c = [c["key"] for c, _ in by_child]
        rank_p = [c["key"] for c, _ in by_parent]
        tau = _kendall_tau(rank_c, rank_p)
        o5 = _topk_overlap(rank_c, rank_p, 5)
        o15 = _topk_overlap(rank_c, rank_p, 15)
        flip = rank_c[0] != rank_p[0]
        if flip:
            top1_flip += 1
        tau_vals.append(tau)
        ov5.append(o5)
        ov15.append(o15)
        per_q.append({
            "user_input": q[:80],
            "question_type": tc.get("question_type"),
            "kendall_tau": round(tau, 4),
            "overlap@5": round(o5, 4),
            "overlap@15": round(o15, 4),
            "top1_flip": flip,
        })

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    summary = {
        "n_questions": len(per_q),
        "mean_kendall_tau": mean(tau_vals),
        "mean_overlap@5": mean(ov5),
        "mean_overlap@15": mean(ov15),
        "top1_flip_rate": round(top1_flip / max(len(per_q), 1), 4),
        "parent_child": pc,
        "ingest": info,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "collection": COLLECTION,
            "CHUNK_SIZE": CHILD_CHUNK,
            "note": (
                "Compare reranker order when scoring child text vs parent_text "
                "on the same fused/dense candidate set. No gold IDs (chunk "
                "boundaries differ from the 98q annotations)."
            ),
        },
        "summary": summary,
        "per_question": per_q,
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = REPORT.with_suffix(".md")
    md.write_text(
        "\n".join([
            "# Parent vs child rerank (c300)",
            "",
            f"Collection `{COLLECTION}`, CHUNK_SIZE={CHILD_CHUNK}, strategy=title_path.",
            f"Chunks={pc['n_chunks']}, parent==child={pc['identical_parent_child']} "
            f"({pc['identical_ratio']}), median len ratio={pc['median_len_ratio']}.",
            "",
            f"| metric | value |",
            f"|---|---|",
            f"| mean Kendall τ | {summary['mean_kendall_tau']} |",
            f"| mean overlap@5 | {summary['mean_overlap@5']} |",
            f"| mean overlap@15 | {summary['mean_overlap@15']} |",
            f"| top1 flip rate | {summary['top1_flip_rate']} |",
            "",
            "τ / overlap 接近 1 → parent 正文重排几乎不改序，展开噪声未稀释排序；",
            "明显下降才说明「只按一层 child 打分会掩盖 parent 稀释」。",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"Wrote {REPORT} and {md}")

    for k, v in original.items():
        setattr(settings, k, v)
    reset_vector_store_cache()


if __name__ == "__main__":
    main()
