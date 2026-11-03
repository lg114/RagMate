"""P1 两层打分：child 检索排序 vs 生产交付（阈值/去重/截断）后的 gold 留存。

检索基线的 reranked 臂故意停在排序：阈值、去重、父展开是生成侧取舍。
本模块把同一 checkpoint 里的 reranked 列表再过一遍生产 `_filter_and_dedup`
（可选 `_expand_to_parent`），用同一套 gold 三元组算 hit/recall/MRR，对照
「排序层」与「交付层」——专门抓同源多证据被 MAX_PER_SOURCE / 断崖掐掉的静默缺口。

父展开不改 chunk_index，因此 expand 本身不改变 ID 指标；若索引里
parent_text == text（CHUNK_SIZE 过大、节未切开），展开是空操作，报告会标明。
真正「用 parent 正文重排是否稀释」要等 parent≫child 的集合（如 c300）再测。
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.core.retriever import _expand_to_parent, _filter_and_dedup
from backend.infrastructure.config import settings
from backend.infrastructure.milvus import get_milvus_client, init_milvus

from .deterministic_metrics import evaluate_questions, normalize_key
from .retrieval_eval import load_retrieval_checkpoint


def load_chunk_text_maps(collection: str | None = None) -> tuple[dict[tuple, str], dict[tuple, str]]:
    """返回 (parent_map, child_map)，key 均为 (source, page, chunk_index)。

    checkpoint 往往只落 text_preview，长度统计必须用索引里的完整 text / parent_text。
    """
    init_milvus()
    client = get_milvus_client()
    name = collection or settings.MILVUS_COLLECTION
    rows = client.query(
        collection_name=name,
        filter="",
        output_fields=["text", "parent_text", "metadata"],
        limit=16384,
    )
    parent_map: dict[tuple, str] = {}
    child_map: dict[tuple, str] = {}
    for row in rows:
        meta = row.get("metadata") or {}
        key = (meta.get("source", ""), meta.get("page"), meta.get("chunk_index"))
        child = row.get("text") or ""
        child_map[key] = child
        parent_map[key] = row.get("parent_text") or child
    return parent_map, child_map


def parent_child_stats(parent_map: dict[tuple, str], child_map: dict[tuple, str]) -> dict:
    """统计 parent 相对 child 是否真拉开。"""
    ratios = []
    identical = 0
    for key, child in child_map.items():
        parent = parent_map.get(key) or child
        if parent.strip() == child.strip():
            identical += 1
        ratios.append((len(parent) + 1) / (len(child) + 1))
    ratios.sort()
    n = len(ratios) or 1
    return {
        "n_chunks": len(child_map),
        "identical_parent_child": identical,
        "identical_ratio": round(identical / n, 4),
        "median_len_ratio": round(ratios[n // 2], 4) if ratios else 1.0,
        "mean_len_ratio": round(sum(ratios) / n, 4) if ratios else 1.0,
    }


def _records_to_candidates(records: list[dict], parent_map: dict[tuple, str]) -> list[dict]:
    cands = []
    for r in records:
        key = normalize_key(r)
        text = r.get("text") or ""
        cands.append({
            "text": text,
            "context": parent_map.get(key) or text,
            "source": r.get("source", ""),
            "page": r.get("page"),
            "chunk_index": r.get("chunk_index"),
            "score": float(r["score"]),
        })
    return cands


def simulate_delivery(
    checkpoint: dict,
    cases: list[dict],
    *,
    parent_map: dict[tuple, str] | None = None,
    child_map: dict[tuple, str] | None = None,
    expand: bool = True,
    ks: tuple[int, ...] = (5, 10, 15, 20),
) -> dict:
    """对 checkpoint 的 reranked 臂跑生产过滤，返回 child / delivered 两层指标。"""
    stage_lists = checkpoint["stages"]
    parent_map = parent_map if parent_map is not None else {}
    child_map = child_map if child_map is not None else {}
    threshold = settings.RERANK_SCORE_THRESHOLD
    final_k = settings.FINAL_CONTEXT_K

    child_items = []
    delivered_items = []
    drops = []

    for idx, tc in enumerate(cases):
        records = stage_lists[str(idx)]["reranked"]

        gold = [normalize_key(g) for g in tc.get("gold_chunks", [])]
        child_ranked = [normalize_key(r) for r in records]
        child_items.append({
            "user_input": tc.get("user_input", ""),
            "source_doc": tc.get("source_doc", ""),
            "question_type": tc.get("question_type", ""),
            "ranked": child_ranked,
            "gold": gold,
        })

        cands = _records_to_candidates(records, parent_map)
        if not cands:
            delivered = []
        else:
            delivered = _filter_and_dedup(cands, threshold, final_k)
            if expand:
                delivered = _expand_to_parent(delivered)

        delivered_keys = [normalize_key(c) for c in delivered]
        delivered_items.append({
            "user_input": tc.get("user_input", ""),
            "source_doc": tc.get("source_doc", ""),
            "question_type": tc.get("question_type", ""),
            "ranked": delivered_keys,
            "gold": gold,
        })

        if gold:
            child_hit = {g for g in gold if g in set(child_ranked)}
            delivered_hit = {g for g in gold if g in set(delivered_keys)}
            lost = sorted(child_hit - delivered_hit)
            if lost:
                drops.append({
                    "index": idx,
                    "question_type": tc.get("question_type", ""),
                    "user_input": tc.get("user_input", ""),
                    "n_gold": len(gold),
                    "child_ranks": {
                        str(normalize_key(g)): child_ranked.index(normalize_key(g)) + 1
                        for g in tc.get("gold_chunks", [])
                        if normalize_key(g) in set(child_ranked)
                    },
                    "delivered_n": len(delivered_keys),
                    "lost_gold": [
                        {"source": s, "page": p, "chunk_index": ci} for s, p, ci in lost
                    ],
                })

    child_eval = evaluate_questions(child_items, ks=ks)
    delivered_eval = evaluate_questions(delivered_items, ks=ks)
    pc_stats = parent_child_stats(parent_map, child_map) if child_map else {}

    return {
        "config": {
            "FINAL_CONTEXT_K": final_k,
            "RERANK_SCORE_THRESHOLD": threshold,
            "MAX_PER_SOURCE": settings.MAX_PER_SOURCE,
            "MIN_PER_SOURCE": settings.MIN_PER_SOURCE,
            "SCORE_GAP_THRESHOLD": settings.SCORE_GAP_THRESHOLD,
            "DYNAMIC_THRESHOLD_RATIO": settings.DYNAMIC_THRESHOLD_RATIO,
            "expand": expand,
            "MILVUS_COLLECTION": settings.MILVUS_COLLECTION,
            "CHUNKING_STRATEGY": settings.CHUNKING_STRATEGY,
            "CHUNK_SIZE": settings.CHUNK_SIZE,
        },
        "parent_child": pc_stats,
        "child": {
            "summary": child_eval["summary"],
            "by_question_type": child_eval["by_question_type"],
            "n_answerable": child_eval["n_answerable"],
            "n_unanswerable": child_eval["n_unanswerable"],
        },
        "delivered": {
            "summary": delivered_eval["summary"],
            "by_question_type": delivered_eval["by_question_type"],
            "n_answerable": delivered_eval["n_answerable"],
            "n_unanswerable": delivered_eval["n_unanswerable"],
        },
        "n_dropped_questions": len(drops),
        "dropped": drops,
    }


def write_delivery_report(result: dict, checkpoint_path, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "checkpoint_path": str(checkpoint_path),
            "note": (
                "Two-layer scoring: child=reranked ranking from checkpoint; "
                "delivered=after production _filter_and_dedup (+ optional parent expand). "
                "No re-retrieval."
            ),
            "config": result["config"],
            "parent_child": result["parent_child"],
        },
        "child": result["child"],
        "delivered": result["delivered"],
        "n_dropped_questions": result["n_dropped_questions"],
        "dropped": result["dropped"],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md_path = output_path.with_suffix(".md")
    cfg = result["config"]
    pc = result["parent_child"]
    lines = [
        "# Two-layer delivery scoring (child rank vs production filter)",
        "",
        f"Checkpoint: `{checkpoint_path}`",
        f"Collection: `{cfg.get('MILVUS_COLLECTION')}`, "
        f"CHUNK_SIZE={cfg.get('CHUNK_SIZE')}, strategy={cfg.get('CHUNKING_STRATEGY')}",
        f"Delivery: FINAL_CONTEXT_K={cfg.get('FINAL_CONTEXT_K')}, "
        f"threshold={cfg.get('RERANK_SCORE_THRESHOLD')}, "
        f"MAX_PER_SOURCE={cfg.get('MAX_PER_SOURCE')}, expand={cfg.get('expand')}",
        "",
        "## Parent vs child text",
        "",
    ]
    if pc:
        lines.extend([
            f"- chunks seen in checkpoint: {pc.get('n_chunks')}",
            f"- parent_text == child text: {pc.get('identical_parent_child')} "
            f"({pc.get('identical_ratio')})",
            f"- median/mean len(parent)/len(child): "
            f"{pc.get('median_len_ratio')} / {pc.get('mean_len_ratio')}",
            "",
        ])
        if pc.get("identical_ratio", 0) >= 0.99:
            lines.extend([
                "Parent expansion is a **no-op** at this CHUNK_SIZE — sections "
                "rarely split, so parent_text equals the child. Re-ranking on "
                "parent text cannot diverge. A c300 (or smaller) collection is "
                "needed before testing parent-rerank dilution.",
                "",
            ])

    def _row(label: str, summary: dict) -> str:
        return (
            f"| {label} | {summary.get('hit@5', 0):.2f} / "
            f"{summary.get('recall@5', 0):.2f} / {summary.get('ndcg@5', 0):.2f} | "
            f"{summary.get('hit@10', 0):.2f} / {summary.get('recall@10', 0):.2f} / "
            f"{summary.get('ndcg@10', 0):.2f} | "
            f"{summary.get('mrr', 0):.2f} | "
            f"{summary.get('failure_rate@20', 0):.2f} |"
        )

    lines.extend([
        "## Metrics",
        "",
        "| layer | hit@5 / recall@5 / nDCG@5 | hit@10 / recall@10 / nDCG@10 | MRR | failure@20 |",
        "|---|---|---|---|---|",
        _row("child (reranked)", result["child"]["summary"]),
        _row("delivered (filter/dedup)", result["delivered"]["summary"]),
        "",
        f"Questions that lose ≥1 gold between child rank and delivery: "
        f"**{result['n_dropped_questions']}**",
        "",
    ])
    if result["dropped"]:
        lines.extend([
            "## Dropped gold (silent production cuts)",
            "",
            "| # | type | lost chunks | delivered_n | question |",
            "|---|---|---|---|---|",
        ])
        for d in result["dropped"]:
            lost = ", ".join(
                f"{x['source']}:{x['chunk_index']}" for x in d["lost_gold"]
            )
            q = (d["user_input"] or "").replace("|", "/")[:48]
            lines.append(
                f"| {d['index']} | {d['question_type']} | {lost} | "
                f"{d['delivered_n']} | {q} |"
            )
        lines.append("")
    else:
        lines.extend([
            "No answerable question loses a gold chunk that was present in the "
            "reranked list — filter/dedup/cutoff did not create a silent gap "
            "on this checkpoint.",
            "",
        ])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def run_delivery_from_paths(
    checkpoint_path,
    cases: list[dict],
    output_path,
    *,
    expand: bool = True,
    load_parents: bool = True,
) -> dict:
    checkpoint = load_retrieval_checkpoint(checkpoint_path)
    parent_map: dict[tuple, str] = {}
    child_map: dict[tuple, str] = {}
    if load_parents:
        try:
            parent_map, child_map = load_chunk_text_maps()
        except Exception:
            parent_map, child_map = {}, {}
    result = simulate_delivery(
        checkpoint,
        cases,
        parent_map=parent_map,
        child_map=child_map,
        expand=expand,
    )
    write_delivery_report(result, checkpoint_path, output_path)
    return result
