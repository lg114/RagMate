"""确定性检索评测：分阶段跑检索、落盘排序、算 tier-1 指标，不经过 LLM。

四个阶段（P0-3 消融臂）：
  dense    —— 仅稠密向量检索；
  sparse   —— 仅稀疏向量检索；
  fused    —— 混合检索 + RRF 融合（即 `_do_search`），不重排；
  reranked —— fused 的候选经 reranker 打分排序，不做阈值过滤/去重/父展开。

检索指标只看“排序里有没有 gold chunk、在第几位”，因此 reranked 阶段
故意停在排序这一步：阈值、去重、父展开是生成侧的取舍，不应污染检索
基线。评测跑单查询，与生产路径一致；基线口径是单查询。

Chunk 身份用 (source, page, chunk_index) 三元组，与 gold 标注一致。
"""
import json
import logging
import math
import time
from pathlib import Path

from tqdm import tqdm

from backend.core.retriever import score_pairs
from backend.infrastructure.config import settings
from backend.infrastructure.encoding import encode_query
from backend.infrastructure.vector_store import init_vector_store

from .deterministic_metrics import evaluate_questions, metric_spread, normalize_key
from .manifest import build_manifest, diff_manifests

logger = logging.getLogger("ragmate")

STAGES = ("dense", "sparse", "fused", "reranked")

SCORE_KINDS = {
    "dense": "pgvector inner product on dense vector",
    "sparse": "pgvector inner product on sparsevec",
    "fused": "RRF fusion score (default k=60) of dense and sparse ranks",
    "reranked": "reranker sigmoid probability on child chunk text",
}

QUESTION_TYPES = ("factoid", "multi-hop", "paraphrase", "ambiguous", "unanswerable")

TEXT_PREVIEW_CHARS = 120


def _hit_entity(hit):
    if isinstance(hit, dict):
        return hit["entity"]
    return hit.entity


def _hit_distance(hit):
    if isinstance(hit, dict):
        return hit.get("distance", hit.get("score"))
    return getattr(hit, "distance", getattr(hit, "score", None))


def _extract_stage_records(results) -> list[dict]:
    """把一页检索结果转成阶段记录（含小片段原文，供重排用）。"""
    records = []
    for rank, hit in enumerate(results[0]):
        entity = _hit_entity(hit)
        meta = entity.get("metadata", {})
        text = entity["text"]
        records.append({
            "rank": rank + 1,
            "source": meta.get("source", ""),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
            "score": _hit_distance(hit),
            "text": text,
            "text_preview": text[:TEXT_PREVIEW_CHARS],
        })
    return records


def _search_dense(store, dense_vec, over_fetch):
    return [store.search_dense(dense_vec, over_fetch)]


def _search_sparse(store, sparse_vec, over_fetch):
    return [store.search_sparse(sparse_vec, over_fetch)]


def _search_fused(store, dense_vec, sparse_vec, over_fetch):
    """混合检索 + RRF：与 retriever._do_search 同一行为。"""
    return [store.search_hybrid(dense_vec, sparse_vec, over_fetch)]


def run_stages(client, question: str, stages, over_fetch: int, timings: dict | None = None) -> dict:
    """对一个问题跑出各阶段的排序。返回 {stage: [record, ...]}。

    编码只做一次，各阶段共用同一对向量；reranked 复用 fused 的候选。
    传 timings 字典时填入各环节耗时（秒，perf_counter）：encode（查询编码）、
    dense / sparse / fused（向量库搜索）、rerank（打分 + 重排）；只记实际跑
    的环节。计时不改变任何排序结果。
    """
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}, expected one of {STAGES}")
    t0 = time.perf_counter()
    dense_vec, sparse_vec = encode_query(question)
    if timings is not None:
        timings["encode"] = time.perf_counter() - t0
    out = {}
    fused_records = None
    if "dense" in stages:
        t0 = time.perf_counter()
        out["dense"] = _extract_stage_records(_search_dense(client, dense_vec, over_fetch))
        if timings is not None:
            timings["dense"] = time.perf_counter() - t0
    if "sparse" in stages:
        t0 = time.perf_counter()
        out["sparse"] = _extract_stage_records(_search_sparse(client, sparse_vec, over_fetch))
        if timings is not None:
            timings["sparse"] = time.perf_counter() - t0
    if "fused" in stages or "reranked" in stages:
        t0 = time.perf_counter()
        fused_records = _extract_stage_records(_search_fused(client, dense_vec, sparse_vec, over_fetch))
        if timings is not None:
            timings["fused"] = time.perf_counter() - t0
    if "fused" in stages:
        out["fused"] = fused_records
    if "reranked" in stages:
        t0 = time.perf_counter()
        reranked = [dict(r) for r in fused_records]
        scores = score_pairs(question, [r["text"] for r in reranked])
        for record, score in zip(reranked, scores):
            record["score"] = score
        reranked.sort(key=lambda r: r["score"], reverse=True)
        for rank, record in enumerate(reranked):
            record["rank"] = rank + 1
        out["reranked"] = reranked
        if timings is not None:
            timings["rerank"] = time.perf_counter() - t0
    return out


# ── Golden set schema（P0-1） ───────────────────────────────────────────────

def validate_gold_schema(cases: list[dict]) -> list[str]:
    """校验 golden set v1.1 的标注完整性，返回问题清单（空即通过）。

    每题必须有 user_input / reference / gold_chunks / expected_claims /
    question_type / source_doc；unanswerable 题的 gold_chunks 必须为空，
    其余题至少一个 gold chunk；gold chunk 须有 source 与 chunk_index。
    """
    errors = []
    seen_questions = set()
    for i, tc in enumerate(cases):
        where = f"case {i}"
        question = tc.get("user_input", "")
        if not question:
            errors.append(f"{where}: missing user_input")
            continue
        if question in seen_questions:
            errors.append(f"{where}: duplicate question: {question[:40]}")
        seen_questions.add(question)
        if not tc.get("reference"):
            errors.append(f"{where}: missing reference")
        if "gold_chunks" not in tc:
            errors.append(f"{where}: missing gold_chunks (annotate with gold chunk IDs first)")
            continue
        gold = tc["gold_chunks"]
        if not isinstance(gold, list):
            errors.append(f"{where}: gold_chunks must be a list")
            continue
        qtype = tc.get("question_type")
        if qtype not in QUESTION_TYPES:
            errors.append(f"{where}: question_type must be one of {QUESTION_TYPES}, got {qtype!r}")
        if qtype == "unanswerable":
            if gold:
                errors.append(f"{where}: unanswerable question must have empty gold_chunks")
        elif not gold:
            errors.append(f"{where}: answerable question must have at least one gold chunk")
        for g in gold:
            if not isinstance(g, dict) or not g.get("source") or g.get("chunk_index") is None:
                errors.append(f"{where}: gold chunk needs source and chunk_index, got {g!r}")
        if not tc.get("expected_claims"):
            errors.append(f"{where}: missing expected_claims")
        if qtype != "unanswerable" and not tc.get("source_doc"):
            errors.append(f"{where}: missing source_doc")
    return errors


def load_golden_testset(path) -> list[dict]:
    """读 golden set 并做 schema 校验；不通过则抛 ValueError 列出全部问题。"""
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Test set must be a non-empty JSON list: {path}")
    errors = validate_gold_schema(cases)
    if errors:
        raise ValueError("Golden set schema errors:\n" + "\n".join(f"  - {e}" for e in errors))
    return cases


def build_chunk_inventory(corpus_dir) -> dict:
    """用真实入库切分逻辑，从语料目录复现 chunk 清单。

    返回 {(source basename, chunk_index): chunk text}。只复现切分，不写
    向量库、不碰数据库；入库的状态上报在离线复现时是噪音，
    这里把它换成空操作——复现的是切分结果，不是入库流程。
    """
    from backend.application.ingest import pipeline as ingest_pipeline

    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise ValueError(f"Corpus directory not found: {corpus_dir}")
    files = sorted(f.name for f in corpus_dir.iterdir() if f.is_file())
    original_reporter = ingest_pipeline.set_ingest_status_sync
    ingest_pipeline.set_ingest_status_sync = lambda data: None
    try:
        chunks, _, _ = ingest_pipeline._load_and_split(str(corpus_dir), files)
    finally:
        ingest_pipeline.set_ingest_status_sync = original_reporter
    import os

    inventory = {}
    for chunk in chunks:
        key = (os.path.basename(chunk.metadata.get("source", "unknown")),
               chunk.metadata.get("chunk_index"))
        inventory[key] = chunk.page_content
    return inventory


def verify_gold_in_inventory(cases: list[dict], inventory: dict) -> list[str]:
    """gold 标注的每个三元组都必须在复现的 chunk 清单里。"""
    missing = []
    for i, tc in enumerate(cases):
        for g in tc.get("gold_chunks", []):
            key = (g.get("source"), g.get("chunk_index"))
            if key not in inventory:
                missing.append(
                    f"case {i} ({tc.get('user_input', '')[:40]}): "
                    f"gold chunk {key} not in chunk inventory "
                    f"(chunking settings or corpus changed?)"
                )
    return missing


def verify_gold_in_index(cases: list[dict]) -> list[str]:
    """gold 标注的每个三元组都必须在 pgvector 表里（--live 校验）。

    只查 (source, chunk_index) 是否存在，不取向量；库不可达时抛错——
    live 校验的意义正在于确认标注与线上索引一致，不可达不能算通过。
    """
    from backend.infrastructure.vector_store import get_vector_store

    store = get_vector_store()
    missing = []
    checked = set()
    for i, tc in enumerate(cases):
        for g in tc.get("gold_chunks", []):
            key = (g.get("source"), g.get("chunk_index"))
            if key in checked:
                continue
            checked.add(key)
            if not store.chunk_exists(g["source"], g["chunk_index"]):
                missing.append(
                    f"case {i} ({tc.get('user_input', '')[:40]}): "
                    f"gold chunk {key} not found in vector table "
                    f"{settings.VECTOR_TABLE}"
                )
    return missing


# ── 运行、落盘与报告 ────────────────────────────────────────────────────────

def _checkpoint_records(records: list[dict]) -> list[dict]:
    """落盘用的记录：去掉小片段原文，只留身份、名次、分数与预览。"""
    return [{
        "rank": r["rank"],
        "source": r["source"],
        "page": r["page"],
        "chunk_index": r["chunk_index"],
        "score": r["score"],
        "text_preview": r["text_preview"],
    } for r in records]


def _percentile(values: list[float], p: float) -> float:
    """最近秩法百分位：p=50 即中位，p=95 即 p95。不引入 numpy 依赖。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(p / 100.0 * len(ordered))))
    return ordered[rank - 1]


def _summarize_times(values: list[float]) -> dict:
    """一组耗时的汇总：个数、均值、p50、p95、总和（单位均为秒）。"""
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "total": 0.0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "total": sum(values),
    }


def summarize_timings(question_timings: list[dict]) -> dict:
    """把每题的计时汇总成报告用的 timings 块。

    per_question_total 是单题端到端耗时（run_stages 墙钟，不含 --delay-seconds
    的 sleep）；per_stage 是各环节耗时。只汇总第一遍（与指标取第一遍一致）。
    """
    totals = [q["total"] for q in question_timings]
    per_stage = {}
    for part in ("encode", "dense", "sparse", "fused", "rerank"):
        vals = [q["parts"][part] for q in question_timings if part in q["parts"]]
        if vals:
            per_stage[part] = _summarize_times(vals)
    return {
        "unit": "seconds",
        "per_question_total": _summarize_times(totals),
        "per_stage": per_stage,
    }


def run_retrieval_eval(cases: list[dict], stages, over_fetch: int, repeat: int,
                       ks, manifest: dict, delay: float = 0.0) -> dict:
    """跑 repeat 遍分阶段检索，返回 checkpoint + 每遍的指标汇总 + 计时。

    repeat 遍的排序应当完全相同（固定索引下的确定性检索）；spread 在
    报告里单独列出，非 0 即说明检索链路里有不确定性。
    计时只取第一遍（与指标取第一遍一致），不含 --delay-seconds 的 sleep。
    """
    client = init_vector_store()
    if client is None:
        raise RuntimeError(
            f"Vector table {settings.VECTOR_TABLE!r} does not exist; "
            "ingest the corpus before running retrieval evaluation."
        )
    checkpoint = None
    evaluations = []
    question_timings = []
    total = repeat * len(cases)
    pbar = tqdm(total=total, desc="Retrieval eval", unit="q")
    try:
        for rep in range(repeat):
            items = {}
            for stage in stages:
                items[stage] = []
            stage_lists = {}
            for idx, tc in enumerate(cases):
                pbar.set_postfix(rep=f"{rep + 1}/{repeat}", refresh=False)
                q_timings = {} if rep == 0 else None
                t0 = time.perf_counter()
                stage_records = run_stages(client, tc["user_input"], stages, over_fetch,
                                           timings=q_timings)
                if rep == 0:
                    question_timings.append({
                        "total": time.perf_counter() - t0,
                        "parts": q_timings,
                    })
                for stage in stages:
                    items[stage].append({
                        "user_input": tc["user_input"],
                        "source_doc": tc.get("source_doc", ""),
                        "question_type": tc.get("question_type", ""),
                        "ranked": [normalize_key(r) for r in stage_records[stage]],
                        "gold": [normalize_key(g) for g in tc.get("gold_chunks", [])],
                    })
                if rep == 0:
                    stage_lists[str(idx)] = {
                        stage: _checkpoint_records(stage_records[stage])
                        for stage in stages
                    }
                pbar.update(1)
                if delay > 0 and not (rep == repeat - 1 and idx == len(cases) - 1):
                    time.sleep(delay)
            if rep == 0:
                checkpoint = {
                    "manifest": manifest,
                    "score_kinds": {stage: SCORE_KINDS[stage] for stage in stages},
                    "cases": [
                        {"index": i, "user_input": tc["user_input"]}
                        for i, tc in enumerate(cases)
                    ],
                    "stages": stage_lists,
                }
            evaluations.append({
                stage: evaluate_questions(items[stage], ks) for stage in stages
            })
    finally:
        pbar.close()
    return {"checkpoint": checkpoint, "evaluations": evaluations,
            "timings": summarize_timings(question_timings)}


def build_retrieval_report(manifest: dict, evaluations: list[dict],
                           checkpoint_path, stages, timings: dict | None = None) -> dict:
    """把多遍评测汇总成报告：每阶段的指标、重复跑的极差、确定性结论。

    timings 不进 manifest（计时不影响分数），只作为报告顶层 timings 块。
    """
    per_stage_summaries = {}
    for stage in stages:
        per_stage_summaries[stage] = [ev[stage]["summary"] for ev in evaluations]
    spread = {stage: metric_spread(per_stage_summaries[stage]) for stage in stages}
    deterministic = all(
        all(v == 0.0 for v in stage_spread.values())
        for stage_spread in spread.values()
    )
    first = evaluations[0]
    report = {
        "meta": {
            **manifest,
            "checkpoint_path": str(checkpoint_path),
            "repeat": len(evaluations),
            "deterministic_across_repeats": deterministic,
            "repeat_spread": spread,
        },
        "stages": {
            stage: {
                "summary": first[stage]["summary"],
                "n_questions": first[stage]["n_questions"],
                "n_answerable": first[stage]["n_answerable"],
                "n_unanswerable": first[stage]["n_unanswerable"],
                "by_source_doc": first[stage]["by_source_doc"],
                "by_question_type": first[stage]["by_question_type"],
                "per_question": first[stage]["per_question"],
            }
            for stage in stages
        },
    }
    if timings is not None:
        report["timings"] = timings
    return report


def write_retrieval_checkpoint(checkpoint: dict, report_path) -> Path:
    """排序落盘：报告同名、后缀换成 .retrieval.json（与 .raw.json 同约定）。"""
    checkpoint_path = Path(str(report_path)).with_suffix(".retrieval.json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    return checkpoint_path


def write_retrieval_markdown(report: dict, report_path) -> Path:
    """人类可读的基线表：每阶段指标 + 失败清单（recall@20 未满的问题）。"""
    md_path = Path(str(report_path)).with_suffix(".md")
    ks = report["meta"]["harness"]["ks"]
    lines = ["# Retrieval baseline (deterministic, no LLM judge)", ""]
    lines.append(
        f"Test set: `{report['meta']['testset']['path']}` "
        f"({report['meta']['testset']['n_questions']} questions, "
        f"sha256 `{report['meta']['testset']['sha256'][:12]}`), "
        f"over_fetch={report['meta']['harness']['over_fetch']}, "
        f"repeat={report['meta']['repeat']}, "
        f"deterministic across repeats: {report['meta']['deterministic_across_repeats']}."
    )
    lines.append("")
    lines.append("Metrics are computed from gold chunk IDs only: hit/recall/MRR/nDCG "
                 "need no LLM. Unanswerable questions are excluded from the means "
                 "and counted separately.")
    lines.append("")
    header = "| stage | " + " | ".join(f"hit@{k} / recall@{k} / nDCG@{k}" for k in ks) + " | MRR | failure_rate@20 |"
    lines.append(header)
    lines.append("|---|---|---|---|---|")
    for stage, data in report["stages"].items():
        summary = data["summary"]
        cells = []
        for k in ks:
            cells.append(
                f"{summary.get(f'hit@{k}', 0):.2f} / {summary.get(f'recall@{k}', 0):.2f} / "
                f"{summary.get(f'ndcg@{k}', 0):.2f}"
            )
        lines.append(
            f"| {stage} | " + " | ".join(cells) +
            f" | {summary.get('mrr', 0):.2f} | {summary.get('failure_rate@20', 0):.2f} |"
        )
    lines.append("")
    lines.append("## Failures (recall@20 < 1)")
    lines.append("")
    for stage, data in report["stages"].items():
        failures = [
            p for p in data["per_question"]
            if not p["unanswerable"] and p["scores"].get("recall@20", 0) < 1.0
        ]
        lines.append(f"### {stage} ({len(failures)} failing)")
        for p in failures:
            lines.append(
                f"- Q{p['index'] + 1} recall@20={p['scores'].get('recall@20', 0):.2f}: "
                f"{p['user_input']}"
            )
        lines.append("")
    timings = report.get("timings")
    if timings:
        lines.append("## Timings (seconds, first repeat only, excludes --delay-seconds sleeps)")
        lines.append("")
        total = timings["per_question_total"]
        lines.append(
            f"Per-question end-to-end: count={total['count']}, mean={total['mean']:.3f}, "
            f"p50={total['p50']:.3f}, p95={total['p95']:.3f}, total={total['total']:.3f}."
        )
        lines.append("")
        lines.append("| part | count | mean | p50 | p95 | total |")
        lines.append("|---|---|---|---|---|---|")
        for part, summary in timings["per_stage"].items():
            lines.append(
                f"| {part} | {summary['count']} | {summary['mean']:.3f} | "
                f"{summary['p50']:.3f} | {summary['p95']:.3f} | {summary['total']:.3f} |"
            )
        lines.append("")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def print_retrieval_report(report: dict):
    """控制台打印：每阶段一行指标 + 确定性结论。"""
    print("\n=== Deterministic Retrieval Report (no LLM judge) ===")
    meta = report["meta"]
    print(f"Test set: {meta['testset']['path']} ({meta['testset']['n_questions']} questions)")
    print(f"Repeat: {meta['repeat']}, deterministic across repeats: "
          f"{meta['deterministic_across_repeats']}")
    if not meta["deterministic_across_repeats"]:
        print(f"Repeat spread: {json.dumps(meta['repeat_spread'], ensure_ascii=False)}")
    print()
    ks = meta["harness"]["ks"]
    for stage, data in report["stages"].items():
        summary = data["summary"]
        parts = [f"hit@{k}={summary.get(f'hit@{k}', 0):.2f} "
                 f"recall@{k}={summary.get(f'recall@{k}', 0):.2f} "
                 f"nDCG@{k}={summary.get(f'ndcg@{k}', 0):.2f}" for k in ks]
        print(f"  {stage:<10} " + "  ".join(parts) +
              f"  MRR={summary.get('mrr', 0):.2f} "
              f"fail@20={summary.get('failure_rate@20', 0):.2f} "
              f"(unanswerable excluded: {data['n_unanswerable']})")
    timings = report.get("timings")
    if timings:
        total = timings["per_question_total"]
        print(f"\nTimings (seconds, first repeat only, excludes --delay-seconds sleeps):")
        print(f"  per-question end-to-end: mean={total['mean']:.3f} "
              f"p50={total['p50']:.3f} p95={total['p95']:.3f} total={total['total']:.3f}")
        for part, summary in timings["per_stage"].items():
            print(f"  {part:<10} mean={summary['mean']:.3f} "
                  f"p50={summary['p50']:.3f} p95={summary['p95']:.3f} "
                  f"total={summary['total']:.3f}")
    print(f"\nFull report saved to {meta.get('report_path', 'N/A')}")


def compare_with_manifest(manifest: dict, previous_report_path) -> dict:
    """把本次 manifest 与上一份报告的 manifest 对比，返回变化的配置项。"""
    with open(previous_report_path, "r", encoding="utf-8") as f:
        previous = json.load(f)
    previous_manifest = previous.get("meta", {})
    return diff_manifests(previous_manifest, manifest)


# ── 最终 k 截断模拟（只读 checkpoint，不重跑检索） ───────────────────────────

def load_retrieval_checkpoint(path) -> dict:
    """读 `.retrieval.json` 排序落盘；结构不对则抛 ValueError。"""
    with open(path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    if not isinstance(checkpoint, dict) or "stages" not in checkpoint:
        raise ValueError(f"Not a retrieval checkpoint (missing stages): {path}")
    return checkpoint


def simulate_final_ks(checkpoint: dict, cases: list[dict], final_ks,
                      stages=None) -> dict:
    """把每阶段排序截到 final_k，当作生产最终上下文窗口重算指标。

    不重跑编码/检索/重排：直接用 checkpoint 里已落盘的 rank。每个 final_k
    只在该截断长度上算 hit/recall/nDCG/MRR（即 hit@{k} 的 k = final_k）。
    incomplete 列表 = answerable 且 recall@{k} < 1 的题（多证据被截断的样例）。
    """
    stages = list(stages) if stages is not None else list(
        checkpoint.get("manifest", {}).get("harness", {}).get("stages", STAGES)
    )
    stage_lists = checkpoint["stages"]
    if len(cases) != len(checkpoint.get("cases", cases)):
        # cases 来自 testset；checkpoint.cases 可能只有 user_input。长度必须一致。
        n_cp = len(checkpoint.get("cases", []))
        if n_cp and n_cp != len(cases):
            raise ValueError(
                f"Test set has {len(cases)} cases but checkpoint has {n_cp}; "
                "use the same testset that produced the checkpoint."
            )

    by_k = {}
    for final_k in final_ks:
        if final_k <= 0:
            raise ValueError(f"final_k must be positive, got {final_k}")
        stage_results = {}
        for stage in stages:
            items = []
            for idx, tc in enumerate(cases):
                records = stage_lists[str(idx)][stage][:final_k]
                items.append({
                    "user_input": tc.get("user_input", ""),
                    "source_doc": tc.get("source_doc", ""),
                    "question_type": tc.get("question_type", ""),
                    "ranked": [normalize_key(r) for r in records],
                    "gold": [normalize_key(g) for g in tc.get("gold_chunks", [])],
                })
            evaluation = evaluate_questions(items, ks=(final_k,))
            incomplete = [
                {
                    "index": p["index"],
                    "question_type": p["question_type"],
                    "user_input": p["user_input"],
                    "recall": p["scores"].get(f"recall@{final_k}"),
                    "hit": p["scores"].get(f"hit@{final_k}"),
                    "mrr": p["scores"].get("mrr"),
                }
                for p in evaluation["per_question"]
                if not p["unanswerable"]
                and p["scores"].get(f"recall@{final_k}", 1.0) < 1.0
            ]
            summary = evaluation["summary"]
            stage_results[stage] = {
                "summary": {
                    "hit": summary.get(f"hit@{final_k}"),
                    "recall": summary.get(f"recall@{final_k}"),
                    "ndcg": summary.get(f"ndcg@{final_k}"),
                    "mrr": summary.get("mrr"),
                    "failure_rate": round(1.0 - summary[f"recall@{final_k}"], 4)
                    if summary.get(f"recall@{final_k}") is not None else None,
                },
                "by_question_type": {
                    qt: {
                        "hit": metrics.get(f"hit@{final_k}"),
                        "recall": metrics.get(f"recall@{final_k}"),
                        "ndcg": metrics.get(f"ndcg@{final_k}"),
                        "mrr": metrics.get("mrr"),
                    }
                    for qt, metrics in evaluation["by_question_type"].items()
                },
                "n_incomplete": len(incomplete),
                "incomplete": incomplete,
                "n_answerable": evaluation["n_answerable"],
                "n_unanswerable": evaluation["n_unanswerable"],
            }
        by_k[str(final_k)] = stage_results
    return {
        "final_ks": list(final_ks),
        "stages": stages,
        "n_questions": len(cases),
        "by_final_k": by_k,
    }


def write_cutoff_report(result: dict, checkpoint_path, output_path) -> Path:
    """落盘 JSON + 同名 Markdown 表。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "checkpoint_path": str(checkpoint_path),
            "final_ks": result["final_ks"],
            "stages": result["stages"],
            "n_questions": result["n_questions"],
            "note": (
                "Simulated FINAL_CONTEXT_K cutoffs from a retrieval checkpoint; "
                "no re-retrieval. Metrics are hit/recall/nDCG/MRR on the truncated list."
            ),
        },
        "by_final_k": result["by_final_k"],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md_path = output_path.with_suffix(".md")
    lines = [
        "# Final-k cutoff simulation (from retrieval checkpoint)",
        "",
        f"Checkpoint: `{checkpoint_path}`",
        f"Questions: {result['n_questions']}, final_ks={result['final_ks']}",
        "",
        "Each row truncates that stage's ranked list to final_k (as if "
        "`FINAL_CONTEXT_K=final_k`), then scores hit/recall/nDCG/MRR on the "
        "truncated list. incomplete = answerable questions with recall < 1.",
        "",
    ]
    for final_k in result["final_ks"]:
        lines.append(f"## final_k = {final_k}")
        lines.append("")
        lines.append("| stage | hit | recall | nDCG | MRR | incomplete |")
        lines.append("|---|---|---|---|---|---|")
        for stage in result["stages"]:
            block = result["by_final_k"][str(final_k)][stage]
            sm = block["summary"]
            lines.append(
                f"| {stage} | {sm['hit']:.4f} | {sm['recall']:.4f} | "
                f"{sm['ndcg']:.4f} | {sm['mrr']:.4f} | {block['n_incomplete']} |"
            )
        lines.append("")
        # Call out incomplete on reranked (production-like) first
        focus = "reranked" if "reranked" in result["stages"] else result["stages"][-1]
        incomplete = result["by_final_k"][str(final_k)][focus]["incomplete"]
        lines.append(f"### Incomplete under `{focus}` @ {final_k} ({len(incomplete)})")
        if not incomplete:
            lines.append("")
            lines.append("_none_")
        else:
            for item in incomplete:
                lines.append(
                    f"- [{item['index']}] ({item['question_type']}) "
                    f"recall={item['recall']:.2f}: {item['user_input']}"
                )
        lines.append("")
        # multi-hop recall row
        lines.append(f"### multi-hop recall by stage @ {final_k}")
        lines.append("")
        lines.append("| stage | multi-hop recall | multi-hop nDCG |")
        lines.append("|---|---|---|")
        for stage in result["stages"]:
            mh = result["by_final_k"][str(final_k)][stage]["by_question_type"].get(
                "multi-hop"
            )
            if not mh:
                lines.append(f"| {stage} | — | — |")
            else:
                lines.append(
                    f"| {stage} | {mh['recall']:.4f} | {mh['ndcg']:.4f} |"
                )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def print_cutoff_report(result: dict):
    """控制台打印各 final_k × stage 的截断指标。"""
    print("\n=== Final-k cutoff simulation (no re-retrieval) ===")
    print(f"Questions: {result['n_questions']}, final_ks={result['final_ks']}")
    for final_k in result["final_ks"]:
        print(f"\nfinal_k={final_k}")
        for stage in result["stages"]:
            block = result["by_final_k"][str(final_k)][stage]
            sm = block["summary"]
            print(
                f"  {stage:<10} hit={sm['hit']:.4f} recall={sm['recall']:.4f} "
                f"nDCG={sm['ndcg']:.4f} MRR={sm['mrr']:.4f} "
                f"incomplete={block['n_incomplete']}"
            )
            mh = block["by_question_type"].get("multi-hop")
            if mh:
                print(
                    f"             multi-hop recall={mh['recall']:.4f} "
                    f"nDCG={mh['ndcg']:.4f}"
                )
