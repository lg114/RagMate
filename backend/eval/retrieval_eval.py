"""确定性检索评测：分阶段跑检索、落盘排序、算 tier-1 指标，不经过 LLM。

四个阶段（P0-3 消融臂）：
  dense    —— 仅稠密向量检索；
  sparse   —— 仅稀疏向量检索；
  fused    —— 混合检索 + RRF 融合（即 `_do_search` 的混合分支），不重排；
  reranked —— fused 的候选经 reranker 打分排序，不做阈值过滤/去重/父展开。

检索指标只看“排序里有没有 gold chunk、在第几位”，因此 reranked 阶段
故意停在排序这一步：阈值、去重、父展开是生成侧的取舍，不应污染检索
基线。评测默认单查询（与 MULTI_QUERY_ENABLED 关闭时的生产路径一致）；
若该开关打开，命令会先告警——基线口径是单查询。

Chunk 身份用 (source, page, chunk_index) 三元组，与 gold 标注一致。
"""
import json
import logging
from pathlib import Path

from pymilvus import AnnSearchRequest, RRFRanker

from backend.core.retriever import score_pairs
from backend.infrastructure.config import settings
from backend.infrastructure.encoding import encode_query
from backend.infrastructure.milvus import init_milvus

from .deterministic_metrics import evaluate_questions, metric_spread, normalize_key
from .manifest import build_manifest, diff_manifests

logger = logging.getLogger("ragmate")

STAGES = ("dense", "sparse", "fused", "reranked")

SCORE_KINDS = {
    "dense": "milvus IP distance on dense field",
    "sparse": "milvus IP distance on sparse field",
    "fused": "RRF fusion score from hybrid_search",
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
    """把 Milvus 返回的一页结果转成阶段记录（含小片段原文，供重排用）。"""
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


def _search_dense(client, dense_vec, over_fetch):
    # NOTE: MilvusClient.search takes the params as `search_params`, not
    # `param` — it forwards `param=search_params` to the gRPC handler
    # itself, so a `param=` kwarg here arrives twice and raises
    # "got multiple values for keyword argument 'param'".
    return client.search(
        collection_name=settings.MILVUS_COLLECTION,
        data=[dense_vec],
        anns_field="dense",
        search_params={"metric_type": "IP", "params": {}},
        limit=over_fetch,
        output_fields=["text", "parent_text", "metadata"],
    )


def _search_sparse(client, sparse_vec, over_fetch):
    return client.search(
        collection_name=settings.MILVUS_COLLECTION,
        data=[sparse_vec],
        anns_field="sparse",
        search_params={"metric_type": "IP", "params": {"drop_ratio_search": settings.DROP_RATIO_SEARCH}},
        limit=over_fetch,
        output_fields=["text", "parent_text", "metadata"],
    )


def _search_fused(client, dense_vec, sparse_vec, over_fetch):
    """混合检索 + RRF：与 retriever._do_search 的混合分支同一行为。

    HYBRID_SEARCH_ENABLED 关闭时退化为 dense 检索（与生产路径一致，
    manifest 里记了该开关，fused==dense 时一看便知）。
    """
    if not settings.HYBRID_SEARCH_ENABLED:
        return _search_dense(client, dense_vec, over_fetch)
    dense_req = AnnSearchRequest(
        data=[dense_vec],
        anns_field="dense",
        param={"metric_type": "IP", "params": {}},
        limit=over_fetch,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vec],
        anns_field="sparse",
        param={"metric_type": "IP", "params": {"drop_ratio_search": settings.DROP_RATIO_SEARCH}},
        limit=over_fetch,
    )
    return client.hybrid_search(
        collection_name=settings.MILVUS_COLLECTION,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(),
        limit=over_fetch,
        output_fields=["text", "parent_text", "metadata"],
    )


def run_stages(client, question: str, stages, over_fetch: int) -> dict:
    """对一个问题跑出各阶段的排序。返回 {stage: [record, ...]}。

    编码只做一次，各阶段共用同一对向量；reranked 复用 fused 的候选。
    """
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}, expected one of {STAGES}")
    dense_vec, sparse_vec = encode_query(question)
    out = {}
    fused_records = None
    if "dense" in stages:
        out["dense"] = _extract_stage_records(_search_dense(client, dense_vec, over_fetch))
    if "sparse" in stages:
        out["sparse"] = _extract_stage_records(_search_sparse(client, sparse_vec, over_fetch))
    if "fused" in stages or "reranked" in stages:
        fused_records = _extract_stage_records(_search_fused(client, dense_vec, sparse_vec, over_fetch))
    if "fused" in stages:
        out["fused"] = fused_records
    if "reranked" in stages:
        reranked = [dict(r) for r in fused_records]
        scores = score_pairs(question, [r["text"] for r in reranked])
        for record, score in zip(reranked, scores):
            record["score"] = score
        reranked.sort(key=lambda r: r["score"], reverse=True)
        for rank, record in enumerate(reranked):
            record["rank"] = rank + 1
        out["reranked"] = reranked
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
    Milvus、不碰数据库；入库的状态上报（Redis）在离线复现时是噪音，
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
    """gold 标注的每个三元组都必须在 live Milvus 索引里（--live 校验）。

    只查 metadata 是否存在，不取向量；索引不可达时抛错——live 校验的
    意义正在于确认标注与线上索引一致，不可达不能算通过。
    """
    from backend.infrastructure.milvus import build_source_filter, get_milvus_client

    client = get_milvus_client()
    missing = []
    checked = set()
    for i, tc in enumerate(cases):
        for g in tc.get("gold_chunks", []):
            key = (g.get("source"), g.get("chunk_index"))
            if key in checked:
                continue
            checked.add(key)
            expr = (
                f"{build_source_filter(g['source'])}"
                f" and metadata[\"chunk_index\"] == {g['chunk_index']}"
            )
            hits = client.query(
                collection_name=settings.MILVUS_COLLECTION,
                filter=expr,
                output_fields=["metadata"],
                limit=1,
            )
            if not hits:
                missing.append(
                    f"case {i} ({tc.get('user_input', '')[:40]}): "
                    f"gold chunk {key} not found in Milvus collection "
                    f"{settings.MILVUS_COLLECTION}"
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


def run_retrieval_eval(cases: list[dict], stages, over_fetch: int, repeat: int,
                       ks, manifest: dict) -> dict:
    """跑 repeat 遍分阶段检索，返回 checkpoint + 每遍的指标汇总。

    repeat 遍的排序应当完全相同（固定索引下的确定性检索）；spread 在
    报告里单独列出，非 0 即说明检索链路里有不确定性。
    """
    client = init_milvus()
    if client is None:
        raise RuntimeError(
            f"Milvus collection {settings.MILVUS_COLLECTION!r} does not exist; "
            "ingest the corpus before running retrieval evaluation."
        )
    if settings.MULTI_QUERY_ENABLED:
        logger.warning(
            "MULTI_QUERY_ENABLED is on: the harness still runs single-query "
            "retrieval, so this baseline is the single-query baseline."
        )
    checkpoint = None
    evaluations = []
    for rep in range(repeat):
        items = {}
        for stage in stages:
            items[stage] = []
        stage_lists = {}
        for idx, tc in enumerate(cases):
            stage_records = run_stages(client, tc["user_input"], stages, over_fetch)
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
    return {"checkpoint": checkpoint, "evaluations": evaluations}


def build_retrieval_report(manifest: dict, evaluations: list[dict],
                           checkpoint_path, stages) -> dict:
    """把多遍评测汇总成报告：每阶段的指标、重复跑的极差、确定性结论。"""
    per_stage_summaries = {}
    for stage in stages:
        per_stage_summaries[stage] = [ev[stage]["summary"] for ev in evaluations]
    spread = {stage: metric_spread(per_stage_summaries[stage]) for stage in stages}
    deterministic = all(
        all(v == 0.0 for v in stage_spread.values())
        for stage_spread in spread.values()
    )
    first = evaluations[0]
    return {
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
    print(f"\nFull report saved to {meta.get('report_path', 'N/A')}")


def compare_with_manifest(manifest: dict, previous_report_path) -> dict:
    """把本次 manifest 与上一份报告的 manifest 对比，返回变化的配置项。"""
    with open(previous_report_path, "r", encoding="utf-8") as f:
        previous = json.load(f)
    previous_manifest = previous.get("meta", {})
    return diff_manifests(previous_manifest, manifest)
