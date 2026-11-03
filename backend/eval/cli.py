"""评估 CLI 入口：RAGAS 打分与确定性检索评测。"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .utils import DEFAULT_TESTSETS_DIR, DEFAULT_REPORTS_DIR, resolve_path
from .document_loader import load_langchain_docs
from .testset_gen import generate_testset
from .retrieval import retrieve_contexts
from .response_gen import generate_response
from .metrics import compute_metrics, probe_judge_llm
from .report import build_report, cases_from_report, print_report, write_raw_checkpoint
from .manifest import build_manifest
from .retrieval_eval import (
    STAGES,
    build_chunk_inventory,
    build_retrieval_report,
    compare_with_manifest,
    load_golden_testset,
    print_retrieval_report,
    run_retrieval_eval,
    validate_gold_schema,
    verify_gold_in_index,
    verify_gold_in_inventory,
    write_retrieval_checkpoint,
    write_retrieval_markdown,
)

logger = logging.getLogger("ragmate")


def cmd_generate(args):
    """Generate a test set from documents."""
    from backend.infrastructure.config import settings

    docs_dir = settings.DOCUMENTS_DIR
    if not os.path.exists(docs_dir):
        print(f"Error: Documents directory not found: {docs_dir}")
        sys.exit(1)

    docs = load_langchain_docs(docs_dir, max_docs=args.max_docs)
    if not docs:
        print("Error: No documents loaded")
        sys.exit(1)

    print(f"Loaded {len(docs)} pages from {docs_dir}")
    print(f"Generating {args.size} test cases (seed={args.seed})...")

    query_distribution = None
    if args.distributions:
        query_distribution = json.loads(args.distributions)

    test_cases = generate_testset(
        documents=docs,
        size=args.size,
        seed=args.seed,
        query_distribution=query_distribution,
    )

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(test_cases)} test cases → {output_path}")
    print("Review and edit the test set before running evaluate.")


def cmd_evaluate(args):
    """Evaluate RAG pipeline against a test set."""
    from backend.infrastructure.config import settings

    testset_path = resolve_path(args.testset)
    if not testset_path.exists():
        print(f"Error: Test set not found: {testset_path}")
        sys.exit(1)

    with open(testset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases from {testset_path}")

    top_k = args.top_k or settings.FINAL_CONTEXT_K
    judge_model = args.judge_model or settings.LLM_MODEL
    mode = args.mode

    print(f"Config: mode={mode}, top_k={top_k}, judge_model={judge_model}")

    print("\nChecking judge model responds with parseable JSON...")
    try:
        probe_judge_llm(judge_model)
    except RuntimeError as e:
        print(f"Error: judge check failed, not running the pipeline: {e}")
        sys.exit(1)
    print("Judge check passed.")

    print("\nRunning RAG pipeline for each test case...")
    errors = 0
    for tc in tqdm(test_cases, desc="Evaluating"):
        question = tc["user_input"]
        try:
            contexts = retrieve_contexts(question, top_k=top_k)
            tc["retrieved_contexts"] = contexts

            response = generate_response(question, contexts)
            tc["response"] = response
        except Exception as e:
            logger.error(f"Failed for question '{question[:50]}': {e}")
            tc["response"] = ""
            tc["retrieved_contexts"] = []
            tc["error"] = str(e)
            errors += 1

    if errors:
        print(f"\nWarning: {errors}/{len(test_cases)} cases failed")

    valid_cases = [tc for tc in test_cases if tc.get("response") and tc.get("retrieved_contexts")]
    if not valid_cases:
        print("Error: No valid test cases to evaluate")
        sys.exit(1)

    print(f"\nComputing RAGAS metrics on {len(valid_cases)} valid cases...")

    report_path = str(resolve_path(args.report) if args.report else DEFAULT_REPORTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    meta = {
        "timestamp": datetime.now().isoformat(),
        "testset": str(testset_path),
        "test_cases": len(test_cases),
        "valid_cases": len(valid_cases),
        "failed_cases": errors,
        "mode": mode,
        "top_k": top_k,
        "judge_model": judge_model,
        "retriever_config": {
            "hybrid_search": settings.HYBRID_SEARCH_ENABLED,
            "reranker": settings.RERANKER_MODEL,
            "score_threshold": settings.RERANK_SCORE_THRESHOLD,
        },
        "report_path": report_path,
    }

    # Checkpoint the pipeline work before scoring: if scoring fails, only
    # scoring has to be redone (see `rescore`).
    raw_path = write_raw_checkpoint(report_path, meta, valid_cases)
    print(f"Raw pipeline results checkpointed to {raw_path}")

    try:
        ragas_result = compute_metrics(valid_cases, judge_model=judge_model)
    except RuntimeError as e:
        print(f"\nError: scoring failed: {e}")
        print(f"Pipeline results are preserved in {raw_path}. Fix the judge, then re-score only with:")
        print(f"  ragmate-eval rescore --report {raw_path}")
        sys.exit(1)

    report = build_report(meta, valid_cases, ragas_result)

    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(report)

    if args.threshold is not None:
        overall = report["summary"].get("overall") or 0
        if overall < args.threshold:
            print(f"\nFAILED: Overall score {overall:.2f} < threshold {args.threshold}")
            sys.exit(1)
        else:
            print(f"\nPASSED: Overall score {overall:.2f} >= threshold {args.threshold}")


def cmd_rescore(args):
    """Recompute RAGAS metrics from a saved report (or raw checkpoint).

    No retrieval or generation is re-run: the saved responses and contexts
    are scored again. Use this after a scoring failure, or to re-score an
    old run with a different judge model.
    """
    from backend.infrastructure.config import settings

    report_path = resolve_path(args.report)
    if not report_path.exists():
        print(f"Error: Report not found: {report_path}")
        sys.exit(1)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    cases = cases_from_report(report)
    if not cases:
        print(f"Error: No scored cases with responses and contexts in {report_path}")
        sys.exit(1)

    judge_model = args.judge_model or report.get("meta", {}).get("judge_model") or settings.LLM_MODEL
    print(f"Re-scoring {len(cases)} cases from {report_path} with judge_model={judge_model}")

    print("\nChecking judge model responds with parseable JSON...")
    try:
        probe_judge_llm(judge_model)
    except RuntimeError as e:
        print(f"Error: judge check failed: {e}")
        sys.exit(1)
    print("Judge check passed.")

    try:
        ragas_result = compute_metrics(cases, judge_model=judge_model)
    except RuntimeError as e:
        print(f"\nError: scoring failed: {e}")
        sys.exit(1)

    output_path = resolve_path(args.output) if args.output else report_path
    meta = dict(report.get("meta", {}))
    meta.update({
        "timestamp": datetime.now().isoformat(),
        "judge_model": judge_model,
        "rescored": True,
        "report_path": str(output_path),
    })

    new_report = build_report(meta, cases, ragas_result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_report, f, ensure_ascii=False, indent=2)

    print_report(new_report)


def cmd_retrieval(args):
    """Deterministic retrieval evaluation: stage-wise ranked lists + tier-1 metrics.

    No LLM calls. The golden set must carry gold chunk IDs (see `check-gold`).
    Writes three files next to --output: the metrics JSON report, a Markdown
    baseline table, and a `<report>.retrieval.json` checkpoint with the raw
    ranked lists (first repeat only) for recomputation without re-running
    retrieval.
    """
    testset_path = resolve_path(args.testset)
    if not testset_path.exists():
        print(f"Error: Test set not found: {testset_path}")
        sys.exit(1)

    try:
        cases = load_golden_testset(testset_path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if not stages or unknown:
        print(f"Error: --stages must be a subset of {STAGES}, got {args.stages!r}")
        sys.exit(1)
    ks = tuple(sorted({k for k in args.ks if k > 0}))
    if not ks:
        print("Error: --ks must list at least one positive k")
        sys.exit(1)

    report_path = str(resolve_path(args.report) if args.report else
                      DEFAULT_REPORTS_DIR / "baseline" / "retrieval-baseline-50q.json")

    manifest = build_manifest(
        testset_path=testset_path, stages=stages, over_fetch=args.over_fetch,
        ks=list(ks), repeat=args.repeat, n_questions=len(cases),
    )
    manifest["report_path"] = report_path

    if args.compare_manifest:
        previous_path = resolve_path(args.compare_manifest)
        if not previous_path.exists():
            print(f"Error: Previous report not found: {previous_path}")
            sys.exit(1)
        changes = compare_with_manifest(manifest, previous_path)
        if changes:
            print("Config changes vs previous run:")
            for key, change in changes.items():
                print(f"  {key}: {change['old']!r} → {change['new']!r}")
        else:
            print("Config identical to previous run (same manifest).")

    print(f"Loaded {len(cases)} golden cases from {testset_path}")
    print(f"Stages: {', '.join(stages)}, over_fetch={args.over_fetch}, "
          f"ks={list(ks)}, repeat={args.repeat}")

    try:
        result = run_retrieval_eval(cases, stages, args.over_fetch, args.repeat, ks, manifest)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    checkpoint_path = write_retrieval_checkpoint(result["checkpoint"], report_path)
    print(f"Raw ranked lists checkpointed to {checkpoint_path}")

    report = build_retrieval_report(manifest, result["evaluations"], checkpoint_path, stages)
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_path = write_retrieval_markdown(report, report_path)
    print(f"Markdown baseline table saved to {md_path}")

    print_retrieval_report(report)


def cmd_check_gold(args):
    """Validate golden set annotations: schema, chunk inventory, live index.

    Schema errors fail fast (annotation incomplete). The inventory check
    reproduces chunking from the corpus directory with the real pipeline
    code and requires every gold triple to exist there. --live additionally
    requires every gold triple to exist in the Milvus collection.
    """
    testset_path = resolve_path(args.testset)
    if not testset_path.exists():
        print(f"Error: Test set not found: {testset_path}")
        sys.exit(1)

    with open(testset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        print("Error: Test set must be a JSON list of cases")
        sys.exit(1)

    errors = validate_gold_schema(cases)
    if errors:
        print(f"Golden set schema: {len(errors)} problem(s):")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print(f"Schema OK: {len(cases)} cases, "
          f"{sum(1 for c in cases if c['gold_chunks'])} answerable, "
          f"{sum(1 for c in cases if not c['gold_chunks'])} unanswerable.")

    corpus_dir = resolve_path(args.corpus)
    try:
        inventory = build_chunk_inventory(corpus_dir)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Chunk inventory from {corpus_dir}: {len(inventory)} chunks.")
    missing = verify_gold_in_inventory(cases, inventory)
    if missing:
        print(f"Inventory check: {len(missing)} gold chunk(s) missing:")
        for item in missing:
            print(f"  - {item}")
        sys.exit(1)
    print("Inventory check OK: every gold chunk exists in the reproduced chunking.")

    if args.live:
        try:
            missing = verify_gold_in_index(cases)
        except Exception as e:
            print(f"Error: live index check failed: {e}")
            sys.exit(1)
        if missing:
            print(f"Live index check: {len(missing)} gold chunk(s) missing from Milvus:")
            for item in missing:
                print(f"  - {item}")
            sys.exit(1)
        print("Live index check OK: every gold chunk exists in Milvus.")


def main():
    # No arguments → interactive menu
    if len(sys.argv) == 1:
        from .interactive import interactive_menu
        interactive_menu()
        return

    # With arguments → CLI mode
    parser = argparse.ArgumentParser(description="RAGAS evaluation for RagMate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate test set from documents")
    gen_parser.add_argument("--size", type=int, default=50, help="Number of test cases to generate")
    gen_parser.add_argument("--max-docs", type=int, default=None, help="Max documents to load")
    gen_parser.add_argument("--output", type=str, default=str(DEFAULT_TESTSETS_DIR / "testset.json"), help="Output JSON path")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    gen_parser.add_argument("--distributions", type=str, default=None, help="Test type distributions (JSON)")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate RAG pipeline")
    eval_parser.add_argument("--testset", type=str, required=True, help="Test set JSON path")
    eval_parser.add_argument("--report", type=str, default=None, help="Report output path")
    eval_parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    eval_parser.add_argument("--threshold", type=float, default=None, help="Min overall score (exit non-zero if below)")
    eval_parser.add_argument("--judge-model", type=str, default=None, help="Judge LLM model name")
    eval_parser.add_argument("--mode", type=str, choices=["single", "agent"], default="single", help="Evaluation mode")

    rescore_parser = subparsers.add_parser("rescore", help="Re-score a saved report without re-running the pipeline")
    rescore_parser.add_argument("--report", type=str, required=True, help="Saved report or raw checkpoint JSON path (overwritten unless --output is given)")
    rescore_parser.add_argument("--output", type=str, default=None, help="Output path for the re-scored report")
    rescore_parser.add_argument("--judge-model", type=str, default=None, help="Judge LLM model name (defaults to the one in the report)")

    retrieval_parser = subparsers.add_parser("retrieval", help="Deterministic retrieval evaluation (no LLM judge)")
    retrieval_parser.add_argument("--testset", type=str, required=True, help="Golden test set JSON path (must carry gold chunk IDs)")
    retrieval_parser.add_argument("--report", type=str, default=None, help="Metrics report output path")
    retrieval_parser.add_argument("--stages", type=str, default=",".join(STAGES), help="Comma-separated stages: dense,sparse,fused,reranked")
    retrieval_parser.add_argument("--over-fetch", type=int, default=50, help="Candidates retrieved per stage before reranking")
    retrieval_parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20], help="k values for hit/recall/nDCG")
    retrieval_parser.add_argument("--repeat", type=int, default=1, help="Repeat runs to measure determinism spread")
    retrieval_parser.add_argument("--compare-manifest", type=str, default=None, help="Previous retrieval report path to diff config against")

    check_gold_parser = subparsers.add_parser("check-gold", help="Validate golden set gold chunk annotations")
    check_gold_parser.add_argument("--testset", type=str, required=True, help="Golden test set JSON path")
    check_gold_parser.add_argument("--corpus", type=str, default="eval/testsets/sample/corpus", help="Corpus directory used to reproduce chunking")
    check_gold_parser.add_argument("--live", action="store_true", help="Also verify gold chunks exist in the live Milvus index")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "rescore":
        cmd_rescore(args)
    elif args.command == "retrieval":
        cmd_retrieval(args)
    elif args.command == "check-gold":
        cmd_check_gold(args)
