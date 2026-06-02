"""RAGAS 评估 CLI 入口。"""
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
from .metrics import compute_metrics
from .report import build_report, print_report

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

    ragas_result = compute_metrics(valid_cases, judge_model=judge_model)

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

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
