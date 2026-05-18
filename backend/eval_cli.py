"""RAGAS evaluation CLI for RagMate RAG pipeline.

Usage:
    ragmate-eval generate --size 50 --output eval/testsets/testset_v1.json
    ragmate-eval evaluate --testset eval/testsets/testset_v1.json --report eval/reports/report.json
"""

import argparse
import json
import logging
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Suppress RAGAS deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

logger = logging.getLogger("ragmate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# Project root: parent of backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TESTSETS_DIR = PROJECT_ROOT / "eval" / "testsets"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"


def resolve_path(p: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _clean_score(value) -> float | None:
    """Convert NaN/Inf to None for JSON serialization."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    return float(value)


# ── Document Loading ────────────────────────────────────────────────────────

def load_langchain_docs(docs_dir: str, max_docs: int = None) -> list[Document]:
    """Load documents from directory as LangChain Document objects.

    Reuses RagMate's existing loaders but skips chunking — RAGAS handles
    its own splitting internally. Metadata includes source filename and page.
    """
    from ingest import load_document, SUPPORTED_EXTENSIONS

    all_files = sorted(
        f for f in os.listdir(docs_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )

    if max_docs:
        all_files = all_files[:max_docs]

    logger.info(f"Loading {len(all_files)} documents from {docs_dir}")

    documents = []
    for filename in all_files:
        filepath = os.path.join(docs_dir, filename)
        try:
            pages = load_document(filepath)
            for page in pages:
                page.metadata["source"] = os.path.basename(page.metadata.get("source", filename))
            documents.extend(pages)
            logger.info(f"  {filename}: {len(pages)} pages")
        except Exception as e:
            logger.warning(f"  Failed to load {filename}: {e}")

    logger.info(f"Loaded {len(documents)} pages total from {len(all_files)} files")
    return documents


# ── Test Set Generation ─────────────────────────────────────────────────────

def generate_testset(
    documents: list[Document],
    size: int = 50,
    seed: int = 42,
    query_distribution: dict = None,
) -> list[dict]:
    """Generate test cases from documents using RAGAS TestsetGenerator.

    Returns list of dicts with user_input, reference, and source fields.
    retrieved_contexts and response are empty — filled during evaluation.
    """
    from ragas.testset import TestsetGenerator
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from model_factory import get_llm, get_embeddings

    llm = LangchainLLMWrapper(get_llm())
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    kwargs = {}
    if query_distribution:
        kwargs["query_distribution"] = query_distribution

    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=size,
        **kwargs,
    )

    test_cases = []
    for sample in testset.samples:
        eval_sample = sample.eval_sample
        test_cases.append({
            "user_input": eval_sample.user_input,
            "reference": eval_sample.reference or "",
            "retrieved_contexts": [],
            "response": "",
            "source": "",
        })

    return test_cases


# ── Context Retrieval ───────────────────────────────────────────────────────

def retrieve_contexts(question: str, top_k: int = None) -> list[str]:
    """Retrieve relevant document chunks for a question.

    Uses the full RagMate retrieval pipeline: hybrid search (dense+sparse)
    → cross-encoder reranking → score filtering → source dedup.
    """
    from retriever import retrieve
    from config import settings

    k = top_k or settings.FINAL_CONTEXT_K
    results = retrieve(question, k=k)
    return [r["text"] for r in results]


# ── Response Generation ─────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """你是一个知识库问答助手。根据提供的上下文回答用户的问题。

规则：
1. 只基于提供的上下文回答，不要编造信息
2. 如果上下文中没有相关信息，明确说明
3. 回答要简洁准确"""

RAG_USER_PROMPT = """上下文：
{contexts}

问题：{question}

请基于上述上下文回答问题。"""


def generate_response(question: str, contexts: list[str]) -> str:
    """Generate an answer using LLM based on retrieved contexts.

    Uses a simple RAG prompt (not the full Deep Agent pipeline).
    This tests the baseline RAG generation capability.
    """
    from model_factory import get_llm

    llm = get_llm()
    context_text = "\n\n---\n\n".join(contexts) if contexts else "（未找到相关上下文）"

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=RAG_USER_PROMPT.format(contexts=context_text, question=question)),
    ]

    response = llm.invoke(messages)
    return response.content


# ── Metric Computation ──────────────────────────────────────────────────────

def compute_metrics(test_cases: list[dict], judge_model: str = None):
    """Compute RAGAS metrics on evaluated test cases.

    Expects each test case to have user_input, response, retrieved_contexts,
    and reference fields populated.
    """
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from model_factory import get_llm, get_embeddings
    from streaming_llm import create_llm, ChatOpenAICompatible
    from config import settings

    # Subclass that forces n=1 (some APIs like MiMo don't support n>1)
    class EvalLLM(ChatOpenAICompatible):
        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            payload.pop("n", None)
            return payload

    # Build judge LLM
    if judge_model:
        base_llm = create_llm(
            model=judge_model,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_API_BASE_URL,
        )
    else:
        base_llm = get_llm()

    judge_llm = LangchainLLMWrapper(EvalLLM(
        model=base_llm.model_name,
        api_key=base_llm.openai_api_key,
        base_url=base_llm.openai_api_base,
    ))

    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    # Add FactualCorrectness if available
    try:
        from ragas.metrics import FactualCorrectness
        metrics.append(FactualCorrectness(llm=judge_llm))
    except ImportError:
        pass

    # Build RAGAS samples
    from ragas.dataset_schema import EvaluationDataset

    samples = []
    for tc in test_cases:
        samples.append(SingleTurnSample(
            user_input=tc["user_input"],
            response=tc["response"],
            retrieved_contexts=tc["retrieved_contexts"],
            reference=tc["reference"],
        ))

    dataset = EvaluationDataset(samples)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return result


# ── Report ──────────────────────────────────────────────────────────────────

def build_report(meta: dict, test_cases: list[dict], ragas_result) -> dict:
    """Build a structured evaluation report."""

    # Extract per-sample scores
    details = []
    scores_list = ragas_result.scores if hasattr(ragas_result, "scores") else []

    for i, tc in enumerate(test_cases):
        scores = {}
        if i < len(scores_list):
            for metric_name, value in scores_list[i].items():
                scores[metric_name] = _clean_score(value)

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

    # Extract summary (mean of each metric, excluding NaN)
    summary = {}
    if scores_list:
        metric_names = list(scores_list[0].keys())
        for metric_name in metric_names:
            values = [
                _clean_score(s[metric_name])
                for s in scores_list
                if _clean_score(s.get(metric_name)) is not None
            ]
            summary[metric_name] = round(sum(values) / len(values), 4) if values else None

    # Overall = mean of non-null metrics
    valid_scores = [v for v in summary.values() if v is not None]
    summary["overall"] = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None

    return {
        "meta": meta,
        "summary": summary,
        "details": details,
    }


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


# ── CLI Commands ────────────────────────────────────────────────────────────

def cmd_generate(args):
    """Generate a test set from documents."""
    from config import settings

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
    from config import settings

    # Load test set
    testset_path = resolve_path(args.testset)
    if not testset_path.exists():
        print(f"Error: Test set not found: {testset_path}")
        sys.exit(1)

    with open(testset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases from {testset_path}")

    # Resolve config
    top_k = args.top_k or settings.FINAL_CONTEXT_K
    judge_model = args.judge_model or settings.LLM_MODEL
    mode = args.mode

    print(f"Config: mode={mode}, top_k={top_k}, judge_model={judge_model}")

    # Step 1: Retrieve contexts and generate responses
    print("\nRunning RAG pipeline for each test case...")
    errors = 0
    for tc in tqdm(test_cases, desc="Evaluating"):
        question = tc["user_input"]
        try:
            contexts = retrieve_contexts(question, top_k=top_k)
            tc["retrieved_contexts"] = contexts

            if mode == "single":
                response = generate_response(question, contexts)
            else:
                # agent mode: placeholder for future
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

    # Filter out failed cases
    valid_cases = [tc for tc in test_cases if tc.get("response") and tc.get("retrieved_contexts")]
    if not valid_cases:
        print("Error: No valid test cases to evaluate")
        sys.exit(1)

    print(f"\nComputing RAGAS metrics on {len(valid_cases)} valid cases...")

    # Step 2: Compute metrics
    ragas_result = compute_metrics(valid_cases, judge_model=judge_model)

    # Step 3: Build report
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

    # Step 4: Save and print
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(report)

    # Step 5: Threshold check
    if args.threshold is not None:
        overall = report["summary"].get("overall") or 0
        if overall < args.threshold:
            print(f"\nFAILED: Overall score {overall:.2f} < threshold {args.threshold}")
            sys.exit(1)
        else:
            print(f"\nPASSED: Overall score {overall:.2f} >= threshold {args.threshold}")


# ── Interactive Menu ────────────────────────────────────────────────────────

def _list_json_files(directory: str) -> list[Path]:
    """List JSON files in a directory, sorted by modification time (newest first)."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _menu_generate():
    """Interactive test set generation."""
    from config import settings

    print("\n--- 生成测试集 ---")
    size = input("测试用例数量 [50]: ").strip()
    size = int(size) if size else 50

    default_output = str(DEFAULT_TESTSETS_DIR / "testset.json")
    output = input(f"输出路径 [{default_output}]: ").strip()
    output = output or default_output

    docs_dir = settings.DOCUMENTS_DIR
    if not os.path.exists(docs_dir):
        print(f"Error: Documents directory not found: {docs_dir}")
        return

    docs = load_langchain_docs(docs_dir)
    if not docs:
        print("Error: No documents loaded")
        return

    print(f"\nLoaded {len(docs)} pages from {docs_dir}")
    print(f"Generating {size} test cases...")

    test_cases = generate_testset(documents=docs, size=size)

    output_path = resolve_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated {len(test_cases)} test cases → {output_path}")
    print("Review and edit the test set before running evaluate.")


def _menu_evaluate():
    """Interactive evaluation."""
    print("\n--- 运行评估 ---")

    # List available test sets
    testsets = _list_json_files(str(DEFAULT_TESTSETS_DIR))
    if not testsets:
        print(f"Error: No test sets found in {DEFAULT_TESTSETS_DIR}")
        print("Run '生成测试集' first.")
        return

    print("可用的测试集：")
    for i, f in enumerate(testsets, 1):
        print(f"  {i}. {f.name}")
    print()

    choice = input("选择测试集 [1]: ").strip()
    choice = int(choice) if choice else 1
    if choice < 1 or choice > len(testsets):
        print("Error: Invalid choice")
        return

    testset_path = testsets[choice - 1]

    threshold_input = input("最低分数线（回车跳过）: ").strip()
    threshold = float(threshold_input) if threshold_input else None

    # Build a mock args object
    class Args:
        pass

    args = Args()
    args.testset = str(testset_path)
    args.report = None
    args.top_k = None
    args.threshold = threshold
    args.judge_model = None
    args.mode = "single"

    cmd_evaluate(args)


def _menu_reports():
    """View historical reports."""
    reports = _list_json_files(str(DEFAULT_REPORTS_DIR))
    if not reports:
        print("\n暂无评估报告。")
        return

    print("\n--- 历史报告 ---")
    for i, f in enumerate(reports, 1):
        print(f"  {i}. {f.name}")
    print()

    choice = input("查看报告（输入编号，回车返回）: ").strip()
    if not choice:
        return

    choice = int(choice)
    if choice < 1 or choice > len(reports):
        print("Error: Invalid choice")
        return

    report_path = reports[choice - 1]
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    print_report(report)


def interactive_menu():
    """Main interactive menu loop."""
    while True:
        print("\n=== RagMate RAGAS 评估工具 ===\n")
        print("  1. 生成测试集")
        print("  2. 运行评估")
        print("  3. 查看历史报告")
        print("  0. 退出")
        print()

        choice = input("请选择 > ").strip()

        if choice == "1":
            _menu_generate()
        elif choice == "2":
            _menu_evaluate()
        elif choice == "3":
            _menu_reports()
        elif choice == "0":
            print("Bye!")
            break
        else:
            print("无效选择，请重试。")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # No arguments → interactive menu
    if len(sys.argv) == 1:
        interactive_menu()
        return

    # With arguments → CLI mode
    parser = argparse.ArgumentParser(description="RAGAS evaluation for RagMate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate test set from documents")
    gen_parser.add_argument("--size", type=int, default=50, help="Number of test cases to generate")
    gen_parser.add_argument("--max-docs", type=int, default=None, help="Max documents to load")
    gen_parser.add_argument("--output", type=str, default=str(DEFAULT_TESTSETS_DIR / "testset.json"), help="Output JSON path")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    gen_parser.add_argument("--distributions", type=str, default=None, help="Test type distributions (JSON)")

    # evaluate subcommand
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


if __name__ == "__main__":
    main()
