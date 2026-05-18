# RAGAS Evaluation Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone `eval/ragas_eval.py` script that can generate test sets from documents and evaluate RagMate's RAG pipeline using RAGAS metrics.

**Architecture:** Single Python script at project root with two CLI modes (generate/evaluate). Imports backend modules (retriever, ingest, model_factory, config) via sys.path manipulation. Uses RAGAS 0.2+ for test set generation and metric computation. Outputs JSON reports with per-question scores and summary statistics.

**Tech Stack:** Python 3.12+, RAGAS 0.2+, LangChain (existing), tqdm

---

### Task 1: Setup — directory structure and dependencies

**Files:**
- Create: `eval/ragas_eval.py` (empty placeholder)
- Create: `eval/testsets/.gitkeep`
- Create: `eval/reports/.gitkeep`
- Modify: `backend/pyproject.toml:34-37`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p eval/testsets eval/reports
touch eval/testsets/.gitkeep eval/reports/.gitkeep
```

- [ ] **Step 2: Create empty eval script placeholder**

Create `eval/ragas_eval.py`:
```python
"""RAGAS evaluation script for RagMate RAG pipeline."""
```

- [ ] **Step 3: Add eval dependencies to pyproject.toml**

In `backend/pyproject.toml`, add `eval` optional dependency group after the existing `study` group (line 37):

```toml
[project.optional-dependencies]
study = [
    "bilibili-api-python>=17.0.0",
]
eval = [
    "ragas>=0.2.0",
    "tqdm>=4.65.0",
]
```

- [ ] **Step 4: Install eval dependencies**

```bash
cd backend && pip install -e ".[eval]"
```

Expected: ragas and tqdm installed successfully.

- [ ] **Step 5: Commit**

```bash
git add eval/ backend/pyproject.toml
git commit -m "feat(eval): scaffold eval directory and add RAGAS dependency"
```

---

### Task 2: CLI entry point with argparse

**Files:**
- Modify: `eval/ragas_eval.py`

- [ ] **Step 1: Write the CLI skeleton**

Replace contents of `eval/ragas_eval.py`:

```python
"""RAGAS evaluation script for RagMate RAG pipeline.

Usage:
    python eval/ragas_eval.py generate --size 50 --output eval/testsets/testset_v1.json
    python eval/ragas_eval.py evaluate --testset eval/testsets/testset_v1.json --report eval/reports/report.json
"""

import argparse
import sys
from pathlib import Path

# Add backend to Python path so we can import RagMate modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


def cmd_generate(args):
    """Generate a test set from documents."""
    print(f"Generating test set with size={args.size}...")
    print(f"Output: {args.output}")
    # TODO: Task 3-4


def cmd_evaluate(args):
    """Evaluate RAG pipeline against a test set."""
    print(f"Evaluating test set: {args.testset}")
    print(f"Report: {args.report}")
    # TODO: Task 5-9


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for RagMate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate test set from documents")
    gen_parser.add_argument("--size", type=int, default=50, help="Number of test cases to generate")
    gen_parser.add_argument("--max-docs", type=int, default=None, help="Max documents to load")
    gen_parser.add_argument("--output", type=str, default="eval/testsets/testset.json", help="Output JSON path")
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
```

- [ ] **Step 2: Verify CLI runs**

```bash
cd c:/Users/19097/Documents/GitHub/RagMate && python eval/ragas_eval.py --help
python eval/ragas_eval.py generate --help
python eval/ragas_eval.py evaluate --help
```

Expected: Help text displayed for each subcommand without import errors.

- [ ] **Step 3: Commit**

```bash
git add eval/ragas_eval.py
git commit -m "feat(eval): add CLI skeleton with generate/evaluate subcommands"
```

---

### Task 3: Document loading for test set generation

**Files:**
- Modify: `eval/ragas_eval.py` (add `load_langchain_docs()` function)

- [ ] **Step 1: Implement load_langchain_docs()**

Add this function to `eval/ragas_eval.py` before `cmd_generate`:

```python
import logging
import os

from langchain_core.documents import Document

logger = logging.getLogger("ragmate")


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
            # Ensure metadata has clean source (filename only, not full path)
            for page in pages:
                page.metadata["source"] = os.path.basename(page.metadata.get("source", filename))
            documents.extend(pages)
            logger.info(f"  {filename}: {len(pages)} pages")
        except Exception as e:
            logger.warning(f"  Failed to load {filename}: {e}")

    logger.info(f"Loaded {len(documents)} pages total from {len(all_files)} files")
    return documents
```

- [ ] **Step 2: Wire into cmd_generate**

Update `cmd_generate` to call the loader and print results:

```python
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
    print(f"Will generate {args.size} test cases → {args.output}")
    # Generation logic in Task 4
```

- [ ] **Step 3: Verify document loading works**

```bash
cd c:/Users/19097/Documents/GitHub/RagMate && python -c "
import sys
sys.path.insert(0, 'backend')
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve() / 'backend'))

# Simulate what eval script does
import os
os.chdir('backend')
from config import settings
from ingest import load_document, SUPPORTED_EXTENSIONS

files = [f for f in os.listdir(settings.DOCUMENTS_DIR) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
print(f'Found files: {files}')

for f in files[:1]:
    pages = load_document(os.path.join(settings.DOCUMENTS_DIR, f))
    print(f'{f}: {len(pages)} pages, first page metadata: {pages[0].metadata}')
"
```

Expected: Files listed, first file's pages and metadata shown.

- [ ] **Step 4: Commit**

```bash
git add eval/ragas_eval.py
git commit -m "feat(eval): add document loading for test set generation"
```

---

### Task 4: Test set generation with RAGAS TestsetGenerator

**Files:**
- Modify: `eval/ragas_eval.py` (add `generate_testset()` function, update `cmd_generate`)

- [ ] **Step 1: Implement generate_testset()**

Add this function to `eval/ragas_eval.py`:

```python
import json
import random
from datetime import datetime


def generate_testset(
    documents: list[Document],
    size: int = 50,
    seed: int = 42,
    distributions: dict = None,
) -> list[dict]:
    """Generate test cases from documents using RAGAS TestsetGenerator.

    Returns list of dicts with user_input, reference, and source fields.
    retrieved_contexts and response are empty — filled during evaluation.
    """
    from ragas.testset import TestsetGenerator
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from model_factory import get_llm, get_embeddings

    llm = LangchainLLMWrapper(get_llm())
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    kwargs = {"language": "chinese"}
    if distributions:
        kwargs["distributions"] = distributions

    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=size,
        **kwargs,
    )

    # Convert to our format
    test_cases = []
    for sample in testset.samples:
        test_cases.append({
            "user_input": sample.user_input,
            "reference": sample.reference,
            "retrieved_contexts": [],
            "response": "",
            "source": "",  # RAGAS doesn't track source; human can add
        })

    return test_cases
```

- [ ] **Step 2: Wire into cmd_generate**

Update `cmd_generate` to call generation and save output:

```python
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

    distributions = None
    if args.distributions:
        distributions = json.loads(args.distributions)

    random.seed(args.seed)
    test_cases = generate_testset(
        documents=docs,
        size=args.size,
        seed=args.seed,
        distributions=distributions,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(test_cases)} test cases → {output_path}")
    print("Review and edit the test set before running evaluate.")
```

- [ ] **Step 3: Verify generation runs (small test)**

```bash
cd c:/Users/19097/Documents/GitHub/RagMate && python eval/ragas_eval.py generate --size 3 --output eval/testsets/test_small.json
```

Expected: 3 test cases generated and saved to JSON. Check the file content.

- [ ] **Step 4: Commit**

```bash
git add eval/ragas_eval.py
git commit -m "feat(eval): implement test set generation with RAGAS TestsetGenerator"
```

---

### Task 5: Context retrieval for evaluation

**Files:**
- Modify: `eval/ragas_eval.py` (add `retrieve_contexts()` function)

- [ ] **Step 1: Implement retrieve_contexts()**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add eval/ragas_eval.py
git commit -m "feat(eval): add context retrieval wrapper"
```

---

### Task 6: Response generation for evaluation

**Files:**
- Modify: `eval/ragas_eval.py` (add `generate_response()` function)

- [ ] **Step 1: Implement generate_response()**

```python
from langchain_core.messages import HumanMessage, SystemMessage


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
```

- [ ] **Step 2: Commit**

```bash
git add eval/ragas_eval.py
git commit -m "feat(eval): add LLM response generation for evaluation"
```

---

### Task 7: RAGAS metric computation

**Files:**
- Modify: `eval/ragas_eval.py` (add `compute_metrics()` function)

- [ ] **Step 1: Implement compute_metrics()**

```python
def compute_metrics(
    test_cases: list[dict],
    judge_model: str = None,
) -> dict:
    """Compute RAGAS metrics on evaluated test cases.

    Expects each test case to have user_input, response, retrieved_contexts,
    and reference fields populated.

    Returns dict with 'summary' (metric means) and 'details' (per-case scores).
    """
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from model_factory import get_llm, get_embeddings
    from streaming_llm import create_llm
    from config import settings

    # Build judge LLM
    if judge_model:
        llm = LangchainLLMWrapper(create_llm(
            model=judge_model,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_API_BASE_URL,
        ))
    else:
        llm = LangchainLLMWrapper(get_llm())

    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    # Try to add FactualCorrectness if available
    try:
        from ragas.metrics import FactualCorrectness
        metrics.append(FactualCorrectness(llm=llm))
    except ImportError:
        pass

    # Build RAGAS samples
    samples = []
    for tc in test_cases:
        samples.append(SingleTurnSample(
            user_input=tc["user_input"],
            response=tc["response"],
            retrieved_contexts=tc["retrieved_contexts"],
            reference=tc["reference"],
        ))

    # Evaluate
    result = evaluate(
        dataset=samples,
        metrics=metrics,
    )

    return result
```

- [ ] **Step 2: Commit**

```bash
git add eval/ragas_eval.py
git commit -m "feat(eval): add RAGAS metric computation"
```

---

### Task 8: Report generation

**Files:**
- Modify: `eval/ragas_eval.py` (add `build_report()` and `print_report()` functions)

- [ ] **Step 1: Implement report functions**

```python
from datetime import datetime


def build_report(
    meta: dict,
    test_cases: list[dict],
    ragas_result,
) -> dict:
    """Build a structured evaluation report."""

    # Extract per-sample scores from RAGAS result
    details = []
    for i, tc in enumerate(test_cases):
        scores = {}
        if hasattr(ragas_result, 'scores') and i < len(ragas_result.scores):
            for metric_name, value in ragas_result.scores[i].items():
                scores[metric_name] = value

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

    # Extract summary
    summary = {}
    if hasattr(ragas_result, 'scores') and ragas_result.scores:
        metric_names = list(ragas_result.scores[0].keys())
        for metric_name in metric_names:
            values = [
                s[metric_name]
                for s in ragas_result.scores
                if s[metric_name] is not None
            ]
            summary[metric_name] = sum(values) / len(values) if values else 0.0

    if summary:
        summary["overall"] = sum(summary.values()) / len(summary)

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
            print(f"  {label:<25} {value:.2f}")
    else:
        print("No metrics computed (all cases may have errored).")

    print(f"\nDetailed report saved to {meta.get('report_path', 'N/A')}")
```

- [ ] **Step 2: Commit**

```bash
git add eval/ragas_eval.py
git commit -m "feat(eval): add report building and console output"
```

---

### Task 9: Wire up evaluate mode end-to-end

**Files:**
- Modify: `eval/ragas_eval.py` (update `cmd_evaluate`)

- [ ] **Step 1: Implement cmd_evaluate**

Replace `cmd_evaluate` with the full implementation:

```python
def cmd_evaluate(args):
    """Evaluate RAG pipeline against a test set."""
    from config import settings
    from tqdm import tqdm

    # Load test set
    testset_path = Path(args.testset)
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
            # Retrieve
            contexts = retrieve_contexts(question, top_k=top_k)
            tc["retrieved_contexts"] = contexts

            # Generate
            if mode == "single":
                response = generate_response(question, contexts)
            else:
                # agent mode: placeholder for future implementation
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

    # Filter out fully failed cases for metric computation
    valid_cases = [tc for tc in test_cases if tc.get("response") and tc.get("retrieved_contexts")]
    if not valid_cases:
        print("Error: No valid test cases to evaluate")
        sys.exit(1)

    print(f"\nComputing RAGAS metrics on {len(valid_cases)} valid cases...")

    # Step 2: Compute metrics
    ragas_result = compute_metrics(valid_cases, judge_model=judge_model)

    # Step 3: Build report
    report_path = args.report or f"eval/reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

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
        overall = report["summary"].get("overall", 0)
        if overall < args.threshold:
            print(f"\nFAILED: Overall score {overall:.2f} < threshold {args.threshold}")
            sys.exit(1)
        else:
            print(f"\nPASSED: Overall score {overall:.2f} >= threshold {args.threshold}")
```

- [ ] **Step 2: Update imports at top of file**

Ensure these imports are at the top of `eval/ragas_eval.py` (some are inside functions, but logging/json/os should be at top):

```python
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

logger = logging.getLogger("ragmate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
```

- [ ] **Step 3: Verify full evaluate flow (dry run)**

First generate a small test set, then evaluate it:

```bash
cd c:/Users/19097/Documents/GitHub/RagMate

# Generate 3 test cases
python eval/ragas_eval.py generate --size 3 --output eval/testsets/test_dryrun.json

# Evaluate
python eval/ragas_eval.py evaluate --testset eval/testsets/test_dryrun.json --report eval/reports/report_dryrun.json
```

Expected: Report generated with metrics. Check `eval/reports/report_dryrun.json`.

- [ ] **Step 4: Commit**

```bash
git add eval/ragas_eval.py
git commit -m "feat(eval): wire up complete evaluate pipeline"
```

---

### Task 10: End-to-end verification and cleanup

**Files:**
- Modify: `eval/ragas_eval.py` (final polish)

- [ ] **Step 1: Run full generate + evaluate cycle**

```bash
cd c:/Users/19097/Documents/GitHub/RagMate

# Generate a meaningful test set
python eval/ragas_eval.py generate --size 10 --output eval/testsets/testset_v1.json

# Review the generated test set
cat eval/testsets/testset_v1.json

# Run evaluation
python eval/ragas_eval.py evaluate --testset eval/testsets/testset_v1.json --report eval/reports/report_v1.json --threshold 0.5

# Check report
cat eval/reports/report_v1.json
```

Expected: Test set generated with 10 cases. Evaluation completes with all 5 metrics. Report saved.

- [ ] **Step 2: Verify threshold exit code**

```bash
# This should fail (threshold too high)
python eval/ragas_eval.py evaluate --testset eval/testsets/testset_v1.json --threshold 0.99
echo "Exit code: $?"
```

Expected: Exit code 1 with "FAILED" message.

- [ ] **Step 3: Verify error handling — missing test set**

```bash
python eval/ragas_eval.py evaluate --testset eval/testsets/nonexistent.json
```

Expected: Error message, exit code 1.

- [ ] **Step 4: Clean up test artifacts**

```bash
rm -f eval/testsets/test_small.json eval/testsets/test_dryrun.json eval/reports/report_dryrun.json
```

- [ ] **Step 5: Final commit**

```bash
git add eval/ragas_eval.py
git commit -m "feat(eval): complete RAGAS evaluation script with end-to-end verification"
```
