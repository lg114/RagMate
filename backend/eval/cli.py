"""评估 CLI：回归门禁是 retrieval-gate。RAGAS 打分命令仍在，已搁置。"""
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
from .response_gen import answer_prompt_fingerprint, generate_response
from .metrics import (
    DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_PARSE_RETRIES,
    build_judge_pin,
    probe_judge_llm,
    score_with_repeats,
)
from .report import (
    build_report,
    cases_from_report,
    load_raw_checkpoint,
    merge_into_raw,
    print_report,
    questions_missing_from_raw,
    save_raw_checkpoint,
    write_raw_checkpoint,
)
from .manifest import build_manifest
from .retrieval_eval import (
    STAGES,
    build_chunk_inventory,
    build_retrieval_report,
    compare_with_manifest,
    load_golden_testset,
    load_retrieval_checkpoint,
    print_cutoff_report,
    print_retrieval_report,
    run_retrieval_eval,
    simulate_final_ks,
    validate_gold_schema,
    verify_gold_in_index,
    verify_gold_in_inventory,
    write_cutoff_report,
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


def _resolve_judge_args(args, meta: dict | None = None):
    """CLI / raw meta → 钉死的 judge 配置。"""
    from backend.infrastructure.config import settings

    meta = meta or {}
    judge_model = (
        getattr(args, "judge_model", None)
        or meta.get("judge_model")
        or settings.LLM_MODEL
    )
    if getattr(args, "judge_temperature", None) is not None:
        temperature = float(args.judge_temperature)
    elif meta.get("judge_temperature") is not None:
        temperature = float(meta["judge_temperature"])
    else:
        temperature = DEFAULT_JUDGE_TEMPERATURE
    parse_retries = (
        args.parse_retries
        if getattr(args, "parse_retries", None) is not None
        else DEFAULT_PARSE_RETRIES
    )
    score_repeats = getattr(args, "score_repeats", None) or 1
    return judge_model, temperature, int(parse_retries), int(score_repeats)


def cmd_evaluate(args):
    """Run RAG pipeline (± optional scoring). Prefer --generate-only then rescore."""
    from backend.infrastructure.config import settings

    testset_path = resolve_path(args.testset)
    if not testset_path.exists():
        print(f"Error: Test set not found: {testset_path}")
        sys.exit(1)

    with open(testset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases from {testset_path}")

    top_k = args.top_k or settings.FINAL_CONTEXT_K
    mode = args.mode
    generate_only = bool(getattr(args, "generate_only", False))
    judge_model, temperature, parse_retries, score_repeats = _resolve_judge_args(args)

    print(f"Config: mode={mode}, top_k={top_k}, generate_only={generate_only}")
    if not generate_only:
        print(
            f"Judge: model={judge_model}, temperature={temperature}, "
            f"parse_retries={parse_retries}, score_repeats={score_repeats}"
        )
        print("\nChecking judge model responds with parseable JSON...")
        try:
            probe_judge_llm(judge_model, temperature=temperature)
        except RuntimeError as e:
            print(f"Error: judge check failed, not running the pipeline: {e}")
            sys.exit(1)
        print("Judge check passed.")

    print(
        f"\nRunning RAG pipeline for each test case "
        f"(timeout={settings.EVAL_LLM_TIMEOUT}s, retries={settings.EVAL_LLM_RETRIES})..."
    )
    errors = 0
    for tc in tqdm(test_cases, desc="Generating"):
        question = tc["user_input"]
        try:
            contexts = retrieve_contexts(question, top_k=top_k)
            tc["retrieved_contexts"] = contexts
            response = generate_response(question, contexts)
            tc["response"] = response
            tc.pop("error", None)
        except Exception as e:
            logger.error(f"Failed for question '{question[:50]}': {e}")
            print(f"Failed for question '{question[:50]}': {e}")
            tc["response"] = ""
            tc["retrieved_contexts"] = []
            tc["error"] = str(e)
            errors += 1

    if errors:
        print(f"\nWarning: {errors}/{len(test_cases)} cases failed")
        print("Patch later with: ragmate-eval patch-raw --raw <run.raw.json> --testset <testset>")

    valid_cases = [tc for tc in test_cases if tc.get("response") and tc.get("retrieved_contexts")]
    if not valid_cases:
        print("Error: No valid test cases to evaluate")
        sys.exit(1)

    report_path = str(
        resolve_path(args.report)
        if args.report
        else DEFAULT_REPORTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    pin = build_judge_pin(judge_model, temperature)
    meta = {
        "timestamp": datetime.now().isoformat(),
        "testset": str(testset_path),
        "test_cases": len(test_cases),
        "valid_cases": len(valid_cases),
        "failed_cases": errors,
        "mode": mode,
        "top_k": top_k,
        "generator_model": settings.LLM_MODEL,
        "retriever_config": {
            "hybrid_search": settings.HYBRID_SEARCH_ENABLED,
            "reranker": settings.RERANKER_MODEL,
            "score_threshold": settings.RERANK_SCORE_THRESHOLD,
            "chunking_strategy": settings.CHUNKING_STRATEGY,
            "final_context_k": settings.FINAL_CONTEXT_K,
        },
        "report_path": report_path,
        **answer_prompt_fingerprint(),
        **pin.as_meta(),
    }

    try:
        # Pass full test_cases so failed_questions land in raw meta for patch-raw.
        raw_path = write_raw_checkpoint(
            report_path, meta, test_cases, force=bool(getattr(args, "force_raw", False)),
        )
    except FileExistsError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Raw pipeline results checkpointed to {raw_path}")
    if errors:
        print(f"Recorded {errors} failed question(s) in raw meta for patch-raw.")

    if generate_only:
        print(
            "Generate-only done. Score later with:\n"
            f"  ragmate-eval rescore --report {raw_path} "
            f"--output {report_path} --score-repeats {score_repeats}"
        )
        if errors:
            print(
                "Fill gaps first:\n"
                f"  ragmate-eval patch-raw --raw {raw_path} --testset {testset_path}"
            )
        return

    print(f"\nComputing RAGAS metrics on {len(valid_cases)} valid cases...")
    try:
        matrix = score_with_repeats(
            valid_cases,
            judge_model=judge_model,
            temperature=temperature,
            parse_retries=parse_retries,
            score_repeats=score_repeats,
        )
    except RuntimeError as e:
        print(f"\nError: scoring failed: {e}")
        print(f"Pipeline results are preserved in {raw_path}. Fix the judge, then re-score only with:")
        print(f"  ragmate-eval rescore --report {raw_path}")
        sys.exit(1)

    report = build_report(meta, valid_cases, matrix)
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


def cmd_patch_raw(args):
    """Re-generate only questions missing from an existing raw checkpoint."""
    from backend.infrastructure.config import settings

    raw_path = resolve_path(args.raw)
    if not raw_path.exists():
        print(f"Error: Raw checkpoint not found: {raw_path}")
        sys.exit(1)
    testset_path = resolve_path(args.testset)
    if not testset_path.exists():
        print(f"Error: Test set not found: {testset_path}")
        sys.exit(1)

    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)
    if not isinstance(testset, list):
        print("Error: Test set must be a JSON list")
        sys.exit(1)

    try:
        raw = load_raw_checkpoint(raw_path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    missing = questions_missing_from_raw(testset, raw)
    if not missing:
        print(f"Nothing to patch: all {len(testset)} testset questions are in {raw_path}")
        return

    top_k = args.top_k or raw.get("meta", {}).get("top_k") or settings.FINAL_CONTEXT_K
    print(
        f"Patching {len(missing)} missing question(s) into {raw_path} "
        f"(timeout={settings.EVAL_LLM_TIMEOUT}s, retries={settings.EVAL_LLM_RETRIES}, top_k={top_k})"
    )

    still_failed = 0
    for tc in tqdm(missing, desc="Patching"):
        question = tc["user_input"]
        try:
            contexts = retrieve_contexts(question, top_k=top_k)
            tc["retrieved_contexts"] = contexts
            tc["response"] = generate_response(question, contexts)
            tc.pop("error", None)
        except Exception as e:
            logger.error(f"Patch failed for '{question[:50]}': {e}")
            print(f"Failed for question '{question[:50]}': {e}")
            tc["response"] = ""
            tc["retrieved_contexts"] = []
            tc["error"] = str(e)
            still_failed += 1

    merged = merge_into_raw(raw, missing)
    meta = merged["meta"]
    meta["timestamp"] = datetime.now().isoformat()
    meta["patched_at"] = meta["timestamp"]
    meta["patched_from_testset"] = str(testset_path)
    meta["test_cases"] = meta.get("test_cases") or len(testset)
    save_raw_checkpoint(raw_path, merged)

    ok = len(missing) - still_failed
    print(
        f"Patched {ok}/{len(missing)} question(s). "
        f"raw now has {merged['meta']['valid_cases']} valid, "
        f"{merged['meta']['failed_cases']} still failed."
    )
    if still_failed:
        print("Re-run patch-raw after the LLM recovers.")
        sys.exit(1)


def cmd_rescore(args):
    """Score-only: read immutable raw/report, never re-run retrieve/generate."""
    from backend.infrastructure.config import settings

    report_path = resolve_path(args.report)
    if not report_path.exists():
        print(f"Error: Report not found: {report_path}")
        sys.exit(1)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    cases = cases_from_report(report)
    if not cases:
        print(f"Error: No cases with responses and contexts in {report_path}")
        sys.exit(1)

    judge_model, temperature, parse_retries, score_repeats = _resolve_judge_args(
        args, report.get("meta", {}),
    )
    print(
        f"Re-scoring {len(cases)} cases from {report_path} "
        f"(judge={judge_model}, temp={temperature}, "
        f"parse_retries={parse_retries}, score_repeats={score_repeats})"
    )

    print("\nChecking judge model responds with parseable JSON...")
    try:
        probe_judge_llm(judge_model, temperature=temperature)
    except RuntimeError as e:
        print(f"Error: judge check failed: {e}")
        sys.exit(1)
    print("Judge check passed.")

    try:
        matrix = score_with_repeats(
            cases,
            judge_model=judge_model,
            temperature=temperature,
            parse_retries=parse_retries,
            score_repeats=score_repeats,
        )
    except RuntimeError as e:
        print(f"\nError: scoring failed: {e}")
        sys.exit(1)

    output_path = resolve_path(args.output) if args.output else report_path
    # 不覆盖 .raw.json：若 --report 本身是 raw，默认写到去 .raw 的报告路径
    if output_path == report_path and report_path.name.endswith(".raw.json"):
        output_path = report_path.with_name(
            report_path.name.replace(".raw.json", ".json")
        )
        print(f"Raw input preserved; writing scored report to {output_path}")

    meta = dict(report.get("meta", {}))
    meta.pop("raw_checkpoint", None)
    meta.update({
        "timestamp": datetime.now().isoformat(),
        "rescored": True,
        "rescored_from": str(report_path),
        "report_path": str(output_path),
        "generator_model": meta.get("generator_model") or settings.LLM_MODEL,
    })
    if matrix.pin:
        meta.update(matrix.pin.as_meta())

    new_report = build_report(meta, cases, matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_report, f, ensure_ascii=False, indent=2)

    print_report(new_report)


def cmd_delivery(args):
    """Two-layer scoring: child rerank vs production filter/dedup delivery."""
    from .delivery_eval import run_delivery_from_paths

    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    from .retrieval_eval import load_retrieval_checkpoint
    try:
        checkpoint = load_retrieval_checkpoint(checkpoint_path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    testset_arg = args.testset
    if not testset_arg:
        testset_arg = checkpoint.get("manifest", {}).get("testset", {}).get("path")
    if not testset_arg:
        print("Error: --testset required (checkpoint has no testset path)")
        sys.exit(1)
    testset_path = resolve_path(testset_arg)
    if not testset_path.exists():
        print(f"Error: Test set not found: {testset_path}")
        sys.exit(1)
    try:
        cases = load_golden_testset(testset_path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    output_path = resolve_path(args.output) if args.output else checkpoint_path.with_name(
        checkpoint_path.name.replace(".retrieval.json", "") + "-delivery.json"
        if checkpoint_path.name.endswith(".retrieval.json")
        else checkpoint_path.stem + "-delivery.json"
    )
    result = run_delivery_from_paths(
        checkpoint_path,
        cases,
        output_path,
        expand=not args.no_expand,
        load_parents=not args.no_parents,
    )
    pc = result.get("parent_child") or {}
    print("=== Two-layer delivery scoring ===")
    print(f"Checkpoint: {checkpoint_path}")
    if pc:
        print(
            f"Parent==child: {pc.get('identical_parent_child')}/{pc.get('n_chunks')} "
            f"(median len ratio {pc.get('median_len_ratio')})"
        )
    cs = result["child"]["summary"]
    ds = result["delivered"]["summary"]
    print(
        f"child     recall@5={cs.get('recall@5')} MRR={cs.get('mrr')} "
        f"fail@20={cs.get('failure_rate@20')}"
    )
    print(
        f"delivered recall@5={ds.get('recall@5')} MRR={ds.get('mrr')} "
        f"fail@20={ds.get('failure_rate@20')}"
    )
    print(f"Questions losing gold after filter/dedup: {result['n_dropped_questions']}")
    for d in result["dropped"][:10]:
        lost = ", ".join(f"{x['source']}:{x['chunk_index']}" for x in d["lost_gold"])
        print(f"  #{d['index']} [{d['question_type']}] lost {lost}")
    print(f"\nReport saved to {output_path}")
    print(f"Markdown saved to {Path(output_path).with_suffix('.md')}")


def cmd_cutoff(args):
    """Simulate FINAL_CONTEXT_K cutoffs from a retrieval checkpoint (no re-retrieval)."""
    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    try:
        checkpoint = load_retrieval_checkpoint(checkpoint_path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    testset_arg = args.testset
    if not testset_arg:
        testset_arg = checkpoint.get("manifest", {}).get("testset", {}).get("path")
    if not testset_arg:
        print("Error: --testset required (checkpoint has no testset path)")
        sys.exit(1)
    testset_path = resolve_path(testset_arg)
    if not testset_path.exists():
        print(f"Error: Test set not found: {testset_path}")
        sys.exit(1)
    try:
        cases = load_golden_testset(testset_path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    final_ks = tuple(sorted({k for k in args.final_ks if k > 0}))
    if not final_ks:
        print("Error: --final-ks must list at least one positive k")
        sys.exit(1)
    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            print(f"Error: --stages must be a subset of {STAGES}, got {args.stages!r}")
            sys.exit(1)

    try:
        result = simulate_final_ks(checkpoint, cases, final_ks, stages=stages)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    output_path = resolve_path(args.output) if args.output else checkpoint_path.with_name(
        checkpoint_path.name.replace(".retrieval.json", "") + "-cutoff.json"
        if checkpoint_path.name.endswith(".retrieval.json")
        else checkpoint_path.stem + "-cutoff.json"
    )
    write_cutoff_report(result, checkpoint_path, output_path)
    print_cutoff_report(result)
    print(f"\nCutoff report saved to {output_path}")
    print(f"Markdown table saved to {Path(output_path).with_suffix('.md')}")


def cmd_retrieval(args):
    """Deterministic retrieval evaluation: stage-wise ranked lists + tier-1 metrics.

    No LLM calls. The golden set must carry gold chunk IDs (see `check-gold`).
    Writes three files next to --output: the metrics JSON report, a Markdown
    baseline table, and a `<report>.retrieval.json` checkpoint with the raw
    ranked lists (first repeat only) for recomputation without re-running
    retrieval.
    """
    from backend.infrastructure.config import settings

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
    print(f"Embedding device: {settings.EMBEDDING_DEVICE}")
    if settings.EMBEDDING_DEVICE == "cuda":
        import torch
        if not torch.cuda.is_available():
            print(
                "Error: EMBEDDING_DEVICE=cuda but torch.cuda.is_available() is False. "
                "This venv likely has a CPU-only torch wheel; install a CUDA build, e.g.\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu126"
            )
            sys.exit(1)
        print(f"CUDA: {torch.cuda.get_device_name(0)}")

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
    requires every gold triple to exist in the pgvector table.
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
            print(f"Live index check: {len(missing)} gold chunk(s) missing from pgvector:")
            for item in missing:
                print(f"  - {item}")
            sys.exit(1)
        print("Live index check OK: every gold chunk exists in pgvector.")


def cmd_retrieval_gate(args):
    """词法重放样本题库，Recall@k / nDCG 低于基线门槛时以非 0 退出。"""
    from .retrieval_gate import run_retrieval_gate

    result = run_retrieval_gate(
        testset_path=args.testset,
        corpus_dir=args.corpus,
        baseline_path=args.baseline,
    )
    print(result.render())
    if not result.ok:
        sys.exit(1)


_BARE_NOTE = (
    "\nRegression guard: ragmate-eval retrieval-gate\n"
    "RAGAS commands (generate, evaluate, rescore, patch-raw, interactive) are shelved. "
    "They are not the CI gate. Python 3.14 RAGAS issues are out of scope.\n"
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "RagMate retrieval regression CLI. The gate is retrieval-gate. "
            "RAGAS scoring commands are shelved."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="[shelved] RAGAS test-set generation")
    gen_parser.add_argument("--size", type=int, default=50, help="Number of test cases to generate")
    gen_parser.add_argument("--max-docs", type=int, default=None, help="Max documents to load")
    gen_parser.add_argument("--output", type=str, default=str(DEFAULT_TESTSETS_DIR / "testset.json"), help="Output JSON path")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    gen_parser.add_argument("--distributions", type=str, default=None, help="Test type distributions (JSON)")

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="[shelved] RAGAS judge scoring; not the CI gate",
    )
    eval_parser.add_argument("--testset", type=str, required=True, help="Test set JSON path")
    eval_parser.add_argument("--report", type=str, default=None, help="Report output path")
    eval_parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    eval_parser.add_argument("--threshold", type=float, default=None, help="Min overall score (exit non-zero if below)")
    eval_parser.add_argument("--judge-model", type=str, default=None, help="Judge LLM model name")
    eval_parser.add_argument(
        "--judge-temperature", type=float, default=None,
        help=f"Judge temperature (default {DEFAULT_JUDGE_TEMPERATURE}; pinned in meta)",
    )
    eval_parser.add_argument(
        "--parse-retries", type=int, default=DEFAULT_PARSE_RETRIES,
        help="Extra full scoring passes to fill NaN/parse failures (default 2)",
    )
    eval_parser.add_argument(
        "--score-repeats", type=int, default=1,
        help="Independent scoring repeats for noise spread (default 1)",
    )
    eval_parser.add_argument(
        "--generate-only", action="store_true",
        help="Only retrieve+generate; write .raw.json and exit (score with rescore)",
    )
    eval_parser.add_argument(
        "--force-raw", action="store_true",
        help="Overwrite an existing .raw.json checkpoint",
    )
    eval_parser.add_argument("--mode", type=str, choices=["single", "agent"], default="single", help="Evaluation mode")

    patch_parser = subparsers.add_parser(
        "patch-raw",
        help="[shelved] Fill RAGAS raw-checkpoint gaps",
    )
    patch_parser.add_argument(
        "--raw", type=str, required=True,
        help="Existing raw checkpoint path (.raw.json)",
    )
    patch_parser.add_argument(
        "--testset", type=str, required=True,
        help="Golden/test set used for the original generate run",
    )
    patch_parser.add_argument(
        "--top-k", type=int, default=None,
        help="Retrieve top-k (default: raw meta.top_k or FINAL_CONTEXT_K)",
    )

    rescore_parser = subparsers.add_parser(
        "rescore",
        help="[shelved] RAGAS score-only from a raw checkpoint",
    )
    rescore_parser.add_argument(
        "--report", type=str, required=True,
        help="Raw checkpoint (.raw.json) or scored report JSON",
    )
    rescore_parser.add_argument(
        "--output", type=str, default=None,
        help="Scored report path (defaults beside raw; never overwrites .raw.json)",
    )
    rescore_parser.add_argument(
        "--judge-model", type=str, default=None,
        help="Judge LLM model (defaults to pinned meta / settings)",
    )
    rescore_parser.add_argument(
        "--judge-temperature", type=float, default=None,
        help=f"Judge temperature (default: pinned meta or {DEFAULT_JUDGE_TEMPERATURE})",
    )
    rescore_parser.add_argument(
        "--parse-retries", type=int, default=DEFAULT_PARSE_RETRIES,
        help="Extra full scoring passes to fill NaN/parse failures (default 2)",
    )
    rescore_parser.add_argument(
        "--score-repeats", type=int, default=1,
        help="Independent scoring repeats for noise spread (default 1; use 3 for P1)",
    )

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
    check_gold_parser.add_argument("--live", action="store_true", help="Also verify gold chunks exist in the live pgvector table")

    gate_parser = subparsers.add_parser(
        "retrieval-gate",
        help="Regression guard: lexical replay vs pinned Recall@k / nDCG floors",
    )
    gate_parser.add_argument(
        "--testset", type=str, default=None,
        help="Golden test set (default: eval/testsets/sample/testset.json)",
    )
    gate_parser.add_argument(
        "--corpus", type=str, default=None,
        help="Corpus directory (default: eval/testsets/sample/corpus)",
    )
    gate_parser.add_argument(
        "--baseline", type=str, default=None,
        help="Committed retrieval report to pin floors from "
             "(default: eval/reports/baseline/retrieval-baseline-98q-titlepath-rerun.json, fused stage)",
    )

    delivery_parser = subparsers.add_parser(
        "delivery",
        help="Two-layer scoring: child rerank vs production filter/dedup (from checkpoint)",
    )
    delivery_parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to <report>.retrieval.json ranked-list checkpoint",
    )
    delivery_parser.add_argument(
        "--testset", type=str, default=None,
        help="Golden test set (defaults to path recorded in the checkpoint manifest)",
    )
    delivery_parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: <checkpoint-stem>-delivery.json)",
    )
    delivery_parser.add_argument(
        "--no-expand", action="store_true",
        help="Skip parent expand after filter/dedup (IDs unchanged either way)",
    )
    delivery_parser.add_argument(
        "--no-parents", action="store_true",
        help="Do not load parent_text from pgvector (skip parent==child stats)",
    )

    cutoff_parser = subparsers.add_parser(
        "cutoff",
        help="Simulate FINAL_CONTEXT_K cutoffs from a .retrieval.json checkpoint (no re-retrieval)",
    )
    cutoff_parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to <report>.retrieval.json ranked-list checkpoint",
    )
    cutoff_parser.add_argument(
        "--testset", type=str, default=None,
        help="Golden test set (defaults to path recorded in the checkpoint manifest)",
    )
    cutoff_parser.add_argument(
        "--final-ks", type=int, nargs="+", default=[5, 10, 15],
        help="Final context window sizes to simulate (default: 5 10 15)",
    )
    cutoff_parser.add_argument(
        "--stages", type=str, default=None,
        help="Comma-separated stages subset (default: stages in the checkpoint)",
    )
    cutoff_parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: <checkpoint-stem>-cutoff.json)",
    )

    subparsers.add_parser(
        "interactive",
        help="[shelved] RAGAS interactive menu",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        print(_BARE_NOTE)
        return

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "patch-raw":
        cmd_patch_raw(args)
    elif args.command == "rescore":
        cmd_rescore(args)
    elif args.command == "retrieval":
        cmd_retrieval(args)
    elif args.command == "check-gold":
        cmd_check_gold(args)
    elif args.command == "retrieval-gate":
        cmd_retrieval_gate(args)
    elif args.command == "delivery":
        cmd_delivery(args)
    elif args.command == "cutoff":
        cmd_cutoff(args)
    elif args.command == "interactive":
        from .interactive import interactive_menu
        interactive_menu()
