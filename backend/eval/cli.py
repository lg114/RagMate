"""评估 CLI：确定性检索评测与回归门禁。"""
import argparse
import json
import sys
from pathlib import Path

from .utils import DEFAULT_REPORTS_DIR, resolve_path
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


def cmd_chunks(args):
    """列出生产切分复现的 chunk 清单，供标注者读文本定 gold。离线，不碰数据库。"""
    from .annotate import list_chunk_rows

    corpus_dir = resolve_path(args.corpus)
    try:
        inventory = build_chunk_inventory(corpus_dir)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    rows = list_chunk_rows(inventory, source=args.source)
    if not rows:
        if args.source:
            print(f"Error: no chunks for source {args.source!r} in {corpus_dir}")
        else:
            print(f"Error: no chunks reproduced from {corpus_dir}")
        sys.exit(1)
    print(f"Chunk inventory from {corpus_dir}: {len(rows)} chunks.")
    for row in rows:
        print(f"\n{row['source']}:{row['chunk_index']}  ({row['chars']} chars)")
        if args.full:
            print(inventory[(row["source"], row["chunk_index"])])
        else:
            print(f"  {row['preview']}")


def cmd_annotate(args):
    """把一道人工标注的题追加进 testset：校验通过才写盘。"""
    from .annotate import append_case, build_case, parse_gold_spec

    try:
        gold = [parse_gold_spec(spec) for spec in args.gold]
        case = build_case(
            question=args.question,
            reference=args.reference,
            question_type=args.question_type,
            gold=gold,
            claims=args.claims,
            source_doc=args.source_doc,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    testset_path = resolve_path(args.testset)
    if not testset_path.exists():
        print(f"Error: Test set not found: {testset_path}")
        sys.exit(1)
    try:
        updated = append_case(
            testset_path, case, resolve_path(args.corpus), dry_run=args.dry_run
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.dry_run:
        print("Dry run: not written. Case that would be appended:")
        print(json.dumps(case, ensure_ascii=False, indent=2))
        return
    print(f"Appended case #{len(updated) - 1} to {testset_path} "
          f"({len(updated)} cases total).")
    print("Next steps before the gate passes again:")
    print("  1. ragmate-eval check-gold --testset <path> [--live after re-ingest]")
    print("  2. Re-ingest the corpus into pgvector, then rerun")
    print("     ragmate-eval retrieval and replace the pinned baseline report.")
    print("  3. ragmate-eval retrieval-gate")


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


_BARE_NOTE = "\nRegression guard: ragmate-eval retrieval-gate\n"


def main():
    parser = argparse.ArgumentParser(
        description="RagMate retrieval regression CLI. The gate is retrieval-gate.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    chunks_parser = subparsers.add_parser(
        "chunks",
        help="List chunks reproduced by production chunking (read them to pick gold IDs)",
    )
    chunks_parser.add_argument(
        "--corpus", type=str, default="eval/testsets/sample/corpus",
        help="Corpus directory used to reproduce chunking",
    )
    chunks_parser.add_argument(
        "--source", type=str, default=None,
        help="Only list chunks from this corpus file (basename)",
    )
    chunks_parser.add_argument(
        "--full", action="store_true",
        help="Print full chunk text instead of a 120-char preview",
    )

    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Append one human-annotated case to a golden test set (validated, atomic)",
    )
    annotate_parser.add_argument(
        "--testset", type=str, required=True,
        help="Golden test set JSON path to append to",
    )
    annotate_parser.add_argument(
        "--corpus", type=str, default="eval/testsets/sample/corpus",
        help="Corpus directory used to verify gold chunk IDs exist",
    )
    annotate_parser.add_argument("--question", type=str, required=True)
    annotate_parser.add_argument("--reference", type=str, required=True)
    annotate_parser.add_argument(
        "--type", dest="question_type", type=str, required=True,
        choices=["factoid", "multi-hop", "paraphrase", "ambiguous", "unanswerable"],
        help="Question type; unanswerable takes no --gold",
    )
    annotate_parser.add_argument(
        "--gold", action="append", type=str, default=[],
        help="Gold chunk as source:chunk_index, repeatable (e.g. 01-docker-compose.md:2)",
    )
    annotate_parser.add_argument(
        "--claims", action="append", type=str, required=True,
        help="Expected claim, repeatable (at least one)",
    )
    annotate_parser.add_argument(
        "--source-doc", type=str, default=None,
        help="Defaults to the single gold source, or '' for unanswerable",
    )
    annotate_parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate only, do not write the test set",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        print(_BARE_NOTE)
        return

    args = parser.parse_args()

    if args.command == "retrieval":
        cmd_retrieval(args)
    elif args.command == "check-gold":
        cmd_check_gold(args)
    elif args.command == "retrieval-gate":
        cmd_retrieval_gate(args)
    elif args.command == "delivery":
        cmd_delivery(args)
    elif args.command == "cutoff":
        cmd_cutoff(args)
    elif args.command == "chunks":
        cmd_chunks(args)
    elif args.command == "annotate":
        cmd_annotate(args)
