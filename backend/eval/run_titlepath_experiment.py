"""P1：title_path 切块 vs 基线 parent_child（独立 collection，同一 98 题）。

用法（仓库根目录，已激活 venv）：
  python -m backend.eval.run_titlepath_experiment
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
COLLECTION = "ragmate_docs__titlepath"
REPORT = ROOT / "eval" / "reports" / "baseline" / "retrieval-baseline-98q-titlepath.json"
BASELINE_CP = ROOT / "eval" / "reports" / "baseline" / "retrieval-baseline-98q.retrieval.json"


def main() -> None:
    from backend.infrastructure.config import settings
    from backend.infrastructure.vector_store import reset_vector_store_cache
    from backend.eval.corpus_ingest import ingest_corpus_to_collection
    from backend.eval.retrieval_eval import (
        build_retrieval_report,
        load_golden_testset,
        print_retrieval_report,
        run_retrieval_eval,
        simulate_final_ks,
        verify_gold_in_index,
        write_cutoff_report,
        write_retrieval_checkpoint,
        write_retrieval_markdown,
    )
    from backend.eval.manifest import build_manifest

    skip_ingest = "--skip-ingest" in sys.argv
    settings.CHUNKING_STRATEGY = "title_path"
    settings.VECTOR_TABLE = COLLECTION
    reset_vector_store_cache()
    if skip_ingest:
        print("=== 1) Skip ingest (--skip-ingest); using existing", COLLECTION)
    else:
        print("=== 1) Ingest corpus with title_path →", COLLECTION)
        info = ingest_corpus_to_collection(CORPUS, COLLECTION)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        if not info.get("row_count"):
            print("ERROR: ingest reported 0 searchable rows; abort")
            sys.exit(1)

    print("\n=== 2) check-gold --live (same testset; chunk ids align) ===")
    settings.VECTOR_TABLE = COLLECTION
    reset_vector_store_cache()
    cases = load_golden_testset(TESTSET)
    missing = verify_gold_in_index(cases)
    if missing:
        print("FAIL live gold check:")
        for m in missing[:20]:
            print(" ", m)
        sys.exit(1)
    print(f"Live index OK: {len(cases)} cases against {COLLECTION}")

    print("\n=== 3) retrieval eval --repeat 1 ===")
    stages = ["dense", "sparse", "fused", "reranked"]
    ks = (5, 10, 20)
    manifest = build_manifest(
        testset_path=TESTSET,
        stages=stages,
        over_fetch=50,
        ks=list(ks),
        repeat=1,
        n_questions=len(cases),
    )
    manifest["report_path"] = str(REPORT)
    manifest["retrieval_config"]["CHUNKING_STRATEGY"] = settings.CHUNKING_STRATEGY
    manifest["retrieval_config"]["VECTOR_TABLE"] = settings.VECTOR_TABLE
    reset_vector_store_cache()
    result = run_retrieval_eval(cases, stages, 50, 1, ks, manifest)
    cp_path = write_retrieval_checkpoint(result["checkpoint"], REPORT)
    report = build_retrieval_report(manifest, result["evaluations"], cp_path, stages)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_retrieval_markdown(report, REPORT)
    print_retrieval_report(report)

    print("\n=== 4) cutoff 5/10/15 ===")
    from backend.eval.retrieval_eval import load_retrieval_checkpoint

    cutoff = simulate_final_ks(
        load_retrieval_checkpoint(cp_path), cases, (5, 10, 15), stages=stages
    )
    cutoff_path = REPORT.with_name("retrieval-baseline-98q-titlepath-cutoff.json")
    write_cutoff_report(cutoff, cp_path, cutoff_path)

    print("\n=== 5) Q67 rank comparison (baseline vs title_path, reranked) ===")
    if BASELINE_CP.exists():
        base = json.loads(BASELINE_CP.read_text(encoding="utf-8"))
        neo = json.loads(cp_path.read_text(encoding="utf-8"))
        gold_keys = {
            (g["source"], g["page"], g["chunk_index"])
            for g in cases[67]["gold_chunks"]
        }

        def ranks(cp, stage):
            out = []
            for r in cp["stages"]["67"][stage]:
                key = (r["source"], r["page"], r["chunk_index"])
                if key in gold_keys:
                    out.append((r["chunk_index"], r["rank"]))
            return out

        print("Q:", cases[67]["user_input"])
        for stage in stages:
            print(f"  {stage:10} baseline={ranks(base, stage)}  titlepath={ranks(neo, stage)}")
    else:
        print("Baseline checkpoint missing; skip Q67 diff")

    # Restore defaults so interactive shells aren't left on the experiment collection
    settings.CHUNKING_STRATEGY = "parent_child"
    settings.VECTOR_TABLE = "ragmate_docs"
    reset_vector_store_cache()
    print("\nDone. Reports:", REPORT, cutoff_path)


if __name__ == "__main__":
    main()
