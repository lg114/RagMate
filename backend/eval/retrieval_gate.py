"""检索质量 CI 门禁：词法重放 130 题，对照已入库的 fused 基线。

线上 `ragmate-eval retrieval` 要嵌入模型、pgvector 和 reranker，不适合
ubuntu-latest 每次 PR 都跑。这里改走三条已有、可复现的链路：

1. `build_chunk_inventory` — 用当前切块配置从样本语料复现 chunk；
2. `LexicalIndex` — 确定性 BM25，不访问网络；
3. `evaluate_questions` — 与检索基线同一套 Recall@k / nDCG。

阈值不是另写一套“看起来很高”的数字。它们等于基线报告 fused 阶段的
Recall@k / nDCG 减去固定余量 `ALLOWANCE`。余量只覆盖词法重放和
Jina 混合检索之间大约一题的差距（121 道可答题里 1 题约占 0.0083）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .deterministic_metrics import evaluate_questions, normalize_key
from .lexical_retrieval import LexicalIndex
from .retrieval_eval import (
    build_chunk_inventory,
    load_golden_testset,
    verify_gold_in_inventory,
)
from .utils import resolve_path

# 当前默认切块（title_path）下的 130 题 Jina 远程混合检索重跑。reranked 依赖
# cross-encoder，CI 的词法重放对不齐那一臂，所以钉 fused。
DEFAULT_BASELINE = "eval/reports/baseline/retrieval-baseline-130q-jina-rerun.json"
DEFAULT_TESTSET = "eval/testsets/sample/testset.json"
DEFAULT_CORPUS = "eval/testsets/sample/corpus"
BASELINE_STAGE = "fused"
GATED_METRICS = (
    "recall@5",
    "ndcg@5",
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
)
# 0.015 = 150 个万分点，略大于 1/91。再掉一整道可答题就会低于门槛。
ALLOWANCE = 0.015
KS = (5, 10, 20)


def floor_metric(published: float, allowance: float = ALLOWANCE) -> float:
    """基线指标（4 位小数）减去余量，再落到 4 位小数。"""
    published_units = int(round(float(published) * 10000))
    allowance_units = int(round(float(allowance) * 10000))
    return (published_units - allowance_units) / 10000


def floors_from_published(published: dict, allowance: float = ALLOWANCE) -> dict:
    """从基线 summary 抽出门禁要看的指标并换成门槛。"""
    missing = [name for name in GATED_METRICS if published.get(name) is None]
    if missing:
        raise ValueError(f"Baseline summary missing metrics: {missing}")
    return {name: floor_metric(published[name], allowance) for name in GATED_METRICS}


def load_baseline_report(path=None) -> dict:
    """读已入库的检索基线报告。"""
    report_path = resolve_path(str(path or DEFAULT_BASELINE))
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    if "stages" not in report or "meta" not in report:
        raise ValueError(f"Not a retrieval baseline report: {report_path}")
    return report


def published_summary(report: dict, stage: str = BASELINE_STAGE) -> dict:
    """取出某一阶段的 summary。阶段不存在时抛 ValueError。"""
    try:
        return report["stages"][stage]["summary"]
    except KeyError as exc:
        raise ValueError(f"Baseline report has no stage {stage!r}") from exc


def check_baseline_consistency(report: dict, stage: str = BASELINE_STAGE) -> list[str]:
    """重算 per_question 均值，确认报告里的 summary 不是手改的。"""
    try:
        stage_data = report["stages"][stage]
    except KeyError:
        return [f"baseline report has no stage {stage!r}"]
    per_question = stage_data.get("per_question") or []
    answerable = [row for row in per_question if not row.get("unanswerable")]
    if not answerable:
        return ["baseline report has no answerable questions"]
    summary = stage_data.get("summary") or {}
    errors = []
    metric_names = list(answerable[0].get("scores") or {})
    for name in metric_names:
        values = []
        for row in answerable:
            score = (row.get("scores") or {}).get(name)
            if score is None:
                errors.append(f"baseline {stage} question {row.get('index')} missing {name}")
                continue
            values.append(score)
        if len(values) != len(answerable):
            continue
        recomputed = round(sum(values) / len(values), 4)
        published = summary.get(name)
        if published != recomputed:
            errors.append(
                f"baseline {stage} {name} summary {published} != mean of per_question {recomputed}"
            )
    if summary.get("recall@20") is not None:
        expected_failure = round(1.0 - summary["recall@20"], 4)
        if summary.get("failure_rate@20") != expected_failure:
            errors.append(
                f"baseline {stage} failure_rate@20 {summary.get('failure_rate@20')} "
                f"!= 1 - recall@20 ({expected_failure})"
            )
    reported_n = stage_data.get("n_answerable")
    if reported_n is not None and reported_n != len(answerable):
        errors.append(
            f"baseline {stage} n_answerable {reported_n} != {len(answerable)} scored rows"
        )
    return errors


def check_question_alignment(cases: list[dict], report: dict, stage: str = BASELINE_STAGE) -> list[str]:
    """题干顺序必须和基线报告一致，否则门槛不再对得上这套题。"""
    try:
        rows = report["stages"][stage]["per_question"]
    except KeyError:
        return [f"baseline report has no per_question for stage {stage!r}"]
    if len(cases) != len(rows):
        return [
            f"testset has {len(cases)} questions but baseline {stage} has {len(rows)}; "
            "regenerate the baseline before moving the gate"
        ]
    errors = []
    for index, (case, row) in enumerate(zip(cases, rows)):
        if case.get("user_input") != row.get("user_input"):
            errors.append(
                f"question {index} text does not match the baseline report; "
                "regenerate the baseline after editing the test set"
            )
            if len(errors) >= 3:
                break
    return errors


def chunks_from_corpus(corpus_dir) -> tuple[dict, list[dict]]:
    """用生产切块复现清单，并转成带 `(source, page, chunk_index)` 的列表。"""
    inventory = build_chunk_inventory(corpus_dir)
    chunks = []
    for (source, chunk_index), text in inventory.items():
        chunks.append({
            "key": (source, None, chunk_index),
            "text": text,
        })
    return inventory, chunks


def evaluate_lexical(cases: list[dict], chunks: list[dict], ranker=None) -> dict:
    """对每道题排序并汇总确定性指标。`ranker(query, index)` 可注入以便测试回归。"""
    index = LexicalIndex(chunks)
    items = []
    for case in cases:
        if ranker is None:
            ranked = index.rank(case.get("user_input", ""))
        else:
            ranked = ranker(case.get("user_input", ""), index)
        items.append({
            "user_input": case.get("user_input", ""),
            "source_doc": case.get("source_doc", ""),
            "question_type": case.get("question_type", ""),
            "ranked": [normalize_key(key) for key in ranked],
            "gold": [normalize_key(gold) for gold in case.get("gold_chunks", [])],
        })
    return evaluate_questions(items, KS)


def compare_to_floors(summary: dict, floors: dict) -> list[str]:
    """指标低于门槛则返回失败说明。门槛本身来自基线减余量。"""
    failures = []
    for name, floor in floors.items():
        observed = summary.get(name)
        if observed is None:
            failures.append(f"{name} missing from retrieval summary")
            continue
        if observed < floor:
            failures.append(f"{name} {observed:.4f} < floor {floor:.4f}")
    return failures


@dataclass
class GateResult:
    ok: bool
    failures: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    floors: dict = field(default_factory=dict)
    published: dict = field(default_factory=dict)
    n_questions: int = 0
    n_answerable: int = 0
    n_chunks: int = 0
    baseline_path: str = ""
    stage: str = BASELINE_STAGE

    def render(self) -> str:
        """人读的通过/失败报告。"""
        lines = [
            "Retrieval quality gate (lexical replay; no pgvector, no BGE, no GPU)",
            f"Baseline: {self.baseline_path} stage={self.stage}",
            f"Allowance below baseline: {ALLOWANCE:.4f}",
            f"Questions: {self.n_questions} ({self.n_answerable} answerable), chunks: {self.n_chunks}",
            "",
        ]
        names = list(self.floors) or list(GATED_METRICS)
        for name in names:
            observed = self.summary.get(name)
            floor = self.floors.get(name)
            published = self.published.get(name)
            observed_text = f"{observed:.4f}" if isinstance(observed, (int, float)) else "missing"
            floor_text = f"{floor:.4f}" if isinstance(floor, (int, float)) else "missing"
            published_text = f"{published:.4f}" if isinstance(published, (int, float)) else "missing"
            metric_failed = any(
                item.startswith(f"{name} ") or item.startswith(f"{name} missing")
                for item in self.failures
            )
            if metric_failed:
                status = "FAIL"
            elif self.ok:
                status = "OK"
            else:
                status = "--"
            lines.append(
                f"  {name:<12} observed={observed_text}  floor={floor_text}  "
                f"baseline={published_text}  {status}"
            )
        lines.append("")
        if self.ok:
            lines.append("PASSED")
        else:
            lines.append("FAILED")
            for item in self.failures:
                lines.append(f"  - {item}")
        return "\n".join(lines)


def run_retrieval_gate(
    testset_path=None,
    corpus_dir=None,
    baseline_path=None,
    *,
    stage: str = BASELINE_STAGE,
    allowance: float = ALLOWANCE,
    ranker=None,
) -> GateResult:
    """跑完整门禁。不调用 `sys.exit`；调用方看 `result.ok`。"""
    testset = resolve_path(str(testset_path or DEFAULT_TESTSET))
    corpus = resolve_path(str(corpus_dir or DEFAULT_CORPUS))
    baseline = resolve_path(str(baseline_path or DEFAULT_BASELINE))
    result = GateResult(ok=False, baseline_path=str(baseline), stage=stage)

    try:
        cases = load_golden_testset(testset)
        report = load_baseline_report(baseline)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result.failures = [str(exc)]
        return result

    result.n_questions = len(cases)
    published = {}
    try:
        published = published_summary(report, stage)
        result.published = {name: published[name] for name in GATED_METRICS}
        result.floors = floors_from_published(published, allowance)
    except (KeyError, ValueError) as exc:
        result.failures = [str(exc)]
        return result

    failures = []
    failures.extend(check_baseline_consistency(report, stage))
    failures.extend(check_question_alignment(cases, report, stage))
    if failures:
        result.failures = failures
        return result

    try:
        inventory, chunks = chunks_from_corpus(corpus)
    except (OSError, ValueError) as exc:
        result.failures = [str(exc)]
        return result
    result.n_chunks = len(chunks)
    missing_gold = verify_gold_in_inventory(cases, inventory)
    if missing_gold:
        result.failures = missing_gold
        return result

    evaluation = evaluate_lexical(cases, chunks, ranker=ranker)
    result.summary = evaluation["summary"]
    result.n_answerable = evaluation["n_answerable"]
    failures.extend(compare_to_floors(result.summary, result.floors))
    result.failures = failures
    result.ok = not failures
    return result
