"""RAGAS 指标计算：judge 钉死、可重复打分、解析失败重试后标 unscored。"""
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field

from backend.infrastructure.streaming_llm import ChatOpenAICompatible

logger = logging.getLogger("ragmate")

DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_PARSE_RETRIES = 2
METRIC_PACK = (
    "Faithfulness",
    "AnswerRelevancy",
    "ContextPrecision",
    "ContextRecall",
    "FactualCorrectness",
)


class EvalLLM(ChatOpenAICompatible):
    """Subclass that forces n=1 (some APIs like MiMo don't support n>1)."""
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload.pop("n", None)
        return payload


@dataclass
class JudgePin:
    """钉死的 judge 配置，写入 raw / 报告 meta，便于 rescore 复现。"""
    model: str
    temperature: float
    metric_pack: str
    metric_pack_sha256: str
    ragas_version: str | None = None

    def as_meta(self) -> dict:
        return {
            "judge_model": self.model,
            "judge_temperature": self.temperature,
            "judge_metric_pack": self.metric_pack,
            "judge_metric_pack_sha256": self.metric_pack_sha256,
            "ragas_version": self.ragas_version,
        }


@dataclass
class ScoreMatrix:
    """一次或多次打分的结果矩阵（与 test_cases 对齐）。"""
    runs: list[list[dict]] = field(default_factory=list)  # [run][case][metric] -> float|None
    pin: JudgePin | None = None

    @property
    def primary(self) -> list[dict]:
        """报告主分数：多遍独立打分用均值，单遍（含解析重试合并）原样。"""
        if len(self.runs) > 1:
            return mean_score_runs(self.runs)
        return merge_score_runs(self.runs)


def _ragas_version() -> str | None:
    try:
        import ragas
        return getattr(ragas, "__version__", None)
    except Exception:
        return None


def build_judge_pin(judge_model: str | None = None, temperature: float | None = None) -> JudgePin:
    from backend.infrastructure.config import settings

    model = judge_model or settings.LLM_MODEL
    temp = DEFAULT_JUDGE_TEMPERATURE if temperature is None else float(temperature)
    pack = "+".join(METRIC_PACK)
    digest = hashlib.sha256(pack.encode("utf-8")).hexdigest()[:16]
    return JudgePin(
        model=model,
        temperature=temp,
        metric_pack=pack,
        metric_pack_sha256=digest,
        ragas_version=_ragas_version(),
    )


def build_judge_llm(judge_model: str = None, temperature: float | None = None):
    """Build the RAGAS judge LLM wrapper with pinned temperature."""
    from ragas.llms.base import LangchainLLMWrapper
    from backend.infrastructure.model_factory import get_llm
    from backend.infrastructure.streaming_llm import create_llm
    from backend.infrastructure.config import settings

    temp = DEFAULT_JUDGE_TEMPERATURE if temperature is None else float(temperature)

    if judge_model:
        base_llm = create_llm(
            model=judge_model,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_API_BASE_URL,
            temperature=temp,
        )
    else:
        base_llm = get_llm()

    return LangchainLLMWrapper(EvalLLM(
        model=base_llm.model_name,
        api_key=base_llm.openai_api_key,
        base_url=base_llm.openai_api_base,
        temperature=temp,
    ))


_PROBE_PROMPT = 'Return exactly this JSON object and nothing else: {"verdict": "yes"}'


def _extract_json_object(text: str):
    """Parse a JSON object out of raw LLM output, tolerating markdown fences."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        body = lines[1:-1] if len(lines) >= 2 and lines[-1].strip() == "```" else lines[1:]
        s = "\n".join(body).strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(s[start:end + 1])


def probe_judge(wrapper, model_name: str = "") -> None:
    """Fail fast if the judge cannot return parseable JSON.

    RAGAS swallows per-row scoring failures as NaN (evaluate() defaults to
    raise_exceptions=False), so a judge that cannot follow the JSON prompts
    produces an all-empty report after a full pipeline run. This probe checks
    the exact capability scoring needs, with one tiny call, before any
    expensive work. Raises RuntimeError showing the raw judge output.
    """
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template("{q}").invoke({"q": _PROBE_PROMPT})
    result = wrapper.generate_text(prompt)
    try:
        text = result.generations[0][0].text
    except (AttributeError, IndexError, TypeError):
        text = ""
    raw = (text or "").strip()
    try:
        parsed = _extract_json_object(raw)
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"Judge model {model_name!r} did not return parseable JSON: {e}. "
            f"Raw judge output (first 500 chars): {raw[:500]!r}. "
            "RAGAS scoring needs strict JSON output; check the judge model and "
            "endpoint (e.g. a reasoning model with thinking left on, or an "
            "endpoint that does not follow JSON instructions)."
        ) from e
    if not isinstance(parsed, dict) or parsed.get("verdict") != "yes":
        raise RuntimeError(
            f"Judge model {model_name!r} returned JSON without the expected verdict. "
            f"Raw judge output (first 500 chars): {raw[:500]!r}."
        )


def probe_judge_llm(judge_model: str = None, temperature: float | None = None):
    """Build the judge wrapper and probe it. Returns the wrapper."""
    from backend.infrastructure.config import settings

    wrapper = build_judge_llm(judge_model, temperature=temperature)
    probe_judge(wrapper, model_name=judge_model or settings.LLM_MODEL)
    return wrapper


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))


def _clean_row(row: dict) -> dict:
    from .utils import clean_score
    if not isinstance(row, dict):
        return {}
    return {k: clean_score(v) for k, v in row.items()}


def validate_scores(scores_list) -> None:
    """Raise when scoring produced no valid score at all; warn on partial loss.

    An all-NaN result means the judge produced nothing RAGAS could parse
    (or every scoring call failed) and must never be mistaken for a baseline.
    """
    if not scores_list:
        return
    total = 0
    missing = 0
    per_metric: dict[str, list[int]] = {}
    for row in scores_list:
        if not isinstance(row, dict):
            continue
        for name, value in row.items():
            total += 1
            counts = per_metric.setdefault(name, [0, 0])
            if _is_missing(value):
                missing += 1
                counts[1] += 1
            else:
                counts[0] += 1
    if total and missing == total:
        raise RuntimeError(
            "RAGAS scoring produced no valid scores: every metric is NaN for every case. "
            "The judge model returned nothing RAGAS could parse (or every scoring call failed); "
            "the pipeline results themselves may be fine. Run the judge probe (the CLI runs it "
            "automatically before scoring) and inspect the raw judge output."
        )
    if missing:
        bad = {name: f"{m}/{o + m} missing" for name, (o, m) in per_metric.items() if m}
        logger.warning("RAGAS scoring missing %d/%d scores: %s", missing, total, bad)


def _run_ragas_once(test_cases: list[dict], judge_model: str | None, temperature: float) -> list[dict]:
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from backend.infrastructure.model_factory import get_embeddings

    judge_llm = build_judge_llm(judge_model, temperature=temperature)
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]
    try:
        from ragas.metrics import FactualCorrectness
        metrics.append(FactualCorrectness(llm=judge_llm))
    except ImportError:
        pass

    samples = [
        SingleTurnSample(
            user_input=tc["user_input"],
            response=tc["response"],
            retrieved_contexts=tc["retrieved_contexts"],
            reference=tc["reference"],
        )
        for tc in test_cases
    ]
    dataset = EvaluationDataset(samples)
    # 注意（Python 3.14）：ragas 0.4.3 会把每次打分包进
    # asyncio.wait_for(..., timeout=run_config.timeout)，而 wait_for 内部
    # 进入 asyncio.timeout()；3.14 对此检查更严，执行器里会报
    # "Timeout should be used inside a task"，250 个任务瞬间全灭、分数
    # 全 NaN（3.12 上复现不出来）。timeout=None 时 wait_for 直接 await，
    # 不再进 timeout 上下文；单次调用的超时仍由 HTTP 客户端默认兜底
    #（OpenAI 客户端默认 read 10 分钟），不会无限挂起。
    result = evaluate(
        dataset=dataset, metrics=metrics, run_config=RunConfig(timeout=None)
    )
    raw_scores = getattr(result, "scores", None) or []
    return [_clean_row(row) for row in raw_scores]


def merge_score_runs(runs: list[list[dict]]) -> list[dict]:
    """多趟打分按「首次非空」合并到每个 case 的每个指标（解析重试用）。"""
    if not runs:
        return []
    n = max(len(run) for run in runs)
    merged = []
    for i in range(n):
        out: dict = {}
        for run in runs:
            if i >= len(run):
                continue
            for name, value in run[i].items():
                if name not in out or out[name] is None:
                    if value is not None:
                        out[name] = value
                    else:
                        out.setdefault(name, None)
        merged.append(out)
    return merged


def mean_score_runs(runs: list[list[dict]]) -> list[dict]:
    """多趟独立打分：每个 case 每个指标取非空均值（噪声分布的点估计）。"""
    if not runs:
        return []
    if len(runs) == 1:
        return [dict(row) for row in runs[0]]
    n = max(len(run) for run in runs)
    names = sorted({k for run in runs for row in run for k in row.keys()})
    out = []
    for i in range(n):
        row = {}
        for name in names:
            vals = [
                run[i][name]
                for run in runs
                if i < len(run) and run[i].get(name) is not None
            ]
            row[name] = round(sum(vals) / len(vals), 4) if vals else None
        out.append(row)
    return out


def _row_has_missing(row: dict) -> bool:
    return (not row) or any(v is None for v in row.values())


def compute_metrics(
    test_cases: list[dict],
    judge_model: str = None,
    *,
    temperature: float | None = None,
    parse_retries: int = DEFAULT_PARSE_RETRIES,
) -> ScoreMatrix:
    """Compute RAGAS metrics; retry full passes to fill parse failures.

    解析失败在 RAGAS 里表现为 NaN→None。整表重跑最多 parse_retries 次，
    按指标「首次非空」合并；仍空的格子在报告里标 unscored，不进均值。
    """
    pin = build_judge_pin(judge_model, temperature)
    temp = pin.temperature
    attempts = max(1, 1 + int(parse_retries))
    runs: list[list[dict]] = []

    for attempt in range(attempts):
        logger.info(
            "RAGAS scoring pass %d/%d (judge=%s temp=%s)",
            attempt + 1, attempts, pin.model, temp,
        )
        rows = _run_ragas_once(test_cases, pin.model, temp)
        runs.append(rows)
        merged = merge_score_runs(runs)
        if merged and not any(_row_has_missing(r) for r in merged):
            break
        if attempt + 1 < attempts:
            missing = sum(
                1 for r in merged for v in r.values() if v is None
            )
            logger.warning(
                "Scoring pass %d still has %d missing cells; retrying",
                attempt + 1, missing,
            )

    merged = merge_score_runs(runs)
    validate_scores(merged)
    # ScoreMatrix.runs 对「解析重试」只保留合并结果作为单次主跑，
    # 真正的噪声分布由 score_with_repeats 的外层 repeats 负责。
    return ScoreMatrix(runs=[merged], pin=pin)


def score_with_repeats(
    test_cases: list[dict],
    judge_model: str = None,
    *,
    temperature: float | None = None,
    parse_retries: int = DEFAULT_PARSE_RETRIES,
    score_repeats: int = 1,
) -> ScoreMatrix:
    """独立打分 score_repeats 遍，供噪声带宽（每遍内部仍可 parse_retries）。"""
    repeats = max(1, int(score_repeats))
    pin = build_judge_pin(judge_model, temperature)
    all_runs: list[list[dict]] = []
    for r in range(repeats):
        logger.info("Score repeat %d/%d", r + 1, repeats)
        matrix = compute_metrics(
            test_cases,
            judge_model=pin.model,
            temperature=pin.temperature,
            parse_retries=parse_retries,
        )
        all_runs.extend(matrix.runs)
    return ScoreMatrix(runs=all_runs, pin=pin)


def summary_spread(runs: list[list[dict]]) -> dict:
    """各次 run 的 per-metric 均值极差（噪声带宽）。"""
    if len(runs) < 2:
        return {}
    per_run_means = []
    for run in runs:
        means = {}
        if not run:
            per_run_means.append(means)
            continue
        names = set().union(*(row.keys() for row in run))
        for name in names:
            vals = [row[name] for row in run if row.get(name) is not None]
            means[name] = round(sum(vals) / len(vals), 4) if vals else None
        per_run_means.append(means)
    from .deterministic_metrics import metric_spread
    return metric_spread(per_run_means)
