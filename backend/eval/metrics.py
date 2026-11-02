"""RAGAS 指标计算。"""
import json
import logging
import math

from backend.infrastructure.streaming_llm import ChatOpenAICompatible

logger = logging.getLogger("ragmate")


class EvalLLM(ChatOpenAICompatible):
    """Subclass that forces n=1 (some APIs like MiMo don't support n>1)."""
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload.pop("n", None)
        return payload


def build_judge_llm(judge_model: str = None):
    """Build the RAGAS judge LLM wrapper (the construction compute_metrics uses)."""
    from ragas.llms.base import LangchainLLMWrapper
    from backend.infrastructure.model_factory import get_llm
    from backend.infrastructure.streaming_llm import create_llm
    from backend.infrastructure.config import settings

    if judge_model:
        base_llm = create_llm(
            model=judge_model,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_API_BASE_URL,
        )
    else:
        base_llm = get_llm()

    return LangchainLLMWrapper(EvalLLM(
        model=base_llm.model_name,
        api_key=base_llm.openai_api_key,
        base_url=base_llm.openai_api_base,
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


def probe_judge_llm(judge_model: str = None):
    """Build the judge wrapper and probe it. Returns the wrapper."""
    from backend.infrastructure.config import settings

    wrapper = build_judge_llm(judge_model)
    probe_judge(wrapper, model_name=judge_model or settings.LLM_MODEL)
    return wrapper


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))


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


def compute_metrics(test_cases: list[dict], judge_model: str = None):
    """Compute RAGAS metrics on evaluated test cases."""
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from backend.infrastructure.model_factory import get_embeddings

    judge_llm = build_judge_llm(judge_model)

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
    result = evaluate(dataset=dataset, metrics=metrics)
    validate_scores(getattr(result, "scores", None))
    return result
